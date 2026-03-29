"""
賭け金最適化分析
・models_2025 + 2026testデータで「変動ベット」vs「一律100円」を比較
・スコア指標(combo_diff, sub_diff, cur_gap, sub_gap)と実際の勝率の関係を分析
・ケリー基準ベースの最適賭け金も試算
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import pandas as pd, numpy as np, os, pickle, json, re, time

base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, 'models_2025')

def get_distance_band(dist):
    m = re.search(r'\d+', str(dist))
    if not m: return None
    d = int(m.group())
    if d <= 1400: return '短距離'
    elif d <= 1800: return 'マイル'
    elif d <= 2200: return '中距離'
    return '長距離'

def get_class_group(r):
    try: r = int(float(r))
    except: return '3勝以上'
    if r == 1: return '新馬'
    elif r == 2: return '未勝利'
    elif r == 3: return '1勝'
    elif r == 4: return '2勝'
    return '3勝以上'

def extract_venue(kaikai):
    m = re.search(r'\d+([^\d]+)', str(kaikai))
    return m.group(1) if m else str(kaikai)

# ── モデル読み込み ──
with open(os.path.join(model_dir, 'model_info.json'), encoding='utf-8') as f: cur_info = json.load(f)
cur_features = cur_info['features']; cur_models = cur_info['models']
with open(os.path.join(model_dir, 'ranker', 'ranker_info.json'), encoding='utf-8') as f:
    cur_rankers = json.load(f).get('rankers', {})
with open(os.path.join(model_dir, 'submodel', 'submodel_info.json'), encoding='utf-8') as f: sub_info = json.load(f)
sub_features = sub_info['features']; sub_models = sub_info['models']
with open(os.path.join(model_dir, 'submodel_ranker', 'class_ranker_info.json'), encoding='utf-8') as f:
    sub_rankers = json.load(f).get('rankers', {})
all_feats = list(set(cur_features + sub_features))

# ── テストデータ ──
test_path = os.path.join(base_dir, 'data', 'processed', 'all_venues_features_2026test.csv')
df = pd.read_csv(test_path, low_memory=False)
df['着順_num']   = pd.to_numeric(df['着順_num'],   errors='coerce')
df['単勝配当']   = pd.to_numeric(df['単勝配当'],   errors='coerce')
df['単勝オッズ'] = pd.to_numeric(df['単勝オッズ'], errors='coerce')
df['複勝配当']   = pd.to_numeric(df['複勝配当'],   errors='coerce')
df = df.dropna(subset=['着順_num', '単勝配当'])
df['target_win']   = (df['着順_num'] == 1).astype(int)
df['会場']       = df['開催'].apply(extract_venue)
df['_surface']   = df['芝・ダ'].astype(str).str.strip()
df['cur_key']    = df['会場'] + '_' + df['距離'].astype(str)
df['_dist_band'] = df['距離'].apply(get_distance_band)
mask_da = (df['_surface'] == 'ダ') & (df['_dist_band'].isin(['中距離','長距離']))
df.loc[mask_da, '_dist_band'] = '中長距離'
df['_cls_group'] = df['クラス_rank'].apply(get_class_group) if 'クラス_rank' in df.columns else '3勝以上'
df['sub_key']    = df['_surface'] + '_' + df['_dist_band'] + '_' + df['_cls_group']
for col in all_feats:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

# ── モデルキャッシュ ──
cur_mc={}; cur_rc={}; sub_mc={}; sub_rc={}
for ck in df['cur_key'].dropna().unique():
    if ck in cur_models:
        p = os.path.join(model_dir, cur_models[ck]['win'])
        if os.path.exists(p):
            with open(p,'rb') as f: m=pickle.load(f)
            cur_mc[ck]=(m,m.booster_.feature_name())
    if ck in cur_rankers:
        p = os.path.join(model_dir,'ranker',cur_rankers[ck])
        if os.path.exists(p):
            with open(p,'rb') as f: cur_rc[ck]=pickle.load(f)
for sk in df['sub_key'].dropna().unique():
    if sk in sub_models:
        p = os.path.join(model_dir,'submodel',sub_models[sk]['win'])
        if os.path.exists(p):
            with open(p,'rb') as f: m=pickle.load(f)
            sub_mc[sk]=(m,m.booster_.feature_name())
    if sk in sub_rankers:
        p = os.path.join(model_dir,'submodel_ranker',sub_rankers[sk])
        if os.path.exists(p):
            with open(p,'rb') as f: sub_rc[sk]=pickle.load(f)

# ── 予測 ──
race_keys = [c for c in ['開催','Ｒ'] if c in df.columns]
all_rows=[]
for gk, idx in df.groupby(race_keys, sort=False).groups.items():
    sub = df.loc[idx].copy()
    ck=sub['cur_key'].iloc[0]; sk=sub['sub_key'].iloc[0]
    sub['cur_score']=np.nan; sub['sub_score']=np.nan
    sub['cur_diff']=np.nan;  sub['cur_rank']=np.nan
    sub['sub_diff']=np.nan;  sub['sub_rank']=np.nan
    sub['cur_prob']=np.nan;  sub['sub_prob']=np.nan
    if ck in cur_mc:
        m,wf=cur_mc[ck]
        for c in wf:
            if c not in sub.columns: sub[c]=np.nan
        prob=m.predict_proba(sub[wf])[:,1]
        st=cur_models[ck].get('stats',{}); wm=st.get('win_mean',prob.mean()); ws=st.get('win_std',prob.std())
        cs=50+10*(prob-wm)/(ws if ws>0 else 1)
        rm=prob.mean(); rs=prob.std()
        sub['cur_score']=cs; sub['cur_diff']=50+10*(prob-rm)/(rs if rs>0 else 1)-cs
        sub['cur_prob']=prob
        if ck in cur_rc:
            sc=cur_rc[ck].predict(sub[cur_features])
            sub['cur_rank']=pd.Series(sc,index=sub.index).rank(ascending=False,method='min').astype(int)
    if sk in sub_mc:
        m,wf=sub_mc[sk]
        for c in wf:
            if c not in sub.columns: sub[c]=np.nan
        prob=m.predict_proba(sub[wf])[:,1]
        st=sub_models[sk].get('stats',{}); wm=st.get('win_mean',prob.mean()); ws=st.get('win_std',prob.std())
        cs=50+10*(prob-wm)/(ws if ws>0 else 1)
        rm=prob.mean(); rs=prob.std()
        sub['sub_score']=cs; sub['sub_diff']=50+10*(prob-rm)/(rs if rs>0 else 1)-cs
        sub['sub_prob']=prob
        if sk in sub_rc:
            sc=sub_rc[sk].predict(sub[wf])
            sub['sub_rank']=pd.Series(sc,index=sub.index).rank(ascending=False,method='min').astype(int)
    all_rows.append(sub)

result = pd.concat(all_rows, ignore_index=True)

cd=pd.to_numeric(result['cur_diff'],errors='coerce')
sd=pd.to_numeric(result['sub_diff'],errors='coerce')
cr=pd.to_numeric(result['cur_rank'],errors='coerce')
sr=pd.to_numeric(result['sub_rank'],errors='coerce')
result['combo_diff']=cd.fillna(0)+sd.fillna(0)
n_h=result.groupby(race_keys)['cur_rank'].transform('count')
result['rank_sum']=cr.fillna(n_h+1)+sr.fillna(n_h+1)
result['combo_rank']=result.groupby(race_keys)['combo_diff'].rank(ascending=False,method='min')

# cur_gap / sub_gap 計算
for key, grp in result.groupby(race_keys):
    idx=grp.index
    for col, gcol in [('cur_score','cur_gap'),('sub_score','sub_gap')]:
        scores=grp[col].dropna().sort_values(ascending=False).values
        gap = scores[0]-scores[1] if len(scores)>=2 else np.nan
        result.loc[idx,gcol]=gap

# ── 分析対象: 両Rnk=1 ──
base = result[(cr==1)&(sr==1)].copy()
base['sd_val'] = sd[base.index]
base['cd_val'] = cd[base.index]
base['cur_gap_v'] = pd.to_numeric(base['cur_gap'], errors='coerce')
base['sub_gap_v'] = pd.to_numeric(base['sub_gap'], errors='coerce')
base['combo_gap'] = base['cur_gap_v'].fillna(0) + base['sub_gap_v'].fillna(0)
base['odds'] = pd.to_numeric(base['単勝オッズ'], errors='coerce')

print("="*65)
print("  賭け金最適化分析  (両Rnk=1 ベース, N={})".format(len(base)))
print("="*65)

# ── ① sub_diff 帯別の勝率・オッズ平均 ──
print("\n【① sub_diff 帯別 勝率・平均オッズ・単勝ROI】")
bins = [(-99,0,'sd<0'), (0,10,'0≤sd<10'), (10,20,'10≤sd<20'), (20,30,'20≤sd<30'), (30,99,'sd≥30')]
for lo,hi,label in bins:
    m = (base['sd_val']>=lo) & (base['sd_val']<hi)
    s = base[m]
    if len(s)==0: continue
    wr = s['target_win'].mean()
    ao = s['odds'].mean()
    roi = s.loc[s['target_win']==1,'単勝配当'].sum()/(len(s)*100)-1
    print(f"  {label:<15} N={len(s):>4}  勝率{wr:.1%}  平均オッズ{ao:.1f}倍  ROI{roi:>+7.1%}")

# ── ② combo_gap 帯別 ──
print("\n【② combo_gap（1位と2位のスコア差）帯別】")
pgbins = [(0,5,'gap<5'), (5,10,'5≤gap<10'), (10,20,'10≤gap<20'), (20,99,'gap≥20')]
for lo,hi,label in pgbins:
    m = (base['combo_gap']>=lo) & (base['combo_gap']<hi)
    s = base[m]
    if len(s)==0: continue
    wr = s['target_win'].mean()
    ao = s['odds'].mean()
    roi = s.loc[s['target_win']==1,'単勝配当'].sum()/(len(s)*100)-1
    print(f"  {label:<15} N={len(s):>4}  勝率{wr:.1%}  平均オッズ{ao:.1f}倍  ROI{roi:>+7.1%}")

# ── ③ 変動ベット: sub_diff + combo_gap でティア分け ──
print("\n【③ 変動ベット シミュレーション（100/200/300円ティア）】")
def assign_tier(row):
    sd_v = row['sd_val']
    cg   = row['combo_gap']
    if pd.isna(cg): cg = 0
    # Tier3(300円): sd≥20 かつ gap≥10
    if sd_v >= 20 and cg >= 10: return 300
    # Tier2(200円): sd≥10 かつ gap≥5
    if sd_v >= 10 and cg >= 5:  return 200
    # Tier1(100円): それ以外
    return 100

base['bet'] = base.apply(assign_tier, axis=1)

for tier in [100,200,300]:
    s = base[base['bet']==tier]
    if len(s)==0: continue
    invest = len(s)*tier
    ret    = s.loc[s['target_win']==1,'単勝配当'].mul(tier/100).sum()
    roi    = ret/invest-1
    wr     = s['target_win'].mean()
    print(f"  Tier {tier}円: N={len(s):>4}  勝率{wr:.1%}  投資{invest:,}円  回収{int(ret):,}円  ROI{roi:>+7.1%}")

invest_flat = len(base)*100
ret_flat    = base.loc[base['target_win']==1,'単勝配当'].sum()
roi_flat    = ret_flat/invest_flat-1
invest_var  = base['bet'].sum()
ret_var     = base.apply(lambda r: r['単勝配当']*(r['bet']/100) if r['target_win']==1 else 0, axis=1).sum()
roi_var     = ret_var/invest_var-1
print(f"\n  一律100円:  投資{invest_flat:,}円  回収{int(ret_flat):,}円  ROI{roi_flat:>+7.1%}")
print(f"  変動ベット: 投資{int(invest_var):,}円  回収{int(ret_var):,}円  ROI{roi_var:>+7.1%}")
print(f"  → 回収差: {int(ret_var-ret_flat):+,}円  投資差: {int(invest_var-invest_flat):+,}円")

# ── ④ ケリー基準（簡易版） ──
print("\n【④ ケリー基準シミュレーション（バンク1万円）】")
print("  ケリー比率 = (p*o - 1)/(o - 1),  p=cur_prob+sub_prob平均, o=単勝オッズ")
base['avg_prob'] = (pd.to_numeric(base['cur_prob'],errors='coerce').fillna(0)
                  + pd.to_numeric(base['sub_prob'],errors='coerce').fillna(0)) / 2
base['kelly'] = (base['avg_prob']*base['odds'] - 1) / (base['odds'] - 1).clip(lower=0.01)
base['kelly'] = base['kelly'].clip(lower=0, upper=0.25)  # 上限25%

bank = 10000
# ケリーベット(バンクの一定割合, 最小100円単位)
base['kelly_bet'] = (base['kelly'] * bank / 100).round(0) * 100
base['kelly_bet'] = base['kelly_bet'].clip(lower=100, upper=1000)

invest_k = base['kelly_bet'].sum()
ret_k    = base.apply(lambda r: r['単勝配当']*(r['kelly_bet']/100) if r['target_win']==1 else 0, axis=1).sum()
roi_k    = ret_k/invest_k-1
print(f"  ケリーベット: 投資{int(invest_k):,}円  回収{int(ret_k):,}円  ROI{roi_k:>+7.1%}")
print(f"  賭け金分布: min={int(base['kelly_bet'].min())}  median={int(base['kelly_bet'].median())}  max={int(base['kelly_bet'].max())}円")

# ケリー×ティア組合わせ
print("\n【⑤ 推奨ルール案まとめ】")
print("  条件: 両Rnk=1")
print(f"  賭け金目安:")
print(f"    100円: sd<10  または combo_gap<5  (自信低め)")
print(f"    200円: sd≥10 かつ combo_gap≥5   (標準)")
print(f"    300円: sd≥20 かつ combo_gap≥10  (高確信)")
print(f"  ※オッズ1〜2倍台は見送りも検討")

print("\n完了")

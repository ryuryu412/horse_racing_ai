"""
複勝 深掘り分析
・sub_gap / combo_gap / オッズ帯 の組み合わせで最適条件を探す
・単勝+複勝の組み合わせベットも試算
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import pandas as pd, numpy as np, os, pickle, json, re

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

with open(os.path.join(model_dir, 'model_info.json'), encoding='utf-8') as f: cur_info = json.load(f)
cur_features = cur_info['features']; cur_models = cur_info['models']
with open(os.path.join(model_dir, 'ranker', 'ranker_info.json'), encoding='utf-8') as f:
    cur_rankers = json.load(f).get('rankers', {})
with open(os.path.join(model_dir, 'submodel', 'submodel_info.json'), encoding='utf-8') as f: sub_info = json.load(f)
sub_features = sub_info['features']; sub_models = sub_info['models']
with open(os.path.join(model_dir, 'submodel_ranker', 'class_ranker_info.json'), encoding='utf-8') as f:
    sub_rankers = json.load(f).get('rankers', {})
all_feats = list(set(cur_features + sub_features))

test_path = os.path.join(base_dir, 'data', 'processed', 'all_venues_features_2026test.csv')
df = pd.read_csv(test_path, low_memory=False)
df['着順_num']   = pd.to_numeric(df['着順_num'],   errors='coerce')
df['単勝配当']   = pd.to_numeric(df['単勝配当'],   errors='coerce')
df['単勝オッズ'] = pd.to_numeric(df['単勝オッズ'], errors='coerce')
df['複勝配当']   = pd.to_numeric(df['複勝配当'],   errors='coerce')
df = df.dropna(subset=['着順_num', '単勝配当'])
df['target_win']   = (df['着順_num'] == 1).astype(int)
df['target_place'] = (df['着順_num'] <= 3).astype(int)
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

race_keys = [c for c in ['開催','Ｒ'] if c in df.columns]
all_rows=[]
for gk, idx in df.groupby(race_keys, sort=False).groups.items():
    sub = df.loc[idx].copy()
    ck=sub['cur_key'].iloc[0]; sk=sub['sub_key'].iloc[0]
    sub['cur_score']=np.nan; sub['sub_score']=np.nan
    sub['cur_diff']=np.nan;  sub['cur_rank']=np.nan
    sub['sub_diff']=np.nan;  sub['sub_rank']=np.nan
    if ck in cur_mc:
        m,wf=cur_mc[ck]
        for c in wf:
            if c not in sub.columns: sub[c]=np.nan
        prob=m.predict_proba(sub[wf])[:,1]
        st=cur_models[ck].get('stats',{}); wm=st.get('win_mean',prob.mean()); ws=st.get('win_std',prob.std())
        cs=50+10*(prob-wm)/(ws if ws>0 else 1)
        rm=prob.mean(); rs=prob.std()
        sub['cur_score']=cs; sub['cur_diff']=50+10*(prob-rm)/(rs if rs>0 else 1)-cs
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
        if sk in sub_rc:
            sc=sub_rc[sk].predict(sub[wf])
            sub['sub_rank']=pd.Series(sc,index=sub.index).rank(ascending=False,method='min').astype(int)
    all_rows.append(sub)

result = pd.concat(all_rows, ignore_index=True)
cr=pd.to_numeric(result['cur_rank'],errors='coerce')
sr=pd.to_numeric(result['sub_rank'],errors='coerce')
sd=pd.to_numeric(result['sub_diff'],errors='coerce')

for key, grp in result.groupby(race_keys):
    idx=grp.index
    for col, gcol in [('cur_score','cur_gap'),('sub_score','sub_gap')]:
        scores=grp[col].dropna().sort_values(ascending=False).values
        gap = scores[0]-scores[1] if len(scores)>=2 else np.nan
        result.loc[idx,gcol]=gap

base = result[(cr==1)&(sr==1)].copy()
base['sd_val']    = sd[base.index]
base['odds']      = pd.to_numeric(base['単勝オッズ'], errors='coerce')
base['cur_gap_v'] = pd.to_numeric(base['cur_gap'], errors='coerce')
base['sub_gap_v'] = pd.to_numeric(base['sub_gap'], errors='coerce')
base['combo_gap'] = base['cur_gap_v'].fillna(0) + base['sub_gap_v'].fillna(0)

def row(label, s, bet=100):
    n=len(s)
    if n==0: return
    hits=int(s['target_place'].sum())
    ret=s.loc[s['target_place']==1,'複勝配当'].mul(bet/100).sum()
    roi=ret/(n*bet)-1
    avg_f=s.loc[s['target_place']==1,'複勝配当'].mean() if hits>0 else 0
    print(f'  {label:<40} N={n:>3}  複勝率{hits/n:.1%}  平均配当{avg_f:>4.0f}円  ROI{roi:>+7.1%}')

# ── ① sub_gap × オッズ帯 ──
print('='*72)
print('  【複勝】sub_gap × 単勝オッズ帯（両Rnk=1ベース）')
print('='*72)
for sg in [0, 5, 8, 10, 12]:
    for olo, ohi, olabel in [(3,5,'odds3〜4'),(5,10,'odds5〜9'),(3,10,'odds3〜9'),(3,999,'odds≥3')]:
        s = base[(base['sub_gap_v']>=sg)&(base['odds']>=olo)&(base['odds']<ohi)]
        if len(s)<5: continue
        row(f'sub_gap≥{sg} & {olabel}', s)
    print()

# ── ② combo_gap × sd ──
print('='*72)
print('  【複勝】combo_gap × sd（両Rnk=1 & odds≥3）')
print('='*72)
b3 = base[base['odds']>=3]
for cg in [10, 15, 20]:
    for sv in [0, 5, 10]:
        s = b3[(b3['combo_gap']>=cg)&(b3['sd_val']>=sv)]
        if len(s)<5: continue
        row(f'combo_gap≥{cg} & sd≥{sv}', s)
    print()

# ── ③ 推奨ルール: 変動ベット ──
print('='*72)
print('  【複勝 推奨変動ベット】')
print('='*72)

# ルールA: sub_gap基準
print('\n  ルールA: sub_gap基準（odds≥3）')
print('    300円: sub_gap≥10')
print('    200円: sub_gap≥5')
print('    見送り: sub_gap<5')
rA = base[base['odds']>=3].copy()
rA['bet'] = rA['sub_gap_v'].apply(lambda x: 300 if x>=10 else (200 if x>=5 else 0))
rA = rA[rA['bet']>0]
invest_A = rA['bet'].sum()
ret_A = rA.apply(lambda r: r['複勝配当']*(r['bet']/100) if r['target_place']==1 else 0, axis=1).sum()
roi_A = ret_A/invest_A-1
print(f'  投資{int(invest_A):,}円  回収{int(ret_A):,}円  損益{int(ret_A-invest_A):+,}円  ROI{roi_A:>+7.1%}')
for t in [200,300]:
    s=rA[rA['bet']==t]; n=len(s)
    if n==0: continue
    hits=int(s['target_place'].sum())
    ret_t=s.loc[s['target_place']==1,'複勝配当'].mul(t/100).sum()
    avg_f=s.loc[s['target_place']==1,'複勝配当'].mean() if hits>0 else 0
    print(f'    {t}円: N={n:>3}  複勝率{hits/n:.1%}  平均配当{avg_f:.0f}円  ROI{ret_t/(n*t)-1:>+7.1%}')

# ルールB: combo_gap+sd基準
print('\n  ルールB: combo_gap+sd基準（odds≥3）')
print('    300円: sd≥10 & combo_gap≥15')
print('    200円: sd≥10 & combo_gap≥5')
print('    見送り: それ以外')
rB = base[base['odds']>=3].copy()
def bet_B(row):
    if row['sd_val']>=10 and row['combo_gap']>=15: return 300
    if row['sd_val']>=10 and row['combo_gap']>=5:  return 200
    return 0
rB['bet'] = rB.apply(bet_B, axis=1)
rB = rB[rB['bet']>0]
invest_B = rB['bet'].sum()
ret_B = rB.apply(lambda r: r['複勝配当']*(r['bet']/100) if r['target_place']==1 else 0, axis=1).sum()
roi_B = ret_B/invest_B-1
print(f'  投資{int(invest_B):,}円  回収{int(ret_B):,}円  損益{int(ret_B-invest_B):+,}円  ROI{roi_B:>+7.1%}')
for t in [200,300]:
    s=rB[rB['bet']==t]; n=len(s)
    if n==0: continue
    hits=int(s['target_place'].sum())
    ret_t=s.loc[s['target_place']==1,'複勝配当'].mul(t/100).sum()
    avg_f=s.loc[s['target_place']==1,'複勝配当'].mean() if hits>0 else 0
    print(f'    {t}円: N={n:>3}  複勝率{hits/n:.1%}  平均配当{avg_f:.0f}円  ROI{ret_t/(n*t)-1:>+7.1%}')

# ── ④ 単勝+複勝 組み合わせ ──
print()
print('='*72)
print('  【単勝+複勝 組み合わせベット】両Rnk=1 & odds≥3')
print('='*72)
combos = [
    ('単勝200+複勝100(全部)',      lambda r: (200,100),   lambda r: True),
    ('単勝200+複勝100(sd≥10)',    lambda r: (200,100),   lambda r: r['sd_val']>=10),
    ('単勝200+複勝200(sd≥10 & sub_gap≥8)', lambda r:(200,200), lambda r: r['sd_val']>=10 and r['sub_gap_v']>=8),
]
for label, bet_fn, cond_fn in combos:
    s = base[base['odds']>=3].copy()
    s = s[s.apply(cond_fn, axis=1)]
    if len(s)==0: continue
    n=len(s)
    tan_bet, fuku_bet = bet_fn(s.iloc[0])
    invest_c = n*(tan_bet+fuku_bet)
    ret_tan  = s.loc[s['target_win']==1,'単勝配当'].mul(tan_bet/100).sum()
    ret_fuku = s.loc[s['target_place']==1,'複勝配当'].mul(fuku_bet/100).sum()
    ret_c    = ret_tan+ret_fuku
    roi_c    = ret_c/invest_c-1
    wr=s['target_win'].mean(); hr=s['target_place'].mean()
    print(f'  {label}')
    print(f'    N={n}  投資{invest_c:,}円  回収{int(ret_c):,}円  損益{int(ret_c-invest_c):+,}円  ROI{roi_c:>+7.1%}')
    print(f'    (単勝率{wr:.1%} / 複勝率{hr:.1%})')
    print()

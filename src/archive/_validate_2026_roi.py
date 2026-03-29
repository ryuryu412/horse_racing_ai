"""
2026年データで2025年モデルのROI検証
・models_2025/ のモデルを使用
・data/processed/all_venues_features_2026test.csv でテスト
・現行モデルは一切使用しない
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import pandas as pd, numpy as np, os, pickle, json, re, time

base_dir  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
print(f"モデル読み込み: {model_dir}")
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
print(f"テストデータ読み込み: {test_path}")
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
print(f"テストデータ: {len(df):,}頭")

# ── モデルキャッシュ ──
print("モデルプリロード中...")
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
print(f"プリロード完了: cur={len(cur_mc)} sub={len(sub_mc)}")

# ── 予測 ──
race_keys = [c for c in ['開催','Ｒ'] if c in df.columns]
print("予測中...")
t0 = time.time(); all_rows=[]
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
print(f"予測完了: {len(result):,}頭 ({time.time()-t0:.0f}秒)")

# ── コンポジット ──
cd=pd.to_numeric(result['cur_diff'],errors='coerce')
sd=pd.to_numeric(result['sub_diff'],errors='coerce')
cr=pd.to_numeric(result['cur_rank'],errors='coerce')
sr=pd.to_numeric(result['sub_rank'],errors='coerce')
result['combo_diff']=cd.fillna(0)+sd.fillna(0)
n_h=result.groupby(race_keys)['cur_rank'].transform('count')
result['rank_sum']=cr.fillna(n_h+1)+sr.fillna(n_h+1)
result['combo_rank']=result.groupby(race_keys)['combo_diff'].rank(ascending=False,method='min')

# ── ROI集計 ──
def roi_table(label, mask):
    sub = result[mask]
    n=len(sub)
    if n==0: return f"  {label}: 該当なし"
    wins=int(sub['target_win'].sum())
    ret=sub.loc[sub['target_win']==1,'単勝配当'].sum()
    roi=ret/(n*100)-1
    place_hits=int(sub['target_place'].sum())
    place_ret=sub.loc[sub['target_place']==1,'複勝配当'].sum()
    place_roi=place_ret/(n*100)-1
    return (f"  {label:<35} N={n:>4}  単勝:{wins:>3}頭 ROI{roi:>+7.1%}"
            f"  複勝:{place_hits:>3}頭 ROI{place_roi:>+7.1%}")

print("\n" + "="*65)
print("  2026年1〜3月 ROI検証（models_2025 使用）")
print("  ※モデルは2026年データを一切見ていない")
print("="*65)

# ── 指数の分布確認 ──
print("\n【指数分布（combo_rank=1のみ）】")
top = result[result['combo_rank']==1].copy()
top['combo_diff'] = pd.to_numeric(top['combo_diff'], errors='coerce')
top['cur_diff2']  = pd.to_numeric(top['cur_diff'],  errors='coerce')
top['sub_diff2']  = pd.to_numeric(top['sub_diff'],  errors='coerce')
for col, label in [('combo_diff','combo_diff'),('cur_diff2','cur_diff'),('sub_diff2','sub_diff')]:
    s = top[col].dropna()
    print(f"  {label:<12} 平均{s.mean():+6.1f}  中央{s.median():+6.1f}  p75={s.quantile(.75):+6.1f}  max={s.max():+6.1f}")

# 月別
result['月'] = pd.to_numeric(result['日付_num'], errors='coerce') // 100 % 100

# ── グリッドサーチ：ROI上位を列挙 ──
print("\n【グリッドサーチ（N≥20、ROI降順TOP20）】")
odds = pd.to_numeric(result['単勝オッズ'], errors='coerce')
combo_diff = pd.to_numeric(result['combo_diff'], errors='coerce')
results_list = []

rank_conds = [
    ("combo=1",        result['combo_rank']==1),
    ("両Rnk=1",        (cr==1)&(sr==1)),
    ("combo=1&両Rnk=1",(result['combo_rank']==1)&(cr==1)&(sr==1)),
    ("rankSum≤3",      result['rank_sum']<=3),
    ("rankSum≤4",      result['rank_sum']<=4),
]
sd_thresholds  = [0, 5, 8, 10, 12, 15, 20]
cd_thresholds  = [0, 5, 10, 15, 20]
odds_thresholds= [(1,999,'全'), (2,999,'odds≥2'), (3,999,'odds≥3'), (3,10,'3〜9倍'), (2,8,'2〜7倍')]

for rname, rmask in rank_conds:
    for sd_th in sd_thresholds:
        for cd_th in cd_thresholds:
            for olo, ohi, oname in odds_thresholds:
                mask = rmask & (sd>=sd_th) & (combo_diff>=cd_th) & (odds>=olo) & (odds<ohi)
                sub  = result[mask]
                n    = len(sub)
                if n < 20: continue
                wins = int(sub['target_win'].sum())
                ret  = sub.loc[sub['target_win']==1,'単勝配当'].sum()
                roi  = ret/(n*100)-1
                label= f"{rname} sd≥{sd_th} cd≥{cd_th} {oname}"
                results_list.append((roi, n, wins, label))

results_list.sort(reverse=True)
for roi, n, wins, label in results_list[:20]:
    print(f"  {label:<45} N={n:>4}  的中{wins:>3}頭  ROI{roi:>+7.1%}")

# ── 偏差値の差（cur_diff / sub_diff）別分析 ──
print("\n【cur_diff vs sub_diff の組み合わせ（combo=1）】")
base = result['combo_rank']==1
cd_arr = pd.to_numeric(result['cur_diff'], errors='coerce')
sd_arr = pd.to_numeric(result['sub_diff'], errors='coerce')
for cd_th in [0, 5, 8, 10, 15]:
    for sd_th in [0, 5, 8, 10, 15]:
        mask = base & (cd_arr>=cd_th) & (sd_arr>=sd_th)
        sub  = result[mask]
        n    = len(sub)
        if n < 15: continue
        wins = int(sub['target_win'].sum())
        ret  = sub.loc[sub['target_win']==1,'単勝配当'].sum()
        roi  = ret/(n*100)-1
        print(f"  cur≥{cd_th:>2} & sub≥{sd_th:>2}   N={n:>4}  的中{wins:>3}頭  ROI{roi:>+7.1%}")

# ── オッズ帯別（最良条件で） ──
print("\n【オッズ帯別（両Rnk=1 & sd≥10）】")
base_mask2 = (cr==1)&(sr==1)&(sd>=10)
for lo, hi, label in [(1,2,'1倍台'),(2,3,'2倍台'),(3,5,'3〜4倍'),(5,10,'5〜9倍'),(10,999,'10倍以上')]:
    mask = base_mask2 & (odds>=lo) & (odds<hi)
    print(roi_table(label, mask))

print("\n【月別（両Rnk=1 & odds≥3 & sd≥10）】")
best_mask = (cr==1)&(sr==1)&(sd>=10)&(odds>=3)
for mo in sorted(result['月'].dropna().unique()):
    sub = result[result['月']==mo & best_mask] if False else result[(result['月']==mo) & best_mask]
    n=len(sub)
    if n==0: continue
    wins=int(sub['target_win'].sum())
    ret=sub.loc[sub['target_win']==1,'単勝配当'].sum()
    roi=ret/(n*100)-1
    print(f"  2026年{int(mo):02d}月  N={n}  的中{wins}  ROI{roi:+.1%}")


# ── 印ティア設計 ──
print("\n" + "="*65)
print("  印ティア設計案（ver2）")
print("="*65)

odds   = pd.to_numeric(result['単勝オッズ'], errors='coerce')
cd_arr = pd.to_numeric(result['cur_diff'],   errors='coerce')
sd_arr = pd.to_numeric(result['sub_diff'],   errors='coerce')
cm_arr = pd.to_numeric(result['combo_diff'], errors='coerce')

# ── ver2: オッズ帯を軸に設計 ──
# 激熱: 両モデルで圧倒 & 5〜9倍（最も旨味のある帯）
# ◎  : 両モデルで圧倒 & 3〜4倍（配当は低いが勝率高い）
# 〇  : 両モデルで圧倒 & 10倍以上（爆発力狙い、少額）
# ☆  : ranksum低くスコア有望 & 5〜19倍（穴馬枠）

base_strong = (result['combo_rank']==1) & (cd_arr>=8) & (sd_arr>=8)

tier_激熱 = base_strong & (odds>=5) & (odds<10)
tier_hon  = base_strong & (odds>=3) & (odds<5)
tier_maru = base_strong & (odds>=10) & (odds<20)
tier_hoshi= (result['rank_sum']<=4) & (sd_arr>=5) & (odds>=5) & (odds<20) \
            & ~tier_激熱 & ~tier_hon & ~tier_maru

tiers = [
    ("激熱(5〜9倍)", tier_激熱),
    ("◎ (3〜4倍)", tier_hon),
    ("〇 (10〜19倍)", tier_maru),
    ("☆ (穴5〜19倍)", tier_hoshi),
]

def show_tier(name, mask):
    sub   = result[mask]
    n     = len(sub)
    if n == 0:
        print(f"  {name:<15}  該当なし")
        return
    wins  = int(sub['target_win'].sum())
    ret_w = sub.loc[sub['target_win']==1,'単勝配当'].sum()
    roi_w = ret_w/(n*100)-1
    places= int(sub['target_place'].sum())
    ret_p = sub.loc[sub['target_place']==1,'複勝配当'].sum()
    roi_p = ret_p/(n*100)-1
    avg_o = sub['単勝オッズ'].dropna().mean()
    per_race_n = n / result['月'].nunique() if result['月'].nunique() > 0 else n
    print(f"  {name:<15}  N={n:>4}  勝率{wins/n:>5.1%}  単勝ROI{roi_w:>+7.1%}  複勝ROI{roi_p:>+7.1%}  avg_odds={avg_o:.1f}  週平均{per_race_n/4:.1f}頭")

print(f"\n{'印':<15}  {'N':>5}  {'勝率':>6}  {'単勝ROI':>8}  {'複勝ROI':>8}  {'avg_odds':>8}  週平均")
print("-"*75)
for name, mask in tiers:
    show_tier(name, mask)

# 月別安定性
print("\n【月別安定性】")
for name, mask in tiers:
    line = f"  {name:<15}"
    for mo in sorted(result['月'].dropna().unique()):
        sub=result[(result['月']==mo)&mask]; n=len(sub)
        if n==0:
            line += f"  {int(mo):02d}月:N=0      "
            continue
        wins=int(sub['target_win'].sum())
        ret=sub.loc[sub['target_win']==1,'単勝配当'].sum()
        roi=ret/(n*100)-1
        line += f"  {int(mo):02d}月:N={n} ROI{roi:>+6.1%}"
    print(line)

# 参考: 単勝のみ vs 単勝+複勝 収支シミュレーション
print("\n【収支シミュレーション（3ヶ月想定）】")
print(f"  {'条件':<20}  {'単勝賭け金':>8}  {'複勝賭け金':>8}  {'合計投資':>8}  {'想定収益':>8}  {'純損益':>8}")
print("-"*75)
sims = [
    ("激熱(5〜9倍)", tier_激熱, 300, 200),
    ("◎ (3〜4倍)", tier_hon,   200,   0),
    ("〇 (10〜19倍)", tier_maru, 100, 100),
    ("☆ (穴5〜19倍)", tier_hoshi,100,   0),
]
total_invest=0; total_return=0
for name, mask, bet_w, bet_p in sims:
    sub=result[mask]; n=len(sub)
    if n==0: continue
    invest = n*(bet_w+bet_p)
    ret_w  = sub.loc[sub['target_win']==1,'単勝配当'].sum() * bet_w / 100
    ret_p  = sub.loc[sub['target_place']==1,'複勝配当'].sum() * bet_p / 100 if bet_p>0 else 0
    total_invest += invest; total_return += ret_w+ret_p
    profit = ret_w+ret_p-invest
    print(f"  {name:<20}  {bet_w:>7}円  {bet_p:>7}円  {invest:>7}円  {ret_w+ret_p:>7.0f}円  {profit:>+7.0f}円")
print(f"  {'合計':<20}  {'':>8}  {'':>8}  {total_invest:>7}円  {total_return:>7.0f}円  {total_return-total_invest:>+7.0f}円")


# ── 複数頭買い検討 ──
print("\n" + "="*65)
print("  複数頭買い検討")
print("="*65)

# combo_rank=2 単体の実力確認
print("\n【combo_rank=2 単体（オッズ帯別）】")
for lo, hi, lbl in [(1,3,'1〜2倍'),(3,5,'3〜4倍'),(5,10,'5〜9倍'),(10,20,'10〜19倍'),(20,999,'20倍以上')]:
    mask = (result['combo_rank']==2) & (cd_arr>=8) & (sd_arr>=8) & (odds>=lo) & (odds<hi)
    sub=result[mask]; n=len(sub)
    if n<5: continue
    wins=int(sub['target_win'].sum())
    ret=sub.loc[sub['target_win']==1,'単勝配当'].sum()
    roi=ret/(n*100)-1
    print(f"  {lbl:<10}  N={n:>4}  勝率{wins/n:>5.1%}  単勝ROI{roi:>+7.1%}")

# 同一レース内で rank=1 と rank=2 を両方買った場合
print("\n【同一レース内 1＋2頭買い（combo_rank≤2 & cur≥8 & sub≥8 & odds≥3）】")
base = (result['combo_rank']<=2) & (cd_arr>=8) & (sd_arr>=8) & (odds>=3)
for lo, hi, lbl in [(3,5,'3〜4倍'),(5,10,'5〜9倍'),(10,20,'10〜19倍'),(3,20,'3〜19倍合計')]:
    mask = base & (odds>=lo) & (odds<hi)
    sub=result[mask]; n=len(sub)
    if n<5: continue
    wins=int(sub['target_win'].sum())
    ret=sub.loc[sub['target_win']==1,'単勝配当'].sum()
    roi=ret/(n*100)-1
    places=int(sub['target_place'].sum())
    ret_p=sub.loc[sub['target_place']==1,'複勝配当'].sum()
    roi_p=ret_p/(n*100)-1
    print(f"  {lbl:<12}  N={n:>4}  勝率{wins/n:>5.1%}  単勝ROI{roi:>+7.1%}  複勝ROI{roi_p:>+7.1%}")

# rank=1が条件を満たすレースでのrank=2の成績（rank=1が激熱条件のレース限定）
print("\n【激熱レース(rank=1が5〜9倍)でのrank=2の成績】")
激熱レース = result[tier_激熱][race_keys].drop_duplicates()
rank2_in_激熱 = result.merge(激熱レース, on=race_keys).query('combo_rank==2')
for lo, hi, lbl in [(3,10,'3〜9倍'),(5,10,'5〜9倍'),(3,999,'全オッズ')]:
    mask = (rank2_in_激熱['単勝オッズ']>=lo) & (rank2_in_激熱['単勝オッズ']<hi)
    sub=rank2_in_激熱[mask]; n=len(sub)
    if n<3: continue
    wins=int(sub['target_win'].sum())
    ret=sub.loc[sub['target_win']==1,'単勝配当'].sum()
    roi=ret/(n*100)-1 if n>0 else 0
    print(f"  {lbl:<10}  N={n:>4}  勝率{wins/n:>5.1%}  単勝ROI{roi:>+7.1%}")

# レース内で複数印が出る頻度
print("\n【1レースに複数印が出る頻度（激熱＋☆）】")
result['has_激熱'] = tier_激熱.astype(int)
result['has_hoshi'] = tier_hoshi.astype(int)
per_race = result.groupby(race_keys).agg(
    n_激熱=('has_激熱','sum'),
    n_hoshi=('has_hoshi','sum')
).reset_index()
print(f"  激熱のみ出るレース:         {(per_race['n_激熱']>=1).sum()}レース")
print(f"  激熱＋☆が同一レースに出る: {((per_race['n_激熱']>=1)&(per_race['n_hoshi']>=1)).sum()}レース")
print(f"  ☆のみ出るレース:           {((per_race['n_激熱']==0)&(per_race['n_hoshi']>=1)).sum()}レース")

# 激熱と☆を同一レースで両方買う戦略
print("\n【激熱＋☆同時買い（同一レース内）】")
激熱_races = result[tier_激熱][race_keys].drop_duplicates()
hoshi_in_激熱_race = result.merge(激熱_races, on=race_keys)[tier_hoshi.reindex(result.merge(激熱_races, on=race_keys).index, fill_value=False)]
n=len(hoshi_in_激熱_race)
if n>0:
    wins=int(hoshi_in_激熱_race['target_win'].sum())
    ret=hoshi_in_激熱_race.loc[hoshi_in_激熱_race['target_win']==1,'単勝配当'].sum()
    roi=ret/(n*100)-1
    print(f"  激熱レース内の☆: N={n}  勝率{wins/n:.1%}  単勝ROI{roi:>+.1%}")
else:
    print("  該当なし")

# 最終推奨まとめ
print("\n" + "="*65)
print("  最終比較：1頭集中 vs 分散買い")
print("="*65)
strategies = [
    ("1頭集中: 激熱のみ",
     tier_激熱, 300, 0),
    ("1頭集中: 激熱+☆",
     tier_激熱 | tier_hoshi, 200, 0),
    ("分散: combo≤2 & cur≥8 & sub≥8 & 5〜9倍",
     (result['combo_rank']<=2)&(cd_arr>=8)&(sd_arr>=8)&(odds>=5)&(odds<10), 200, 0),
]
for label, mask, bet, _ in strategies:
    sub=result[mask]; n=len(sub)
    if n==0: continue
    wins=int(sub['target_win'].sum())
    ret=sub.loc[sub['target_win']==1,'単勝配当'].sum()
    roi=ret/(n*100)-1
    invest=n*bet
    profit=ret*bet/100-invest
    print(f"  {label}")
    print(f"    N={n}  勝率{wins/n:.1%}  ROI{roi:>+.1%}  投資{invest}円  純損益{profit:>+.0f}円")


# ── 賭け金シミュレーション ──
print("\n" + "="*65)
print("  賭け金シミュレーション（3ヶ月実績ベース）")
print("="*65)

tiers_final = [
    ("激熱(5〜9倍)",   tier_激熱),
    ("◎ (3〜4倍)",   tier_hon),
    ("〇 (10〜19倍)", tier_maru),
    ("☆ (穴5〜19倍)", tier_hoshi),
]

# ケリー基準も表示
print("\n【ケリー基準（参考）】")
for name, mask in tiers_final:
    sub=result[mask]; n=len(sub)
    if n==0: continue
    win_rate = sub['target_win'].sum()/n
    avg_odds = sub['単勝オッズ'].dropna().mean()
    b = avg_odds - 1
    kelly = (b * win_rate - (1-win_rate)) / b if b>0 else 0
    half_kelly = max(kelly/2, 0)
    print(f"  {name:<15}  勝率{win_rate:.1%}  平均odds{avg_odds:.1f}  Kelly={kelly:.1%}  半Kelly={half_kelly:.1%}")

# 複数の賭け金パターンでシミュレーション
print("\n【賭け金パターン別 3ヶ月収支】")
print(f"  {'パターン':<35}  {'合計投資':>8}  {'合計収益':>8}  {'純損益':>9}  {'ROI':>7}")
print("-"*75)

bet_patterns = [
    ("現状(激熱300/☆100/◎100/〇100)",  [300, 100, 100, 100]),
    ("小増し(激熱500/☆200/◎100/〇100)", [500, 100, 200, 100]),
    ("中増し(激熱1000/☆300/◎200/〇200)",[1000,200, 300, 200]),
    ("激熱重点(激熱2000/☆500/◎300/〇300)",[2000,300,500,300]),
    ("均等増し(激熱1000/☆500/◎500/〇500)",[1000,500,500,500]),
    ("激熱のみ集中(激熱3000/他なし)",    [3000,  0,   0,   0]),
    ("激熱1000+☆500のみ",               [1000,  0, 500,   0]),
]

for label, bets in bet_patterns:
    total_invest=0; total_return=0
    for (name,mask), bet in zip(tiers_final, bets):
        if bet==0: continue
        sub=result[mask]; n=len(sub)
        if n==0: continue
        invest=n*bet
        ret=sub.loc[sub['target_win']==1,'単勝配当'].sum()*bet/100
        total_invest+=invest; total_return+=ret
    if total_invest==0: continue
    roi=(total_return/total_invest)-1
    print(f"  {label:<35}  {total_invest:>7}円  {total_return:>7.0f}円  {total_return-total_invest:>+8.0f}円  {roi:>+6.1%}")

# 月別安定性（推奨パターンで）
print("\n【月別収支（激熱1000+☆500）】")
bet_map = [(tier_激熱,1000),(tier_hoshi,500)]
for mo in sorted(result['月'].dropna().unique()):
    invest=0; ret=0
    for mask, bet in bet_map:
        sub=result[(result['月']==mo)&mask]
        n=len(sub)
        invest+=n*bet
        ret+=sub.loc[sub['target_win']==1,'単勝配当'].sum()*bet/100
    if invest==0: continue
    print(f"  {int(mo):02d}月  投資{invest:>6}円  収益{ret:>7.0f}円  純損益{ret-invest:>+7.0f}円  ROI{(ret/invest-1):>+7.1%}")

print("\n完了")

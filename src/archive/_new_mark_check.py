"""
新印体系 ROI確認
◎ 両Rnk=1 & gap≥10
▲ 両Rnk=1 & gap<10
☆ 両Rnk≤3（片方が2か3）
激熱 両Rnk=1 & gap≥15 & sd≥10
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

def extract_venue(k):
    m = re.search(r'\d+([^\d]+)', str(k))
    return m.group(1) if m else str(k)

with open(os.path.join(model_dir, 'model_info.json'), encoding='utf-8') as f: ci = json.load(f)
cf = ci['features']; cm = ci['models']
with open(os.path.join(model_dir, 'ranker', 'ranker_info.json'), encoding='utf-8') as f:
    cr2 = json.load(f).get('rankers', {})
with open(os.path.join(model_dir, 'submodel', 'submodel_info.json'), encoding='utf-8') as f: si = json.load(f)
sf = si['features']; sm = si['models']
with open(os.path.join(model_dir, 'submodel_ranker', 'class_ranker_info.json'), encoding='utf-8') as f:
    sr2 = json.load(f).get('rankers', {})
af = list(set(cf + sf))

df = pd.read_csv(os.path.join(base_dir, 'data', 'processed', 'all_venues_features_2026test.csv'), low_memory=False)
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
mask_da = (df['_surface'] == 'ダ') & (df['_dist_band'].isin(['中距離', '長距離']))
df.loc[mask_da, '_dist_band'] = '中長距離'
df['_cls_group'] = df['クラス_rank'].apply(get_class_group) if 'クラス_rank' in df.columns else '3勝以上'
df['sub_key']    = df['_surface'] + '_' + df['_dist_band'] + '_' + df['_cls_group']
for col in af:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

cmc = {}; crc = {}; smc = {}; src = {}
for ck in df['cur_key'].dropna().unique():
    if ck in cm:
        p = os.path.join(model_dir, cm[ck]['win'])
        if os.path.exists(p):
            with open(p, 'rb') as f: m = pickle.load(f)
            cmc[ck] = (m, m.booster_.feature_name())
    if ck in cr2:
        p = os.path.join(model_dir, 'ranker', cr2[ck])
        if os.path.exists(p):
            with open(p, 'rb') as f: crc[ck] = pickle.load(f)
for sk in df['sub_key'].dropna().unique():
    if sk in sm:
        p = os.path.join(model_dir, 'submodel', sm[sk]['win'])
        if os.path.exists(p):
            with open(p, 'rb') as f: m = pickle.load(f)
            smc[sk] = (m, m.booster_.feature_name())
    if sk in sr2:
        p = os.path.join(model_dir, 'submodel_ranker', sr2[sk])
        if os.path.exists(p):
            with open(p, 'rb') as f: src[sk] = pickle.load(f)

rk = [c for c in ['開催', 'Ｒ'] if c in df.columns]
rows = []
for gk, idx in df.groupby(rk, sort=False).groups.items():
    sub = df.loc[idx].copy()
    ck = sub['cur_key'].iloc[0]; sk = sub['sub_key'].iloc[0]
    sub['cur_score'] = np.nan; sub['sub_score'] = np.nan
    sub['cur_rank']  = np.nan; sub['sub_rank']  = np.nan
    sub['cur_diff']  = np.nan; sub['sub_diff']  = np.nan
    if ck in cmc:
        m, wf = cmc[ck]
        for c in wf:
            if c not in sub.columns: sub[c] = np.nan
        prob = m.predict_proba(sub[wf])[:, 1]
        st = cm[ck].get('stats', {}); wm = st.get('win_mean', prob.mean()); ws = st.get('win_std', prob.std())
        cs = 50 + 10*(prob - wm)/(ws if ws > 0 else 1)
        rm = prob.mean(); rs = prob.std()
        sub['cur_score'] = cs
        sub['cur_diff']  = 50 + 10*(prob - rm)/(rs if rs > 0 else 1) - cs
        if ck in crc:
            sc = crc[ck].predict(sub[cf])
            sub['cur_rank'] = pd.Series(sc, index=sub.index).rank(ascending=False, method='min').astype(int)
    if sk in smc:
        m, wf = smc[sk]
        for c in wf:
            if c not in sub.columns: sub[c] = np.nan
        prob = m.predict_proba(sub[wf])[:, 1]
        st = sm[sk].get('stats', {}); wm = st.get('win_mean', prob.mean()); ws = st.get('win_std', prob.std())
        cs = 50 + 10*(prob - wm)/(ws if ws > 0 else 1)
        rm = prob.mean(); rs = prob.std()
        sub['sub_score'] = cs
        sub['sub_diff']  = 50 + 10*(prob - rm)/(rs if rs > 0 else 1) - cs
        if sk in src:
            sc = src[sk].predict(sub[wf])
            sub['sub_rank'] = pd.Series(sc, index=sub.index).rank(ascending=False, method='min').astype(int)
    for col, gcol in [('cur_score', 'cur_gap'), ('sub_score', 'sub_gap')]:
        sc2 = sub[col].dropna().sort_values(ascending=False).values
        sub[gcol] = (sc2[0] - sc2[1]) if len(sc2) >= 2 else np.nan
    sub['combo_gap'] = sub['cur_gap'].fillna(0) + sub['sub_gap'].fillna(0)
    rows.append(sub)

res = pd.concat(rows, ignore_index=True)
cr = pd.to_numeric(res['cur_rank'],  errors='coerce')
sr = pd.to_numeric(res['sub_rank'],  errors='coerce')
sd = pd.to_numeric(res['sub_diff'],  errors='coerce')
cg = pd.to_numeric(res['combo_gap'], errors='coerce')
res['odds'] = pd.to_numeric(res['単勝オッズ'], errors='coerce')

both_r1 = (cr == 1) & (sr == 1)
star    = (cr <= 3) & (sr <= 3) & ~both_r1   # 片方が2か3

def show(label, s):
    n = len(s)
    if n == 0: print(f'  {label}: 該当なし'); return
    ret_t = s.loc[s['target_win']==1,   '単勝配当'].sum()
    ret_f = s.loc[s['target_place']==1, '複勝配当'].sum()
    roi_t = ret_t/(n*100)-1
    roi_f = ret_f/(n*100)-1
    ao    = s['odds'].mean()
    print(f'  {label:<42} N={n:>4}  平均{ao:.1f}倍  単勝ROI{roi_t:>+7.1%}  複勝ROI{roi_f:>+6.1%}')

print('='*75)
print('  新印体系 ROI確認（2026年 out-of-sample）')
print('='*75)
print()
print('【全体・オッズ無制限】')
show('◎ 両Rnk=1 & gap≥10',             res[both_r1 & (cg>=10)])
show('▲ 両Rnk=1 & gap<10',             res[both_r1 & (cg<10)])
show('☆ 両Rnk≤3（片方2か3）',          res[star])
show('激熱 両Rnk=1 & gap≥15 & sd≥10', res[both_r1 & (cg>=15) & (sd>=10)])

print()
print('【オッズフィルター追加（推奨運用）】')
show('◎ & odds≥3',                     res[both_r1 & (cg>=10) & (res['odds']>=3)])
show('▲ & odds≥3',                     res[both_r1 & (cg<10)  & (res['odds']>=3)])
show('☆ & odds≥5',                     res[star    & (res['odds']>=5)])
show('☆ & odds≥5 & sd≥5',             res[star    & (res['odds']>=5) & (sd>=5)])
show('激熱 & odds≥3',                  res[both_r1 & (cg>=15) & (sd>=10) & (res['odds']>=3)])

print()
print('【☆ 内訳（odds≥5）】')
show('  cur=2 & sub=2',                res[(cr==2) & (sr==2) & (res['odds']>=5)])
show('  cur=2 & sub=3',                res[(cr==2) & (sr==3) & (res['odds']>=5)])
show('  cur=3 & sub=2',                res[(cr==3) & (sr==2) & (res['odds']>=5)])
show('  cur=3 & sub=3',                res[(cr==3) & (sr==3) & (res['odds']>=5)])
show('  ☆ & sd≥5 & odds≥5',          res[star & (sd>=5) & (res['odds']>=5)])
show('  ☆ & sd≥10 & odds≥5',         res[star & (sd>=10) & (res['odds']>=5)])

print()
print('【週間本数イメージ（3ヶ月÷12週）】')
n_weeks = 12
for label, mask in [
    ('◎(odds≥3)',  both_r1 & (cg>=10) & (res['odds']>=3)),
    ('▲(odds≥3)',  both_r1 & (cg<10)  & (res['odds']>=3)),
    ('☆(odds≥5)',  star    & (res['odds']>=5)),
    ('激熱(odds≥3)',both_r1 & (cg>=15) & (sd>=10) & (res['odds']>=3)),
]:
    n = mask.sum()
    print(f'  {label:<20} 計{n:>4}本 → 週{n/n_weeks:.1f}本')

# coding: utf-8
"""強さPT別ROI分析"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd, numpy as np, os, pickle, json, re, glob

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, 'models_2025')

def get_distance_band(dist):
    m = re.search(r'\d+', str(dist))
    if not m: return None
    d = int(m.group())
    if d <= 1400: return '短距離'
    elif d <= 1800: return 'マイル'
    elif d <= 2200: return '中距離'
    else: return '長距離'

def get_class_group(r):
    try: r = int(float(r))
    except: return '3勝以上'
    if r == 1: return '新馬'
    elif r == 2: return '未勝利'
    elif r == 3: return '1勝'
    elif r == 4: return '2勝'
    return '3勝以上'

def _zen(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    m = re.search(r'\d+', s)
    return int(m.group()) if m else np.nan

# モデル読み込み
with open(os.path.join(model_dir, 'model_info.json'), encoding='utf-8') as f:
    cur_info = json.load(f)
cur_features = cur_info['features']
cur_models   = cur_info['models']
with open(os.path.join(model_dir, 'submodel', 'submodel_info.json'), encoding='utf-8') as f:
    sub_info = json.load(f)
sub_features = sub_info['features']
sub_models   = sub_info['models']

# Parquet最新特徴量
import pyarrow.parquet as pq_mod
pq_path = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.parquet')
all_feats = list(set(cur_features + sub_features))
avail = set(pq_mod.read_schema(pq_path).names)
load_cols = ['馬名S', '日付', '距離'] + [c for c in avail if c in set(all_feats)]
df_pq = pd.read_parquet(pq_path, columns=load_cols)
df_pq['日付_n'] = pd.to_numeric(df_pq['日付'], errors='coerce')
pq_latest = df_pq.sort_values('日付_n').groupby('馬名S', sort=False).last().reset_index()
print(f"Parquet読込完了: {len(pq_latest)}頭分の最新特徴量")

VENUE_MAP = {'中山':'中','東京':'東','阪神':'阪','中京':'名','京都':'京',
             '函館':'函','新潟':'新','小倉':'小','札幌':'札','福島':'福'}

# モデルキャッシュ（初回読み込みのみ）
cur_model_cache = {}
sub_model_cache = {}

def get_cur_model(ck):
    if ck not in cur_model_cache and ck in cur_models:
        p = os.path.join(model_dir, cur_models[ck]['win'])
        if os.path.exists(p):
            with open(p, 'rb') as f:
                m = pickle.load(f)
            cur_model_cache[ck] = (m, m.booster_.feature_name())
    return cur_model_cache.get(ck)

def get_sub_model(sk):
    if sk not in sub_model_cache and sk in sub_models:
        p = os.path.join(model_dir, 'submodel', sub_models[sk]['win'])
        if os.path.exists(p):
            with open(p, 'rb') as f:
                m = pickle.load(f)
            sub_model_cache[sk] = (m, m.booster_.feature_name())
    return sub_model_cache.get(sk)

# 全結果CSV処理
result_csvs = sorted(glob.glob(os.path.join(base_dir, 'data', 'raw', 'results', '出馬表形式*結果*.csv')))
all_records = []

for rf in result_csvs:
    fname = os.path.basename(rf)
    try:    dfr = pd.read_csv(rf, encoding='cp932', low_memory=False)
    except: dfr = pd.read_csv(rf, encoding='utf-8',  low_memory=False)
    if '日付S' not in dfr.columns:
        continue
    ds   = str(dfr['日付S'].iloc[0]).replace('/', '.').split('.')
    dnum = (int(ds[0]) - 2000) * 10000 + int(ds[1]) * 100 + int(ds[2])

    dfr['着_num']  = dfr['着'].apply(_zen)
    dfr['_tan']    = pd.to_numeric(dfr['単勝'],   errors='coerce')
    dfr['_fuku']   = pd.to_numeric(dfr['複勝'],   errors='coerce')
    dfr['_odds']   = pd.to_numeric(dfr['単オッズ'], errors='coerce')
    dfr['会場']    = dfr['場所'].astype(str).map(VENUE_MAP).fillna(dfr['場所'].astype(str))
    dfr['_surface']= dfr['芝ダ'].astype(str).str.strip() if '芝ダ' in dfr.columns else 'ダ'
    dfr['cur_key'] = dfr['会場'] + '_' + dfr['_surface'] + dfr['距離'].astype(str)
    dfr['_dist_band'] = dfr['距離'].apply(get_distance_band)
    _mask = (dfr['_surface'] == 'ダ') & (dfr['_dist_band'].isin(['中距離', '長距離']))
    dfr.loc[_mask, '_dist_band'] = '中長距離'
    if 'クラス' in dfr.columns:
        _cls_map = {'新馬': 1, '未勝利': 2, '1勝': 3, '2勝': 4}
        dfr['クラス_rank'] = dfr['クラス'].apply(
            lambda v: next((r for k, r in _cls_map.items() if k in str(v)), 5))
    dfr['_cls_group'] = dfr['クラス_rank'].apply(get_class_group) if 'クラス_rank' in dfr.columns else '3勝以上'
    dfr['sub_key']  = dfr['_surface'] + '_' + dfr['_dist_band'] + '_' + dfr['_cls_group']
    dfr['性別_num'] = dfr['性別'].map({'牡': 0, '牝': 1, 'セ': 2}).astype(float)

    feat_cols = ['馬名S'] + [c for c in all_feats if c in pq_latest.columns]
    merged = dfr.merge(pq_latest[feat_cols], on='馬名S', how='left', suffixes=('', '_p'))
    if '距離' in pq_latest.columns:
        merged['前距離'] = pq_latest.set_index('馬名S')['距離'].reindex(
            merged['馬名S'].values).apply(
            lambda x: float(re.search(r'\d+', str(x)).group())
            if re.search(r'\d+', str(x)) else np.nan).values
    for c in all_feats:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors='coerce')

    # 勝ち馬の単勝をレース全馬に展開
    rkey = merged['場所'].astype(str) + '_' + merged['Ｒ'].astype(str)
    merged['_race_key'] = rkey
    win_tan = merged[merged['着_num'] == 1].drop_duplicates('_race_key').set_index('_race_key')['_tan']
    merged['_tansho'] = merged['_race_key'].map(win_tan)

    # レース別予測
    for gk, idx in merged.groupby(['場所', 'Ｒ'], sort=False).groups.items():
        sub = merged.loc[idx].copy()
        ck = sub['cur_key'].iloc[0]
        sk = sub['sub_key'].iloc[0]

        cur_cs = np.full(len(sub), np.nan)
        sub_cs = np.full(len(sub), np.nan)

        cm = get_cur_model(ck)
        if cm:
            m, wf = cm
            for c in wf:
                if c not in sub.columns: sub[c] = np.nan
            prob = m.predict_proba(sub[wf])[:, 1]
            st = cur_models[ck].get('stats', {})
            wm = st.get('win_mean', prob.mean()); ws = st.get('win_std', prob.std())
            cur_cs = 50 + 10 * (prob - wm) / (ws if ws > 0 else 1)

        sm = get_sub_model(sk)
        if sm:
            m, wf = sm
            for c in wf:
                if c not in sub.columns: sub[c] = np.nan
            prob = m.predict_proba(sub[wf])[:, 1]
            st = sub_models[sk].get('stats', {})
            wm = st.get('win_mean', prob.mean()); ws = st.get('win_std', prob.std())
            sub_cs = 50 + 10 * (prob - wm) / (ws if ws > 0 else 1)

        # 強さPT = 両モデル平均（片方だけでもOK）
        cur_s = pd.Series(cur_cs, index=sub.index)
        sub_s = pd.Series(sub_cs, index=sub.index)
        both  = ~(cur_s.isna() | sub_s.isna())
        pt    = pd.Series(np.nan, index=sub.index)
        pt[both]  = (cur_s[both] + sub_s[both]) / 2
        pt[~both] = cur_s.fillna(sub_s)[~both]

        for i, (row_idx, row) in enumerate(sub.iterrows()):
            all_records.append({
                '日付_num': dnum,
                '日付_str': f"20{str(dnum)[:2]}/{str(dnum)[2:4]}/{str(dnum)[4:]}",
                '場所':     str(row.get('場所', '')),
                'Ｒ':       row.get('Ｒ'),
                '馬名S':    row.get('馬名S', ''),
                '着_num':   row['着_num'],
                '_tansho':  row['_tansho'],
                '_fuku':    row['_fuku'],
                '_odds':    row['_odds'],
                '強さPT':   pt.iloc[i],
                'cur_cs':   cur_s.iloc[i],
                'sub_cs':   sub_s.iloc[i],
            })

    n_races = merged.groupby(['場所', 'Ｒ']).ngroups
    print(f"  {fname[:20]}... dnum={dnum} {n_races}R")

df = pd.DataFrame(all_records)
df = df.dropna(subset=['着_num', '強さPT', '_tansho'])
print(f"\n総データ: {len(df)}頭 / {df['日付_num'].nunique()}日 / 勝ち馬あり")
print(f"強さPT 分布: min={df['強さPT'].min():.1f} / mean={df['強さPT'].mean():.1f} / max={df['強さPT'].max():.1f}")
print()

# ── PT閾値別ROI分析 ──
print("=" * 65)
print(f"{'PT閾値':>8}  {'頭数':>6}  {'勝率':>6}  {'平均オッズ':>8}  {'単勝ROI':>8}  {'損益':>8}")
print("=" * 65)

thresholds = [50, 52, 54, 55, 56, 57, 58, 59, 60, 62, 65, 68, 70]
BET = 100

results_by_thresh = {}
for t in thresholds:
    mask  = df['強さPT'] >= t
    sub2  = df[mask]
    if len(sub2) == 0:
        continue
    n     = len(sub2)
    wins  = int(sub2['着_num'].eq(1).sum())
    wr    = wins / n
    avg_o = sub2['_odds'].mean()
    tb    = n * BET
    ret   = sub2[sub2['着_num'] == 1]['_tansho'].sum() * BET / 100
    roi   = ret / tb - 1.0
    pf    = int(ret - tb)
    results_by_thresh[t] = {'n': n, 'wins': wins, 'wr': wr, 'avg_odds': avg_o, 'roi': roi, 'pf': pf}
    marker = ' ◀' if roi > 0 else ''
    print(f"  {t}以上:  {n:5d}頭  {wr:5.1%}  {avg_o:7.1f}倍  {roi:+7.1%}  {pf:+8,}円{marker}")

print()

# ── 日別×PT閾値 ──
print("=" * 55)
print("日別収支（PT58以上 / PT60以上 / PT62以上）")
print("=" * 55)
print(f"{'日付':>12}  {'PT58':>12}  {'PT60':>12}  {'PT62':>12}")
for dnum, grp in df.groupby('日付_num'):
    date_str = f"20{str(int(dnum))[:2]}/{str(int(dnum))[2:4]}/{str(int(dnum))[4:]}"
    row_parts = [f"{date_str:>12}"]
    for t in [58, 60, 62]:
        sub3 = grp[grp['強さPT'] >= t]
        if len(sub3) == 0:
            row_parts.append(f"{'  -':>12}")
            continue
        tb  = len(sub3) * BET
        ret = sub3[sub3['着_num'] == 1]['_tansho'].sum() * BET / 100
        pf  = int(ret - tb)
        roi = ret / tb - 1.0
        s   = f"{pf:+,}円({roi:+.0%})"
        row_parts.append(f"{s:>12}")
    print("  ".join(row_parts))

print()

# ── 複勝も見る ──
df2 = pd.DataFrame(all_records)
df2 = df2.dropna(subset=['着_num', '強さPT', '_fuku'])
print("=" * 65)
print("複勝ROI（PT閾値別）")
print(f"{'PT閾値':>8}  {'頭数':>6}  {'複勝率':>6}  {'単勝ROI':>8}  {'複勝ROI':>8}")
print("=" * 65)
for t in [55, 57, 58, 59, 60, 62, 65]:
    mask  = df2['強さPT'] >= t
    sub4  = df2[mask]
    if len(sub4) == 0: continue
    n     = len(sub4)
    fuku_hits = int((sub4['着_num'] <= 3).sum())
    fuku_rate = fuku_hits / n
    # 単勝
    sub4b = df.copy(); sub4b = sub4b[sub4b['強さPT'] >= t]
    if len(sub4b) > 0:
        ret_t = sub4b[sub4b['着_num'] == 1]['_tansho'].sum() * BET / 100
        roi_t = ret_t / (len(sub4b) * BET) - 1.0
    else:
        roi_t = np.nan
    # 複勝
    ret_f = sub4[sub4['着_num'] <= 3]['_fuku'].sum() * BET / 100
    roi_f = ret_f / (n * BET) - 1.0
    marker = ' ◀' if roi_f > 0 else ''
    print(f"  {t}以上:  {n:5d}頭  {fuku_rate:5.1%}  {roi_t:+7.1%}  {roi_f:+7.1%}{marker}")

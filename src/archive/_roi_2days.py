"""2戦略（距離Rnk1単勝 / 両Rnk1複勝）の3/21・3/22損益計算
条件: 単オッズ下限 × ランカースコア差（1位-2位）"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import pandas as pd, numpy as np, os, pickle, json, re

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, 'models')

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

def zen_to_num(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    m = re.search(r'\d+', s)
    return int(m.group()) if m else np.nan

VENUE_MAP = {'中山':'中','東京':'東','阪神':'阪','中京':'名','京都':'京',
             '函館':'函','新潟':'新','小倉':'小','札幌':'札','福島':'福'}

with open(os.path.join(model_dir, 'model_info.json'), encoding='utf-8') as f: cur_info = json.load(f)
cur_features = cur_info['features']; cur_models = cur_info['models']
with open(os.path.join(model_dir, 'ranker', 'ranker_info.json'), encoding='utf-8') as f:
    cur_rankers = json.load(f).get('rankers', {})
with open(os.path.join(model_dir, 'submodel', 'submodel_info.json'), encoding='utf-8') as f: sub_info = json.load(f)
sub_features = sub_info['features']; sub_models = sub_info['models']
with open(os.path.join(model_dir, 'submodel_ranker', 'class_ranker_info.json'), encoding='utf-8') as f:
    sub_rankers = json.load(f).get('rankers', {})
all_feats = list(set(cur_features + sub_features))

df_feat = pd.read_parquet(os.path.join(base_dir, 'data', 'processed', 'all_venues_features.parquet'))
df_latest = df_feat.sort_values('日付').groupby('馬名S').last().reset_index()
feat_subset = ['馬名S'] + [c for c in all_feats if c in df_latest.columns]
for ec in ['日付', '距離']:
    if ec in df_latest.columns and ec not in feat_subset: feat_subset.append(ec)

cur_model_cache = {}; cur_ranker_cache = {}
sub_model_cache = {}; sub_ranker_cache = {}

def load_models(df_m):
    for ck in df_m['cur_key'].dropna().unique():
        if ck in cur_models:
            p = os.path.join(model_dir, cur_models[ck]['win'])
            if os.path.exists(p):
                with open(p, 'rb') as f: m = pickle.load(f)
                cur_model_cache[ck] = (m, m.booster_.feature_name())
        if ck in cur_rankers:
            p = os.path.join(model_dir, 'ranker', cur_rankers[ck])
            if os.path.exists(p):
                with open(p, 'rb') as f: cur_ranker_cache[ck] = pickle.load(f)
    for sk in df_m['sub_key'].dropna().unique():
        if sk in sub_models:
            p = os.path.join(model_dir, 'submodel', sub_models[sk]['win'])
            if os.path.exists(p):
                with open(p, 'rb') as f: m = pickle.load(f)
                sub_model_cache[sk] = (m, m.booster_.feature_name())
        if sk in sub_rankers:
            p = os.path.join(model_dir, 'submodel_ranker', sub_rankers[sk])
            if os.path.exists(p):
                with open(p, 'rb') as f: sub_ranker_cache[sk] = pickle.load(f)

def predict(df_res, cur_date):
    df_res = df_res.copy()
    df_res['会場'] = df_res['場所'].astype(str).map(VENUE_MAP).fillna(df_res['場所'].astype(str))
    if 'コースマーク' in df_res.columns:
        cm = df_res['コースマーク'].astype(str).str.strip()
        df_res['会場'] = df_res['会場'] + cm.where(cm.isin(['A','B','C']), '')
    df_res['_surface'] = df_res['芝ダ'].astype(str).str.strip()
    df_res['cur_key'] = df_res['会場'] + '_' + df_res['_surface'] + df_res['距離'].astype(str)
    df_res['_dist_band'] = df_res['距離'].apply(get_distance_band)
    mask = (df_res['_surface'] == 'ダ') & (df_res['_dist_band'].isin(['中距離', '長距離']))
    df_res.loc[mask, '_dist_band'] = '中長距離'
    df_res['_cls_group'] = df_res['クラス_rank'].apply(get_class_group) if 'クラス_rank' in df_res.columns else '3勝以上'
    df_res['sub_key'] = df_res['_surface'] + '_' + df_res['_dist_band'].astype(str) + '_' + df_res['_cls_group'].astype(str)
    df_m = df_res.merge(df_latest[feat_subset], on='馬名S', how='left', suffixes=('', '_f'))
    df_m['性別_num'] = df_m['性別'].map({'牡': 0, '牝': 1, 'セ': 2}).astype(float)
    if '距離_f' in df_m.columns:
        df_m['前距離'] = df_m['距離_f'].astype(str).str.extract(r'(\d+)').iloc[:, 0].astype(float)
    if '日付_f' in df_m.columns:
        def _d(v):
            try:
                v = int(v)
                return pd.Timestamp(2000 + v // 10000, (v // 100) % 100, v % 100)
            except: return pd.NaT
        df_m['間隔'] = ((cur_date - df_m['日付_f'].apply(_d)).dt.days / 7).round(0)
    for col in all_feats:
        if col in df_m.columns: df_m[col] = pd.to_numeric(df_m[col], errors='coerce')
    load_models(df_m)
    rows = []
    for gk, idx in df_m.groupby(['場所', 'Ｒ'], sort=False).groups.items():
        sub = df_m.loc[idx].copy()
        ck = sub['cur_key'].iloc[0]; sk = sub['sub_key'].iloc[0]
        sub['cur_diff'] = np.nan; sub['cur_rank'] = np.nan; sub['cur_gap'] = np.nan
        sub['sub_diff'] = np.nan; sub['sub_rank'] = np.nan; sub['sub_gap'] = np.nan
        if ck in cur_model_cache:
            m, wf = cur_model_cache[ck]
            for c in wf:
                if c not in sub.columns: sub[c] = np.nan
            prob = m.predict_proba(sub[wf])[:, 1]
            st = cur_models[ck].get('stats', {})
            wm = st.get('win_mean', prob.mean()); ws = st.get('win_std', prob.std())
            cs = 50 + 10*(prob - wm)/(ws if ws > 0 else 1)
            rm = prob.mean(); rs = prob.std()
            sub['cur_diff'] = 50 + 10*(prob - rm)/(rs if rs > 0 else 1) - cs
            if ck in cur_ranker_cache:
                sc = cur_ranker_cache[ck].predict(sub[cur_features])
                sc_s = pd.Series(sc, index=sub.index)
                sub['cur_rank'] = sc_s.rank(ascending=False, method='min').astype(int)
                if len(sc) >= 2:
                    top2 = sorted(sc, reverse=True)[:2]
                    gap = top2[0] - top2[1]
                    sub.loc[sc_s.idxmax(), 'cur_gap'] = gap
        if sk in sub_model_cache:
            m, wf = sub_model_cache[sk]
            for c in wf:
                if c not in sub.columns: sub[c] = np.nan
            prob = m.predict_proba(sub[wf])[:, 1]
            st = sub_models[sk].get('stats', {})
            wm = st.get('win_mean', prob.mean()); ws = st.get('win_std', prob.std())
            cs = 50 + 10*(prob - wm)/(ws if ws > 0 else 1)
            rm = prob.mean(); rs = prob.std()
            sub['sub_diff'] = 50 + 10*(prob - rm)/(rs if rs > 0 else 1) - cs
            if sk in sub_ranker_cache:
                sc = sub_ranker_cache[sk].predict(sub[wf])
                sc_s = pd.Series(sc, index=sub.index)
                sub['sub_rank'] = sc_s.rank(ascending=False, method='min').astype(int)
                if len(sc) >= 2:
                    top2 = sorted(sc, reverse=True)[:2]
                    gap = top2[0] - top2[1]
                    sub.loc[sc_s.idxmax(), 'sub_gap'] = gap
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)

# ── スコアギャップの分布確認用 ──
all_results = {}

print()
print('=' * 70)
print('  距離Rnk1単勝 ＆ 両Rnk1複勝  条件別損益シミュレーション')
print('=' * 70)

for fname, ds, cur_date in [
    ('出馬表形式3月21日結果確認.csv', '3月21日', pd.Timestamp(2026, 3, 21)),
    ('出馬表形式3月22日結果確認.csv', '3月22日', pd.Timestamp(2026, 3, 22)),
]:
    df_res = pd.read_csv(os.path.join(base_dir, 'data', 'raw', fname), encoding='cp932', low_memory=False)
    df_res['着_num'] = df_res['着'].apply(zen_to_num)
    df_res['target_win']   = (df_res['着_num'] == 1).astype(int)
    df_res['target_place'] = (df_res['着_num'] <= 3).astype(int)
    df_res['単勝_num'] = pd.to_numeric(df_res['単勝'], errors='coerce')
    df_res['複勝_num'] = pd.to_numeric(df_res['複勝'], errors='coerce')
    df_res['単オッズ_num'] = pd.to_numeric(df_res['単オッズ'], errors='coerce')
    race_w = df_res[df_res['target_win']==1][['場所','Ｒ','単勝_num']].rename(columns={'単勝_num': 'tansho'})
    df_res = df_res.merge(race_w, on=['場所','Ｒ'], how='left')
    result = predict(df_res, cur_date)
    all_results[ds] = result

    cr  = result['cur_rank']
    sr  = result['sub_rank']
    cg  = pd.to_numeric(result['cur_gap'], errors='coerce')
    sg  = pd.to_numeric(result['sub_gap'], errors='coerce')
    ods = pd.to_numeric(result['単オッズ_num'], errors='coerce')

    # ギャップ分布を確認
    rank1_gaps = result[cr == 1]['cur_gap'].dropna()
    print(f'\n--- {ds}  (距離ランカー1位のスコアギャップ分布) ---')
    if len(rank1_gaps) > 0:
        print(f'  件数:{len(rank1_gaps)}  中央値:{rank1_gaps.median():.3f}  25%:{rank1_gaps.quantile(0.25):.3f}  75%:{rank1_gaps.quantile(0.75):.3f}  最大:{rank1_gaps.max():.3f}')
    rank1_ods = result[cr == 1]['単オッズ_num'].dropna()
    if len(rank1_ods) > 0:
        print(f'  単オッズ: 中央値:{rank1_ods.median():.1f}  1.9以下:{( rank1_ods <= 1.9).sum()}件  2.0-3.9:{((rank1_ods >= 2.0)&(rank1_ods <= 3.9)).sum()}件  4.0以上:{(rank1_ods >= 4.0).sum()}件')

    print()
    print(f'  {"条件":<28}  {"◎単勝":>10}  {"○複勝":>10}  {"合計損益":>10}  {"合計ROI":>8}')
    print(f'  {"-"*28}  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*8}')

    for odds_min, gap_min in [
        (0,   0),
        (2.0, 0),    (3.0, 0),    (4.0, 0),
        (0,   0.05), (0,   0.1),  (0,   0.2),
        (2.0, 0.05), (2.0, 0.1),
        (3.0, 0.05), (3.0, 0.1),
        (4.0, 0.05), (4.0, 0.1),
    ]:
        odds_ok = (ods >= odds_min) if odds_min > 0 else pd.Series(True, index=result.index)
        gap_ok  = (cg >= gap_min)   if gap_min  > 0 else pd.Series(True, index=result.index)

        # 戦略1: 距離Rnk1 & オッズ条件 & ギャップ条件 → 単勝
        m1 = (cr == 1) & odds_ok & gap_ok
        b1 = result[m1]
        n1 = len(b1)
        if n1 == 0:
            w1_str = '-'; roi1_str = '-'; prf1 = 0; ret1 = 0
        else:
            w1 = int(b1['target_win'].sum())
            ret1 = b1[b1['target_win']==1]['tansho'].sum()
            prf1 = ret1 - n1*100
            roi1 = ret1/(n1*100) - 1
            w1_str = f'{w1}/{n1}R {roi1:+.0%}'
            roi1_str = f'{roi1:+.1%}'

        # 戦略2: 両Rnk1 & ギャップ条件（オッズは複勝なので下限なし）→ 複勝
        m2 = (cr==1) & (sr==1) & gap_ok
        b2 = result[m2]
        n2 = len(b2)
        if n2 == 0:
            p2_str = '-'; roi2_str = '-'; prf2 = 0; ret2 = 0
        else:
            p2 = int(b2['target_place'].sum())
            ret2 = b2[b2['target_place']==1]['複勝_num'].sum()
            prf2 = ret2 - n2*100
            roi2 = ret2/(n2*100) - 1
            p2_str = f'{p2}/{n2}R {roi2:+.0%}'

        total_inv = n1*100 + n2*100
        total_ret = ret1 + ret2
        total_prf = total_ret - total_inv
        total_roi = total_ret/total_inv - 1 if total_inv > 0 else 0

        ods_label = f'オッズ≥{odds_min:.1f}' if odds_min > 0 else 'オッズ制限なし'
        gap_label = f'gap≥{gap_min:.2f}' if gap_min > 0 else 'gap制限なし'
        label = f'{ods_label} & {gap_label}'
        print(f'  {label:<28}  {w1_str:>10}  {p2_str:>10}  {total_prf:>+10,.0f}  {total_roi:>+8.1%}')

print()
print('=' * 70)
print('※ gap = 距離ランカーの1位スコア − 2位スコア（ランカー内部スコア差）')
print('=' * 70)

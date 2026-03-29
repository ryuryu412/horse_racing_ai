"""
勝負レース判定 & 単勝 vs 馬連 判断ロジック
バックテストデータから最適な閾値を導出する

出力:
  1. 勝負レース条件別ROI（フィルター有効性）
  2. cur_gap/sub_gap 別 単勝ROI vs 馬連ROI
  3. 推奨ルールサマリー
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import pandas as pd, numpy as np, os, pickle, json, re, time

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

def extract_venue(kaikai):
    m = re.search(r'\d+([^\d]+)', str(kaikai))
    return m.group(1) if m else str(kaikai)

# ── モデル読み込み ──
print("モデル読み込み中...")
with open(os.path.join(model_dir, 'model_info.json'), encoding='utf-8') as f: cur_info = json.load(f)
cur_features = cur_info['features']; cur_models = cur_info['models']
with open(os.path.join(model_dir, 'ranker', 'ranker_info.json'), encoding='utf-8') as f:
    cur_rankers = json.load(f).get('rankers', {})
with open(os.path.join(model_dir, 'submodel', 'submodel_info.json'), encoding='utf-8') as f: sub_info = json.load(f)
sub_features = sub_info['features']; sub_models = sub_info['models']
with open(os.path.join(model_dir, 'submodel_ranker', 'class_ranker_info.json'), encoding='utf-8') as f:
    sub_rankers = json.load(f).get('rankers', {})
all_feats = list(set(cur_features + sub_features))

# ── テストデータ読み込み ──
test_path = os.path.join(base_dir, 'data', 'processed', 'features_2012_test.csv')
print(f"データ読み込み中...")
df = pd.read_csv(test_path, low_memory=False)
df['着順_num']   = pd.to_numeric(df['着順_num'], errors='coerce')
df['単勝配当']   = pd.to_numeric(df['単勝配当'], errors='coerce')
df['馬連']       = pd.to_numeric(df['馬連'],     errors='coerce')
df['単勝オッズ'] = pd.to_numeric(df['単勝オッズ'], errors='coerce')
df = df.dropna(subset=['着順_num', '単勝配当'])
df['target_win']   = (df['着順_num'] == 1).astype(int)
df['target_place'] = (df['着順_num'] <= 2).astype(int)  # 2着以内
df['会場']       = df['開催'].apply(extract_venue)
df['_surface']   = df['芝・ダ'].astype(str).str.strip()
df['cur_key']    = df['会場'] + '_' + df['距離'].astype(str)
df['_dist_band'] = df['距離'].apply(get_distance_band)
mask_da = (df['_surface'] == 'ダ') & (df['_dist_band'].isin(['中距離', '長距離']))
df.loc[mask_da, '_dist_band'] = '中長距離'
df['_cls_group'] = df['クラス_rank'].apply(get_class_group) if 'クラス_rank' in df.columns else '3勝以上'
df['sub_key']    = df['_surface'] + '_' + df['_dist_band'] + '_' + df['_cls_group']
for col in all_feats:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

# ── モデルキャッシュ ──
print("モデルプリロード中...")
cur_model_cache = {}; cur_ranker_cache = {}
sub_model_cache = {}; sub_ranker_cache = {}
for ck in df['cur_key'].dropna().unique():
    if ck in cur_models:
        p = os.path.join(model_dir, cur_models[ck]['win'])
        if os.path.exists(p):
            with open(p,'rb') as f: m = pickle.load(f)
            cur_model_cache[ck] = (m, m.booster_.feature_name())
    if ck in cur_rankers:
        p = os.path.join(model_dir, 'ranker', cur_rankers[ck])
        if os.path.exists(p):
            with open(p,'rb') as f: cur_ranker_cache[ck] = pickle.load(f)
for sk in df['sub_key'].dropna().unique():
    if sk in sub_models:
        p = os.path.join(model_dir, 'submodel', sub_models[sk]['win'])
        if os.path.exists(p):
            with open(p,'rb') as f: m = pickle.load(f)
            sub_model_cache[sk] = (m, m.booster_.feature_name())
    if sk in sub_rankers:
        p = os.path.join(model_dir, 'submodel_ranker', sub_rankers[sk])
        if os.path.exists(p):
            with open(p,'rb') as f: sub_ranker_cache[sk] = pickle.load(f)
print(f"プリロード完了: cur={len(cur_model_cache)} sub={len(sub_model_cache)}")

# ── レース別予測 ──
race_keys = [c for c in ['開催','Ｒ'] if c in df.columns]
df['Ｒ'] = pd.to_numeric(df.get('Ｒ', np.nan), errors='coerce')
print("予測中...")
t0 = time.time()
all_rows = []
for gk, idx in df.groupby(race_keys, sort=False).groups.items():
    sub = df.loc[idx].copy()
    ck = sub['cur_key'].iloc[0]; sk = sub['sub_key'].iloc[0]
    sub['cur_score'] = np.nan; sub['sub_score'] = np.nan
    sub['cur_diff']  = np.nan; sub['cur_rank']  = np.nan
    sub['sub_diff']  = np.nan; sub['sub_rank']  = np.nan

    if ck in cur_model_cache:
        m, wf = cur_model_cache[ck]
        for c in wf:
            if c not in sub.columns: sub[c] = np.nan
        prob = m.predict_proba(sub[wf])[:, 1]
        st = cur_models[ck].get('stats', {})
        wm = st.get('win_mean', prob.mean()); ws = st.get('win_std', prob.std())
        cs = 50 + 10*(prob - wm)/(ws if ws > 0 else 1)
        rm = prob.mean(); rs_p = prob.std()
        sub['cur_score'] = cs
        sub['cur_diff']  = 50 + 10*(prob - rm)/(rs_p if rs_p > 0 else 1) - cs
        if ck in cur_ranker_cache:
            sc = cur_ranker_cache[ck].predict(sub[cur_features])
            sub['cur_rank'] = pd.Series(sc, index=sub.index).rank(ascending=False, method='min').astype(int)

    if sk in sub_model_cache:
        m, wf = sub_model_cache[sk]
        for c in wf:
            if c not in sub.columns: sub[c] = np.nan
        prob = m.predict_proba(sub[wf])[:, 1]
        st = sub_models[sk].get('stats', {})
        wm = st.get('win_mean', prob.mean()); ws = st.get('win_std', prob.std())
        cs = 50 + 10*(prob - wm)/(ws if ws > 0 else 1)
        rm = prob.mean(); rs_p = prob.std()
        sub['sub_score'] = cs
        sub['sub_diff']  = 50 + 10*(prob - rm)/(rs_p if rs_p > 0 else 1) - cs
        if sk in sub_ranker_cache:
            sc = sub_ranker_cache[sk].predict(sub[wf])
            sub['sub_rank'] = pd.Series(sc, index=sub.index).rank(ascending=False, method='min').astype(int)

    all_rows.append(sub)

result = pd.concat(all_rows, ignore_index=True)
print(f"予測完了: {len(result):,}頭 ({time.time()-t0:.0f}秒)")

# ── コンポジット & gap計算 ──
cd = pd.to_numeric(result['cur_diff'], errors='coerce')
sd = pd.to_numeric(result['sub_diff'], errors='coerce')
cr = pd.to_numeric(result['cur_rank'], errors='coerce')
sr = pd.to_numeric(result['sub_rank'], errors='coerce')
cs_cur = pd.to_numeric(result['cur_score'], errors='coerce')
cs_sub = pd.to_numeric(result['sub_score'], errors='coerce')

result['combo_diff']    = cd.fillna(0) + sd.fillna(0)
n_h = result.groupby(race_keys)['cur_rank'].transform('count')
result['rank_sum']      = cr.fillna(n_h+1) + sr.fillna(n_h+1)
result['combo_rank']    = result.groupby(race_keys)['combo_diff'].rank(ascending=False, method='min')
result['rank_sum_rank'] = result.groupby(race_keys)['rank_sum'].rank(ascending=True, method='min')

# cur_gap / sub_gap: 1位スコアと2位スコアの差
def calc_gap(df_group, score_col, rank_col):
    scores = df_group[score_col].dropna().sort_values(ascending=False).values
    if len(scores) >= 2:
        return scores[0] - scores[1]
    return np.nan

for key, grp in result.groupby(race_keys):
    idx = grp.index
    cur_g = calc_gap(grp, 'cur_score', 'cur_rank')
    sub_g = calc_gap(grp, 'sub_score', 'sub_rank')
    result.loc[idx, 'cur_gap'] = cur_g
    result.loc[idx, 'sub_gap'] = sub_g

result['cur_gap'] = pd.to_numeric(result['cur_gap'], errors='coerce')
result['sub_gap'] = pd.to_numeric(result['sub_gap'], errors='coerce')

print(f"gap計算完了")

# ── 馬連配当の付与（レース内top2の組み合わせ） ──
# top2 = combo_rank 1位・2位の馬 → その組み合わせで馬連配当
def get_umaren_payout(grp):
    top2_idx = grp.nsmallest(2, 'rank_sum_rank').index if len(grp) >= 2 else grp.index[:1]
    top2_places = grp.loc[top2_idx, '着順_num'].values
    if len(top2_places) == 2 and set(top2_places) == {1.0, 2.0}:
        pay = grp['馬連'].dropna().iloc[0] if '馬連' in grp.columns and len(grp['馬連'].dropna()) > 0 else 0
        return top2_idx, True, pay
    return top2_idx, False, 0

# ──────────────────────────────────────────────────────
# 分析 ① 勝負レース条件別 単勝ROI
# ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("① 勝負レース条件別 単勝ROI（N≥100）")
print("="*60)

conditions = {
    '全馬':                  result,
    'combo_rank=1':          result[result['combo_rank']==1],
    'combo_rank=1 & sd≥10':  result[(result['combo_rank']==1) & (sd>=10)],
    'combo_rank=1 & sd≥20':  result[(result['combo_rank']==1) & (sd>=20)],
    'combo_rank=1 & sd≥25':  result[(result['combo_rank']==1) & (sd>=25)],
    '両Rnk=1':               result[(cr==1) & (sr==1)],
    '両Rnk=1 & sd≥10':       result[(cr==1) & (sr==1) & (sd>=10)],
    '両Rnk=1 & sd≥20':       result[(cr==1) & (sr==1) & (sd>=20)],
    'odds≥3 & combo=1':      result[(result['combo_rank']==1) & (result['単勝オッズ']>=3)],
    'odds≥3 & 両Rnk=1':      result[(cr==1) & (sr==1) & (result['単勝オッズ']>=3)],
}

print(f"{'条件':<30} {'N':>6} {'勝率':>6} {'単勝ROI':>9}")
print("-"*55)
for name, sub in conditions.items():
    n = len(sub)
    if n < 50: continue
    wins = int(sub['target_win'].sum())
    ret  = sub.loc[sub['target_win']==1, '単勝配当'].sum()
    roi  = ret / (n*100) - 1
    print(f"{name:<30} {n:>6,} {wins/n:>6.1%} {roi:>+9.1%}")

# ──────────────────────────────────────────────────────
# 分析 ② cur_gap 別 単勝ROI（勝負レース内で）
# ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("② cur_gap 別 単勝ROI（combo_rank=1 & sd≥20）")
print("="*60)

base = result[(result['combo_rank']==1) & (sd>=20)].copy()
print(f"{'cur_gap閾値':<15} {'N':>6} {'勝率':>6} {'単勝ROI':>9}")
print("-"*40)
for thresh in [0, 2, 4, 6, 8, 10, 12, 15, 20]:
    sub = base[base['cur_gap'] >= thresh] if thresh > 0 else base
    n = len(sub)
    if n < 30: break
    wins = int(sub['target_win'].sum())
    ret  = sub.loc[sub['target_win']==1, '単勝配当'].sum()
    roi  = ret / (n*100) - 1
    print(f"cur_gap≥{thresh:<8} {n:>6,} {wins/n:>6.1%} {roi:>+9.1%}")

# ──────────────────────────────────────────────────────
# 分析 ③ sub_gap 別 単勝ROI
# ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("③ sub_gap 別 単勝ROI（combo_rank=1 & sd≥20）")
print("="*60)

print(f"{'sub_gap閾値':<15} {'N':>6} {'勝率':>6} {'単勝ROI':>9}")
print("-"*40)
for thresh in [0, 2, 4, 6, 8, 10, 12, 15, 20]:
    sub = base[base['sub_gap'] >= thresh] if thresh > 0 else base
    n = len(sub)
    if n < 30: break
    wins = int(sub['target_win'].sum())
    ret  = sub.loc[sub['target_win']==1, '単勝配当'].sum()
    roi  = ret / (n*100) - 1
    print(f"sub_gap≥{thresh:<8} {n:>6,} {wins/n:>6.1%} {roi:>+9.1%}")

# ──────────────────────────────────────────────────────
# 分析 ④ gap大 vs gap小: 単勝 vs 馬連 どちらが有利か
# ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("④ gap大小別: 単勝ROI vs 馬連ROI（combo_rank=1 & sd≥20）")
print("   馬連 = rank_sum_rank 1位+2位の組み合わせ")
print("="*60)

# レース単位で馬連的中/払戻を計算
race_results = []
for key, grp in base.groupby(race_keys):
    top1 = grp[grp['rank_sum_rank']==1]
    top2 = grp[grp['rank_sum_rank']<=2]
    if len(top1) == 0: continue

    row = top1.iloc[0]
    cur_g = float(row.get('cur_gap', np.nan))
    sub_g = float(row.get('sub_gap', np.nan))

    # 単勝
    tan_win = int(row['target_win'])
    tan_pay = float(row['単勝配当']) if tan_win else 0

    # 馬連: top2の2頭の着順が1・2位か
    if len(top2) >= 2:
        places = sorted(top2['着順_num'].values)
        uma_win = 1 if places[0] == 1 and places[1] == 2 else 0
        uma_pay = float(top2['馬連'].dropna().iloc[0]) if uma_win and '馬連' in top2.columns and len(top2['馬連'].dropna()) > 0 else 0
    else:
        uma_win = 0; uma_pay = 0

    race_results.append({
        'cur_gap': cur_g, 'sub_gap': sub_g,
        'tan_win': tan_win, 'tan_pay': tan_pay,
        'uma_win': uma_win, 'uma_pay': uma_pay,
        'odds': float(row.get('単勝オッズ', np.nan)),
    })

rdf = pd.DataFrame(race_results)

print(f"\n{'gap区分':<20} {'N':>5} | {'単勝的中率':>8} {'単勝ROI':>8} | {'馬連的中率':>8} {'馬連ROI':>8}")
print("-"*70)

gap_splits = [
    ('cur_gap < 5（拮抗）',  rdf['cur_gap'] < 5),
    ('cur_gap 5-10',         (rdf['cur_gap'] >= 5) & (rdf['cur_gap'] < 10)),
    ('cur_gap 10-15',        (rdf['cur_gap'] >= 10) & (rdf['cur_gap'] < 15)),
    ('cur_gap ≥ 15（独走）',  rdf['cur_gap'] >= 15),
]

for label, mask in gap_splits:
    sub = rdf[mask]
    n = len(sub)
    if n < 10: continue
    tan_roi = sub['tan_pay'].sum() / (n*100) - 1
    uma_roi = sub['uma_pay'].sum() / (n*100) - 1
    tan_rate = sub['tan_win'].mean()
    uma_rate = sub['uma_win'].mean()
    winner = '← 単勝' if tan_roi > uma_roi else '← 馬連'
    print(f"{label:<20} {n:>5} | {tan_rate:>8.1%} {tan_roi:>+8.1%} | {uma_rate:>8.1%} {uma_roi:>+8.1%}  {winner}")

# sub_gap版
print()
gap_splits_sub = [
    ('sub_gap < 3（拮抗）',  rdf['sub_gap'] < 3),
    ('sub_gap 3-6',          (rdf['sub_gap'] >= 3) & (rdf['sub_gap'] < 6)),
    ('sub_gap 6-10',         (rdf['sub_gap'] >= 6) & (rdf['sub_gap'] < 10)),
    ('sub_gap ≥ 10（独走）',  rdf['sub_gap'] >= 10),
]

for label, mask in gap_splits_sub:
    sub = rdf[mask]
    n = len(sub)
    if n < 10: continue
    tan_roi = sub['tan_pay'].sum() / (n*100) - 1
    uma_roi = sub['uma_pay'].sum() / (n*100) - 1
    tan_rate = sub['tan_win'].mean()
    uma_rate = sub['uma_win'].mean()
    winner = '← 単勝' if tan_roi > uma_roi else '← 馬連'
    print(f"{label:<20} {n:>5} | {tan_rate:>8.1%} {tan_roi:>+8.1%} | {uma_rate:>8.1%} {uma_roi:>+8.1%}  {winner}")

# ──────────────────────────────────────────────────────
# 分析 ⑤ オッズ帯別 単勝ROI（勝負レース内）
# ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("⑤ オッズ帯別 単勝ROI（combo_rank=1 & sd≥20）")
print("="*60)

print(f"{'オッズ帯':<15} {'N':>6} {'勝率':>6} {'単勝ROI':>9}")
print("-"*40)
odds_bands = [
    ('1.0-1.9倍', (base['単勝オッズ'] < 2)),
    ('2.0-2.9倍', (base['単勝オッズ'] >= 2) & (base['単勝オッズ'] < 3)),
    ('3.0-4.9倍', (base['単勝オッズ'] >= 3) & (base['単勝オッズ'] < 5)),
    ('5.0-9.9倍', (base['単勝オッズ'] >= 5) & (base['単勝オッズ'] < 10)),
    ('10倍以上',  (base['単勝オッズ'] >= 10)),
]
for label, mask in odds_bands:
    sub = base[mask]
    n = len(sub)
    if n < 20: continue
    wins = int(sub['target_win'].sum())
    ret  = sub.loc[sub['target_win']==1, '単勝配当'].sum()
    roi  = ret / (n*100) - 1
    print(f"{label:<15} {n:>6,} {wins/n:>6.1%} {roi:>+9.1%}")

# ──────────────────────────────────────────────────────
# 推奨ルールサマリー
# ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("【推奨ルール（案）】")
print("="*60)
print("""
■ 勝負レース条件
  → combo_rank=1 AND sub_diff≥20 AND 単勝オッズ≥3.0

■ 単勝 vs 馬連 判断
  → cur_gap≥10 AND sub_gap≥6  : 単勝（1頭に絞る）
  → cur_gap<10 OR  sub_gap<6  : 馬連（rank_sum_rank 1位+2位）

■ 見送り
  → 単勝オッズ1倍台
  → 両モデルとも欠損（モデル未対応コース）
""")
print("完了")

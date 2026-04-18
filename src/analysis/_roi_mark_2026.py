"""
2026年1〜3月 印別ROI計算（単勝＋複勝）
・models_2025/ のモデル使用
・all_venues_features_2026test.csv でテスト
・現在のHTML印ロジックを適用し、推奨金額ベースのROIを計算
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

# ── モデル読み込み ──────────────────────────────────────────
print("モデル読み込み中...")
with open(os.path.join(model_dir, 'model_info.json'), encoding='utf-8') as f:
    cur_info = json.load(f)
cur_features = cur_info['features']; cur_models = cur_info['models']
with open(os.path.join(model_dir, 'ranker', 'ranker_info.json'), encoding='utf-8') as f:
    cur_rankers = json.load(f).get('rankers', {})
with open(os.path.join(model_dir, 'submodel', 'submodel_info.json'), encoding='utf-8') as f:
    sub_info = json.load(f)
sub_features = sub_info['features']; sub_models = sub_info['models']
with open(os.path.join(model_dir, 'submodel_ranker', 'class_ranker_info.json'), encoding='utf-8') as f:
    sub_rankers = json.load(f).get('rankers', {})
all_feats = list(set(cur_features + sub_features))

# ── テストデータ ────────────────────────────────────────────
test_path = os.path.join(base_dir, 'data', 'processed', 'all_venues_features_2026test.csv')
print(f"データ読み込み: {test_path}")
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
mask_da = (df['_surface'] == 'ダ') & (df['_dist_band'].isin(['中距離', '長距離']))
df.loc[mask_da, '_dist_band'] = '中長距離'
df['_cls_group'] = df['クラス_rank'].apply(get_class_group) if 'クラス_rank' in df.columns else '3勝以上'
df['sub_key']    = df['_surface'] + '_' + df['_dist_band'] + '_' + df['_cls_group']
df['日付_num']   = pd.to_numeric(df['日付'], errors='coerce')
df['月']         = df['日付_num'] // 100 % 100
for col in all_feats:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
print(f"テストデータ: {len(df):,}頭  期間: {int(df['日付_num'].min())}〜{int(df['日付_num'].max())}")

# ── モデルキャッシュ ────────────────────────────────────────
print("モデルキャッシュ中...")
cur_mc={}; cur_rc={}; sub_mc={}; sub_rc={}
for ck in df['cur_key'].dropna().unique():
    if ck in cur_models:
        p = os.path.join(model_dir, cur_models[ck]['win'])
        if os.path.exists(p):
            with open(p,'rb') as f2: m=pickle.load(f2)
            cur_mc[ck]=(m, m.booster_.feature_name())
    if ck in cur_rankers:
        p = os.path.join(model_dir, 'ranker', cur_rankers[ck])
        if os.path.exists(p):
            with open(p,'rb') as f2: cur_rc[ck]=pickle.load(f2)
for sk in df['sub_key'].dropna().unique():
    if sk in sub_models:
        p = os.path.join(model_dir, 'submodel', sub_models[sk]['win'])
        if os.path.exists(p):
            with open(p,'rb') as f2: m=pickle.load(f2)
            sub_mc[sk]=(m, m.booster_.feature_name())
    if sk in sub_rankers:
        p = os.path.join(model_dir, 'submodel_ranker', sub_rankers[sk])
        if os.path.exists(p):
            with open(p,'rb') as f2: sub_rc[sk]=pickle.load(f2)
print(f"キャッシュ完了: cur={len(cur_mc)} sub={len(sub_mc)}")

# ── 予測（レース別） ────────────────────────────────────────
race_keys = [c for c in ['開催','Ｒ'] if c in df.columns]
print("予測中...")
t0=time.time(); all_rows=[]

for gk, idx in df.groupby(race_keys, sort=False).groups.items():
    sub = df.loc[idx].copy()
    ck = sub['cur_key'].iloc[0]
    sk = sub['sub_key'].iloc[0]

    sub['cur_rank'] = np.nan; sub['cur_diff'] = np.nan; sub['cur_score'] = np.nan
    sub['sub_rank'] = np.nan; sub['sub_diff'] = np.nan; sub['sub_score'] = np.nan

    if ck in cur_mc:
        m, wf = cur_mc[ck]
        for c in wf:
            if c not in sub.columns: sub[c] = np.nan
        prob = m.predict_proba(sub[wf])[:, 1]
        st = cur_models[ck].get('stats', {}); wm=st.get('win_mean',prob.mean()); ws=st.get('win_std',prob.std())
        cs = 50 + 10*(prob-wm)/(ws if ws>0 else 1)
        rm = prob.mean(); rs = prob.std()
        sub['cur_score'] = cs
        sub['cur_diff']  = 50 + 10*(prob-rm)/(rs if rs>0 else 1) - cs
        if ck in cur_rc:
            sc = cur_rc[ck].predict(sub[cur_features])
            sub['cur_rank'] = pd.Series(sc, index=sub.index).rank(ascending=False, method='min').astype(int)

    if sk in sub_mc:
        m, wf = sub_mc[sk]
        for c in wf:
            if c not in sub.columns: sub[c] = np.nan
        prob = m.predict_proba(sub[wf])[:, 1]
        st = sub_models[sk].get('stats', {}); wm=st.get('win_mean',prob.mean()); ws=st.get('win_std',prob.std())
        cs = 50 + 10*(prob-wm)/(ws if ws>0 else 1)
        rm = prob.mean(); rs = prob.std()
        sub['sub_score'] = cs
        sub['sub_diff']  = 50 + 10*(prob-rm)/(rs if rs>0 else 1) - cs
        if sk in sub_rc:
            sc = sub_rc[sk].predict(sub[wf])
            sub['sub_rank'] = pd.Series(sc, index=sub.index).rank(ascending=False, method='min').astype(int)

    # combo_gap（1位〜2位のスコア差）
    for score_col, gap_col in [('cur_score','cur_gap'), ('sub_score','sub_gap')]:
        sc2 = sub[score_col].dropna().sort_values(ascending=False).values
        sub[gap_col] = (sc2[0]-sc2[1]) if len(sc2) >= 2 else np.nan
    sub['combo_gap'] = sub.get('cur_gap', pd.Series(0, index=sub.index)).fillna(0) + \
                       sub.get('sub_gap', pd.Series(0, index=sub.index)).fillna(0)

    all_rows.append(sub)

result = pd.concat(all_rows, ignore_index=True)
print(f"予測完了: {len(result):,}頭 ({time.time()-t0:.0f}秒)")

# ── 印ロジック（06_predict_from_card.py の generate_html と同一）──
cr = pd.to_numeric(result['cur_rank'], errors='coerce')
sr = pd.to_numeric(result['sub_rank'], errors='coerce')
sd = pd.to_numeric(result['sub_diff'], errors='coerce')
cg = pd.to_numeric(result['combo_gap'], errors='coerce')
od = pd.to_numeric(result['単勝オッズ'], errors='coerce')

both_r1   = (cr == 1) & (sr == 1)
star      = (cr <= 3) & (sr <= 3) & ~both_r1
odds_ok3  = od.isna() | (od >= 3)
odds_ok5  = od.isna() | (od >= 5)

cd = pd.to_numeric(result['cur_diff'], errors='coerce')
mask_gekiatu = both_r1 & (cd >= 10) & (sd >= 10) & odds_ok5
mask_maru2   = both_r1 & (sd >= 10) & odds_ok3 & ~mask_gekiatu
cr2          = pd.to_numeric(result['cur_rank'], errors='coerce')
sr2          = pd.to_numeric(result['sub_rank'], errors='coerce')
mask_diamond = (cr2 <= 2) & (sr2 <= 2) & ~both_r1 & (sd >= 10) & odds_ok5
mask_hoshi   = star & ~((cr2 <= 2) & (sr2 <= 2)) & odds_ok5 & (sd >= 10)

def mark_of(i):
    if mask_gekiatu.iloc[i]: return '激熱'
    if mask_maru2.iloc[i]:   return '〇'
    if mask_diamond.iloc[i]: return '▲'
    if mask_hoshi.iloc[i]:   return '☆'
    return ''

result['_印'] = [mark_of(i) for i in range(len(result))]

# ── 推奨金額（HTML現在値）──────────────────────────────────
# 激熱: 単2000   ◎: 単1000   〇: 単500   ☆: 単300+複200
BETS = {
    '激熱': {'単': 500, '複': 0},
    '〇':   {'単': 500, '複': 0},
    '▲':   {'単': 300, '複': 0},
    '☆':   {'単': 300, '複': 0},
}

# ── ROI集計 ──────────────────────────────────────────────
print(f"\n{'='*70}")
print("  2026年1〜3月 印別ROI（HTML推奨金額ベース）")
print(f"{'='*70}")
print(f"{'印':<5} {'頭数':>5} {'単的中':>6} {'単勝率':>7}  {'投資':>10}  {'払戻':>10}  {'損益':>10}  {'ROI':>8}")
print('-'*70)

summary = {}
for mark in ['激熱', '〇', '▲', '☆']:
    sub = result[result['_印'] == mark]
    if sub.empty:
        print(f"{mark:<5} {'該当なし':>5}")
        continue

    n          = len(sub)
    bet_tan    = BETS[mark]['単']
    bet_fuku   = BETS[mark]['複']

    wins       = int(sub['target_win'].sum())
    win_pay    = sub.loc[sub['target_win']==1, '単勝配当'].sum()   # 100円あたり払戻
    tan_inv    = n * bet_tan
    tan_pay    = win_pay * (bet_tan / 100) if bet_tan > 0 else 0

    places     = int(sub['target_place'].sum())
    place_pay  = sub.loc[sub['target_place']==1, '複勝配当'].sum()
    fuku_inv   = n * bet_fuku
    fuku_pay   = place_pay * (bet_fuku / 100) if bet_fuku > 0 else 0

    total_inv  = tan_inv + fuku_inv
    total_pay  = tan_pay + fuku_pay
    roi        = (total_pay - total_inv) / total_inv * 100 if total_inv > 0 else 0

    summary[mark] = {'n':n,'wins':wins,'places':places,
                     'tan_inv':tan_inv,'tan_pay':tan_pay,
                     'fuku_inv':fuku_inv,'fuku_pay':fuku_pay,
                     'total_inv':total_inv,'total_pay':total_pay,'roi':roi}

    fuku_str = f"  複({fuku_inv//1000:.0f}k→{fuku_pay/1000:.1f}k)" if bet_fuku > 0 else ''
    print(f"{mark:<5} {n:>5} {wins:>6} {wins/n:>7.1%}  {total_inv:>10,.0f}円  {total_pay:>10,.0f}円  {total_pay-total_inv:>+10,.0f}円  {roi:>+7.1f}%{fuku_str}")

print('-'*70)
total_inv_all = sum(v['total_inv'] for v in summary.values())
total_pay_all = sum(v['total_pay'] for v in summary.values())
total_roi_all = (total_pay_all - total_inv_all) / total_inv_all * 100 if total_inv_all else 0
total_n = sum(v['n'] for v in summary.values())
total_w = sum(v['wins'] for v in summary.values())
print(f"{'合計':<5} {total_n:>5} {total_w:>6} {total_w/total_n if total_n else 0:>7.1%}  "
      f"{total_inv_all:>10,.0f}円  {total_pay_all:>10,.0f}円  {total_pay_all-total_inv_all:>+10,.0f}円  {total_roi_all:>+7.1f}%")

# ── 月別内訳 ──────────────────────────────────────────────
print(f"\n{'='*70}")
print("  月別ROI（全印合算・推奨金額ベース）")
print(f"{'='*70}")
for month in sorted(result['月'].dropna().unique()):
    m_df = result[result['月'] == month]
    month_rows = []
    for mark in ['激熱', '〇', '▲', '☆']:
        sub = m_df[m_df['_印'] == mark]
        if sub.empty: continue
        n = len(sub)
        bet_tan  = BETS[mark]['単']
        bet_fuku = BETS[mark]['複']
        win_pay  = sub.loc[sub['target_win']==1, '単勝配当'].sum()
        place_pay= sub.loc[sub['target_place']==1, '複勝配当'].sum()
        inv = n*bet_tan + n*bet_fuku
        pay = win_pay*(bet_tan/100) + place_pay*(bet_fuku/100 if bet_fuku else 0)
        month_rows.append((n, inv, pay))
    if month_rows:
        mn = sum(r[0] for r in month_rows)
        mi = sum(r[1] for r in month_rows)
        mp = sum(r[2] for r in month_rows)
        mr = (mp-mi)/mi*100 if mi else 0
        print(f"  {int(month)}月: {mn:>3}頭  投資{mi:>8,.0f}円  払戻{mp:>8,.0f}円  ROI{mr:>+7.1f}%")

# ── ☆詳細（sd・gap条件別）──────────────────────────────────
print(f"\n{'='*70}")
print("  ☆ 条件絞り込み別ROI（単勝300+複勝200円ベース）")
print(f"{'='*70}")
hoshi_df = result[result['_印'] == '☆'].copy()
hoshi_sd  = pd.to_numeric(hoshi_df['sub_diff'],  errors='coerce')
hoshi_gap = pd.to_numeric(hoshi_df['combo_gap'], errors='coerce')
print(f"{'条件':<20} {'頭数':>5} {'単的中':>6} {'複的中':>6}  {'投資':>9}  {'払戻':>9}  {'ROI':>8}")
print('-'*70)
for sd_th in [0, 5, 10, 15, 20]:
    for gap_th in [0, 10, 20]:
        cond = (hoshi_sd >= sd_th) & (hoshi_gap >= gap_th)
        sub = hoshi_df[cond]
        if len(sub) == 0: continue
        n = len(sub)
        wp = sub.loc[sub['target_win']==1,   '単勝配当'].sum()
        pp = sub.loc[sub['target_place']==1, '複勝配当'].sum()
        inv = n*300 + n*200
        pay = wp*(300/100) + pp*(200/100)
        roi = (pay-inv)/inv*100 if inv else 0
        label = f"sd≥{sd_th} gap≥{gap_th}"
        print(f"  {label:<18} {n:>5} {int(sub['target_win'].sum()):>6} {int(sub['target_place'].sum()):>6}  {inv:>9,.0f}  {pay:>9,.0f}  {roi:>+7.1f}%")

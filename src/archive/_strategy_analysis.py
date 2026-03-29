"""
戦略多角的分析: 距離モデル × クラスモデルの組み合わせROI比較
バックテスト（2012テストデータ）＋ 実績（3/21・3/22）
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import pandas as pd, numpy as np, os, pickle, json, re, time

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, 'models')

# ─────────────────────────────────────────
# 共通ユーティリティ
# ─────────────────────────────────────────
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

def extract_venue(kaikai):
    m = re.search(r'\d+([^\d]+)', str(kaikai))
    return m.group(1) if m else str(kaikai)

# ─────────────────────────────────────────
# モデル読み込み
# ─────────────────────────────────────────
print("モデル情報読み込み中...")
with open(os.path.join(model_dir, 'model_info.json'), encoding='utf-8') as f: cur_info = json.load(f)
cur_features = cur_info['features']; cur_models = cur_info['models']
with open(os.path.join(model_dir, 'ranker', 'ranker_info.json'), encoding='utf-8') as f:
    cur_rankers = json.load(f).get('rankers', {})
with open(os.path.join(model_dir, 'submodel', 'submodel_info.json'), encoding='utf-8') as f: sub_info = json.load(f)
sub_features = sub_info['features']; sub_models = sub_info['models']
with open(os.path.join(model_dir, 'submodel_ranker', 'class_ranker_info.json'), encoding='utf-8') as f:
    sub_rankers = json.load(f).get('rankers', {})
all_feats = list(set(cur_features + sub_features))
print(f"距離モデル: {len(cur_models)}グループ / クラスモデル: {len(sub_models)}グループ")

# ─────────────────────────────────────────
# コンポジット戦略をレース単位で計算して付加
# ─────────────────────────────────────────
def add_combo_signals(result, race_keys):
    """result DFにコンポジット順位列を追加"""
    cd = pd.to_numeric(result['cur_diff'], errors='coerce')
    sd = pd.to_numeric(result['sub_diff'], errors='coerce')
    cr = pd.to_numeric(result['cur_rank'], errors='coerce')
    sr = pd.to_numeric(result['sub_rank'], errors='coerce')

    # 両モデル有効フラグ
    result['_both_valid'] = cd.notna() & sd.notna() & cr.notna() & sr.notna()

    # コンポジットdiff: 両方NaNでなければ足す（片方NaNは0扱い）
    result['combo_diff'] = cd.fillna(0) + sd.fillna(0)
    # コンポジットdiff 重み付き (距離重視 / クラス重視)
    result['combo_diff_cur'] = cd.fillna(0) * 1.5 + sd.fillna(0) * 0.5
    result['combo_diff_sub'] = cd.fillna(0) * 0.5 + sd.fillna(0) * 1.5

    # ランク和 (NaN → 大きい数字 = ペナルティ)
    n_horses = result.groupby(race_keys)['cur_rank'].transform('count')
    result['rank_sum'] = cr.fillna(n_horses + 1) + sr.fillna(n_horses + 1)

    # レース内コンポジットランク
    grp = result.groupby(race_keys)
    result['combo_rank']      = grp['combo_diff'].rank(ascending=False, method='min')
    result['combo_rank_cur']  = grp['combo_diff_cur'].rank(ascending=False, method='min')
    result['combo_rank_sub']  = grp['combo_diff_sub'].rank(ascending=False, method='min')
    result['rank_sum_rank']   = grp['rank_sum'].rank(ascending=True, method='min')

    return result

# ─────────────────────────────────────────
# ROI表示
# ─────────────────────────────────────────
def show_strategies(result, payout_col='単勝配当', place_col='複勝配当', odds_col=None):
    cd = pd.to_numeric(result['cur_diff'], errors='coerce')
    sd = pd.to_numeric(result['sub_diff'], errors='coerce')
    cr = pd.to_numeric(result['cur_rank'], errors='coerce')
    sr = pd.to_numeric(result['sub_rank'], errors='coerce')
    combo_r  = result['combo_rank']
    combo_rc = result['combo_rank_cur']
    combo_rs = result['combo_rank_sub']
    rsr      = result['rank_sum_rank']

    ods = pd.to_numeric(result[odds_col], errors='coerce') if odds_col and odds_col in result.columns else None

    def roi(mask, use_odds_min=0):
        if use_odds_min > 0 and ods is not None:
            mask = mask & (ods >= use_odds_min)
        bets = result[mask]
        n = len(bets)
        if n == 0: return None, 0, 0, 0, 0, 0
        wins   = int(bets['target_win'].sum())
        places = int(bets['target_place'].sum())
        ret_w  = bets.loc[bets['target_win']==1,   payout_col].sum()
        ret_p  = bets.loc[bets['target_place']==1, place_col].sum() if place_col in bets.columns else 0
        roi_w  = ret_w  / (n*100) - 1
        roi_p  = ret_p  / (n*100) - 1 if n > 0 else 0
        return roi_w, wins, n, roi_p, places/n, places

    def line(label, mask, use_odds_min=0):
        r, w, n, rp, pp, pl = roi(mask, use_odds_min)
        if r is None:
            print(f"  {label:<38} 該当なし")
            return
        ods_str = f" [オッズ≥{use_odds_min:.0f}]" if use_odds_min > 0 else ""
        print(f"  {label:<38} 単ROI{r:+.1%}({w}/{n}頭)  複勝率{pp:.0%} 複ROI{rp:+.1%}{ods_str}")

    print()
    print("  【既存戦略】")
    line("距離ランカー1位",              cr == 1)
    line("クラスランカー1位",            sr == 1)
    line("両ランカー1位",               (cr==1)&(sr==1))
    line("クラス◎(+20) & 距離Rnk1",    (sd>=20)&(cr==1))
    line("クラス〇以上(+15) & 距離Rnk1",(sd>=15)&(cr==1))
    line("距離◎(+20) & クラスRnk1",    (cd>=20)&(sr==1))

    print()
    print("  【新戦略: コンポジットスコア順位】")
    line("合計diff ランク1位 (均等)",    combo_r  == 1)
    line("合計diff ランク1位 (距離重視)",combo_rc == 1)
    line("合計diff ランク1位 (クラス重視)",combo_rs == 1)
    line("ランク和 最小",               rsr == 1)

    print()
    print("  【新戦略: クロスフィルター】")
    line("距離Rnk1 & クラスtop3以内",   (cr==1)&(sr<=3))
    line("クラスRnk1 & 距離top3以内",   (sr==1)&(cr<=3))
    line("距離Rnk1 & クラスtop5以内",   (cr==1)&(sr<=5))
    line("クラスRnk1 & 距離top5以内",   (sr==1)&(cr<=5))

    print()
    print("  【新戦略: diff閾値 + ランカー1位】")
    line("距離Rnk1 & 合計diff+10以上",  (cr==1)&((cd.fillna(0)+sd.fillna(0))>=10))
    line("距離Rnk1 & 合計diff+20以上",  (cr==1)&((cd.fillna(0)+sd.fillna(0))>=20))
    line("クラスRnk1 & 合計diff+10以上",(sr==1)&((cd.fillna(0)+sd.fillna(0))>=10))
    line("クラスRnk1 & 合計diff+20以上",(sr==1)&((cd.fillna(0)+sd.fillna(0))>=20))
    line("合計Rnk1 & 距離diff+0以上",   (combo_r==1)&(cd>=0))
    line("合計Rnk1 & クラスdiff+0以上", (combo_r==1)&(sd>=0))
    line("合計Rnk1 & 両方diff+0以上",   (combo_r==1)&(cd>=0)&(sd>=0))

    if ods is not None:
        print()
        print("  【オッズ下限フィルター (距離Rnk1単勝)】")
        for om in [2.0, 3.0, 4.0]:
            line(f"距離Rnk1", cr==1, use_odds_min=om)
        print()
        print("  【オッズ下限フィルター (クラス◎&距離Rnk1)】")
        for om in [2.0, 3.0]:
            line(f"クラス◎(+20)&距離Rnk1", (sd>=20)&(cr==1), use_odds_min=om)
        print()
        print("  【オッズ下限フィルター (合計diff Rnk1)】")
        for om in [2.0, 3.0]:
            line(f"合計diffランク1位", combo_r==1, use_odds_min=om)

# ─────────────────────────────────────────
# ① バックテスト（features_2012_test.csv）
# ─────────────────────────────────────────
test_path = os.path.join(base_dir, 'data', 'processed', 'features_2012_test.csv')
print(f"\nテストデータ読み込み中... ({os.path.getsize(test_path)//1024//1024}MB)")
t0 = time.time()
df = pd.read_csv(test_path, low_memory=False)
print(f"読み込み完了: {len(df):,}行 ({time.time()-t0:.1f}秒)")

df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
df['単勝配当'] = pd.to_numeric(df['単勝配当'], errors='coerce')
df = df.dropna(subset=['着順_num', '単勝配当'])
df['target_win']   = (df['着順_num'] == 1).astype(int)
df['target_place'] = (df['着順_num'] <= 3).astype(int)
df['複勝配当'] = pd.to_numeric(df['複勝配当'], errors='coerce')
df['会場']    = df['開催'].apply(extract_venue)
df['cur_key'] = df['会場'] + '_' + df['距離'].astype(str)
df['_surface']   = df['芝・ダ'].astype(str).str.strip()
df['_dist_band'] = df['距離'].apply(get_distance_band)
mask_da = (df['_surface'] == 'ダ') & (df['_dist_band'].isin(['中距離', '長距離']))
df.loc[mask_da, '_dist_band'] = '中長距離'
df['_cls_group'] = df['クラス_rank'].apply(get_class_group) if 'クラス_rank' in df.columns else '3勝以上'
df['sub_key'] = df['_surface'] + '_' + df['_dist_band'] + '_' + df['_cls_group']
for col in all_feats:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

# モデルキャッシュ
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

# レース別予測
race_keys_bt = [c for c in ['開催','Ｒ'] if c in df.columns]
df['Ｒ'] = pd.to_numeric(df.get('Ｒ', np.nan), errors='coerce')
print("レース別予測中...")
t0 = time.time()
all_rows = []
races = df.groupby(race_keys_bt, sort=False).groups
for i, (gk, idx) in enumerate(races.items()):
    sub = df.loc[idx].copy()
    ck = sub['cur_key'].iloc[0]; sk = sub['sub_key'].iloc[0]
    sub['cur_diff'] = np.nan; sub['cur_rank'] = np.nan
    sub['sub_diff'] = np.nan; sub['sub_rank'] = np.nan
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
            sub['cur_rank'] = pd.Series(sc, index=sub.index).rank(ascending=False, method='min').astype(int)
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
            sub['sub_rank'] = pd.Series(sc, index=sub.index).rank(ascending=False, method='min').astype(int)
    all_rows.append(sub)
    if (i+1) % 1000 == 0:
        print(f"  {i+1}/{len(races)}レース ({time.time()-t0:.0f}秒)")

result_bt = pd.concat(all_rows, ignore_index=True)
result_bt = add_combo_signals(result_bt, race_keys_bt)
print(f"バックテスト予測完了: {len(result_bt):,}頭 / {len(races)}レース ({time.time()-t0:.0f}秒)")

print()
print('=' * 70)
print('  ① バックテスト結果 (2012テストデータ)')
print('=' * 70)
show_strategies(result_bt, payout_col='単勝配当', place_col='複勝配当')

# ─────────────────────────────────────────
# ② 実績データ（3/21・3/22）
# ─────────────────────────────────────────
df_feat = pd.read_parquet(os.path.join(base_dir, 'data', 'processed', 'all_venues_features.parquet'))
df_latest = df_feat.sort_values('日付').groupby('馬名S').last().reset_index()
feat_subset = ['馬名S'] + [c for c in all_feats if c in df_latest.columns]
for ec in ['日付', '距離']:
    if ec in df_latest.columns and ec not in feat_subset: feat_subset.append(ec)

def predict_real(df_res, cur_date):
    df_res = df_res.copy()
    df_res['会場'] = df_res['場所'].astype(str).map(VENUE_MAP).fillna(df_res['場所'].astype(str))
    if 'コースマーク' in df_res.columns:
        cm = df_res['コースマーク'].astype(str).str.strip()
        df_res['会場'] = df_res['会場'] + cm.where(cm.isin(['A','B','C']), '')
    df_res['_surface'] = df_res['芝ダ'].astype(str).str.strip()
    df_res['cur_key']  = df_res['会場'] + '_' + df_res['_surface'] + df_res['距離'].astype(str)
    df_res['_dist_band'] = df_res['距離'].apply(get_distance_band)
    mask = (df_res['_surface'] == 'ダ') & (df_res['_dist_band'].isin(['中距離', '長距離']))
    df_res.loc[mask, '_dist_band'] = '中長距離'
    df_res['_cls_group'] = df_res['クラス_rank'].apply(get_class_group) if 'クラス_rank' in df_res.columns else '3勝以上'
    df_res['sub_key'] = df_res['_surface'] + '_' + df_res['_dist_band'] + '_' + df_res['_cls_group']
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
    rows = []
    for gk, idx in df_m.groupby(['場所','Ｒ'], sort=False).groups.items():
        sub = df_m.loc[idx].copy()
        ck = sub['cur_key'].iloc[0]; sk = sub['sub_key'].iloc[0]
        sub['cur_diff'] = np.nan; sub['cur_rank'] = np.nan
        sub['sub_diff'] = np.nan; sub['sub_rank'] = np.nan
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
                sub['cur_rank'] = pd.Series(sc, index=sub.index).rank(ascending=False, method='min').astype(int)
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
                sub['sub_rank'] = pd.Series(sc, index=sub.index).rank(ascending=False, method='min').astype(int)
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)

print()
print('=' * 70)
print('  ② 実績データ（3/21・3/22）')
print('=' * 70)

for fname, ds, cur_date in [
    ('出馬表形式3月21日結果確認.csv', '3月21日', pd.Timestamp(2026, 3, 21)),
    ('出馬表形式3月22日結果確認.csv', '3月22日', pd.Timestamp(2026, 3, 22)),
]:
    df_res = pd.read_csv(os.path.join(base_dir, 'data', 'raw', fname), encoding='cp932', low_memory=False)
    df_res['着_num'] = df_res['着'].apply(zen_to_num)
    df_res['target_win']   = (df_res['着_num'] == 1).astype(int)
    df_res['target_place'] = (df_res['着_num'] <= 3).astype(int)
    df_res['単勝配当'] = pd.to_numeric(df_res['単勝'], errors='coerce')
    df_res['複勝配当'] = pd.to_numeric(df_res['複勝'], errors='coerce')
    df_res['単オッズ_num'] = pd.to_numeric(df_res['単オッズ'], errors='coerce')
    result_r = predict_real(df_res, cur_date)
    result_r = add_combo_signals(result_r, ['場所','Ｒ'])

    print()
    print(f'  ─── {ds} ({len(result_r[["場所","Ｒ"]].drop_duplicates())}レース / {len(result_r)}頭) ───')
    show_strategies(result_r, payout_col='単勝配当', place_col='複勝配当', odds_col='単オッズ_num')

print()
print('=' * 70)
print('完了')
print('=' * 70)

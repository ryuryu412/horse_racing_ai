"""
オーバーナイト グリッドサーチ
① diff閾値 全組み合わせROI（バックテスト）
② 最有望戦略をクラス別・距離帯別・会場別に詳細分析
③ 複勝戦略の最適化
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import pandas as pd, numpy as np, os, pickle, json, re, time
from itertools import product

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

# ─────────────────────────────────────────
# テストデータ読み込み・前処理
# ─────────────────────────────────────────
test_path = os.path.join(base_dir, 'data', 'processed', 'features_2012_test.csv')
print(f"テストデータ読み込み中... ({os.path.getsize(test_path)//1024//1024}MB)")
t0 = time.time()
df = pd.read_csv(test_path, low_memory=False)
print(f"読み込み完了: {len(df):,}行 ({time.time()-t0:.1f}秒)")

df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
df['単勝配当'] = pd.to_numeric(df['単勝配当'], errors='coerce')
df = df.dropna(subset=['着順_num', '単勝配当'])
df['target_win']   = (df['着順_num'] == 1).astype(int)
df['target_place'] = (df['着順_num'] <= 3).astype(int)
df['複勝配当'] = pd.to_numeric(df['複勝配当'], errors='coerce')
df['会場']       = df['開催'].apply(extract_venue)
df['_surface']   = df['芝・ダ'].astype(str).str.strip()
df['cur_key']    = df['会場'] + '_' + df['距離'].astype(str)
df['_dist_band'] = df['距離'].apply(get_distance_band)
mask_da = (df['_surface'] == 'ダ') & (df['_dist_band'].isin(['中距離', '長距離']))
df.loc[mask_da, '_dist_band'] = '中長距離'
df['_cls_group'] = df['クラス_rank'].apply(get_class_group) if 'クラス_rank' in df.columns else '3勝以上'
df['sub_key'] = df['_surface'] + '_' + df['_dist_band'] + '_' + df['_cls_group']
for col in all_feats:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

# ─────────────────────────────────────────
# モデルキャッシュ
# ─────────────────────────────────────────
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

# ─────────────────────────────────────────
# レース別予測
# ─────────────────────────────────────────
race_keys = [c for c in ['開催','Ｒ'] if c in df.columns]
df['Ｒ'] = pd.to_numeric(df.get('Ｒ', np.nan), errors='coerce')
print("レース別予測中...")
t0 = time.time()
all_rows = []
races = df.groupby(race_keys, sort=False).groups
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
        rm = prob.mean(); rs_p = prob.std()
        sub['cur_diff'] = 50 + 10*(prob - rm)/(rs_p if rs_p > 0 else 1) - cs
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
        sub['sub_diff'] = 50 + 10*(prob - rm)/(rs_p if rs_p > 0 else 1) - cs
        if sk in sub_ranker_cache:
            sc = sub_ranker_cache[sk].predict(sub[wf])
            sub['sub_rank'] = pd.Series(sc, index=sub.index).rank(ascending=False, method='min').astype(int)
    all_rows.append(sub)
    if (i+1) % 1000 == 0:
        print(f"  {i+1}/{len(races)}レース ({time.time()-t0:.0f}秒)")

result = pd.concat(all_rows, ignore_index=True)
print(f"予測完了: {len(result):,}頭 ({time.time()-t0:.0f}秒)")

# コンポジット信号
cd = pd.to_numeric(result['cur_diff'], errors='coerce')
sd = pd.to_numeric(result['sub_diff'], errors='coerce')
cr = pd.to_numeric(result['cur_rank'], errors='coerce')
sr = pd.to_numeric(result['sub_rank'], errors='coerce')
result['combo_diff'] = cd.fillna(0) + sd.fillna(0)
n_h = result.groupby(race_keys)['cur_rank'].transform('count')
result['rank_sum']   = cr.fillna(n_h+1) + sr.fillna(n_h+1)
result['combo_rank'] = result.groupby(race_keys)['combo_diff'].rank(ascending=False, method='min')
result['rank_sum_rank'] = result.groupby(race_keys)['rank_sum'].rank(ascending=True, method='min')

# ─────────────────────────────────────────
# ヘルパー
# ─────────────────────────────────────────
def roi_stats(df_bets):
    n = len(df_bets)
    if n == 0: return None
    wins   = int(df_bets['target_win'].sum())
    places = int(df_bets['target_place'].sum())
    ret_w  = df_bets.loc[df_bets['target_win']==1, '単勝配当'].sum()
    ret_p  = df_bets.loc[df_bets['target_place']==1, '複勝配当'].sum()
    roi_w  = ret_w  / (n*100) - 1
    roi_p  = ret_p  / (n*100) - 1 if pd.notna(ret_p) else np.nan
    return {'n': n, 'wins': wins, 'places': places,
            'roi_w': roi_w, 'roi_p': roi_p, 'place_rate': places/n}

# ─────────────────────────────────────────
# ① 全diff閾値グリッドサーチ（単勝）
# ─────────────────────────────────────────
print()
print('=' * 80)
print('  ① diff閾値グリッドサーチ（バックテスト 単勝ROI / N≥200頭）')
print('=' * 80)

THRESHOLDS = list(range(0, 45, 5))  # 0,5,10,15,20,25,30,35,40

records = []
# (a) クラスRnk1 & 距離diff≥X
for ct in THRESHOLDS:
    mask = (sr == 1) & (cd >= ct)
    s = roi_stats(result[mask])
    if s: records.append({'戦略': f'クラスRnk1 & 距離diff≥{ct}', **s})

# (b) 距離Rnk1 & クラスdiff≥X
for ct in THRESHOLDS:
    mask = (cr == 1) & (sd >= ct)
    s = roi_stats(result[mask])
    if s: records.append({'戦略': f'距離Rnk1 & クラスdiff≥{ct}', **s})

# (c) 両Rnk1 & クラスdiff≥X
for ct in THRESHOLDS:
    mask = (cr == 1) & (sr == 1) & (sd >= ct)
    s = roi_stats(result[mask])
    if s: records.append({'戦略': f'両Rnk1 & クラスdiff≥{ct}', **s})

# (d) 両Rnk1 & 距離diff≥X
for ct in THRESHOLDS:
    mask = (cr == 1) & (sr == 1) & (cd >= ct)
    s = roi_stats(result[mask])
    if s: records.append({'戦略': f'両Rnk1 & 距離diff≥{ct}', **s})

# (e) 合計Rnk1 & クラスdiff≥X
for ct in THRESHOLDS:
    mask = (result['combo_rank'] == 1) & (sd >= ct)
    s = roi_stats(result[mask])
    if s: records.append({'戦略': f'合計Rnk1 & クラスdiff≥{ct}', **s})

# (f) 合計Rnk1 & 距離diff≥X
for ct in THRESHOLDS:
    mask = (result['combo_rank'] == 1) & (cd >= ct)
    s = roi_stats(result[mask])
    if s: records.append({'戦略': f'合計Rnk1 & 距離diff≥{ct}', **s})

# (g) クラスdiff≥X & 距離diff≥Y（マトリクス）
for ct, st2 in product([0,10,15,20,25], [0,10,15,20,25]):
    if ct == 0 and st2 == 0: continue
    mask = (cd >= ct) & (sd >= st2)
    s = roi_stats(result[mask])
    if s: records.append({'戦略': f'距離diff≥{ct} & クラスdiff≥{st2}', **s})

# (h) ランカーtopN クロス
for n1, n2 in [(1,1),(1,2),(1,3),(1,5),(2,1),(3,1),(5,1)]:
    mask = (cr <= n1) & (sr <= n2)
    s = roi_stats(result[mask])
    if s: records.append({'戦略': f'距離top{n1} & クラスtop{n2}', **s})

df_grid = pd.DataFrame(records)
df_grid = df_grid[df_grid['n'] >= 200].sort_values('roi_w', ascending=False)

print(f"\n{'戦略':<40} {'単ROI':>8}  {'N':>6}  {'的中率':>6}  {'複勝率':>6}  {'複ROI':>8}")
print('-' * 85)
for _, r in df_grid.head(40).iterrows():
    print(f"  {r['戦略']:<38} {r['roi_w']:>+8.1%}  {r['n']:>6,}  {r['wins']/r['n']:>6.1%}  {r['place_rate']:>6.1%}  {r['roi_p']:>+8.1%}")

# ─────────────────────────────────────────
# ② クラス別 × 戦略別詳細
# ─────────────────────────────────────────
print()
print('=' * 80)
print('  ② クラス別 × 戦略別 単勝ROI（バックテスト）')
print('=' * 80)

# バックテストROI上位だった戦略を詳細分析
top_strategies = [
    ('クラスRnk1 & 合計diff≥20',  (sr==1) & ((cd.fillna(0)+sd.fillna(0))>=20)),
    ('距離◎(+20) & クラスRnk1',   (cd>=20) & (sr==1)),
    ('合計Rnk1 & 両方diff+0以上',  (result['combo_rank']==1) & (cd>=0) & (sd>=0)),
    ('距離Rnk1 & クラスtop3',      (cr==1) & (sr<=3)),
    ('クラスRnk1 & 距離top3',      (sr==1) & (cr<=3)),
    ('両ランカー1位',               (cr==1) & (sr==1)),
    ('クラス◎ & 距離Rnk1',         (sd>=20) & (cr==1)),
]

CLASS_ORDER = ['新馬', '未勝利', '1勝', '2勝', '3勝以上']
for sname, smask in top_strategies:
    print(f"\n  【{sname}】")
    print(f"  {'クラス':<10} {'単ROI':>8}  {'N':>5}  {'的中率':>6}  {'複ROI':>8}")
    for cls in CLASS_ORDER:
        cmask = smask & (result['_cls_group'] == cls)
        s = roi_stats(result[cmask])
        if s and s['n'] >= 5:
            print(f"    {cls:<8} {s['roi_w']:>+8.1%}  {s['n']:>5,}  {s['wins']/s['n']:>6.1%}  {s['roi_p']:>+8.1%}")

# ─────────────────────────────────────────
# ③ 距離帯別 × 戦略別詳細
# ─────────────────────────────────────────
print()
print('=' * 80)
print('  ③ 距離帯別 × 戦略別 単勝ROI（バックテスト）')
print('=' * 80)

DIST_ORDER = ['短距離', 'マイル', '中距離', '中長距離', '長距離']
for sname, smask in top_strategies:
    print(f"\n  【{sname}】")
    print(f"  {'距離帯':<10} {'芝/ダ':<4} {'単ROI':>8}  {'N':>5}  {'的中率':>6}  {'複ROI':>8}")
    for dist in DIST_ORDER:
        for surf in ['芝', 'ダ']:
            dmask = smask & (result['_dist_band'] == dist) & (result['_surface'] == surf)
            s = roi_stats(result[dmask])
            if s and s['n'] >= 5:
                print(f"    {dist:<8} {surf:<4} {s['roi_w']:>+8.1%}  {s['n']:>5,}  {s['wins']/s['n']:>6.1%}  {s['roi_p']:>+8.1%}")

# ─────────────────────────────────────────
# ④ 会場別 × 最有望戦略
# ─────────────────────────────────────────
print()
print('=' * 80)
print('  ④ 会場別 ROI（トップ3戦略）')
print('=' * 80)

top3 = top_strategies[:3]
venues = sorted(result['会場'].dropna().unique())
for sname, smask in top3:
    print(f"\n  【{sname}】")
    print(f"  {'会場':<8} {'単ROI':>8}  {'N':>5}  {'的中率':>6}")
    for v in venues:
        vmask = smask & (result['会場'] == v)
        s = roi_stats(result[vmask])
        if s and s['n'] >= 10:
            print(f"    {v:<6} {s['roi_w']:>+8.1%}  {s['n']:>5,}  {s['wins']/s['n']:>6.1%}")

# ─────────────────────────────────────────
# ⑤ 複勝専用戦略の最適化
# ─────────────────────────────────────────
print()
print('=' * 80)
print('  ⑤ 複勝専用グリッドサーチ（N≥200頭）')
print('=' * 80)

place_records = []
for ct in THRESHOLDS:
    for st2 in THRESHOLDS:
        if ct == 0 and st2 == 0: continue
        mask = (cr==1) & (sr==1) & (cd >= ct) & (sd >= st2)
        s = roi_stats(result[mask])
        if s and s['n'] >= 20:
            place_records.append({'戦略': f'両Rnk1 cd≥{ct} sd≥{st2}',
                                   'n': s['n'], 'places': s['places'],
                                   'place_rate': s['place_rate'], 'roi_p': s['roi_p'],
                                   'roi_w': s['roi_w']})
# 単独条件も
for ct in THRESHOLDS:
    mask = (cr==1) & (cd >= ct)
    s = roi_stats(result[mask])
    if s and s['n'] >= 20:
        place_records.append({'戦略': f'距離Rnk1 cd≥{ct}',
                               'n': s['n'], 'places': s['places'],
                               'place_rate': s['place_rate'], 'roi_p': s['roi_p'],
                               'roi_w': s['roi_w']})
    mask = (sr==1) & (sd >= ct)
    s = roi_stats(result[mask])
    if s and s['n'] >= 20:
        place_records.append({'戦略': f'クラスRnk1 sd≥{ct}',
                               'n': s['n'], 'places': s['places'],
                               'place_rate': s['place_rate'], 'roi_p': s['roi_p'],
                               'roi_w': s['roi_w']})

df_place = pd.DataFrame(place_records).sort_values('roi_p', ascending=False)
print(f"\n{'戦略':<35} {'複ROI':>8}  {'複勝率':>6}  {'N':>6}  {'単ROI':>8}")
print('-' * 75)
for _, r in df_place.head(30).iterrows():
    print(f"  {r['戦略']:<33} {r['roi_p']:>+8.1%}  {r['place_rate']:>6.1%}  {r['n']:>6,}  {r['roi_w']:>+8.1%}")

# ─────────────────────────────────────────
# ⑥ 曜日・月別 安定性チェック
# ─────────────────────────────────────────
if '日付' in result.columns or '年月日' in result.columns:
    date_col = '日付' if '日付' in result.columns else '年月日'
    print()
    print('=' * 80)
    print('  ⑥ 月別安定性（上位3戦略）')
    print('=' * 80)
    try:
        result['_date'] = pd.to_numeric(result[date_col], errors='coerce')
        result['_month'] = result['_date'].apply(
            lambda v: (2000 + int(v)//10000)*100 + (int(v)//100)%100 if pd.notna(v) and v > 0 else np.nan)
        for sname, smask in top3:
            print(f"\n  【{sname}】")
            monthly = []
            for mon, grp in result[smask].groupby('_month'):
                s = roi_stats(grp)
                if s and s['n'] >= 3:
                    monthly.append({'月': int(mon), **s})
            df_mon = pd.DataFrame(monthly)
            if len(df_mon) > 0:
                print(f"  月別ROI中央値: {df_mon['roi_w'].median():+.1%}  標準偏差: {df_mon['roi_w'].std():.1%}  プラス月: {(df_mon['roi_w']>0).mean():.0%}")
                print(f"  {'月':>8} {'単ROI':>8}  {'N':>5}")
                for _, r in df_mon.sort_values('月').iterrows():
                    bar = '█' * max(0, int((r['roi_w']+1)*5)) if r['roi_w'] > -1 else ''
                    print(f"    {int(r['月']):>6} {r['roi_w']:>+8.1%}  {r['n']:>5,}  {bar}")
    except Exception as e:
        print(f"  (月別分析エラー: {e})")

print()
print('=' * 80)
print('グリッドサーチ完了')
print('=' * 80)

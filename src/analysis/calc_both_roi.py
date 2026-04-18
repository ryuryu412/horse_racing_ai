"""
両モデル一致時のROI計算スクリプト
2012テストデータで 距離モデル × クラスモデル の組み合わせROIを算出
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import os, pickle, json, re, time

def extract_venue(kaikai):
    m = re.search(r'\d+([^\d]+)', str(kaikai))
    return m.group(1) if m else str(kaikai)

def get_distance_band(dist):
    m = re.search(r'\d+', str(dist))
    if not m: return None
    d = int(m.group())
    if d <= 1400:   return '短距離'
    elif d <= 1800: return 'マイル'
    elif d <= 2200: return '中距離'
    else:           return '長距離'

def get_class_group(class_rank):
    try:
        r = float(class_rank)
    except: return '3勝以上'
    if np.isnan(r): return '3勝以上'
    r = int(r)
    if r == 1:   return '新馬'
    elif r == 2: return '未勝利'
    elif r == 3: return '1勝'
    elif r == 4: return '2勝'
    elif r >= 5: return '3勝以上'
    return '3勝以上'

def calc_roi(df_bets, payout_col='単勝配当'):
    n = len(df_bets)
    if n == 0: return None, 0, 0
    wins = df_bets['target_win'].sum()
    paid = df_bets.loc[df_bets['target_win']==1, payout_col].sum()
    roi = paid / (n * 100) - 1.0
    return roi, int(wins), n

def calc_place_roi(df_bets, payout_col='複勝配当'):
    n = len(df_bets)
    if n == 0: return None, None, 0, 0
    places = df_bets['target_place'].sum()
    place_rate = places / n
    paid = df_bets.loc[df_bets['target_place']==1, payout_col].sum()
    roi = paid / (n * 100) - 1.0
    return roi, place_rate, int(places), n

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, 'models')

# ── モデル読み込み ──
print("モデル情報読み込み中...")
with open(os.path.join(model_dir, 'model_info.json'), encoding='utf-8') as f:
    cur_info = json.load(f)
cur_features = cur_info['features']
cur_models   = cur_info['models']

with open(os.path.join(model_dir, 'ranker', 'ranker_info.json'), encoding='utf-8') as f:
    cur_rankers = json.load(f).get('rankers', {})

sub_info_path = os.path.join(model_dir, 'submodel', 'submodel_info.json')
with open(sub_info_path, encoding='utf-8') as f:
    sub_info = json.load(f)
sub_features = sub_info['features']
sub_models   = sub_info['models']

with open(os.path.join(model_dir, 'submodel_ranker', 'class_ranker_info.json'), encoding='utf-8') as f:
    sub_rankers = json.load(f).get('rankers', {})

print(f"距離モデル: {len(cur_models)}グループ / クラスモデル: {len(sub_models)}グループ")

# ── テストデータ読み込み ──
test_path = os.path.join(base_dir, 'data', 'processed', 'features_2012_test.csv')
print(f"テストデータ読み込み中... ({os.path.getsize(test_path)//1024//1024}MB)")
t0 = time.time()
df = pd.read_csv(test_path, low_memory=False)
print(f"読み込み完了: {len(df):,}行 ({time.time()-t0:.1f}秒)")

# 前処理
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
mask_da_ml = (df['_surface'] == 'ダ') & (df['_dist_band'].isin(['中距離', '長距離']))
df.loc[mask_da_ml, '_dist_band'] = '中長距離'
df['_cls_group'] = df['クラス_rank'].apply(get_class_group) if 'クラス_rank' in df.columns else '3勝以上'
df['sub_key'] = df['_surface'] + '_' + df['_dist_band'].astype(str) + '_' + df['_cls_group'].astype(str)

all_feats = list(set(cur_features + sub_features))
for col in all_feats:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ── モデルキャッシュ ──
print("モデルをプリロード中...")
t0 = time.time()
cur_model_cache  = {}
cur_ranker_cache = {}
sub_model_cache  = {}
sub_ranker_cache = {}

for ck in df['cur_key'].dropna().unique():
    if ck in cur_models:
        p = os.path.join(model_dir, cur_models[ck]['win'])
        if os.path.exists(p):
            with open(p, 'rb') as f: m = pickle.load(f)
            cur_model_cache[ck] = (m, m.booster_.feature_name())
    if ck in cur_rankers:
        p = os.path.join(model_dir, 'ranker', cur_rankers[ck])
        if os.path.exists(p):
            with open(p, 'rb') as f: cur_ranker_cache[ck] = pickle.load(f)

for sk in df['sub_key'].dropna().unique():
    if sk in sub_models:
        p = os.path.join(model_dir, 'submodel', sub_models[sk]['win'])
        if os.path.exists(p):
            with open(p, 'rb') as f: m = pickle.load(f)
            sub_model_cache[sk] = (m, m.booster_.feature_name())
    if sk in sub_rankers:
        p = os.path.join(model_dir, 'submodel_ranker', sub_rankers[sk])
        if os.path.exists(p):
            with open(p, 'rb') as f: sub_ranker_cache[sk] = pickle.load(f)

print(f"プリロード完了: {time.time()-t0:.1f}秒  (cur:{len(cur_model_cache)} sub:{len(sub_model_cache)})")

# ── レース別予測 ──
race_keys = [c for c in ['開催','Ｒ','レース名'] if c in df.columns]
if 'Ｒ' in df.columns:
    df['Ｒ'] = pd.to_numeric(df['Ｒ'], errors='coerce')

print("レース別予測中...")
t0 = time.time()
all_rows = []
races = df.groupby(race_keys, sort=False).groups

for i, (gk, idx) in enumerate(races.items()):
    sub = df.loc[idx].copy()
    ck = sub['cur_key'].iloc[0]
    sk = sub['sub_key'].iloc[0]

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
        rs_val = 50 + 10*(prob - rm)/(rs if rs > 0 else 1)
        sub['cur_diff'] = rs_val - cs
        if ck in cur_ranker_cache:
            scores = cur_ranker_cache[ck].predict(sub[cur_features])
            sub['cur_rank'] = pd.Series(scores, index=sub.index).rank(ascending=False, method='min').astype(int)

    if sk in sub_model_cache:
        m, wf = sub_model_cache[sk]
        for c in wf:
            if c not in sub.columns: sub[c] = np.nan
        prob = m.predict_proba(sub[wf])[:, 1]
        st = sub_models[sk].get('stats', {})
        wm = st.get('win_mean', prob.mean()); ws = st.get('win_std', prob.std())
        cs = 50 + 10*(prob - wm)/(ws if ws > 0 else 1)
        rm = prob.mean(); rs = prob.std()
        rs_val = 50 + 10*(prob - rm)/(rs if rs > 0 else 1)
        sub['sub_diff'] = rs_val - cs
        if sk in sub_ranker_cache:
            scores = sub_ranker_cache[sk].predict(sub[wf])
            sub['sub_rank'] = pd.Series(scores, index=sub.index).rank(ascending=False, method='min').astype(int)

    all_rows.append(sub)
    if (i+1) % 500 == 0:
        print(f"  {i+1}/{len(races)}レース ({time.time()-t0:.0f}秒)")

result = pd.concat(all_rows, ignore_index=True)
print(f"予測完了: {len(result):,}頭 ({time.time()-t0:.0f}秒)")

# ── ROI計算 ──
cd = pd.to_numeric(result['cur_diff'], errors='coerce')
sd = pd.to_numeric(result['sub_diff'], errors='coerce')
cr = result['cur_rank']
sr = result['sub_rank']

def show_roi(label, mask):
    roi, wins, n = calc_roi(result[mask])
    p_roi, p_rate, places, _ = calc_place_roi(result[mask])
    if roi is None:
        print(f"  {label}: 該当なし")
    else:
        p_str = f"複勝率{p_rate:.1%} ROI{p_roi:+.1%}" if p_roi is not None else "-"
        print(f"  {label}: 単勝ROI {roi:+.1%}({n}頭)  / {p_str}")

print()
print("=" * 60)
print("  距離モデル単体")
print("=" * 60)
show_roi("◎ 差+20以上",         cd >= 20)
show_roi("〇 差+15以上",         cd >= 15)
show_roi("▲ 差+10以上",         cd >= 10)
show_roi("ランカー1位",          cr == 1)
show_roi("ランカー1位 & 差+20",  (cr == 1) & (cd >= 20))
show_roi("ランカー1位 & 差+15",  (cr == 1) & (cd >= 15))

print()
print("=" * 60)
print("  クラスモデル単体")
print("=" * 60)
show_roi("◎ 差+20以上",         sd >= 20)
show_roi("〇 差+15以上",         sd >= 15)
show_roi("▲ 差+10以上",         sd >= 10)
show_roi("ランカー1位",          sr == 1)
show_roi("ランカー1位 & 差+20",  (sr == 1) & (sd >= 20))
show_roi("ランカー1位 & 差+15",  (sr == 1) & (sd >= 15))

print()
print("=" * 60)
print("  両モデル一致（AND条件）")
print("=" * 60)
show_roi("両◎ (両方差+20以上)",          (cd >= 20) & (sd >= 20))
show_roi("両〇 (両方差+15以上)",          (cd >= 15) & (sd >= 15))
show_roi("両▲ (両方差+10以上)",          (cd >= 10) & (sd >= 10))
show_roi("両ランカー1位",                 (cr == 1)  & (sr == 1))
show_roi("両ランカー1位 & 両差+20",       (cr == 1)  & (sr == 1) & (cd >= 20) & (sd >= 20))
show_roi("両ランカー1位 & 両差+15",       (cr == 1)  & (sr == 1) & (cd >= 15) & (sd >= 15))
show_roi("両ランカー1位 & 両差+10",       (cr == 1)  & (sr == 1) & (cd >= 10) & (sd >= 10))
show_roi("距離◎ & クラスランカー1位",    (cd >= 20) & (sr == 1))
show_roi("クラス◎ & 距離ランカー1位",    (sd >= 20) & (cr == 1))
show_roi("距離〇以上 & クラスランカー1位",(cd >= 15) & (sr == 1))
show_roi("クラス〇以上 & 距離ランカー1位",(sd >= 15) & (cr == 1))

print()
print("完了")

"""日別ROI集計 → G:\マイドライブ\競馬AI\daily_roi_2026.html
元の_validate_new_marks.pyと同じtest CSVを使用"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import os, pickle, json, re, time

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, 'models_2025')

def get_distance_band(dist):
    m = re.search(r'\d+', str(dist))
    if not m: return None
    d = int(m.group())
    if d <= 1400:   return '短距離'
    elif d <= 1800: return 'マイル'
    elif d <= 2200: return '中距離'
    else:           return '長距離'

def get_class_group(r):
    try: r = int(float(r))
    except: return '3勝以上'
    if r == 1: return '新馬'
    elif r == 2: return '未勝利'
    elif r == 3: return '1勝'
    elif r == 4: return '2勝'
    return '3勝以上'

def extract_venue(v):
    m = re.search(r'\d+([^\d]+)', str(v))
    return m.group(1) if m else str(v)

# ── モデル読み込み ──────────────────────────────────────────────
with open(os.path.join(model_dir, 'model_info.json'), encoding='utf-8') as f:
    cur_info = json.load(f)
cur_features = cur_info['features']
cur_models   = cur_info['models']
with open(os.path.join(model_dir, 'ranker', 'ranker_info.json'), encoding='utf-8') as f:
    cur_rankers = json.load(f).get('rankers', {})
with open(os.path.join(model_dir, 'submodel', 'submodel_info.json'), encoding='utf-8') as f:
    sub_info = json.load(f)
sub_features = sub_info['features']
sub_models   = sub_info['models']
with open(os.path.join(model_dir, 'submodel_ranker', 'class_ranker_info.json'), encoding='utf-8') as f:
    sub_rankers = json.load(f).get('rankers', {})

# ── test CSV 読み込み ──────────────────────────────────────────
test_csv = os.path.join(base_dir, 'data', 'processed', 'all_venues_features_2026test.csv')
print("test CSV 読み込み中...")
t0 = time.time()
df_all = pd.read_csv(test_csv, low_memory=False)
df_all['日付_num'] = pd.to_numeric(df_all['日付'], errors='coerce')
print(f"完了: {time.time()-t0:.1f}秒 / {len(df_all)}行")

# 特徴量を数値型に変換
all_feats = list(set(cur_features + sub_features))
for c in all_feats:
    if c in df_all.columns:
        df_all[c] = pd.to_numeric(df_all[c], errors='coerce')

# ── コースキー設定 ──────────────────────────────────────────────
df_all['_surface'] = df_all['芝・ダ'].astype(str).str.strip() if '芝・ダ' in df_all.columns else 'None'
df_all['_dist_band'] = df_all['距離'].apply(get_distance_band)
mask_da = (df_all['_surface'] == 'ダ') & (df_all['_dist_band'].isin(['中距離', '長距離']))
df_all.loc[mask_da, '_dist_band'] = '中長距離'
df_all['_cls_group'] = df_all['クラス_rank'].apply(get_class_group) if 'クラス_rank' in df_all.columns else '3勝以上'
df_all['sub_key'] = df_all['_surface'] + '_' + df_all['_dist_band'] + '_' + df_all['_cls_group']
df_all['会場'] = df_all['開催'].apply(extract_venue)
df_all['cur_key'] = df_all['会場'] + '_' + df_all['距離'].astype(str)

dates_2026 = sorted(df_all['日付_num'].dropna().unique())
print(f"日付: {len(dates_2026)}日")

# ── 結果確認CSV の補完処理 ──────────────────────────────────
import glob

VENUE_MAP = {'中山': '中', '東京': '東', '阪神': '阪', '中京': '名',
             '京都': '京', '函館': '函', '新潟': '新', '小倉': '小',
             '札幌': '札', '福島': '福'}

def _zen(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    m = re.search(r'\d+', s)
    return int(m.group()) if m else np.nan

result_csvs = sorted(glob.glob(os.path.join(base_dir, 'data', 'raw', 'results', '出馬表形式*結果*.csv')))
_pq_latest = None  # parquet は必要な場合のみ読む

# 結果確認CSVが存在する日付は test CSV から除外（parquetで再処理）
for _rf in result_csvs:
    try:    _tmp = pd.read_csv(_rf, encoding='cp932', low_memory=False, nrows=1)
    except: _tmp = pd.read_csv(_rf, encoding='utf-8',  low_memory=False, nrows=1)
    if '日付S' not in _tmp.columns: continue
    _ds = str(_tmp['日付S'].iloc[0]).replace('/', '.').split('.')
    _excl = (int(_ds[0])-2000)*10000 + int(_ds[1])*100 + int(_ds[2])
    df_all = df_all[df_all['日付_num'] != _excl]

for _rf in result_csvs:
    try:    _dfr = pd.read_csv(_rf, encoding='cp932', low_memory=False)
    except: _dfr = pd.read_csv(_rf, encoding='utf-8',  low_memory=False)
    if '日付S' not in _dfr.columns: continue
    _ds = str(_dfr['日付S'].iloc[0]).replace('/', '.').split('.')
    _dnum = (int(_ds[0])-2000)*10000 + int(_ds[1])*100 + int(_ds[2])

    _dfr['着_num']  = _dfr['着'].apply(_zen)
    _dfr['_tan']    = pd.to_numeric(_dfr['単勝'],  errors='coerce')
    _dfr['_fuku']   = pd.to_numeric(_dfr['複勝'],  errors='coerce')
    _dfr['_odds']   = pd.to_numeric(_dfr['単オッズ'], errors='coerce')

    if False:  # test CSV パッチは廃止 → 結果確認CSVは全て parquet ベースで処理
        pass
    else:
        # ── test CSVにない日（3/21等）→ parquetから特徴量構築 ──
        if _pq_latest is None:
            print("Parquet読み込み中（追加日付用）...")
            _t0 = time.time()
            _pq_path = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.parquet')
            _all_feats_set = set(all_feats)
            try:
                import pyarrow.parquet as _pq_mod
                _avail = set(_pq_mod.read_schema(_pq_path).names)
                _load_cols = ['馬名S', '日付', '距離'] + [c for c in _avail if c in _all_feats_set]
                _df_pq = pd.read_parquet(_pq_path, columns=_load_cols)
            except Exception:
                _df_pq = pd.read_parquet(_pq_path)
            _df_pq['日付_n'] = pd.to_numeric(_df_pq['日付'], errors='coerce')
            _pq_latest = _df_pq.sort_values('日付_n').groupby('馬名S', sort=False).last().reset_index()
            print(f"完了: {time.time()-_t0:.1f}秒")

        # 会場コード・コースキー
        _dfr['会場'] = _dfr['場所'].astype(str).map(VENUE_MAP).fillna(_dfr['場所'].astype(str))
        _dfr['_surface'] = _dfr['芝ダ'].astype(str).str.strip() if '芝ダ' in _dfr.columns else 'ダ'
        _dfr['cur_key'] = _dfr['会場'] + '_' + _dfr['_surface'] + _dfr['距離'].astype(str)
        _dfr['_dist_band'] = _dfr['距離'].apply(get_distance_band)
        _dfmask_da = (_dfr['_surface'] == 'ダ') & (_dfr['_dist_band'].isin(['中距離', '長距離']))
        _dfr.loc[_dfmask_da, '_dist_band'] = '中長距離'
        if 'クラス' in _dfr.columns:
            _cls_map = {'新馬': 1, '未勝利': 2, '1勝': 3, '2勝': 4}
            def _cls_r(v):
                s = str(v).strip()
                for k, r in _cls_map.items():
                    if k in s: return r
                return 5
            _dfr['クラス_rank'] = _dfr['クラス'].apply(_cls_r)
        _dfr['_cls_group'] = _dfr['クラス_rank'].apply(get_class_group) if 'クラス_rank' in _dfr.columns else '3勝以上'
        _dfr['sub_key'] = _dfr['_surface'] + '_' + _dfr['_dist_band'] + '_' + _dfr['_cls_group']
        _dfr['性別_num'] = _dfr['性別'].map({'牡': 0, '牝': 1, 'セ': 2}).astype(float)

        # parquet特徴量マージ
        _feat_cols = ['馬名S'] + [c for c in all_feats if c in _pq_latest.columns]
        _merged = _dfr.merge(_pq_latest[_feat_cols], on='馬名S', how='left', suffixes=('', '_p'))
        # 間隔・前距離を再計算
        if '距離' in _pq_latest.columns:
            _merged['前距離'] = _pq_latest.set_index('馬名S')['距離'].reindex(_merged['馬名S'].values).apply(
                lambda x: float(re.search(r'\d+', str(x)).group()) if re.search(r'\d+', str(x)) else np.nan).values
        if '日付_n' in _pq_latest.columns:
            _date_map = _pq_latest.set_index('馬名S')['日付_n']
            _cur_date = pd.Timestamp(2000 + _dnum // 10000, (_dnum // 100) % 100, _dnum % 100)
            def _yymmdd(v):
                try:
                    v = int(v); return pd.Timestamp(2000 + v//10000, (v//100)%100, v%100)
                except: return pd.NaT
            _merged['間隔'] = ((_cur_date - _merged['馬名S'].map(_date_map).apply(_yymmdd)).dt.days / 7).round(0)
        for _c in all_feats:
            if _c in _merged.columns:
                _merged[_c] = pd.to_numeric(_merged[_c], errors='coerce')

        # 結果列設定
        _merged['日付_num'] = _dnum
        _merged['日付'] = _dnum
        _merged['開催'] = _merged['場所'].astype(str)
        # 単勝配当: 勝ち馬のみ
        _merged['着順_num'] = _merged['着_num']
        _merged['単勝配当'] = np.where(_merged['着_num'] == 1, _merged['_tan'], np.nan)
        _merged['複勝配当'] = np.where(_merged['着_num'] <= 3, _merged['_fuku'], np.nan)
        _merged['単勝オッズ'] = _merged['_odds']

        # df_all に追加（必要列だけ揃える）
        _add_cols = [c for c in df_all.columns if c in _merged.columns]
        _add = _merged[_add_cols].copy()
        df_all = pd.concat([df_all, _add], ignore_index=True)
        dates_2026 = sorted(df_all['日付_num'].dropna().unique())
        print(f"追加日付: {_dnum} ({len(_merged)}頭)")
# ────────────────────────────────────────────────────────────

# ── モデルキャッシュ（全日付分を一括ロード）──────────────────
cur_cache = {}; cur_rk_cache = {}
sub_cache = {}; sub_rk_cache = {}
for ck in df_all['cur_key'].dropna().unique():
    if ck in cur_models:
        p = os.path.join(model_dir, cur_models[ck]['win'])
        if os.path.exists(p):
            with open(p,'rb') as f: m = pickle.load(f)
            cur_cache[ck] = (m, m.booster_.feature_name())
    if ck in cur_rankers:
        p = os.path.join(model_dir, 'ranker', cur_rankers[ck])
        if os.path.exists(p):
            with open(p,'rb') as f: cur_rk_cache[ck] = pickle.load(f)
for sk in df_all['sub_key'].dropna().unique():
    if sk in sub_models:
        p = os.path.join(model_dir, 'submodel', sub_models[sk]['win'])
        if os.path.exists(p):
            with open(p,'rb') as f: m = pickle.load(f)
            sub_cache[sk] = (m, m.booster_.feature_name())
    if sk in sub_rankers:
        p = os.path.join(model_dir, 'submodel_ranker', sub_rankers[sk])
        if os.path.exists(p):
            with open(p,'rb') as f: sub_rk_cache[sk] = pickle.load(f)
print(f"距離{len(cur_cache)} クラス{len(sub_cache)} ランカー{len(cur_rk_cache)}/{len(sub_rk_cache)}")

# ── 日付別予測・ROI集計 ──────────────────────────────────────
daily_rows = []

for dnum in dates_2026:
    day = df_all[df_all['日付_num'] == dnum].copy()
    if len(day) == 0:
        continue

    # 勝ち馬・配当列（test CSVは着順_numが数値、単勝配当はyenper100yen）
    ord_col  = '着順_num' if '着順_num' in day.columns else '着順'
    pay_col  = '単勝配当'
    fuku_col = '複勝配当'
    odds_col = '単勝オッズ'

    day['_ord']  = pd.to_numeric(day[ord_col],  errors='coerce') if ord_col  in day.columns else np.nan
    day['_tan']  = pd.to_numeric(day[pay_col],  errors='coerce') if pay_col  in day.columns else np.nan
    day['_fuku'] = pd.to_numeric(day[fuku_col], errors='coerce') if fuku_col in day.columns else np.nan
    day['_odds'] = pd.to_numeric(day[odds_col], errors='coerce') if odds_col in day.columns else np.nan

    # レース単位キー（開催+R）
    race_keys = [c for c in ['開催', 'Ｒ'] if c in day.columns]

    # 単勝配当をレース全馬に展開（勝ち馬の単勝配当 → 同じレースの全馬に）
    day['_race_key'] = day[race_keys].astype(str).agg('_'.join, axis=1)
    win_pay = (day[day['_ord'] == 1]
               .drop_duplicates('_race_key')
               .set_index('_race_key')['_tan'])
    day['_tansho'] = day['_race_key'].map(win_pay)

    fuku_pay = (day[day['_ord'] <= 3]
                .groupby('_race_key')['_fuku'].mean())
    day['_fukusho'] = day['_race_key'].map(fuku_pay)

    # 予測
    day['cur_diff'] = np.nan; day['cur_rank'] = np.nan; day['cur_cs'] = np.nan
    day['sub_diff'] = np.nan; day['sub_rank'] = np.nan; day['sub_cs'] = np.nan

    all_rows = []
    for gk, idx in day.groupby(race_keys, sort=False).groups.items():
        sub = day.loc[idx].copy()
        ck = sub['cur_key'].iloc[0]
        sk = sub['sub_key'].iloc[0]
        if ck in cur_cache:
            m, wf = cur_cache[ck]
            for c in wf:
                if c not in sub.columns: sub[c] = np.nan
            prob = m.predict_proba(sub[wf])[:, 1]
            st = cur_models[ck].get('stats', {})
            wm = st.get('win_mean', prob.mean()); ws = st.get('win_std', prob.std())
            cs = 50 + 10*(prob - wm)/(ws if ws > 0 else 1)
            rm = prob.mean(); rs = prob.std()
            sub['cur_cs']   = cs
            sub['cur_diff'] = 50 + 10*(prob - rm)/(rs if rs > 0 else 1) - cs
            if ck in cur_rk_cache:
                scores = cur_rk_cache[ck].predict(sub[cur_features])
                sub['cur_rank'] = pd.Series(scores, index=sub.index).rank(ascending=False, method='min').astype(int)
        if sk in sub_cache:
            m, wf = sub_cache[sk]
            for c in wf:
                if c not in sub.columns: sub[c] = np.nan
            prob = m.predict_proba(sub[wf])[:, 1]
            st = sub_models[sk].get('stats', {})
            wm = st.get('win_mean', prob.mean()); ws = st.get('win_std', prob.std())
            cs = 50 + 10*(prob - wm)/(ws if ws > 0 else 1)
            rm = prob.mean(); rs = prob.std()
            sub['sub_cs']   = cs
            sub['sub_diff'] = 50 + 10*(prob - rm)/(rs if rs > 0 else 1) - cs
            if sk in sub_rk_cache:
                scores = sub_rk_cache[sk].predict(sub[wf])
                sub['sub_rank'] = pd.Series(scores, index=sub.index).rank(ascending=False, method='min').astype(int)
        all_rows.append(sub)

    res = pd.concat(all_rows, ignore_index=True)

    # combo_gap
    res['cur_gap'] = np.nan; res['sub_gap'] = np.nan
    for gk, idx in res.groupby(race_keys, sort=False).groups.items():
        s = res.loc[idx]
        for sc, gc in [('cur_cs','cur_gap'),('sub_cs','sub_gap')]:
            v = s[sc].dropna().sort_values(ascending=False).values
            res.loc[idx, gc] = (v[0]-v[1]) if len(v) >= 2 else np.nan
    res['combo_gap'] = res['cur_gap'].fillna(0) + res['sub_gap'].fillna(0)

    cr = res['cur_rank']; sr = res['sub_rank']
    sd = pd.to_numeric(res['sub_diff'], errors='coerce')
    combo_gap = res['combo_gap']
    odds = res['_odds']
    ok3 = odds.isna() | (odds >= 3)
    ok5 = odds.isna() | (odds >= 5)
    both_r1 = (cr==1)&(sr==1)
    star = (cr<=3)&(sr<=3)&~both_r1

    # 新印ロジック（2026-03-28 改訂）
    # 激熱: 両Rnk=1 & cur_diff≥10 & sd≥10 & odds≥5  単1000円  ROI+253%
    # 〇  : 両Rnk=1 & sd≥10 & odds≥3 & ~激熱           単300円  ROI+35%
    # ▲  : 両Rnk≤2（片方2） & sd≥10 & odds≥5          単500円  ROI+91%
    # ☆  : 両Rnk≤3（片方3） & sd≥10 & odds≥5          単200円
    cd = pd.to_numeric(res.get('cur_diff', pd.Series(np.nan, index=res.index)), errors='coerce')
    res['_印'] = ''
    res.loc[star & ~((cr<=2)&(sr<=2)) & (sd>=10) & ok5, '_印'] = '☆'
    res.loc[(cr<=2)&(sr<=2)&~both_r1 & (sd>=10) & ok5, '_印'] = '▲'
    res.loc[both_r1 & (sd>=10) & ok3, '_印'] = '〇'
    res.loc[both_r1 & (cd>=10) & (sd>=10) & ok5, '_印'] = '激熱'

    # 結果データがあるか確認
    has_result = res['_tansho'].notna().any()

    def _roi(mask, bet, fuku_bet=0):
        b = res[mask]
        n = len(b)
        if n == 0: return 0, 0, 0, 0
        tb = n * bet + n * fuku_bet
        if not has_result:
            return n, None, None, None
        hits = int((b['_ord'] <= 3).sum()) if fuku_bet > 0 and bet == 0 else int((b['_ord'] == 1).sum())
        tan_ret = b[b['_ord']==1]['_tansho'].sum() * bet / 100
        fuku_ret = 0
        if fuku_bet > 0:
            fuku_ret = b[b['_ord']<=3]['_fukusho'].sum() * fuku_bet / 100
        ret = tan_ret + fuku_ret
        pf = int(ret - tb)
        roi = ret/tb - 1.0 if tb > 0 else 0
        return n, pf, roi, hits

    n_g, pf_g, roi_g, w_g = _roi(res['_印']=='激熱', 1000)
    n_o, pf_o, roi_o, w_o = _roi(res['_印']=='〇',    300)
    n_d, pf_d, roi_d, w_d = _roi(res['_印']=='▲',    500)
    n_s, pf_s, roi_s, w_s = _roi(res['_印']=='☆',    200)

    if has_result:
        total_tb = n_g*1000 + n_o*300 + n_d*500 + n_s*200
        total_ret = 0
        for mask, bet in [(res['_印']=='激熱',1000),(res['_印']=='〇',300),
                          (res['_印']=='▲',500),(res['_印']=='☆',200)]:
            b = res[mask]
            total_ret += b[b['_ord']==1]['_tansho'].sum() * bet / 100
        total_pf = int(total_ret - total_tb)
        total_roi = total_ret/total_tb - 1.0 if total_tb > 0 else 0
        sign = '+' if total_pf >= 0 else ''
        roi_str = f"{sign}{total_pf:,}円 ({total_roi:+.1%})"
    else:
        total_pf = None; total_roi = None
        roi_str = "結果未格納"

    d = str(int(dnum))
    date_str = f"20{d[:2]}/{d[2:4]}/{d[4:6]}"
    print(f"{date_str}  激熱{n_g}/{w_g}  〇{n_o}/{w_o}  ▲{n_d}/{w_d}  ☆{n_s}/{w_s}  計{roi_str}")

    daily_rows.append({
        '日付': date_str, '日付_num': dnum,
        'has_result': has_result,
        '激熱_n': n_g, '激熱_w': w_g or 0, '激熱_pf': pf_g or 0, '激熱_roi': roi_g or 0,
        '〇_n': n_o,   '〇_w': w_o or 0,   '〇_pf': pf_o or 0,   '〇_roi': roi_o or 0,
        '▲_n': n_d,   '▲_w': w_d or 0,   '▲_pf': pf_d or 0,   '▲_roi': roi_d or 0,
        '☆_n': n_s,   '☆_w': w_s or 0,   '☆_pf': pf_s or 0,   '☆_roi': roi_s or 0,
        '計_pf': total_pf or 0, '計_roi': total_roi or 0,
        '計_tb': n_g*1000 + n_o*300 + n_d*500 + n_s*200,
    })

# ── 4/4 キャッシュ処理（ノーヘッダー形式のため通常フローで処理不可）──
_cache44  = os.path.join(base_dir, 'data', 'raw', 'cache', '出馬表形式4月4日.cache.pkl')
_result44 = os.path.join(base_dir, 'data', 'raw', 'results', '出馬表形式4月4日結果入力.csv')
_dnum44   = 260404

if (os.path.exists(_cache44) and os.path.exists(_result44)
        and _dnum44 not in [r.get('日付_num') for r in daily_rows]):
    print("4/4 キャッシュ処理中...")
    with open(_cache44, 'rb') as _f44: _c44 = pickle.load(_f44)
    _res44 = _c44['result'].copy()

    # 結果CSV（33列・ヘッダーなし）
    _r4 = pd.read_csv(_result44, encoding='cp932', header=None)
    _r4['馬名S']   = _r4[7].astype(str)
    _r4['着順_r']  = pd.to_numeric(_r4[31], errors='coerce')
    _r4['odds_r']  = pd.to_numeric(_r4[30], errors='coerce')
    # 実際の単勝払戻が取得できないためオッズ×100で近似
    _r4['tansho_r'] = _r4['odds_r'] * 100

    # キャッシュとマージ（馬名で突合）
    _res44 = _res44.merge(_r4[['馬名S','着順_r','tansho_r']], on='馬名S', how='left')
    _res44['_ord']    = _res44['着順_r']
    _res44['_tansho'] = np.where(_res44['_ord'] == 1, _res44['tansho_r'], np.nan)
    _res44['_odds']   = pd.to_numeric(_res44['単勝オッズ'], errors='coerce')

    # 印ロジック（_daily_roi_2026 と同一）
    cr44 = pd.to_numeric(_res44['cur_ランカー順位'], errors='coerce')
    sr44 = pd.to_numeric(_res44['sub_ランカー順位'], errors='coerce')
    sd44 = pd.to_numeric(_res44['sub_偏差値の差'],   errors='coerce')
    cd44 = pd.to_numeric(_res44['cur_偏差値の差'],   errors='coerce')
    ok3_44     = _res44['_odds'].isna() | (_res44['_odds'] >= 3)
    ok5_44     = _res44['_odds'].isna() | (_res44['_odds'] >= 5)
    both_r1_44 = (cr44 == 1) & (sr44 == 1)
    star_44    = (cr44 <= 3) & (sr44 <= 3) & ~both_r1_44

    _res44['_印'] = ''
    _res44.loc[star_44 & ~((cr44<=2)&(sr44<=2)) & (sd44>=10) & ok5_44, '_印'] = '☆'
    _res44.loc[(cr44<=2)&(sr44<=2)&~both_r1_44 & (sd44>=10) & ok5_44, '_印'] = '▲'
    _res44.loc[both_r1_44 & (sd44>=10) & ok3_44, '_印'] = '〇'
    _res44.loc[both_r1_44 & (cd44>=10) & (sd44>=10) & ok5_44, '_印'] = '激熱'

    has_result44 = _res44['_ord'].notna().any()

    def _roi44(mask, bet):
        b = _res44[mask]
        n = len(b)
        if n == 0: return 0, 0, 0, 0
        tb = n * bet
        if not has_result44:
            return n, None, None, None
        hits = int((b['_ord'] == 1).sum())
        tan_ret = b[b['_ord']==1]['_tansho'].sum() * bet / 100
        pf = int(tan_ret - tb)
        roi = tan_ret/tb - 1.0 if tb > 0 else 0
        return n, pf, roi, hits

    ng44, pfg44, roig44, wg44 = _roi44(_res44['_印']=='激熱', 1000)
    no44, pfo44, roio44, wo44 = _roi44(_res44['_印']=='〇',    300)
    nd44, pfd44, roid44, wd44 = _roi44(_res44['_印']=='▲',    500)
    ns44, pfs44, rois44, ws44 = _roi44(_res44['_印']=='☆',    200)

    ttb44 = ng44*1000 + no44*300 + nd44*500 + ns44*200
    tret44 = sum(
        _res44[_res44['_印']==mk][_res44[_res44['_印']==mk]['_ord']==1]['_tansho'].sum() * bt / 100
        for mk, bt in [('激熱',1000),('〇',300),('▲',500),('☆',200)]
    )
    tpf44  = int(tret44 - ttb44) if has_result44 else None
    troi44 = tret44/ttb44 - 1.0 if (ttb44 > 0 and has_result44) else 0

    print(f"4/4  激熱{ng44}/{wg44}  〇{no44}/{wo44}  ▲{nd44}/{wd44}  ☆{ns44}/{ws44}")

    daily_rows.append({
        '日付': '2026/04/04', '日付_num': _dnum44,
        'has_result': has_result44,
        '激熱_n': ng44, '激熱_w': wg44 or 0, '激熱_pf': pfg44 or 0, '激熱_roi': roig44 or 0,
        '〇_n':  no44,  '〇_w':  wo44 or 0,  '〇_pf':  pfo44 or 0,  '〇_roi':  roio44 or 0,
        '▲_n':  nd44,  '▲_w':  wd44 or 0,  '▲_pf':  pfd44 or 0,  '▲_roi':  roid44 or 0,
        '☆_n':  ns44,  '☆_w':  ws44 or 0,  '☆_pf':  pfs44 or 0,  '☆_roi':  rois44 or 0,
        '計_pf': tpf44 or 0, '計_roi': troi44 or 0,
        '計_tb': ttb44,
    })
    print(f"4/4 追加完了: {len(_res44)}頭")

df_daily = pd.DataFrame(daily_rows).sort_values('日付_num').reset_index(drop=True)
df_res_only = df_daily[df_daily['has_result']]

# 累計（結果あり日のみ）
cum_pf = 0; cum_tb = 0; cum_ret = 0
for _, r in df_daily.iterrows():
    if r['has_result']:
        cum_tb += r['計_tb']
        cum_ret += r['計_tb'] * (r['計_roi'] + 1)
    df_daily.loc[_, '累計_pf'] = int(cum_pf + (r['計_pf'] if r['has_result'] else 0))
    if r['has_result']:
        cum_pf += r['計_pf']
df_daily['累計_pf'] = df_daily['累計_pf'].fillna(0)
cum_roi_final = cum_ret/cum_tb - 1.0 if cum_tb > 0 else 0

# ── HTML生成 ──────────────────────────────────────────────────
def pf_cell(pf, roi, has_res=True):
    if not has_res:
        return '<td style="text-align:center;color:#888">-</td>'
    if pf is None: return '<td>-</td>'
    sign = '+' if pf >= 0 else ''
    col = '#2d862d' if pf >= 0 else '#c0392b'
    return f'<td style="color:{col};font-weight:bold;text-align:right">{sign}{pf:,}円<br><small>({roi:+.1%})</small></td>'

def mark_cell(n, w, pf, roi, has_res):
    if n == 0: return '<td style="text-align:center;color:#555">-</td>'
    col = '#2d862d' if (pf or 0) >= 0 else '#c0392b'
    pf_str = f'<br><small style="color:{col}">{("+" if (pf or 0)>=0 else "")}{int(pf or 0):,}円</small>' if has_res else ''
    return f'<td style="text-align:center">{n}頭/{w}的{pf_str}</td>'

rows_html = ''
for _, r in df_daily.iterrows():
    if not bool(r['has_result']):
        continue
    hr = bool(r['has_result'])
    cum_col = '#2d862d' if r['累計_pf'] >= 0 else '#c0392b'
    rows_html += f'''<tr>
<td style="text-align:center">{r["日付"]}</td>
{mark_cell(r["激熱_n"], r["激熱_w"], r["激熱_pf"], r["激熱_roi"], hr)}
{mark_cell(r["〇_n"],    r["〇_w"],    r["〇_pf"],    r["〇_roi"],    hr)}
{mark_cell(r["▲_n"],    r["▲_w"],    r["▲_pf"],    r["▲_roi"],    hr)}
{mark_cell(r["☆_n"],    r["☆_w"],    r["☆_pf"],    r["☆_roi"],    hr)}
{pf_cell(r["計_pf"] if hr else None, r["計_roi"] if hr else None, hr)}
<td style="color:{cum_col};font-weight:bold;text-align:right">{("+" if r["累計_pf"]>=0 else "")}{int(r["累計_pf"]):,}円</td>
</tr>'''

plus_days = int((df_res_only['計_pf'] >= 0).sum())
total_days = len(df_res_only)
col_all = '#2d862d' if cum_pf >= 0 else '#c0392b'

html = f'''<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8">
<title>2026年 日別ROI</title>
<style>
body{{font-family:"Hiragino Kaku Gothic Pro",Meiryo,sans-serif;background:#1a1a2e;color:#e0e0e0;padding:20px}}
h2{{color:#f0c040;text-align:center}}
.summary{{display:flex;gap:20px;justify-content:center;margin:10px 0 20px;flex-wrap:wrap}}
.card{{background:#16213e;border-radius:8px;padding:12px 20px;text-align:center;min-width:120px}}
.card .val{{font-size:1.6em;font-weight:bold}}
table{{width:100%;border-collapse:collapse;font-size:0.85em}}
th{{background:#16213e;color:#f0c040;padding:6px 8px;text-align:center;position:sticky;top:0}}
td{{padding:5px 8px;border-bottom:1px solid #2a2a4a}}
tr:nth-child(even){{background:#16213e88}}
tr:hover{{background:#1a3a5a}}
</style></head><body>
<h2>2026年 日別ROI　（最終更新: {df_res_only["日付"].iloc[-1]}）</h2>
<div class="summary">
  <div class="card"><div>累計損益</div><div class="val" style="color:{col_all}">{("+" if cum_pf>=0 else "")}{cum_pf:,}円</div></div>
  <div class="card"><div>累計ROI</div><div class="val" style="color:{col_all}">{cum_roi_final:+.1%}</div></div>
  <div class="card"><div>プラス日数</div><div class="val">{plus_days}/{total_days}日</div></div>
  <div class="card"><div>激熱的中率</div><div class="val">{int(df_res_only["激熱_w"].sum())}/{int(df_res_only["激熱_n"].sum())}頭</div></div>
  <div class="card"><div>▲的中率</div><div class="val">{int(df_res_only["▲_w"].sum())}/{int(df_res_only["▲_n"].sum())}頭</div></div>
</div>
<table><thead><tr>
<th>日付</th>
<th>激熱<br>単1000円</th>
<th>〇<br>単300円</th>
<th>▲<br>単500円</th>
<th>☆<br>単200円</th>
<th>日計</th><th>累計損益</th>
</tr></thead><tbody>
{rows_html}
</tbody></table></body></html>'''

out = r'G:\マイドライブ\競馬AI\daily_roi_2026.html'
try:
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n出力: {out}")
except Exception as e:
    print(f"\nGドライブ書き込みスキップ: {e}")

docs_out = 'G:/マイドライブ/horse_racing_ai/docs/daily_roi_2026.html'
try:
    os.makedirs(os.path.dirname(docs_out), exist_ok=True)
    with open(docs_out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"出力: {docs_out}")
except Exception as e:
    print(f"Gドライブ(docs)書き込みスキップ: {e}")

# ローカルdocsへ出力（GitHub Pages用）
local_docs = os.path.join(base_dir, 'docs', 'daily_roi_2026.html')
os.makedirs(os.path.dirname(local_docs), exist_ok=True)
with open(local_docs, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"出力: {local_docs}")
print(f"累計: {('+' if cum_pf>=0 else '')}{cum_pf:,}円  ROI{cum_roi_final:+.1%}  {plus_days}/{total_days}日プラス")

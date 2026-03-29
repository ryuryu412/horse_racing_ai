"""
2026年実績ROI計算スクリプト
現在の印ロジックで3/21・3/22を再予測し結果CSVと突合してROIを計算する
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os, re, json, pickle, unicodedata
import pandas as pd
import numpy as np
import time as _time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── ユーティリティ ──────────────────────────────────────────

def norm_int(s):
    try:
        return int(unicodedata.normalize('NFKC', str(s)).strip())
    except Exception:
        return np.nan

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
    except (ValueError, TypeError):
        return '3勝以上'
    if np.isnan(r): return '3勝以上'
    r = int(r)
    if r == 1:   return '新馬'
    elif r == 2: return '未勝利'
    elif r == 3: return '1勝'
    elif r == 4: return '2勝'
    else:        return '3勝以上'

def extract_venue(kaikai):
    m = re.search(r'\d+([^\d]+)', str(kaikai))
    return m.group(1) if m else str(kaikai)

# ─── 出馬表CSV変換 ───────────────────────────────────────────

def convert_card(card_path):
    try:
        df = pd.read_csv(card_path, encoding='cp932', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(card_path, encoding='utf-8', low_memory=False)

    if '日付S' in df.columns:
        def parse_date_s(s):
            try:
                parts = str(s).replace('/', '.').split('.')
                y, mo, d = int(parts[0]), int(parts[1]), int(parts[2])
                return (y - 2000) * 10000 + mo * 100 + d
            except Exception:
                return np.nan
        df['日付'] = df['日付S'].apply(parse_date_s)
    else:
        df['日付'] = pd.to_numeric(df['日付'], errors='coerce')

    if '場 R' in df.columns and '開催' not in df.columns:
        def make_kaikai(bar):
            s = str(bar).strip()
            m = re.match(r'([^\d]+)', s)
            vc = m.group(1) if m else s
            return f'1{vc}1'
        df['開催'] = df['場 R'].apply(make_kaikai)

    col_map = {'芝ダ': '芝・ダ', '単オッズ': '単勝オッズ'}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if '馬名S' not in df.columns and '馬名' in df.columns:
        df['馬名S'] = df['馬名']
    if '距離' in df.columns:
        df['距離'] = pd.to_numeric(df['距離'], errors='coerce')
    return df

# ─── 印ロジック（generate_html と同一）──────────────────────

def apply_marks(df):
    sub_diff  = pd.to_numeric(df['sub_偏差値の差'],  errors='coerce')
    combo_gap = pd.to_numeric(df.get('combo_gap', np.nan), errors='coerce') if 'combo_gap' in df.columns else pd.Series(np.nan, index=df.index)
    cur_r     = pd.to_numeric(df['cur_ランカー順位'], errors='coerce')
    sub_r     = pd.to_numeric(df['sub_ランカー順位'], errors='coerce')
    both_r1   = (cur_r == 1) & (sub_r == 1)
    star      = (cur_r <= 3) & (sub_r <= 3) & ~both_r1
    _odds     = pd.to_numeric(df.get('単勝オッズ', np.nan), errors='coerce') if '単勝オッズ' in df.columns else pd.Series(np.nan, index=df.index)
    odds_ok3  = _odds.isna() | (_odds >= 3)
    odds_ok5  = _odds.isna() | (_odds >= 5)
    mask_gekiatu = both_r1 & (combo_gap >= 15) & (sub_diff >= 10) & odds_ok3
    mask_maru    = both_r1 & (combo_gap >= 10) & odds_ok3 & ~mask_gekiatu
    mask_maru2   = both_r1 & (combo_gap <  10) & odds_ok3
    mask_hoshi   = star    & odds_ok5
    def mark_of(i):
        if mask_gekiatu.iloc[i]: return '激熱'
        if mask_maru.iloc[i]:    return '◎'
        if mask_maru2.iloc[i]:   return '〇'
        if mask_hoshi.iloc[i]:   return '☆'
        return ''
    df = df.copy()
    df['_印'] = [mark_of(i) for i in range(len(df))]
    return df

# ─── 予測（models_2025 使用）────────────────────────────────

def predict_day(date_num, card_df):
    sub_info_path = os.path.join(BASE_DIR, 'models_2025', 'submodel', 'submodel_info.json')
    cur_info_path = os.path.join(BASE_DIR, 'models_2025', 'model_info.json')

    sub_features, sub_models, sub_rankers = None, {}, {}
    if os.path.exists(sub_info_path):
        with open(sub_info_path, encoding='utf-8') as f:
            si = json.load(f)
        sub_features = si['features']
        sub_models   = si['models']
        rpath = os.path.join(BASE_DIR, 'models_2025', 'submodel_ranker', 'class_ranker_info.json')
        if os.path.exists(rpath):
            with open(rpath, encoding='utf-8') as f:
                sub_rankers = json.load(f).get('rankers', {})

    cur_features, cur_models, cur_rankers = None, {}, {}
    if os.path.exists(cur_info_path):
        with open(cur_info_path, encoding='utf-8') as f:
            ci = json.load(f)
        cur_features = ci['features']
        cur_models   = ci['models']
        rpath = os.path.join(BASE_DIR, 'models_2025', 'ranker', 'ranker_info.json')
        if os.path.exists(rpath):
            with open(rpath, encoding='utf-8') as f:
                cur_rankers = json.load(f).get('rankers', {})

    # 特徴量読み込み
    feat_pq  = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
    feat_csv = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.csv')
    t0 = _time.time()
    if os.path.exists(feat_pq) and os.path.getmtime(feat_pq) >= os.path.getmtime(feat_csv):
        need_cols = set(['馬名S', '日付', '距離', '開催', '芝・ダ', 'クラス_rank', '単勝オッズ'] +
                        (sub_features or []) + (cur_features or []))
        try:
            import pyarrow.parquet as _pq
            avail = set(_pq.read_schema(feat_pq).names)
            cols  = [c for c in avail if c in need_cols]
            df_all = pd.read_parquet(feat_pq, columns=cols)
        except Exception:
            df_all = pd.read_parquet(feat_pq)
    else:
        df_all = pd.read_csv(feat_csv, low_memory=False)
    df_all['_d'] = pd.to_numeric(df_all['日付'], errors='coerce')
    day = df_all[df_all['_d'] == date_num].copy()
    print(f"  特徴量: {len(day)}頭 ({_time.time()-t0:.1f}秒)")

    if day.empty:
        print("  該当日付のデータなし")
        return None

    # オッズを card_df から補完
    if '単勝オッズ' in card_df.columns:
        odds_map = card_df.drop_duplicates('馬名S').set_index('馬名S')['単勝オッズ']
        odds_map = pd.to_numeric(odds_map, errors='coerce')
        if '単勝オッズ' not in day.columns:
            day['単勝オッズ'] = np.nan
        day['単勝オッズ'] = day['単勝オッズ'].combine_first(day['馬名S'].map(odds_map))

    # 特徴量列を数値に変換
    all_feats = list(set((sub_features or []) + (cur_features or [])))
    for col in all_feats:
        if col in day.columns:
            day[col] = pd.to_numeric(day[col], errors='coerce')

    # sub_key / cur_key
    day['_surface']   = day['芝・ダ'].astype(str).str.strip() if '芝・ダ' in day.columns else 'None'
    day['_dist_band'] = day['距離'].apply(get_distance_band)
    mask_da_ml = (day['_surface'] == 'ダ') & (day['_dist_band'].isin(['中距離', '長距離']))
    day.loc[mask_da_ml, '_dist_band'] = '中長距離'
    day['_cls_group'] = day['クラス_rank'].apply(get_class_group) if 'クラス_rank' in day.columns else None
    day['sub_key'] = day['_surface'] + '_' + day['_dist_band'].astype(str) + '_' + day['_cls_group'].astype(str)
    day['会場']    = day['開催'].apply(extract_venue) if '開催' in day.columns else ''
    day['cur_key'] = day['会場'] + '_' + day['距離'].astype(str)

    if '距離' in day.columns:
        day['距離'] = pd.to_numeric(day['距離'], errors='coerce')

    # モデルプリロード
    model_cache, ranker_cache, cur_model_cache, cur_ranker_cache = {}, {}, {}, {}
    for sk in day['sub_key'].dropna().unique():
        if sk in sub_models:
            p = os.path.join(BASE_DIR, 'models_2025', 'submodel', sub_models[sk]['win'])
            if os.path.exists(p):
                with open(p, 'rb') as f: m = pickle.load(f)
                model_cache[sk] = (m, m.booster_.feature_name())
        if sk in sub_rankers:
            p = os.path.join(BASE_DIR, 'models_2025', 'submodel_ranker', sub_rankers[sk])
            if os.path.exists(p):
                with open(p, 'rb') as f: ranker_cache[sk] = pickle.load(f)
    for ck in day['cur_key'].dropna().unique():
        if ck in cur_models:
            p = os.path.join(BASE_DIR, 'models_2025', cur_models[ck]['win'])
            if os.path.exists(p):
                with open(p, 'rb') as f: m = pickle.load(f)
                cur_model_cache[ck] = (m, m.booster_.feature_name())
        if ck in cur_rankers:
            p = os.path.join(BASE_DIR, 'models_2025', 'ranker', cur_rankers[ck])
            if os.path.exists(p):
                with open(p, 'rb') as f: cur_ranker_cache[ck] = pickle.load(f)

    # レース別予測
    race_keys = ['開催', 'Ｒ'] if 'Ｒ' in day.columns else ['開催']
    if 'Ｒ' in day.columns:
        day['Ｒ'] = pd.to_numeric(day['Ｒ'], errors='coerce')
    all_rows = []
    for gk, idx in day.groupby(race_keys, sort=True).groups.items():
        sub = day.loc[idx].copy()
        sk = sub['sub_key'].iloc[0]
        ck = sub['cur_key'].iloc[0]

        for col in ['sub_prob_win', 'sub_偏差値の差', 'sub_コース偏差値', 'sub_レース内偏差値', 'sub_ランカー順位']:
            sub[col] = np.nan
        if sk in model_cache:
            m, wf = model_cache[sk]
            for c in wf:
                if c not in sub.columns: sub[c] = np.nan
            sub['sub_prob_win'] = m.predict_proba(sub[wf])[:, 1]
            st = sub_models[sk].get('stats', {})
            wm = st.get('win_mean', sub['sub_prob_win'].mean())
            ws = st.get('win_std',  sub['sub_prob_win'].std())
            sub['sub_コース偏差値']   = 50 + 10 * (sub['sub_prob_win'] - wm) / (ws if ws > 0 else 1)
            rm, rs = sub['sub_prob_win'].mean(), sub['sub_prob_win'].std()
            sub['sub_レース内偏差値'] = 50 + 10 * (sub['sub_prob_win'] - rm) / (rs if rs > 0 else 1)
            sub['sub_偏差値の差']     = sub['sub_レース内偏差値'] - sub['sub_コース偏差値']
            if sk in ranker_cache:
                scores = ranker_cache[sk].predict(sub[wf])
                sub['sub_ランカー順位'] = pd.Series(scores, index=sub.index).rank(ascending=False, method='min').astype(int)

        for col in ['cur_prob_win', 'cur_偏差値の差', 'cur_コース偏差値', 'cur_レース内偏差値', 'cur_ランカー順位']:
            sub[col] = np.nan
        if ck in cur_model_cache:
            m, wf = cur_model_cache[ck]
            for c in wf:
                if c not in sub.columns: sub[c] = np.nan
            sub['cur_prob_win'] = m.predict_proba(sub[wf])[:, 1]
            st = cur_models[ck].get('stats', {})
            wm = st.get('win_mean', sub['cur_prob_win'].mean())
            ws = st.get('win_std',  sub['cur_prob_win'].std())
            sub['cur_コース偏差値']   = 50 + 10 * (sub['cur_prob_win'] - wm) / (ws if ws > 0 else 1)
            rm, rs = sub['cur_prob_win'].mean(), sub['cur_prob_win'].std()
            sub['cur_レース内偏差値'] = 50 + 10 * (sub['cur_prob_win'] - rm) / (rs if rs > 0 else 1)
            sub['cur_偏差値の差']     = sub['cur_レース内偏差値'] - sub['cur_コース偏差値']
            if ck in cur_ranker_cache:
                scores = cur_ranker_cache[ck].predict(sub[cur_features])
                sub['cur_ランカー順位'] = pd.Series(scores, index=sub.index).rank(ascending=False, method='min').astype(int)

        for sc, gc in [('cur_コース偏差値', 'cur_gap'), ('sub_コース偏差値', 'sub_gap')]:
            sc2 = sub[sc].dropna().sort_values(ascending=False).values
            sub[gc] = (sc2[0] - sc2[1]) if len(sc2) >= 2 else np.nan
        sub['combo_gap'] = sub.get('cur_gap', pd.Series(0, index=sub.index)).fillna(0) + \
                           sub.get('sub_gap', pd.Series(0, index=sub.index)).fillna(0)
        all_rows.append(sub)

    return pd.concat(all_rows, ignore_index=True) if all_rows else None


# ─── メイン ─────────────────────────────────────────────────

WEEKS = [
    ('3/21', 260321,
     os.path.join(BASE_DIR, 'data', 'raw', '出馬表形式3月21日.csv'),
     os.path.join(BASE_DIR, 'data', 'raw', '出馬表形式3月21日結果確認.csv')),
    ('3/22', 260322,
     os.path.join(BASE_DIR, 'data', 'raw', '出馬表形式3月22日.csv'),
     os.path.join(BASE_DIR, 'data', 'raw', '出馬表形式3月22日結果確認.csv')),
]
HTML_BET = {'激熱': 2000, '◎': 1000, '〇': 500, '☆': 300}

all_rows = []

for label, date_num, card_path, result_path in WEEKS:
    print(f"\n{'='*60}")
    print(f"  {label} (date={date_num})")
    print(f"{'='*60}")

    card_df = convert_card(card_path)
    result  = predict_day(date_num, card_df)
    if result is None:
        continue

    result = apply_marks(result)

    # 結果CSV
    res = pd.read_csv(result_path, encoding='cp932', low_memory=False)
    res['着順'] = res['着'].apply(norm_int)
    res['配当'] = pd.to_numeric(res['単勝'], errors='coerce')  # 100円あたり払戻

    # 場 R + Ｒ で1着の配当マップ
    res1 = res[res['着順'] == 1][['場 R', 'Ｒ', '配当', '馬名S']].drop_duplicates(['場 R', 'Ｒ'])
    payout_map = {(r['場 R'], int(r['Ｒ'])): r['配当'] for _, r in res1.iterrows()}
    winner_map  = {(r['場 R'], int(r['Ｒ'])): r['馬名S'] for _, r in res1.iterrows()}

    for _, row in result.iterrows():
        mark = row['_印']
        if not mark:
            continue

        kaikai = str(row.get('開催', ''))
        r_num  = row.get('Ｒ', np.nan)
        m = re.search(r'\d+([^\d]+)', kaikai)
        venue_abbr = m.group(1) if m else ''
        try:
            r_int = int(float(r_num))
        except Exception:
            continue

        bar_key = f'{venue_abbr}{r_int}'
        payout  = payout_map.get((bar_key, r_int), np.nan)
        winner  = winner_map.get((bar_key, r_int), '')
        win     = (row['馬名S'] == winner) if winner else False
        odds_v  = pd.to_numeric(row.get('単勝オッズ', np.nan), errors='coerce')

        all_rows.append({
            'week':       label,
            'race':       f"{venue_abbr}{r_int}R",
            'horse':      row['馬名S'],
            '印':          mark,
            '着順':        row.get('着順', np.nan),
            'オッズ':      odds_v,
            '的中':        win,
            '払戻100':     float(payout) if (win and pd.notna(payout)) else 0.0,
            'sub_diff':   row.get('sub_偏差値の差'),
            'cur_rank':   row.get('cur_ランカー順位'),
            'sub_rank':   row.get('sub_ランカー順位'),
            'combo_gap':  row.get('combo_gap'),
        })

df_all = pd.DataFrame(all_rows)
if df_all.empty:
    print("\nデータなし")
    sys.exit(0)

print(f"\n\n{'='*60}")
print("  ROI集計（印別・HTML推奨金額ベース）")
print(f"{'='*60}")
print(f"{'印':<5} {'頭数':>4} {'的中':>4} {'勝率':>7}  {'投資合計':>10}  {'払戻合計':>10}  {'損益':>10}  {'ROI':>8}")
print('-'*60)

summary = {}
for mark in ['激熱', '◎', '〇', '☆']:
    sub = df_all[df_all['印'] == mark]
    if sub.empty:
        continue
    n     = len(sub)
    wins  = int(sub['的中'].sum())
    wr    = wins / n
    bet   = HTML_BET.get(mark, 100)
    inv   = n * bet
    pay   = sub['払戻100'].sum() * (bet / 100)
    roi   = (pay - inv) / inv * 100 if inv > 0 else 0
    summary[mark] = {'n': n, 'wins': wins, 'inv': inv, 'pay': pay, 'roi': roi}
    print(f"{mark:<5} {n:>4} {wins:>4} {wr:>7.1%}  {inv:>10,.0f}円  {pay:>10,.0f}円  {pay-inv:>+10,.0f}円  {roi:>+7.1f}%")

print('-'*60)
total_inv = sum(v['inv'] for v in summary.values())
total_pay = sum(v['pay'] for v in summary.values())
total_roi = (total_pay - total_inv) / total_inv * 100 if total_inv else 0
print(f"{'合計':<5} {sum(v['n'] for v in summary.values()):>4} {sum(v['wins'] for v in summary.values()):>4}         {total_inv:>10,.0f}円  {total_pay:>10,.0f}円  {total_pay-total_inv:>+10,.0f}円  {total_roi:>+7.1f}%")

print(f"\n\n{'='*60}")
print("  的中馬一覧")
print(f"{'='*60}")
wins_df = df_all[df_all['的中'] == True]
if wins_df.empty:
    print("  的中なし")
else:
    for _, r in wins_df.iterrows():
        bet = HTML_BET.get(r['印'], 100)
        pay = r['払戻100'] * bet / 100
        print(f"  {r['week']} {r['race']} {r['horse']} 【{r['印']}】 {r['オッズ']:.1f}倍  払戻{pay:,.0f}円（投資{bet:,}円）")

print(f"\n\n{'='*60}")
print("  全買い目一覧")
print(f"{'='*60}")
marked = df_all[df_all['印'] != ''].sort_values(['week', 'race'])
for _, r in marked.iterrows():
    sd = f"{r['sub_diff']:+.1f}" if pd.notna(r['sub_diff']) else '-'
    cg = f"{r['combo_gap']:.1f}" if pd.notna(r['combo_gap']) else '-'
    win_str = '✓' if r['的中'] else '✗'
    print(f"  {r['week']} {r['race']:8s} 【{r['印']}】 {r['horse']:<16} オッズ{r['オッズ']:.1f}  sd{sd}  gap{cg}  {win_str}")

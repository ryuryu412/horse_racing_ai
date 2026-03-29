import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import os
import pickle
import json
import re
import argparse
import subprocess

# ============================================================
# サブモデル予測スクリプト（芝ダ × 距離帯 × クラスグループ）
# 使い方: python src/10_predict_submodel.py data/raw/出馬表形式XX月XX日.csv
# 06_predict_from_card.py の現行モデルとサブモデルを並べて比較表示
# ============================================================

def extract_venue(kaikai):
    m = re.search(r'\d+([^\d]+)', str(kaikai))
    return m.group(1) if m else str(kaikai)

def get_distance_band(dist):
    m = re.search(r'\d+', str(dist))
    if not m:
        return None
    d = int(m.group())
    if d <= 1400:   return '短距離'
    elif d <= 1800: return 'マイル'
    elif d <= 2200: return '中距離'
    else:           return '長距離'

def get_class_group(class_rank):
    try:
        r = float(class_rank)
    except (ValueError, TypeError):
        return '3勝以上'  # OP等クラス不明は最上位クラスとして扱う
    if np.isnan(r):
        return '3勝以上'  # NaN（一部OP）も同様
    r = int(r)
    if r == 1:   return '新馬'
    elif r == 2: return '未勝利'
    elif r == 3: return '1勝'
    elif r == 4: return '2勝'
    elif r >= 5: return '3勝以上'
    return '3勝以上'

def convert_card_to_base_format(card_path):
    """出馬表形式CSVを基本.csv 互換形式に変換（06と同じ処理）"""
    try:
        df = pd.read_csv(card_path, encoding='cp932', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(card_path, encoding='utf-8', low_memory=False)

    if '日付S' in df.columns:
        def parse_date_s(s):
            try:
                parts = str(s).replace('/', '.').split('.')
                y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                return (y - 2000) * 10000 + m * 100 + d
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

    col_map = {
        '芝ダ': '芝・ダ', '単オッズ': '単勝オッズ',
        '着': '着順', '前着順': '前走着順', '前人気': '前走人気',
        '前走騎手': '前騎手',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if '馬名S' not in df.columns and '馬名' in df.columns:
        df['馬名S'] = df['馬名']
    if '距離' in df.columns:
        df['距離'] = pd.to_numeric(df['距離'], errors='coerce')
    return df

def run_feature_engineering(base_dir, extra_df):
    """card_extra.csv に書いて 01_make_features.py を実行（06と同じ）"""
    card_path = os.path.join(base_dir, 'data', 'raw', 'card_extra.csv')
    extra_df.to_csv(card_path, index=False, encoding='utf-8-sig')
    print(f"card_extra.csv 書込: {len(extra_df):,}行")
    script = os.path.join(base_dir, 'src', '01_make_features.py')
    print("特徴量生成中...")
    result = subprocess.run(
        [sys.executable, script],
        cwd=base_dir, capture_output=True, text=True, encoding='utf-8'
    )
    if result.returncode != 0:
        print("エラー:", result.stderr[-500:] if result.stderr else "不明")
    else:
        lines = [l for l in result.stdout.strip().split('\n') if l]
        for l in lines[-3:]:
            print(l)
        # Parquet キャッシュを更新（次回以降の高速読み込み用）
        csv_path = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.csv')
        pq_path  = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.parquet')
        try:
            import pyarrow  # noqa
            print("Parquetキャッシュ更新中...")
            pd.read_csv(csv_path, low_memory=False).to_parquet(pq_path, index=False)
            print("Parquetキャッシュ更新完了")
        except ImportError:
            pass  # pyarrow未インストール時はスキップ

def predict_with_submodel(base_dir, target_date_num, card_df=None):
    """サブモデルで予測を実行し、現行モデルとの比較を表示する"""

    # ── サブモデル情報を読み込む ──────────────────────────
    sub_info_path = os.path.join(base_dir, 'models', 'submodel', 'submodel_info.json')
    if not os.path.exists(sub_info_path):
        print(f"エラー: {sub_info_path} が見つかりません。先に 09_train_submodel.py を実行してください。")
        return None
    with open(sub_info_path, 'r', encoding='utf-8') as f:
        sub_info = json.load(f)
    sub_features = sub_info['features']
    sub_models   = sub_info['models']

    # ── クラスランカー情報を読み込む ──────────────────────
    sub_rankers = {}
    rinfo_sub_path = os.path.join(base_dir, 'models', 'submodel_ranker', 'class_ranker_info.json')
    if os.path.exists(rinfo_sub_path):
        with open(rinfo_sub_path, 'r', encoding='utf-8') as f:
            sub_rankers = json.load(f).get('rankers', {})

    # ── 現行モデル情報を読み込む（比較用・任意）──────────
    cur_info_path = os.path.join(base_dir, 'models', 'model_info.json')
    cur_features  = None
    cur_models    = {}
    cur_rankers   = {}
    if os.path.exists(cur_info_path):
        with open(cur_info_path, 'r', encoding='utf-8') as f:
            cur_info = json.load(f)
        cur_features = cur_info['features']
        cur_models   = cur_info['models']
        rinfo_path   = os.path.join(base_dir, 'models', 'ranker', 'ranker_info.json')
        if os.path.exists(rinfo_path):
            with open(rinfo_path, 'r', encoding='utf-8') as f:
                cur_rankers = json.load(f).get('rankers', {})

    # ── 特徴量データを読み込む（Parquet優先 / なければCSVチャンク）────────────────────────────
    import time as _time
    feat_csv = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.csv')
    feat_pq  = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.parquet')
    t0 = _time.time()
    if os.path.exists(feat_pq) and os.path.getmtime(feat_pq) >= os.path.getmtime(feat_csv):
        df_all = pd.read_parquet(feat_pq)
        df_all['日付_num'] = pd.to_numeric(df_all['日付'], errors='coerce')
        day = df_all[df_all['日付_num'] == target_date_num].copy()
        print(f"Parquet読み込み: {_time.time()-t0:.1f}秒")
    else:
        chunks = []
        for chunk in pd.read_csv(feat_csv, low_memory=False, chunksize=10000):
            chunk['日付_num'] = pd.to_numeric(chunk['日付'], errors='coerce')
            filtered = chunk[chunk['日付_num'] == target_date_num]
            if not filtered.empty:
                chunks.append(filtered)
        day = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        print(f"CSVチャンク読み込み: {_time.time()-t0:.1f}秒")
    if day.empty:
        print(f"日付 {target_date_num} のデータが見つかりません。")
        return None

    # 馬体重パッチ: card_dfに馬体重があれば Parquet の値を上書き（当日発表後の再実行用）
    _weight_cols = [c for c in ['馬体重', '馬体重増減'] if card_df is not None and c in card_df.columns]
    if _weight_cols:
        _w = card_df[['馬名S'] + _weight_cols].copy()
        for c in _weight_cols:
            _w[c] = pd.to_numeric(_w[c], errors='coerce')
        _w = _w.dropna(subset=_weight_cols, how='all')
        if not _w.empty:
            # 既存列を一時退避して上書き
            for c in _weight_cols:
                if c in day.columns:
                    day = day.drop(columns=[c])
            day = day.merge(_w.drop_duplicates('馬名S'), on='馬名S', how='left')
            _patched = day[_weight_cols[0]].notna().sum()
            print(f"馬体重パッチ適用: {_patched}頭")

    # 現行モデルキー（会場×距離）
    day['会場']        = day['開催'].apply(extract_venue)
    day['cur_key']     = day['会場'] + '_' + day['距離'].astype(str)

    # サブモデルキー（芝ダ×距離帯×クラス）
    day['_surface']    = day['芝・ダ'].astype(str).str.strip()
    day['_dist_band']  = day['距離'].apply(get_distance_band)
    # ダートは中距離・長距離を統合
    mask_da_ml = (day['_surface'] == 'ダ') & (day['_dist_band'].isin(['中距離', '長距離']))
    day.loc[mask_da_ml, '_dist_band'] = '中長距離'
    day['_cls_group']  = day['クラス_rank'].apply(get_class_group) if 'クラス_rank' in day.columns else None
    day['sub_key']     = (day['_surface'].astype(str) + '_' +
                          day['_dist_band'].astype(str) + '_' +
                          day['_cls_group'].astype(str))

    if '単勝オッズ' in day.columns:
        day['単勝オッズ'] = pd.to_numeric(day['単勝オッズ'], errors='coerce')

    # 数値変換（全特徴量）
    all_feats = list(set((sub_features or []) + (cur_features or [])))
    for col in all_feats:
        if col in day.columns:
            day[col] = pd.to_numeric(day[col], errors='coerce')

    print(f"\n日付: {target_date_num}  /  {len(day)}頭\n")

    # ── モデルを一括プリロード（同一sub_keyの重複ロードを防ぐ）──
    t0 = _time.time()
    needed_keys = day['sub_key'].dropna().unique()
    model_cache  = {}  # sub_key -> (model, feature_names)
    ranker_cache = {}  # sub_key -> ranker
    cur_model_cache  = {}  # cur_key -> (model, feature_names)
    cur_ranker_cache = {}  # cur_key -> ranker
    for sk in needed_keys:
        if sk in sub_models:
            win_path = os.path.join(base_dir, 'models', 'submodel', sub_models[sk]['win'])
            if os.path.exists(win_path):
                with open(win_path, 'rb') as f:
                    m = pickle.load(f)
                model_cache[sk] = (m, m.booster_.feature_name())
        if sk in sub_rankers:
            rpath = os.path.join(base_dir, 'models', 'submodel_ranker', sub_rankers[sk])
            if os.path.exists(rpath):
                with open(rpath, 'rb') as f:
                    ranker_cache[sk] = pickle.load(f)
    if cur_features:
        needed_cur_keys = day['cur_key'].dropna().unique()
        for ck in needed_cur_keys:
            if ck in cur_models:
                win_path = os.path.join(base_dir, 'models', cur_models[ck]['win'])
                if os.path.exists(win_path):
                    with open(win_path, 'rb') as f:
                        m = pickle.load(f)
                    cur_model_cache[ck] = (m, m.booster_.feature_name())
            if ck in cur_rankers:
                rpath = os.path.join(base_dir, 'models', 'ranker', cur_rankers[ck])
                if os.path.exists(rpath):
                    with open(rpath, 'rb') as f:
                        cur_ranker_cache[ck] = pickle.load(f)
    print(f"モデルプリロード: {_time.time()-t0:.1f}秒 "
          f"(subモデル{len(model_cache)}+ランカー{len(ranker_cache)} / "
          f"現行{len(cur_model_cache)}+ランカー{len(cur_ranker_cache)})")

    # ── レース別に予測 ──────────────────────────────────
    race_keys = ['開催']
    if 'Ｒ' in day.columns:
        day['Ｒ'] = pd.to_numeric(day['Ｒ'], errors='coerce')
        race_keys.append('Ｒ')
    if 'レース名' in day.columns:
        race_keys.append('レース名')
    races = day.groupby(race_keys, sort=True).groups

    all_rows = []
    for group_key, idx in races.items():
        kaikai    = group_key[0]
        race_name = group_key[-1] if 'レース名' in day.columns else str(group_key)
        r_num     = group_key[1] if 'Ｒ' in day.columns else ''
        sub       = day.loc[idx].copy()
        sub_key   = sub['sub_key'].iloc[0]
        cur_key   = sub['cur_key'].iloc[0]

        # ── サブモデル予測（キャッシュ使用）──
        sub['sub_prob_win']    = np.nan
        sub['sub_偏差値の差']  = np.nan
        sub['sub_コース偏差値'] = np.nan
        sub['sub_ランカー順位'] = np.nan
        if sub_key in model_cache:
            m, wf = model_cache[sub_key]
            for c in wf:
                if c not in sub.columns: sub[c] = np.nan
            sub['sub_prob_win'] = m.predict_proba(sub[wf])[:, 1]

            st    = sub_models[sub_key].get('stats', {})
            w_m   = st.get('win_mean', sub['sub_prob_win'].mean())
            w_s   = st.get('win_std',  sub['sub_prob_win'].std())
            sub['sub_コース偏差値']  = 50 + 10 * (sub['sub_prob_win'] - w_m) / (w_s if w_s > 0 else 1)
            r_m   = sub['sub_prob_win'].mean()
            r_s   = sub['sub_prob_win'].std()
            sub['sub_レース内偏差値'] = 50 + 10 * (sub['sub_prob_win'] - r_m) / (r_s if r_s > 0 else 1)
            sub['sub_偏差値の差']    = sub['sub_レース内偏差値'] - sub['sub_コース偏差値']

            if sub_key in ranker_cache:
                scores = ranker_cache[sub_key].predict(sub[wf])
                sub['sub_ランカー順位'] = pd.Series(scores, index=sub.index).rank(
                    ascending=False, method='min').astype(int)

        # ── 現行モデル予測（キャッシュ使用）──
        sub['cur_prob_win']       = np.nan
        sub['cur_ランカー順位']    = np.nan
        sub['cur_偏差値の差']      = np.nan
        sub['cur_コース偏差値']    = np.nan
        sub['cur_レース内偏差値']  = np.nan
        if cur_key in cur_model_cache:
            m, wf = cur_model_cache[cur_key]
            for c in wf:
                if c not in sub.columns: sub[c] = np.nan
            sub['cur_prob_win'] = m.predict_proba(sub[wf])[:, 1]

            st   = cur_models[cur_key].get('stats', {})
            w_m  = st.get('win_mean', sub['cur_prob_win'].mean())
            w_s  = st.get('win_std',  sub['cur_prob_win'].std())
            sub['cur_コース偏差値']   = 50 + 10 * (sub['cur_prob_win'] - w_m) / (w_s if w_s > 0 else 1)
            r_m  = sub['cur_prob_win'].mean()
            r_s  = sub['cur_prob_win'].std()
            sub['cur_レース内偏差値'] = 50 + 10 * (sub['cur_prob_win'] - r_m) / (r_s if r_s > 0 else 1)
            sub['cur_偏差値の差']     = sub['cur_レース内偏差値'] - sub['cur_コース偏差値']

            if cur_key in cur_ranker_cache:
                scores = cur_ranker_cache[cur_key].predict(sub[cur_features])
                sub['cur_ランカー順位'] = pd.Series(scores, index=sub.index).rank(
                    ascending=False, method='min').astype(int)

        # ターミナル表示
        venue = extract_venue(kaikai)
        shiba_da = sub['芝・ダ'].iloc[0] if '芝・ダ' in sub.columns else ''
        kyori    = sub['距離'].iloc[0] if '距離' in sub.columns else ''
        print(f"{'='*70}")
        print(f"  {venue} {r_num}R　{race_name}  [{shiba_da}{kyori}m / {sub_key}]")
        print(f"{'='*70}")

        disp = sub[['馬名S']].copy()
        disp['現行_差']   = sub['cur_偏差値の差'].map(lambda x: f'{x:+.1f}' if pd.notna(x) else '-')
        disp['現行_ランカー'] = sub['cur_ランカー順位'].map(lambda x: f'{int(x)}位' if pd.notna(x) else '-')
        disp['sub_差']      = sub['sub_偏差値の差'].map(lambda x: f'{x:+.1f}' if pd.notna(x) else 'モデルなし')
        disp['sub_ランカー'] = sub['sub_ランカー順位'].map(lambda x: f'{int(x)}位' if pd.notna(x) else '-')
        disp['sub_コース偏差値'] = sub.get('sub_コース偏差値', pd.Series(np.nan, index=sub.index)).map(
            lambda x: f'{x:.1f}' if pd.notna(x) else '-')
        disp['sub_%']      = sub['sub_prob_win'].map(lambda x: f'{x*100:.1f}' if pd.notna(x) else '-')
        if '単勝オッズ' in sub.columns:
            disp['オッズ'] = sub['単勝オッズ'].map(lambda x: f'{x:.1f}' if pd.notna(x) else '-')
        print(disp.to_string(index=False))
        print()

        sub['レース名'] = race_name
        all_rows.append(sub)

    if not all_rows:
        print("対応サブモデルが見つかりませんでした。")
        return None

    result = pd.concat(all_rows, ignore_index=True)

    # ── サマリー: サブモデルで差+15以上の馬 ──
    diff_num = pd.to_numeric(result['sub_偏差値の差'], errors='coerce')
    hits15 = result[diff_num >= 15].copy()
    hits20 = result[diff_num >= 20].copy()

    print(f"\n{'='*70}")
    print(f"  サブモデル 注目馬サマリー")
    print(f"{'='*70}")

    for label, df_sub in [('◎ sub差+20以上', hits20), ('○ sub差+15〜19', hits15[diff_num[hits15.index] < 20])]:
        if df_sub.empty:
            print(f"{label}: 該当なし")
            continue
        print(f"\n{label}（{len(df_sub)}頭）")
        s = df_sub[['レース名', '馬名S']].copy()
        s['sub_差']      = df_sub['sub_偏差値の差'].map(lambda x: f'{x:+.1f}' if pd.notna(x) else '-')
        s['sub_ランカー'] = df_sub['sub_ランカー順位'].map(lambda x: f'{int(x)}位' if pd.notna(x) else '-')
        s['現行_差']     = df_sub['cur_偏差値の差'].map(lambda x: f'{x:+.1f}' if pd.notna(x) else '-')
        s['現行_ランカー'] = df_sub['cur_ランカー順位'].map(lambda x: f'{int(x)}位' if pd.notna(x) else '-')
        s['オッズ']      = df_sub['単勝オッズ'].map(lambda x: f'{x:.1f}' if pd.notna(x) else '-') if '単勝オッズ' in df_sub.columns else '-'
        print(s.to_string(index=False))

    return result


def generate_html(result, card_df, target_date_num, out_path):
    """両モデル予測結果をHTMLレポートとして出力（横向き印刷対応）"""

    # ROI統計JSON読み込み（事前計算済み）
    _roi_stats_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'roi_stats.json')
    _roi_stats = {}
    if os.path.exists(_roi_stats_path):
        with open(_roi_stats_path, encoding='utf-8') as _f:
            _roi_stats = json.load(_f)
    _roi_cur = _roi_stats.get('distance_model', {})
    _roi_sub = _roi_stats.get('class_model', {})

    def roi_line(stats_dict, key):
        """ROI統計を1行テキストで返す。データなしなら空文字"""
        s = stats_dict.get(key)
        if not s:
            return ''
        def r(v): return f'{v:+.1%}' if v is not None else 'N/A'
        n = s.get('n_total', 0)
        return (f"2012テスト {n:,}頭　"
                f"ROI　◎{r(s['diff20']['roi'])}({s['diff20']['n']}頭)　"
                f"〇{r(s['diff15']['roi'])}({s['diff15']['n']}頭)　"
                f"▲{r(s['diff10']['roi'])}({s['diff10']['n']}頭)　"
                f"ランカー1位{r(s['ranker1']['roi'])}({s['ranker1']['n']}頭)")

    d = str(target_date_num)
    date_str = f"20{d[0:2]}/{d[2:4]}/{d[4:6]}" if len(d) == 6 else str(target_date_num)

    # ── カードデータをマージ（枠番・馬番・騎手等の表示用） ──
    card_map = {'枠番':'dc_枠番','馬番':'dc_馬番','性':'dc_性','齢':'dc_齢',
                '性別':'dc_性別','年齢':'dc_年齢','騎手':'dc_騎手',
                '調教師':'dc_調教師','斤量':'dc_斤量','馬体重':'dc_馬体重',
                '増減':'dc_増減','前走着順':'dc_前走着順','前走人気':'dc_前走人気',
                '単勝オッズ':'dc_単勝オッズ'}
    card_cols = ['馬名S'] + [c for c in card_map if c in card_df.columns]
    card_disp = card_df[card_cols].drop_duplicates('馬名S').rename(
        columns={k:v for k,v in card_map.items() if k in card_df.columns})
    df = result.merge(card_disp, on='馬名S', how='left')

    # ── 各種フラグ ──
    cur_diff  = pd.to_numeric(df['cur_偏差値の差'],  errors='coerce')
    sub_diff  = pd.to_numeric(df['sub_偏差値の差'],  errors='coerce')
    cur_r1    = df['cur_ランカー順位'].apply(lambda x: pd.notna(x) and int(x)==1 if pd.notna(x) else False)
    sub_r1    = df['sub_ランカー順位'].apply(lambda x: pd.notna(x) and int(x)==1 if pd.notna(x) else False)
    cur_nota  = cur_diff >= 10   # Page1の▲以上すべて
    sub_nota  = sub_diff >= 10   # Page2の▲以上すべて
    both_nota = cur_nota & sub_nota

    def fmt(val, fmt_str='+.1f', default='-'):
        try:
            return format(float(val), fmt_str) if pd.notna(val) else default
        except: return default

    def rank_str(val):
        try: return f"{int(float(val))}位" if pd.notna(val) else '-'
        except: return '-'

    def seir(row):
        if 'dc_性別' in row.index and pd.notna(row.get('dc_性別')):
            return f"{row.get('dc_性別','')}{row.get('dc_年齢','')}"
        return f"{row.get('dc_性','')}{row.get('dc_齢','')}"

    # ── CSS ──
    css = """<style>
      @page { size: A4 landscape; margin: 8mm; }
      body { font-family:'Yu Gothic','Hiragino Sans',sans-serif; font-size:11px; margin:6px; background:#f0f0f0; }
      h1 { font-size:16px; margin:6px 0; color:#1a252f; }
      h2 { font-size:13px; background:#2c3e50; color:white; padding:5px 10px; margin:14px 0 0; border-radius:4px 4px 0 0; }
      .race-info { font-size:10px; color:#555; padding:2px 10px; background:#dde4ea; margin:0; }
      .model-stats { font-size:8.5px; padding:1px 10px; background:#f7f9f9; margin:0; }
      .page { page-break-after:always; break-after:page; padding:4px; }
      .page:last-child { page-break-after:avoid; }
      table { border-collapse:collapse; width:100%; background:white; margin-bottom:4px; }
      th { background:#2c3e50; color:white; padding:3px 5px; font-size:9px; text-align:center; white-space:nowrap; border:1px solid #1a252f; }
      td { padding:2px 4px; text-align:center; white-space:nowrap; font-size:10px; border:1px solid #ccc; }
      td.name { text-align:left; font-weight:bold; font-size:11px; }
      td.sub-detail { font-size:9px; color:#555; text-align:left; }
      /* 行ハイライト */
      .both  td { background:#fde8e8 !important; }
      .cur15 td { background:#fff9c4 !important; }
      .sub15 td { background:#e8f4fd !important; }
      /* 数値色 */
      .hi { color:#c0392b; font-weight:bold; }
      .md { color:#e67e22; font-weight:bold; }
      /* セクション見出し */
      .sec-cur  h2 { background:#1e8449; }
      .sec-cur     { background:#f0faf4; }
      .sec-sub  h2 { background:#1f618d; }
      .sec-sub     { background:#eaf4fb; }
      .sec-both h2 { background:#6c3483; }
      .sec-both    { background:#f5eef8; }
      /* バッジ */
      .badge { display:inline-block; font-size:8px; font-weight:bold; border-radius:3px;
               padding:1px 3px; margin:1px; }
      .b-both { background:#8e44ad; color:white; }
      .b-cur  { background:#27ae60; color:white; }
      .b-sub  { background:#2980b9; color:white; }
      input.memo { width:120px; border:1px solid #ccc; border-radius:3px; padding:1px 3px; font-size:10px; }
      /* 両モデル一致カード */
      .both-list { display:flex; flex-wrap:wrap; gap:8px; padding:4px; background:white; }
      .both-card { border:2px solid #8e44ad; border-radius:6px; padding:6px 10px; min-width:220px; max-width:320px; background:#fdf5ff; }
      .both-race { font-size:9px; color:#666; margin-bottom:3px; }
      .both-horse { display:flex; align-items:baseline; gap:6px; margin-bottom:4px; }
      .both-banum { font-size:10px; color:#888; }
      .both-name  { font-size:15px; font-weight:bold; color:#1a1a2e; }
      .both-jockey { font-size:10px; color:#555; }
      .both-odds  { font-size:10px; color:#e67e22; margin-left:auto; }
      .both-eval  { display:flex; flex-direction:column; gap:2px; }
      .both-cur   { font-size:10px; color:#27ae60; font-weight:bold; }
      .both-sub   { font-size:10px; color:#2980b9; font-weight:bold; }
      @media print { input { display:none; } body { margin:0; } h2 { margin-top:0; } }
    </style>"""

    def diff_cell(val):
        s = fmt(val)
        try:
            v = float(val)
            if v >= 20: return f'<td class="hi">{s}</td>'
            if v >= 15: return f'<td class="md">{s}</td>'
        except: pass
        return f'<td>{s}</td>'

    # ── サマリーテーブル共通 ──
    def summary_table(mask, show_cur=True, show_sub=True):
        rows = df[mask].copy()
        if rows.empty: return '<p style="padding:6px;background:white;">該当なし</p>'
        race_keys = [c for c in ['開催','Ｒ','レース名'] if c in rows.columns]
        rows = rows.sort_values(race_keys)
        html = ['<table><thead><tr>']
        html.append('<th>レース</th><th>馬番</th><th>馬名</th><th>騎手</th>')
        if show_cur: html.append('<th>会場×距離<br>ランカー</th><th>会場×距離<br>偏差値差</th>')
        if show_sub: html.append('<th>クラス<br>ランカー</th><th>クラス<br>偏差値差</th>')
        html.append('<th>単勝%</th><th>オッズ</th></tr></thead><tbody>')
        for _, r in rows.iterrows():
            venue = extract_venue(r.get('開催',''))
            r_num = int(r['Ｒ']) if 'Ｒ' in rows.columns and pd.notna(r.get('Ｒ')) else ''
            race_lbl = f"{venue}{r_num}R {r.get('レース名','')}"
            banum = int(r['dc_馬番']) if pd.notna(r.get('dc_馬番')) else '-'
            # 行クラス
            bc = 'both' if both_nota[r.name] else ('cur15' if cur_nota[r.name] else 'sub15')
            row_html = f'<tr class="{bc}"><td>{race_lbl}</td><td>{banum}</td>'
            row_html += f'<td class="name">{r.get("馬名S","")}</td>'
            row_html += f'<td>{r.get("dc_騎手", r.get("騎手",""))}</td>'
            if show_cur:
                row_html += f'<td>{rank_str(r.get("cur_ランカー順位"))}</td>'
                row_html += diff_cell(r.get('cur_偏差値の差'))
            if show_sub:
                row_html += f'<td>{rank_str(r.get("sub_ランカー順位"))}</td>'
                row_html += diff_cell(r.get('sub_偏差値の差'))
            prob = r.get('cur_prob_win') if show_cur and pd.notna(r.get('cur_prob_win')) else r.get('sub_prob_win')
            row_html += f'<td>{fmt(prob,":.1%","-") if prob else "-"}</td>'
            _odds = r.get('dc_単勝オッズ') if pd.notna(r.get('dc_単勝オッズ')) else r.get('単勝オッズ')
            row_html += f'<td>{fmt(_odds,":.1f")}</td></tr>'
            html.append(row_html)
        html.append('</tbody></table>')
        return ''.join(html)

    # ── 印別セクション生成（◎〇▲ + テーマカラー）──
    # 会場×距離=緑, クラス=青 の3段階カラー
    GREEN = {'◎': '#1a6e3c', '〇': '#27ae60', '▲': '#58d68d', 'th': '#1e8449', 'bg': '#eafaf1'}
    BLUE  = {'◎': '#1a3f6e', '〇': '#2471a3', '▲': '#5dade2', 'th': '#1f618d', 'bg': '#eaf4fb'}

    def tier_table(label, mask, diff_col, rank_col, palette):
        """ランカー順位込みの印別馬リストを生成"""
        horses = df[mask].copy()
        if horses.empty:
            return f'<p style="padding:4px 8px;color:#999;font-size:10px">{label}: 該当なし</p>'
        rk = [c for c in ['開催','Ｒ','レース名'] if c in horses.columns]
        horses = horses.sort_values(rk)
        tier_color = palette.get(label[0], palette['▲'])
        rows_html = []
        for _, r in horses.iterrows():
            venue  = extract_venue(r.get('開催',''))
            r_num  = int(r['Ｒ']) if 'Ｒ' in horses.columns and pd.notna(r.get('Ｒ')) else ''
            r_name = r.get('レース名','')
            banum  = int(r['dc_馬番']) if pd.notna(r.get('dc_馬番')) else '-'
            jockey = r.get('dc_騎手', r.get('騎手',''))
            diff_v = r.get(diff_col)
            rank_v = r.get(rank_col)
            odds_v = r.get('dc_単勝オッズ') if pd.notna(r.get('dc_単勝オッズ')) else r.get('単勝オッズ')
            rank_s = rank_str(rank_v)
            odds_s = fmt(odds_v, '.1f')
            rows_html.append(
                f'<tr>'
                f'<td style="text-align:left;padding:2px 6px">{venue} {r_num}R　{r_name}</td>'
                f'<td>{banum}番</td>'
                f'<td style="text-align:left;font-weight:bold;font-size:12px">{r.get("馬名S","")}</td>'
                f'<td>{jockey}</td>'
                f'<td style="font-weight:bold;color:{palette["th"]}">{rank_s}</td>'
                f'{diff_cell(diff_v)}'
                f'<td>{odds_s}</td>'
                f'</tr>'
            )
        return (f'<div style="font-size:12px;font-weight:bold;color:{tier_color};'
                f'padding:4px 8px;margin:10px 0 2px;border-left:5px solid {tier_color};'
                f'background:{palette["bg"]}">'
                f'{label}（{len(horses)}頭）</div>'
                f'<table><thead><tr style="background:{palette["th"]}">'
                f'<th style="text-align:left">レース</th><th>馬番</th><th style="text-align:left">馬名</th>'
                f'<th>騎手</th><th>ランカー順位</th><th>偏差値差</th><th>オッズ</th>'
                f'</tr></thead><tbody>{"".join(rows_html)}</tbody></table>')

    # ── Page1: 会場×距離モデル（緑）──
    page1 = f"""<div class="page sec-cur">
      <h1>競馬AI予測レポート　{date_str}</h1>
      <h2>【会場×距離モデル】注目馬</h2>
      <p style="font-size:9px;color:#666;margin:2px 0 6px">※ バックテスト: ランカー1位 ROI+31.9% / 差+15 ROI+33.4% / 差+20 ROI+37.8%（後ろ20%・約12.3万頭）</p>
      {tier_table('◎偏差値差20以上', cur_diff >= 20, 'cur_偏差値の差', 'cur_ランカー順位', GREEN)}
      {tier_table('〇偏差値差15以上', (cur_diff >= 15) & (cur_diff < 20), 'cur_偏差値の差', 'cur_ランカー順位', GREEN)}
      {tier_table('▲偏差値差10以上', (cur_diff >= 10) & (cur_diff < 15), 'cur_偏差値の差', 'cur_ランカー順位', GREEN)}
    </div>"""

    # ── Page2: クラスモデル（青）──
    page2 = f"""<div class="page sec-sub">
      <h2>【クラスモデル】注目馬</h2>
      <p style="font-size:9px;color:#666;margin:2px 0 6px">※ 2012テスト: クラスランク1位 ROI+55.5% / 差+15 ROI+99.6% / 差+20 ROI+137.8%</p>
      {tier_table('◎クラス差20以上', sub_diff >= 20, 'sub_偏差値の差', 'sub_ランカー順位', BLUE)}
      {tier_table('〇クラス差15以上', (sub_diff >= 15) & (sub_diff < 20), 'sub_偏差値の差', 'sub_ランカー順位', BLUE)}
      {tier_table('▲クラス差10以上', (sub_diff >= 10) & (sub_diff < 15), 'sub_偏差値の差', 'sub_ランカー順位', BLUE)}
    </div>"""

    # ── Page3: 両モデル一致（ツリー形式）──
    both_r1 = cur_r1 & sub_r1   # 両モデルでランカー1位

    def make_both_cards(mask):
        rows = df[mask].copy()
        if rows.empty:
            return '<p style="padding:8px;background:white;color:#999;font-size:10px">該当なし</p>'
        rk = [c for c in ['開催','Ｒ','レース名'] if c in rows.columns]
        rows = rows.sort_values(rk)
        items = []
        for _, r in rows.iterrows():
            venue  = extract_venue(r.get('開催',''))
            r_num  = int(r['Ｒ']) if 'Ｒ' in rows.columns and pd.notna(r.get('Ｒ')) else ''
            r_name = r.get('レース名','')
            banum  = int(r['dc_馬番']) if pd.notna(r.get('dc_馬番')) else '-'
            horse  = r.get('馬名S','')
            jockey = r.get('dc_騎手', r.get('騎手',''))
            odds   = fmt(r.get('dc_単勝オッズ') if pd.notna(r.get('dc_単勝オッズ')) else r.get('単勝オッズ'), '.1f')

            cdv = r.get('cur_偏差値の差')
            cur_tier = '◎' if pd.notna(cdv) and cdv >= 20 else ('〇' if pd.notna(cdv) and cdv >= 15 else ('▲' if pd.notna(cdv) and cdv >= 10 else ''))
            cur_lbl_parts = [cur_tier] if cur_tier else []
            if pd.notna(r.get('cur_ランカー順位')):
                cur_lbl_parts.append(f"ランカー{int(r['cur_ランカー順位'])}位")
            if pd.notna(cdv):
                cur_lbl_parts.append(f"偏差値差{fmt(cdv)}")
            cur_lbl = '　'.join(cur_lbl_parts) if cur_lbl_parts else '-'

            sdv = r.get('sub_偏差値の差')
            sub_tier = '◎' if pd.notna(sdv) and sdv >= 20 else ('〇' if pd.notna(sdv) and sdv >= 15 else ('▲' if pd.notna(sdv) and sdv >= 10 else ''))
            sub_lbl_parts = [sub_tier] if sub_tier else []
            if pd.notna(r.get('sub_ランカー順位')):
                sub_lbl_parts.append(f"ランカー{int(r['sub_ランカー順位'])}位")
            if pd.notna(sdv):
                sub_lbl_parts.append(f"クラス差{fmt(sdv)}")
            sub_lbl = '　'.join(sub_lbl_parts) if sub_lbl_parts else '-'

            items.append(f"""
            <div class="both-card">
              <div class="both-race">{venue} {r_num}R　{r_name}</div>
              <div class="both-horse">
                <span class="both-banum">{banum}番</span>
                <span class="both-name">{horse}</span>
                <span class="both-jockey">{jockey}</span>
                <span class="both-odds">オッズ {odds}</span>
              </div>
              <div class="both-eval">
                <span class="both-cur">┗ 距離モデル：{cur_lbl}</span>
                <span class="both-sub">┗ クラスモデル：{sub_lbl}</span>
              </div>
            </div>""")
        return '<div class="both-list">' + ''.join(items) + '</div>'

    page3 = f"""<div class="page sec-both">
      <h2>【両モデル一致】穴馬候補　（両モデルで偏差値差10以上）</h2>
      {make_both_cards(both_nota)}
    </div>"""

    page4_ranker = f"""<div class="page sec-both">
      <h2>【両モデル一致】本命候補　（両モデルでランカー1位）　※複勝向き</h2>
      {make_both_cards(both_r1)}
    </div>"""

    # ── レース別詳細（1レース1ページ）──
    def horse_tier(diff_val):
        try:
            v = float(diff_val)
            if v >= 20: return '◎'
            if v >= 15: return '〇'
            if v >= 10: return '▲'
        except: pass
        return ''

    race_keys = [c for c in ['開催','Ｒ','レース名'] if c in df.columns]
    race_pages = []
    for gk, idx in df.groupby(race_keys, sort=True).groups.items():
        grp = df.loc[idx].copy()
        grp = grp.sort_values('dc_馬番', na_position='last')
        kaikai    = gk[0]
        race_name = gk[-1] if 'レース名' in df.columns else ''
        r_num     = int(gk[1]) if 'Ｒ' in df.columns and pd.notna(gk[1]) else ''
        venue     = extract_venue(kaikai)
        shiba_da  = grp['芝・ダ'].iloc[0] if '芝・ダ' in grp.columns else ''
        _kyori_raw = grp['距離'].iloc[0] if '距離' in grp.columns else ''
        try: kyori = int(re.search(r'\d+', str(_kyori_raw)).group())
        except: kyori = _kyori_raw
        sub_key   = grp['sub_key'].iloc[0] if 'sub_key' in grp.columns else ''
        cur_key_r = grp['cur_key'].iloc[0] if 'cur_key' in grp.columns else ''
        # sub_key から距離帯・クラスだけ取り出す（例: "芝_マイル_1勝" → "マイル・1勝クラス"）
        sk_parts  = str(sub_key).split('_')
        cls_label = '・'.join(sk_parts[1:]) if len(sk_parts) >= 3 else sub_key
        surface_label = {'芝': '芝', 'ダ': 'ダート'}.get(str(shiba_da).strip(), str(shiba_da))
        cur_roi_txt = roi_line(_roi_cur, cur_key_r)
        sub_roi_txt = roi_line(_roi_sub, sub_key)

        rows_html = []
        for _, r in grp.iterrows():
            cur_t = horse_tier(r.get('cur_偏差値の差'))
            sub_t = horse_tier(r.get('sub_偏差値の差'))
            if cur_t and sub_t:   row_bg = 'background:#fde8e8'
            elif cur_t:           row_bg = 'background:#eafaf1'
            elif sub_t:           row_bg = 'background:#eaf4fb'
            else:                 row_bg = ''

            bw = r.get('dc_馬体重', '-')
            bw_d = r.get('dc_増減', '')
            try: taiju = f"{bw}({'+' if float(bw_d)>=0 else ''}{int(float(bw_d))})" if pd.notna(bw_d) and bw_d != '' else str(bw)
            except: taiju = str(bw)

            odds_v = r.get('dc_単勝オッズ') if pd.notna(r.get('dc_単勝オッズ')) else r.get('単勝オッズ')

            rows_html.append(
                f'<tr style="{row_bg}">'
                f'<td>{r.get("dc_枠番","-")}</td>'
                f'<td>{r.get("dc_馬番","-")}</td>'
                f'<td style="text-align:left;font-weight:bold;white-space:nowrap">{r.get("馬名S","")}'
                f' <span style="font-weight:normal;color:#888;font-size:9px">{seir(r)}</span></td>'
                f'<td style="white-space:nowrap">{r.get("dc_騎手", r.get("騎手",""))}'
                f'<span style="color:#888;font-size:9px"> {r.get("dc_斤量","")}kg</span></td>'
                f'<td>{fmt(odds_v, ".1f")}</td>'
                f'<td style="font-size:14px;color:#1e8449;font-weight:bold">{cur_t}</td>'
                f'<td>{fmt(r.get("cur_コース偏差値"), ".1f")}</td>'
                f'<td>{fmt(r.get("cur_レース内偏差値"), ".1f")}</td>'
                f'<td>{fmt(r.get("cur_偏差値の差"), ".1f")}</td>'
                f'<td style="color:#1e8449">{fmt(r.get("cur_prob_win"), ".1%")}</td>'
                f'<td>{rank_str(r.get("cur_ランカー順位"))}</td>'
                f'<td style="font-size:14px;color:#1f618d;font-weight:bold">{sub_t}</td>'
                f'<td>{fmt(r.get("sub_コース偏差値"), ".1f")}</td>'
                f'<td>{fmt(r.get("sub_レース内偏差値"), ".1f")}</td>'
                f'<td>{fmt(r.get("sub_偏差値の差"), ".1f")}</td>'
                f'<td style="color:#1f618d">{fmt(r.get("sub_prob_win"), ".1%")}</td>'
                f'<td>{rank_str(r.get("sub_ランカー順位"))}</td>'
                f'</tr>'
            )

        cur_roi_html = f'<div class="model-stats" style="color:#1e8449">【距離】{cur_roi_txt}</div>' if cur_roi_txt else ''
        sub_roi_html = f'<div class="model-stats" style="color:#1f618d">【クラス】{sub_roi_txt}</div>' if sub_roi_txt else ''
        race_pages.append(f"""<div class="page">
          <h2>{venue} {r_num}R　{race_name}</h2>
          <div class="race-info">{surface_label} {kyori}m　／　{cls_label}</div>
          {cur_roi_html}{sub_roi_html}
          <table style="font-size:9px">
            <thead><tr>
              <th>枠</th><th>馬番</th><th style="text-align:left">馬名/性齢</th><th>騎手/斤量</th><th>オッズ</th>
              <th style="color:#58d68d">距離<br>印</th><th style="color:#58d68d">距離<br>絶対偏差</th><th style="color:#58d68d">距離<br>レース内偏差</th><th style="color:#58d68d">距離<br>偏差値差</th><th style="color:#58d68d">距離<br>勝率</th><th style="color:#58d68d">距離<br>ランカー</th>
              <th style="color:#5dade2">クラス<br>印</th><th style="color:#5dade2">クラス<br>絶対偏差</th><th style="color:#5dade2">クラス<br>レース内偏差</th><th style="color:#5dade2">クラス<br>偏差値差</th><th style="color:#5dade2">クラス<br>勝率</th><th style="color:#5dade2">クラス<br>ランカー</th>
            </tr></thead>
            <tbody>{''.join(rows_html)}</tbody>
          </table>
        </div>""")

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>競馬AI予測 {date_str}</title>
  {css}
</head>
<body>
  {page1}
  {page2}
  {page3}
  {page4_ranker}
  {''.join(race_pages)}
</body>
</html>"""

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\nHTML出力: {out_path}")

    # ── Chrome ヘッドレスで PDF 生成 ──
    pdf_path = out_path.replace('.html', '.pdf')
    chrome_candidates = [
        r'C:\Program Files\Google\Chrome\Application\chrome.exe',
        r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
    ]
    chrome = next((c for c in chrome_candidates if os.path.exists(c)), None)
    if chrome:
        try:
            result_pdf = subprocess.run([
                chrome,
                '--headless=new',
                '--disable-gpu',
                '--no-sandbox',
                '--no-pdf-header-footer',
                f'--print-to-pdf={pdf_path}',
                f'file:///{out_path.replace(chr(92), "/")}',
            ], capture_output=True, timeout=60)
            if os.path.exists(pdf_path):
                print(f"PDF出力: {pdf_path}")
            else:
                print(f"PDF生成失敗: {result_pdf.stderr.decode('utf-8','ignore')[-200:]}")
        except Exception as e:
            print(f"PDF生成エラー: {e}")
    else:
        print("Chrome未検出のためPDF生成スキップ")


def main():
    parser = argparse.ArgumentParser(description='サブモデル（芝ダ×距離帯×クラス）予測')
    parser.add_argument('card_file', help='出馬表形式CSVのパス')
    parser.add_argument('--no-rebuild', action='store_true', help='特徴量再生成をスキップ')
    parser.add_argument('--no-html',   action='store_true', help='HTMLレポート出力をスキップ')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if 'src' in os.path.abspath(__file__) else '.'

    print(f"=== 競馬AI サブモデル予測 ===")
    print(f"入力: {args.card_file}")

    # ── recent_all.csv の取り込み忘れチェック ──
    _recent_path  = os.path.join(base_dir, 'data', 'raw', 'recent_all.csv')
    _master_path  = os.path.join(base_dir, 'data', 'raw', 'master_all.csv')
    if os.path.exists(_recent_path) and os.path.exists(_master_path):
        try:
            _r = pd.read_csv(_recent_path, encoding='cp932', low_memory=False, usecols=['日付'])
            _r_max = pd.to_numeric(_r['日付'], errors='coerce').max()
            _master_max = 0
            for _chunk in pd.read_csv(_master_path, encoding='cp932', low_memory=False,
                                       chunksize=50000, usecols=['日付']):
                _d = pd.to_numeric(_chunk['日付'], errors='coerce').max()
                if _d > _master_max: _master_max = _d
            if _r_max > _master_max:
                print(f'\n⚠️  警告: recent_all.csv に未取り込みデータがあります')
                print(f'   recent_all 最終日付: {int(_r_max)}  /  master_all 最終日付: {int(_master_max)}')
                print(f'   先に「python src/weekly_update.py」を実行することを推奨します。')
                print()
        except Exception:
            pass

    card_df     = convert_card_to_base_format(args.card_file)
    dates       = card_df['日付'].dropna().unique()
    print(f"日付: {[int(d) for d in sorted(dates)]}")
    target_date = int(sorted(dates)[-1])

    # 既存Parquetに今日のデータが含まれていればスキップ（--no-rebuildと同等）
    feat_pq = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.parquet')
    feat_csv = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.csv')
    _already_built = False
    if not args.no_rebuild and os.path.exists(feat_pq):
        try:
            import pyarrow.parquet as pq
            pq_dates = pq.read_table(feat_pq, columns=['日付']).to_pandas()['日付']
            if target_date in pd.to_numeric(pq_dates, errors='coerce').values:
                print(f"特徴量再生成スキップ（Parquetに {target_date} のデータ確認済み）")
                _already_built = True
        except Exception:
            pass

    if not args.no_rebuild and not _already_built:
        run_feature_engineering(base_dir, card_df)
    elif args.no_rebuild:
        print("特徴量再生成スキップ")

    result = predict_with_submodel(base_dir, target_date, card_df)

    if result is not None and not args.no_html:
        d = str(target_date)
        date_str = f"20{d[0:2]}{d[2:4]}{d[4:6]}" if len(d) == 6 else str(target_date)
        gdrive_dir = r'G:\マイドライブ\競馬AI'
        out_dir  = gdrive_dir if os.path.isdir(gdrive_dir) else os.path.join(base_dir, 'output')
        out_path = os.path.join(out_dir, f"predict_{date_str}.html")
        generate_html(result, card_df, target_date, out_path)


if __name__ == '__main__':
    main()

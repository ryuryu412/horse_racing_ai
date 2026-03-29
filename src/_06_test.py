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

# Yahoo競馬オッズ取得（cloudscraper/bs4 が入っていない場合は無効化）
try:
    import cloudscraper
    from bs4 import BeautifulSoup
    _SCRAPER_AVAILABLE = True
except ImportError:
    _SCRAPER_AVAILABLE = False

# JRA-VAN会場略称 → netkeiba場コード
_VENUE_CODE = {
    '札': '01', '函': '02', '福': '03', '新': '04',
    '東': '05', '中山': '06', '中': '06', '中京': '07',
    '京都': '08', '京': '08', '阪神': '09', '阪': '09',
    '小倉': '10', '小': '10',
}

def _make_race_id(kaikai, date_num, r_num):
    """開催(例:'1東8') + 日付(260315) + R番号(11) → 12桁netkeiba race_id"""
    try:
        d = str(int(date_num))          # '260315'
        year = '20' + d[:2]             # '2026'
        s = str(kaikai).strip()
        kai_m   = re.match(r'(\d+)', s)
        venue_m = re.search(r'\d+([^\d]+)', s)
        day_m   = re.search(r'(\d+)$', s)
        kai   = kai_m.group(1).zfill(2)   if kai_m   else '01'
        abbr  = venue_m.group(1)          if venue_m else ''
        day   = day_m.group(1).zfill(2)   if day_m   else '01'
        code  = _VENUE_CODE.get(abbr, '00')
        r_str = str(int(r_num)).zfill(2)
        return f"{year}{code}{kai}{day}{r_str}"
    except Exception:
        return None

def _yahoo_odds(race_id):
    """Yahoo競馬から単勝オッズを取得: {馬番str: float}"""
    if not _SCRAPER_AVAILABLE:
        return {}
    yahoo_id = race_id[2:] if len(race_id) == 12 else race_id
    url = f"https://sports.yahoo.co.jp/keiba/race/odds/tfw/{yahoo_id}/"
    try:
        scraper = cloudscraper.create_scraper()
        soup = BeautifulSoup(scraper.get(url).text, 'html.parser')
        result = {}
        for row in soup.find_all('tr'):
            tds   = row.find_all(['td', 'th'])
            texts = [td.text.strip() for td in tds if td.text.strip()]
            if len(texts) < 5:
                continue
            digits = [t for t in texts[:3] if t.isdigit() and 1 <= int(t) <= 18]
            if not digits:
                continue
            umaban = digits[-1]
            for txt in texts:
                if re.match(r'^\d+\.\d+$', txt):
                    result[umaban] = float(txt)
                    break
        return result
    except Exception as e:
        print(f"  Yahoo競馬取得エラー: {e}")
        return {}

def fetch_odds_if_missing(card_df):
    """
    CSVに単勝オッズがあればそのまま使い、なければYahoo競馬からスクレイピングして補完する。
    """
    if '単勝オッズ' not in card_df.columns:
        card_df['単勝オッズ'] = np.nan

    race_keys = [c for c in ['開催', 'Ｒ'] if c in card_df.columns]
    if not race_keys:
        return card_df

    fetched_any = False
    for gk, idx in card_df.groupby(race_keys).groups.items():
        grp = card_df.loc[idx]
        odds_vals = pd.to_numeric(grp['単勝オッズ'], errors='coerce')
        if odds_vals.notna().sum() >= len(grp) * 0.5:
            continue  # 半数以上埋まっていればスクレイピング不要

        # race_id を構築してYahooから取得
        kaikai   = grp['開催'].iloc[0]
        date_num = grp['日付'].iloc[0] if '日付' in grp.columns else None
        r_num    = gk[1] if len(race_keys) > 1 else grp['Ｒ'].iloc[0]
        if date_num is None:
            continue
        race_id = _make_race_id(kaikai, date_num, r_num)
        if race_id is None:
            continue

        odds_dict = _yahoo_odds(race_id)
        if not odds_dict:
            continue

        fetched_any = True
        for i in idx:
            try:
                umaban = str(int(float(card_df.at[i, '馬番'])))
            except Exception:
                continue
            if umaban in odds_dict:
                card_df.at[i, '単勝オッズ'] = odds_dict[umaban]

    if fetched_any:
        filled = pd.to_numeric(card_df['単勝オッズ'], errors='coerce').notna().sum()
        print(f"Yahoo競馬オッズ取得完了: {filled}頭分")
    return card_df


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
    """出馬表形式CSVを基本.csv / 全て.csv 互換形式に変換して返す"""
    try:
        df = pd.read_csv(card_path, encoding='cp932', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(card_path, encoding='utf-8', low_memory=False)

    # ── 日付変換: 日付S (2026.3.15) → 260315 ──────────
    if '日付S' in df.columns:
        def parse_date_s(s):
            try:
                parts = str(s).replace('/', '.').split('.')
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                return (year - 2000) * 10000 + month * 100 + day
            except Exception:
                return np.nan
        df['日付'] = df['日付S'].apply(parse_date_s)
    else:
        df['日付'] = pd.to_numeric(df['日付'], errors='coerce')

    # ── 開催コード生成: 場 R (中1) → 1中1 ─────────────
    if '場 R' in df.columns and '開催' not in df.columns:
        def make_kaikai(bar):
            s = str(bar).strip()
            m = re.match(r'([^\d]+)', s)
            vc = m.group(1) if m else s
            return f'1{vc}1'
        df['開催'] = df['場 R'].apply(make_kaikai)

    # ── 列名マッピング ─────────────────────────────────
    col_map = {
        '芝ダ':   '芝・ダ',
        '単オッズ': '単勝オッズ',
        '着':     '着順',
        '前着順': '前走着順',
        '前人気': '前走人気',
        '前走騎手': '前騎手',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # 馬名S が無くて馬名だけある場合
    if '馬名S' not in df.columns and '馬名' in df.columns:
        df['馬名S'] = df['馬名']

    # 距離を数値に
    if '距離' in df.columns:
        df['距離'] = pd.to_numeric(df['距離'], errors='coerce')

    return df


def run_feature_engineering(base_dir, extra_df):
    """card_extra.csv に変換済みデータを書き、01_make_features.py を実行"""
    card_path = os.path.join(base_dir, 'data', 'raw', 'card_extra.csv')

    # 既存の card_extra.csv と日付単位でマージ（他日付分を消さない）
    new_dates = set(pd.to_numeric(extra_df['日付'], errors='coerce').dropna())
    if os.path.exists(card_path):
        try:
            existing = pd.read_csv(card_path, encoding='utf-8-sig', low_memory=False)
            existing['_d'] = pd.to_numeric(existing['日付'], errors='coerce')
            keep = existing[~existing['_d'].isin(new_dates)].drop(columns=['_d'])
            if len(keep) > 0:
                common = [c for c in keep.columns if c in extra_df.columns]
                extra_df = pd.concat([keep[common], extra_df], ignore_index=True)
                print(f"  既存card_extra: {len(keep)}行を保持（他日付分）")
        except Exception:
            pass

    extra_df.to_csv(card_path, index=False, encoding='utf-8-sig')
    print(f"card_extra.csv 書込: {len(extra_df):,}行")

    script = os.path.join(base_dir, 'src', '01_make_features.py')
    print("特徴量生成中（数分かかります）...")
    result = subprocess.run(
        [sys.executable, script],
        cwd=base_dir,
        capture_output=True, text=True, encoding='utf-8'
    )
    if result.returncode != 0:
        print("エラー:", result.stderr[-500:] if result.stderr else "不明")
    else:
        lines = [l for l in result.stdout.strip().split('\n') if l]
        for l in lines[-3:]:
            print(l)
        # Parquet キャッシュを更新
        csv_path = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.csv')
        pq_path  = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.parquet')
        try:
            import pyarrow  # noqa
            print("Parquetキャッシュ更新中...")
            pd.read_csv(csv_path, low_memory=False).to_parquet(pq_path, index=False)
            print("Parquetキャッシュ更新完了")
        except ImportError:
            pass


def predict_date(base_dir, target_date_num, card_df=None):
    """指定日付のレース予測を実行（距離モデル + クラスモデルの両モデル）"""
    import time as _time

    # ── サブモデル情報を読み込む ──────────────────────────
    sub_info_path = os.path.join(base_dir, 'models_2025', 'submodel', 'submodel_info.json')
    sub_features  = None
    sub_models    = {}
    sub_rankers   = {}
    if os.path.exists(sub_info_path):
        with open(sub_info_path, 'r', encoding='utf-8') as f:
            sub_info = json.load(f)
        sub_features = sub_info['features']
        sub_models   = sub_info['models']
        rinfo_sub_path = os.path.join(base_dir, 'models_2025', 'submodel_ranker', 'class_ranker_info.json')
        if os.path.exists(rinfo_sub_path):
            with open(rinfo_sub_path, 'r', encoding='utf-8') as f:
                sub_rankers = json.load(f).get('rankers', {})

    # ── 現行モデル情報を読み込む ──────────────────────────
    cur_info_path = os.path.join(base_dir, 'models_2025', 'model_info.json')
    cur_features  = None
    cur_models    = {}
    cur_rankers   = {}
    if os.path.exists(cur_info_path):
        with open(cur_info_path, 'r', encoding='utf-8') as f:
            cur_info = json.load(f)
        cur_features = cur_info['features']
        cur_models   = cur_info['models']
        rinfo_path   = os.path.join(base_dir, 'models_2025', 'ranker', 'ranker_info.json')
        if os.path.exists(rinfo_path):
            with open(rinfo_path, 'r', encoding='utf-8') as f:
                cur_rankers = json.load(f).get('rankers', {})

    if not cur_models and not sub_models:
        print("エラー: モデルが見つかりません。")
        return None

    # ── 特徴量データを読み込む（Parquet優先 / なければCSV）─────────────
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

    # 馬体重パッチ: card_dfに馬体重があれば上書き
    if card_df is not None:
        _weight_cols = [c for c in ['馬体重', '馬体重増減'] if c in card_df.columns]
        if _weight_cols:
            _w = card_df[['馬名S'] + _weight_cols].copy()
            for c in _weight_cols:
                _w[c] = pd.to_numeric(_w[c], errors='coerce')
            _w = _w.dropna(subset=_weight_cols, how='all')
            if not _w.empty:
                for c in _weight_cols:
                    if c in day.columns:
                        day = day.drop(columns=[c])
                day = day.merge(_w.drop_duplicates('馬名S'), on='馬名S', how='left')
                print(f"馬体重パッチ適用: {day[_weight_cols[0]].notna().sum()}頭")

    # ── モデルキー計算 ──
    day['会場']       = day['開催'].apply(extract_venue)
    day['cur_key']    = day['会場'] + '_' + day['距離'].astype(str)
    day['_surface']   = day['芝・ダ'].astype(str).str.strip() if '芝・ダ' in day.columns else 'None'
    day['_dist_band'] = day['距離'].apply(get_distance_band)
    mask_da_ml = (day['_surface'] == 'ダ') & (day['_dist_band'].isin(['中距離', '長距離']))
    day.loc[mask_da_ml, '_dist_band'] = '中長距離'
    day['_cls_group'] = day['クラス_rank'].apply(get_class_group) if 'クラス_rank' in day.columns else None
    day['sub_key']    = (day['_surface'].astype(str) + '_' +
                         day['_dist_band'].astype(str) + '_' +
                         day['_cls_group'].astype(str))

    if '単勝オッズ' in day.columns:
        day['単勝オッズ'] = pd.to_numeric(day['単勝オッズ'], errors='coerce')

    all_feats = list(set((sub_features or []) + (cur_features or [])))
    for col in all_feats:
        if col in day.columns:
            day[col] = pd.to_numeric(day[col], errors='coerce')

    print(f"\n日付: {target_date_num}  /  {len(day)}頭\n")

    # ── モデルを一括プリロード ──
    t0 = _time.time()
    model_cache  = {}
    ranker_cache = {}
    cur_model_cache  = {}
    cur_ranker_cache = {}
    for sk in day['sub_key'].dropna().unique():
        if sk in sub_models:
            win_path = os.path.join(base_dir, 'models_2025', 'submodel', sub_models[sk]['win'])
            if os.path.exists(win_path):
                with open(win_path, 'rb') as f:
                    m = pickle.load(f)
                model_cache[sk] = (m, m.booster_.feature_name())
        if sk in sub_rankers:
            rpath = os.path.join(base_dir, 'models_2025', 'submodel_ranker', sub_rankers[sk])
            if os.path.exists(rpath):
                with open(rpath, 'rb') as f:
                    ranker_cache[sk] = pickle.load(f)
    if cur_features:
        for ck in day['cur_key'].dropna().unique():
            if ck in cur_models:
                win_path = os.path.join(base_dir, 'models_2025', cur_models[ck]['win'])
                if os.path.exists(win_path):
                    with open(win_path, 'rb') as f:
                        m = pickle.load(f)
                    cur_model_cache[ck] = (m, m.booster_.feature_name())
            if ck in cur_rankers:
                rpath = os.path.join(base_dir, 'models_2025', 'ranker', cur_rankers[ck])
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

        # ── サブモデル予測 ──
        sub['sub_prob_win']       = np.nan
        sub['sub_偏差値の差']      = np.nan
        sub['sub_コース偏差値']    = np.nan
        sub['sub_レース内偏差値']  = np.nan
        sub['sub_ランカー順位']    = np.nan
        if sub_key in model_cache:
            m, wf = model_cache[sub_key]
            for c in wf:
                if c not in sub.columns: sub[c] = np.nan
            sub['sub_prob_win'] = m.predict_proba(sub[wf])[:, 1]
            st  = sub_models[sub_key].get('stats', {})
            w_m = st.get('win_mean', sub['sub_prob_win'].mean())
            w_s = st.get('win_std',  sub['sub_prob_win'].std())
            sub['sub_コース偏差値']  = 50 + 10 * (sub['sub_prob_win'] - w_m) / (w_s if w_s > 0 else 1)
            r_m = sub['sub_prob_win'].mean()
            r_s = sub['sub_prob_win'].std()
            sub['sub_レース内偏差値'] = 50 + 10 * (sub['sub_prob_win'] - r_m) / (r_s if r_s > 0 else 1)
            sub['sub_偏差値の差']    = sub['sub_レース内偏差値'] - sub['sub_コース偏差値']
            if sub_key in ranker_cache:
                scores = ranker_cache[sub_key].predict(sub[wf])
                sub['sub_ランカー順位'] = pd.Series(scores, index=sub.index).rank(
                    ascending=False, method='min').astype(int)

        # ── 現行モデル予測 ──
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
            st  = cur_models[cur_key].get('stats', {})
            w_m = st.get('win_mean', sub['cur_prob_win'].mean())
            w_s = st.get('win_std',  sub['cur_prob_win'].std())
            sub['cur_コース偏差値']   = 50 + 10 * (sub['cur_prob_win'] - w_m) / (w_s if w_s > 0 else 1)
            r_m = sub['cur_prob_win'].mean()
            r_s = sub['cur_prob_win'].std()
            sub['cur_レース内偏差値'] = 50 + 10 * (sub['cur_prob_win'] - r_m) / (r_s if r_s > 0 else 1)
            sub['cur_偏差値の差']     = sub['cur_レース内偏差値'] - sub['cur_コース偏差値']
            if cur_key in cur_ranker_cache:
                scores = cur_ranker_cache[cur_key].predict(sub[cur_features])
                sub['cur_ランカー順位'] = pd.Series(scores, index=sub.index).rank(
                    ascending=False, method='min').astype(int)

        # ── combo_gap 計算（1位と2位のスコア差）──
        for score_col, gap_col in [('cur_コース偏差値', 'cur_gap'), ('sub_コース偏差値', 'sub_gap')]:
            sc2 = sub[score_col].dropna().sort_values(ascending=False).values
            sub[gap_col] = (sc2[0] - sc2[1]) if len(sc2) >= 2 else np.nan
        sub['combo_gap'] = sub['cur_gap'].fillna(0) + sub['sub_gap'].fillna(0)

        # ターミナル表示
        venue    = extract_venue(kaikai)
        shiba_da = sub['芝・ダ'].iloc[0] if '芝・ダ' in sub.columns else ''
        kyori    = sub['距離'].iloc[0] if '距離' in sub.columns else ''
        print(f"{'='*70}")
        print(f"  {venue} {r_num}R　{race_name}  [{shiba_da}{kyori}m / {sub_key}]")
        print(f"{'='*70}")
        disp = sub[['馬名S']].copy()
        disp['現行_差']      = sub['cur_偏差値の差'].map(lambda x: f'{x:+.1f}' if pd.notna(x) else '-')
        disp['現行_ランカー'] = sub['cur_ランカー順位'].map(lambda x: f'{int(x)}位' if pd.notna(x) else '-')
        disp['sub_差']       = sub['sub_偏差値の差'].map(lambda x: f'{x:+.1f}' if pd.notna(x) else 'モデルなし')
        disp['sub_ランカー']  = sub['sub_ランカー順位'].map(lambda x: f'{int(x)}位' if pd.notna(x) else '-')
        if '単勝オッズ' in sub.columns:
            disp['オッズ'] = sub['単勝オッズ'].map(lambda x: f'{x:.1f}' if pd.notna(x) else '-')
        print(disp.to_string(index=False))
        print()

        sub['レース名'] = race_name
        all_rows.append(sub)

    if not all_rows:
        print("対応モデルが見つかりませんでした。")
        return None

    return pd.concat(all_rows, ignore_index=True)


def generate_html(result, card_df, target_date_num, out_path):
    """両モデル予測結果をHTMLレポートとして出力（横向き印刷対応）"""

    # ROI統計JSON読み込み
    _roi_stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'models', 'roi_stats.json')
    _roi_stats = {}
    if os.path.exists(_roi_stats_path):
        with open(_roi_stats_path, encoding='utf-8') as _f:
            _roi_stats = json.load(_f)
    _roi_cur = _roi_stats.get('distance_model', {})
    _roi_sub = _roi_stats.get('class_model', {})

    def roi_line(stats_dict, key):
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

    # ── カードデータをマージ ──
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
    cur_diff  = pd.to_numeric(df['cur_偏差値の差'], errors='coerce')
    sub_diff  = pd.to_numeric(df['sub_偏差値の差'], errors='coerce')
    combo_gap = pd.to_numeric(df.get('combo_gap'), errors='coerce') if 'combo_gap' in df.columns else pd.Series(np.nan, index=df.index)
    cur_r     = pd.to_numeric(df['cur_ランカー順位'], errors='coerce')
    sub_r     = pd.to_numeric(df['sub_ランカー順位'], errors='coerce')
    cur_r1    = (cur_r == 1)
    sub_r1    = (sub_r == 1)
    both_r1   = cur_r1 & sub_r1
    star      = (cur_r <= 3) & (sub_r <= 3) & ~both_r1  # 片方が2か3
    # オッズ（あれば使う、なければフィルターなし）
    _odds = pd.to_numeric(df.get('dc_単勝オッズ'), errors='coerce')
    if _odds.isna().all() and '単勝オッズ' in df.columns:
        _odds = pd.to_numeric(df['単勝オッズ'], errors='coerce')
    odds_ok3  = _odds.isna() | (_odds >= 3)   # NaN = フィルターなし扱い
    odds_ok5  = _odds.isna() | (_odds >= 5)
    # 新印マスク
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
    df['_印'] = [mark_of(i) for i in range(len(df))]
    # 旧フラグ（行ハイライト用に残す）
    cur_nota  = cur_diff >= 10
    sub_nota  = sub_diff >= 10
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
      .sec-hot  h2 { background:#b7410e; }
      .sec-hot     { background:#fef5f0; }
      /* 印バッジ */
      .mark { display:inline-block; font-size:13px; font-weight:bold; min-width:28px; text-align:center; }
      .mark-gekiatu { color:#b7410e; }
      .mark-maru    { color:#1a6e3c; }
      .mark-maru2   { color:#27ae60; }
      .mark-hoshi   { color:#1f618d; }
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
        race_keys_s = [c for c in ['開催','Ｒ','レース名'] if c in rows.columns]
        rows = rows.sort_values(race_keys_s)
        html = ['<table><thead><tr>']
        html.append('<th>レース</th><th>馬番</th><th>馬名</th><th>騎手</th>')
        if show_cur: html.append('<th>会場×距離<br>ランカー</th><th>会場×距離<br>偏差値差</th>')
        if show_sub: html.append('<th>クラス<br>ランカー</th><th>クラス<br>偏差値差</th>')
        html.append('<th>単勝%</th><th>オッズ</th></tr></thead><tbody>')
        for _, r in rows.iterrows():
            venue    = extract_venue(r.get('開催',''))
            r_n      = int(r['Ｒ']) if 'Ｒ' in rows.columns and pd.notna(r.get('Ｒ')) else ''
            race_lbl = f"{venue}{r_n}R {r.get('レース名','')}"
            try:
                banum = int(float(r['dc_馬番'])) if pd.notna(r.get('dc_馬番')) else '-'
            except (ValueError, TypeError):
                banum = r.get('dc_馬番', '-')
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

    GREEN = {'◎': '#1a6e3c', '〇': '#27ae60', '▲': '#58d68d', 'th': '#1e8449', 'bg': '#eafaf1'}
    BLUE  = {'◎': '#1a3f6e', '〇': '#2471a3', '▲': '#5dade2', 'th': '#1f618d', 'bg': '#eaf4fb'}

    def tier_table(label, mask, diff_col, rank_col, palette):
        horses = df[mask].copy()
        if horses.empty:
            return f'<p style="padding:4px 8px;color:#999;font-size:10px">{label}: 該当なし</p>'
        rk = [c for c in ['開催','Ｒ','レース名'] if c in horses.columns]
        horses = horses.sort_values(rk)
        tier_color = palette.get(label[0], palette['▲'])
        rows_html = []
        for _, r in horses.iterrows():
            venue  = extract_venue(r.get('開催',''))
            r_n    = int(r['Ｒ']) if 'Ｒ' in horses.columns and pd.notna(r.get('Ｒ')) else ''
            r_name = r.get('レース名','')
            try:
                banum = int(float(r['dc_馬番'])) if pd.notna(r.get('dc_馬番')) else '-'
            except (ValueError, TypeError):
                banum = r.get('dc_馬番', '-')
            jockey = r.get('dc_騎手', r.get('騎手',''))
            diff_v = r.get(diff_col)
            rank_v = r.get(rank_col)
            odds_v = r.get('dc_単勝オッズ') if pd.notna(r.get('dc_単勝オッズ')) else r.get('単勝オッズ')
            rows_html.append(
                f'<tr>'
                f'<td style="text-align:left;padding:2px 6px">{venue} {r_n}R　{r_name}</td>'
                f'<td>{banum}番</td>'
                f'<td style="text-align:left;font-weight:bold;font-size:12px">{r.get("馬名S","")}</td>'
                f'<td>{jockey}</td>'
                f'<td style="font-weight:bold;color:{palette["th"]}">{rank_str(rank_v)}</td>'
                f'{diff_cell(diff_v)}'
                f'<td>{fmt(odds_v, ".1f")}</td>'
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

    # ── 新印カード生成 ──
    def make_mark_cards(mask):
        rows = df[mask].copy()
        if rows.empty:
            return '<p style="padding:8px;background:white;color:#999;font-size:10px">該当なし</p>'
        rk = [c for c in ['開催','Ｒ','レース名'] if c in rows.columns]
        rows = rows.sort_values(rk)
        items = []
        for _, r in rows.iterrows():
            venue  = extract_venue(r.get('開催',''))
            r_n    = int(r['Ｒ']) if 'Ｒ' in rows.columns and pd.notna(r.get('Ｒ')) else ''
            r_name = r.get('レース名','')
            try:
                banum = int(float(r['dc_馬番'])) if pd.notna(r.get('dc_馬番')) else '-'
            except (ValueError, TypeError):
                banum = r.get('dc_馬番', '-')
            horse   = r.get('馬名S','')
            jockey  = r.get('dc_騎手', r.get('騎手',''))
            odds_v  = r.get('dc_単勝オッズ') if pd.notna(r.get('dc_単勝オッズ')) else r.get('単勝オッズ')
            mark    = r.get('_印','')
            cgap    = fmt(r.get('combo_gap'), '.1f')
            sd_v    = fmt(r.get('sub_偏差値の差'), '+.1f')
            cr_s    = rank_str(r.get('cur_ランカー順位'))
            sr_s    = rank_str(r.get('sub_ランカー順位'))
            mark_cls = {'激熱':'mark-gekiatu','◎':'mark-maru','〇':'mark-maru2','☆':'mark-hoshi'}.get(mark,'')
            items.append(f"""
            <div class="both-card">
              <div class="both-race">{venue} {r_n}R　{r_name}</div>
              <div class="both-horse">
                <span class="both-banum">{banum}番</span>
                <span class="both-name">{horse}</span>
                <span class="both-jockey">{jockey}</span>
                <span class="both-odds">オッズ {fmt(odds_v,'.1f')}</span>
              </div>
              <div class="both-eval">
                <span class="mark {mark_cls}" style="font-size:18px">{mark}</span>
                <span style="font-size:10px;color:#555">　距離{cr_s} / クラス{sr_s} / gap{cgap} / sd{sd_v}</span>
              </div>
            </div>""")
        return '<div class="both-list">' + ''.join(items) + '</div>'

    # 新印テーブル（一覧用）
    def mark_table(mask, header_color='#2c3e50'):
        rows = df[mask].copy()
        if rows.empty:
            return '<p style="padding:4px 8px;color:#999;font-size:10px">該当なし</p>'
        rk = [c for c in ['開催','Ｒ','レース名'] if c in rows.columns]
        rows = rows.sort_values(rk)
        rows_html = []
        for _, r in rows.iterrows():
            venue  = extract_venue(r.get('開催',''))
            r_n    = int(r['Ｒ']) if 'Ｒ' in rows.columns and pd.notna(r.get('Ｒ')) else ''
            try:
                banum = int(float(r['dc_馬番'])) if pd.notna(r.get('dc_馬番')) else '-'
            except (ValueError, TypeError):
                banum = r.get('dc_馬番', '-')
            jockey  = r.get('dc_騎手', r.get('騎手',''))
            odds_v  = r.get('dc_単勝オッズ') if pd.notna(r.get('dc_単勝オッズ')) else r.get('単勝オッズ')
            mark    = r.get('_印','')
            mark_cls = {'激熱':'mark-gekiatu','◎':'mark-maru','〇':'mark-maru2','☆':'mark-hoshi'}.get(mark,'')
            rows_html.append(
                f'<tr>'
                f'<td style="text-align:left">{venue} {r_n}R　{r.get("レース名","")}</td>'
                f'<td>{banum}番</td>'
                f'<td style="text-align:left;font-weight:bold;font-size:12px">{r.get("馬名S","")}</td>'
                f'<td>{jockey}</td>'
                f'<td class="mark {mark_cls}">{mark}</td>'
                f'<td>{rank_str(r.get("cur_ランカー順位"))}</td>'
                f'<td>{rank_str(r.get("sub_ランカー順位"))}</td>'
                f'<td>{fmt(r.get("combo_gap"),".1f")}</td>'
                f'<td>{fmt(r.get("sub_偏差値の差"),"+.1f")}</td>'
                f'<td>{fmt(odds_v,".1f")}</td>'
                f'</tr>'
            )
        return (f'<table><thead><tr style="background:{header_color}">'
                f'<th style="text-align:left">レース</th><th>馬番</th><th style="text-align:left">馬名</th>'
                f'<th>騎手</th><th>印</th><th>距離<br>ランカー</th><th>クラス<br>ランカー</th>'
                f'<th>combo<br>gap</th><th>sub<br>diff</th><th>オッズ</th>'
                f'</tr></thead><tbody>{"".join(rows_html)}</tbody></table>')

    # ── Page1: 激熱専用 ──
    page1 = f"""<div class="page sec-hot">
      <h1>競馬AI予測レポート　{date_str}</h1>
      <h2>【激熱】両Rnk=1 &amp; gap≥15 &amp; sd≥10</h2>
      <p style="font-size:9px;color:#666;margin:2px 0 6px">※ 2026年out-of-sample ROI +227%（週2本）  賭け金推奨: 2,000円</p>
      {make_mark_cards(mask_gekiatu)}
    </div>"""

    # ── Page2: ◎/〇/☆ 一覧 ──
    page2 = f"""<div class="page sec-both">
      <h2>【印一覧】◎ 両Rnk=1 gap≥10 / 〇 両Rnk=1 gap&lt;10 / ☆ 両Rnk≤3</h2>
      <p style="font-size:9px;color:#666;margin:2px 0 4px">
        ◎ ROI +60%（週8本・推奨1,000円）　〇 ROI +30%（週3本・推奨500円）　☆ ROI +46%（週47本・推奨単300+複200円）
      </p>
      {mark_table(mask_maru | mask_maru2 | mask_hoshi)}
    </div>"""

    # ── Page3 & 4: 両モデル一致 詳細（参考）──
    def make_both_cards(mask):
        rows = df[mask].copy()
        if rows.empty:
            return '<p style="padding:8px;background:white;color:#999;font-size:10px">該当なし</p>'
        rk = [c for c in ['開催','Ｒ','レース名'] if c in rows.columns]
        rows = rows.sort_values(rk)
        items = []
        for _, r in rows.iterrows():
            venue  = extract_venue(r.get('開催',''))
            r_n    = int(r['Ｒ']) if 'Ｒ' in rows.columns and pd.notna(r.get('Ｒ')) else ''
            r_name = r.get('レース名','')
            try:
                banum = int(float(r['dc_馬番'])) if pd.notna(r.get('dc_馬番')) else '-'
            except (ValueError, TypeError):
                banum = r.get('dc_馬番', '-')
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
              <div class="both-race">{venue} {r_n}R　{r_name}</div>
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

    # ── レース別詳細（発走時刻順）──
    def horse_tier(row):
        """新印体系: _印列から取得"""
        return row.get('_印', '')

    race_keys_html = [c for c in ['開催','Ｒ','レース名'] if c in df.columns]

    # 発走時刻でソート用のマップを card_df から作成
    race_time_map = {}
    if '発走時刻' in card_df.columns and 'Ｒ' in card_df.columns:
        for _, row in card_df[['開催', 'Ｒ', '発走時刻']].drop_duplicates().iterrows():
            try:
                r_int = int(float(row['Ｒ']))
            except (ValueError, TypeError):
                continue
            t = str(row.get('発走時刻', '') or '').replace(':', '').replace('：', '')
            race_time_map[(str(row['開催']), r_int)] = t

    def race_sort_key(gk):
        kaikai = str(gk[0])
        try:
            r_int = int(float(gk[1])) if len(gk) > 1 else 0
        except (ValueError, TypeError):
            r_int = 0
        t = race_time_map.get((kaikai, r_int), '9999')
        return (t, r_int)

    sorted_race_groups = sorted(
        df.groupby(race_keys_html, sort=False).groups.items(),
        key=lambda x: race_sort_key(x[0])
    )

    race_pages = []
    for gk, idx in sorted_race_groups:
        grp = df.loc[idx].copy()
        grp = grp.sort_values('dc_馬番', na_position='last')
        kaikai    = gk[0]
        race_name = gk[-1] if 'レース名' in df.columns else ''
        r_n       = int(gk[1]) if 'Ｒ' in df.columns and pd.notna(gk[1]) else ''
        venue     = extract_venue(kaikai)
        shiba_da  = grp['芝・ダ'].iloc[0] if '芝・ダ' in grp.columns else ''
        _kyori_raw = grp['距離'].iloc[0] if '距離' in grp.columns else ''
        try: kyori = int(re.search(r'\d+', str(_kyori_raw)).group())
        except: kyori = _kyori_raw
        sub_key   = grp['sub_key'].iloc[0] if 'sub_key' in grp.columns else ''
        cur_key_r = grp['cur_key'].iloc[0] if 'cur_key' in grp.columns else ''
        sk_parts  = str(sub_key).split('_')
        cls_label = '・'.join(sk_parts[1:]) if len(sk_parts) >= 3 else sub_key
        surface_label = {'芝': '芝', 'ダ': 'ダート'}.get(str(shiba_da).strip(), str(shiba_da))
        cur_roi_txt = roi_line(_roi_cur, cur_key_r)
        sub_roi_txt = roi_line(_roi_sub, sub_key)

        rows_html = []
        for _, r in grp.iterrows():
            mark     = horse_tier(r)
            mark_cls = {'激熱':'mark-gekiatu','◎':'mark-maru','〇':'mark-maru2','☆':'mark-hoshi'}.get(mark,'')
            if mark == '激熱':  row_bg = 'background:#fde8e8'
            elif mark == '◎':   row_bg = 'background:#fef9ec'
            elif mark == '〇':   row_bg = 'background:#eafaf1'
            elif mark == '☆':   row_bg = 'background:#eaf4fb'
            else:               row_bg = ''

            bw   = r.get('dc_馬体重', '-')
            bw_d = r.get('dc_増減', '')
            try: taiju = f"{bw}({'+' if float(bw_d)>=0 else ''}{int(float(bw_d))})" if pd.notna(bw_d) and bw_d != '' else str(bw)
            except: taiju = str(bw)

            odds_v = r.get('dc_単勝オッズ') if pd.notna(r.get('dc_単勝オッズ')) else r.get('単勝オッズ')

            # 詳細行（距離+クラス 1行）
            detail = (
                f'<span style="color:#1e8449">距離</span> '
                f'{rank_str(r.get("cur_ランカー順位"))} '
                f'絶対{fmt(r.get("cur_コース偏差値"),".1f")} '
                f'内{fmt(r.get("cur_レース内偏差値"),".1f")} '
                f'差{fmt(r.get("cur_偏差値の差"),"+.1f")} '
                f'勝{fmt(r.get("cur_prob_win"),".1%")}'
                f'　／　'
                f'<span style="color:#2471a3">クラス</span> '
                f'{rank_str(r.get("sub_ランカー順位"))} '
                f'絶対{fmt(r.get("sub_コース偏差値"),".1f")} '
                f'内{fmt(r.get("sub_レース内偏差値"),".1f")} '
                f'差{fmt(r.get("sub_偏差値の差"),"+.1f")} '
                f'勝{fmt(r.get("sub_prob_win"),".1%")}'
            )

            # メイン行
            mark_disp = f'<span style="font-size:13px;font-weight:bold;letter-spacing:1px">{mark}</span>' if mark else ''
            rows_html.append(
                f'<tr style="{row_bg}">'
                f'<td>{r.get("dc_枠番","-")}</td>'
                f'<td>{r.get("dc_馬番","-")}</td>'
                f'<td class="{mark_cls}" style="font-size:13px;font-weight:bold;min-width:28px">{mark}</td>'
                f'<td style="text-align:left;font-weight:bold;white-space:nowrap">{r.get("馬名S","")}'
                f' <span style="font-weight:normal;color:#888;font-size:8px">{seir(r)}</span></td>'
                f'<td style="white-space:nowrap;font-size:9px">{r.get("dc_騎手", r.get("騎手",""))}'
                f' <span style="color:#888;font-size:8px">{r.get("dc_斤量","")}kg</span></td>'
                f'<td style="white-space:nowrap">{taiju}</td>'
                f'<td>{fmt(odds_v, ".1f")}</td>'
                f'<td>{rank_str(r.get("cur_ランカー順位"))}</td>'
                f'<td>{rank_str(r.get("sub_ランカー順位"))}</td>'
                f'<td>{fmt(r.get("combo_gap"),".1f")}</td>'
                f'<td>{fmt(r.get("sub_偏差値の差"),"+.1f")}</td>'
                f'</tr>'
                # 詳細行（7px・全列結合）
                f'<tr style="{row_bg}">'
                f'<td colspan="11" style="font-size:7px;color:#666;text-align:left;padding:1px 6px;border-top:none">{detail}</td>'
                f'</tr>'
            )

        cur_roi_html = f'<div class="model-stats" style="color:#1e8449">【距離】{cur_roi_txt}</div>' if cur_roi_txt else ''
        sub_roi_html = f'<div class="model-stats" style="color:#1f618d">【クラス】{sub_roi_txt}</div>' if sub_roi_txt else ''
        race_pages.append(f"""<div class="page">
          <h2>{venue} {r_n}R　{race_name}</h2>
          <div class="race-info">{surface_label} {kyori}m　／　{cls_label}</div>
          {cur_roi_html}{sub_roi_html}
          <table style="font-size:10px">
            <thead><tr>
              <th>枠</th><th>馬番</th><th>印</th><th style="text-align:left">馬名/性齢</th><th>騎手/斤量</th><th>馬体重</th><th>オッズ</th>
              <th style="color:#58d68d">距離<br>ランカー</th>
              <th style="color:#5dade2">クラス<br>ランカー</th>
              <th>combo<br>gap</th><th>sub<br>diff</th>
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
    parser = argparse.ArgumentParser(description='出馬表形式CSVから予測を実行')
    parser.add_argument('card_file', help='出馬表形式CSVのパス（例: data/raw/出馬表形式3月15日.csv）')
    parser.add_argument('--no-rebuild', action='store_true', help='特徴量再生成をスキップ')
    parser.add_argument('--no-html', action='store_true', help='HTML出力をスキップ')
    parser.add_argument('--html-dir', default=r'G:\マイドライブ\競馬AI', help='HTML保存先フォルダ')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if 'src' in os.path.abspath(__file__) else '.'

    print(f"=== 競馬AI 出馬表予測 ===")
    print(f"入力: {args.card_file}")

    # 1. 出馬表を変換
    card_df = convert_card_to_base_format(args.card_file)

    # 1b. オッズ補完（CSVにあればそのまま / なければYahoo競馬からスクレイピング）
    card_df = fetch_odds_if_missing(card_df)

    dates = card_df['日付'].dropna().unique()
    print(f"日付: {[int(d) for d in sorted(dates)]}")
    print(f"レース: {card_df['レース名'].nunique() if 'レース名' in card_df.columns else '?'}レース / {len(card_df)}頭")

    target_date = int(sorted(dates)[-1])

    # 2. 特徴量再生成
    if not args.no_rebuild:
        run_feature_engineering(base_dir, card_df)
    else:
        print("特徴量再生成スキップ")

    # 3. 予測（両モデル）
    result = predict_date(base_dir, target_date, card_df)

    # 4. HTML生成
    if result is not None and not args.no_html:
        html_dir = args.html_dir
        if not os.path.exists(os.path.splitdrive(html_dir)[0] + '\\'):
            html_dir = os.path.join(base_dir, 'output')
        out_path = os.path.join(html_dir, f'card_{target_date}.html')
        generate_html(result, card_df, target_date, out_path)


if __name__ == '__main__':
    main()

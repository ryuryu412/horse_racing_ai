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

MIN_SAMPLES = 30   # 閾値判断に必要な最低馬数
THRESHOLDS  = [-5, 0, 5, 10, 15, 20, 25, 30]

def extract_venue(kaikai):
    m = re.search(r'\d+([^\d]+)', str(kaikai))
    return m.group(1) if m else str(kaikai)

def deviation_score(values, mean, std):
    if std == 0:
        return pd.Series([50.0] * len(values), index=values.index)
    return 50 + 10 * (values - mean) / std

def find_best_threshold(df, target_col):
    """
    偏差値の差の閾値ごとに的中率を計算し、
    - MIN_SAMPLES 以上のサンプルがある
    - 的中率が最も高い
    閾値を返す。該当なしは None。
    """
    best_thr = None
    best_rate = -1.0
    for t in THRESHOLDS:
        subset = df[df['偏差値の差'] >= t]
        if len(subset) < MIN_SAMPLES:
            continue
        rate = subset[target_col].mean()
        if rate > best_rate:
            best_rate = rate
            best_thr = t
    return best_thr, best_rate

def calc_deviation_diff(test, m_win, m_plc, stats):
    """テストデータに偏差値の差を付与して返す"""
    wf = m_win.booster_.feature_name()
    pf = m_plc.booster_.feature_name()
    test = test.copy()
    test['prob_win']   = m_win.predict_proba(test[wf])[:, 1]
    test['prob_place'] = m_plc.predict_proba(test[pf])[:, 1]

    w_mean = stats.get('win_mean', test['prob_win'].mean())
    w_std  = stats.get('win_std',  test['prob_win'].std())
    test['コース偏差値'] = deviation_score(test['prob_win'], w_mean, w_std).values

    test['レース内偏差値'] = np.nan
    for (_, _), race_df in test.groupby(['日付_num', 'レース名']):
        if len(race_df) < 3:
            continue
        r_mean = race_df['prob_win'].mean()
        r_std  = race_df['prob_win'].std()
        test.loc[race_df.index, 'レース内偏差値'] = deviation_score(
            race_df['prob_win'], r_mean, r_std if r_std > 0 else 1
        ).values

    test['偏差値の差'] = test['レース内偏差値'] - test['コース偏差値']
    return test

def main():
    print("--- 会場×距離別 閾値分析を開始します ---")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if 'src' in os.path.abspath(__file__) else '.'
    input_file = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.csv')
    model_dir  = os.path.join(base_dir, 'models')
    info_path  = os.path.join(model_dir, 'model_info.json')

    with open(info_path, 'r', encoding='utf-8') as f:
        model_info = json.load(f)
    features       = model_info['features']
    trained_models = model_info['models']

    print("加工済みデータを読み込んでいます...")
    df = pd.read_csv(input_file, low_memory=False)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num'] = pd.to_numeric(df['着順'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num'])
    df['target_win']   = (df['着順_num'] == 1).astype(int)
    df['target_place'] = (df['着順_num'] <= 3).astype(int)
    df['会場']      = df['開催'].apply(extract_venue)
    df['model_key'] = df['会場'] + '_' + df['距離'].astype(str)
    for col in features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"データ: {len(df):,}件 / モデル: {len(trained_models)}グループ\n")

    all_rows   = []
    per_model  = {}   # key -> {best_threshold_win, best_threshold_place, n_test, ...}

    # ── モデルごとに閾値を探す ──────────────────────────────
    for key, files in trained_models.items():
        win_path = os.path.join(model_dir, files['win'])
        plc_path = os.path.join(model_dir, files['place'])
        if not os.path.exists(win_path) or not os.path.exists(plc_path):
            continue

        with open(win_path, 'rb') as f: m_win = pickle.load(f)
        with open(plc_path, 'rb') as f: m_plc = pickle.load(f)

        g = df[df['model_key'] == key].sort_values('日付_num').reset_index(drop=True)
        n = len(g)
        split = int(n * 0.8)
        test  = g.iloc[split:].copy()
        if len(test) < MIN_SAMPLES:
            per_model[key] = {'n_test': len(test), 'fallback': True}
            continue

        test = calc_deviation_diff(test, m_win, m_plc, files.get('stats', {}))
        test = test.dropna(subset=['偏差値の差'])

        best_thr_win,   best_rate_win   = find_best_threshold(test, 'target_win')
        best_thr_place, best_rate_place = find_best_threshold(test, 'target_place')

        per_model[key] = {
            'n_test':              len(test),
            'fallback':            False,
            'best_threshold_win':  best_thr_win,
            'best_win_rate':       round(best_rate_win,   4) if best_thr_win   is not None else None,
            'best_threshold_place': best_thr_place,
            'best_place_rate':     round(best_rate_place, 4) if best_thr_place is not None else None,
        }

        test['model_key'] = key
        all_rows.append(test[['日付_num', 'レース名', '馬名S', '着順_num',
                               'target_win', 'target_place',
                               'コース偏差値', 'レース内偏差値', '偏差値の差', 'model_key']])

    # ── 全体フォールバック閾値（サンプル不足モデル用）──────────
    all_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    fallback_win   = None
    fallback_place = None
    avg_horses     = 0.0

    if not all_df.empty:
        all_df = all_df.dropna(subset=['偏差値の差'])
        avg_horses = all_df.groupby(['日付_num', 'レース名']).size().mean()
        fallback_win,   _ = find_best_threshold(all_df, 'target_win')
        fallback_place, _ = find_best_threshold(all_df, 'target_place')

    # ── 結果表示 ────────────────────────────────────────────
    print(f"{'='*72}")
    print(f" 会場×距離別 最適閾値一覧（偏差値の差）")
    print(f"{'='*72}")
    print(f"{'モデル':<18} {'テスト数':>7}  {'単勝閾値':>8}  {'単勝的中率':>9}  {'複勝閾値':>8}  {'複勝的中率':>9}")
    print(f"{'-'*72}")

    fallback_count = 0
    for key, info in sorted(per_model.items()):
        if info.get('fallback'):
            fallback_count += 1
            print(f"  {key:<16} {info['n_test']:>7}頭  {'(不足)':>8}  {'---':>9}  {'(不足)':>8}  {'---':>9}")
            continue

        thr_w = info['best_threshold_win']
        thr_p = info['best_threshold_place']
        rate_w = f"{info['best_win_rate']:.1%}"   if thr_w is not None else '---'
        rate_p = f"{info['best_place_rate']:.1%}" if thr_p is not None else '---'
        thr_w_str = f"+{thr_w}" if thr_w is not None and thr_w >= 0 else str(thr_w) if thr_w is not None else '---'
        thr_p_str = f"+{thr_p}" if thr_p is not None and thr_p >= 0 else str(thr_p) if thr_p is not None else '---'
        print(f"  {key:<16} {info['n_test']:>7}頭  {thr_w_str:>8}  {rate_w:>9}  {thr_p_str:>8}  {rate_p:>9}")

    print(f"{'-'*72}")
    thr_w_str = f"+{fallback_win}" if fallback_win is not None and fallback_win >= 0 else str(fallback_win)
    thr_p_str = f"+{fallback_place}" if fallback_place is not None and fallback_place >= 0 else str(fallback_place)
    print(f"  【全体フォールバック】単勝閾値 {thr_w_str} / 複勝閾値 {thr_p_str}（サンプル不足{fallback_count}モデルに適用）")
    print(f"  ランダム参考: 単勝 {1/avg_horses:.1%} / 複勝 {3/avg_horses:.1%}（平均{avg_horses:.0f}頭立て）\n" if avg_horses else "")

    # ── model_info.json に書き戻し ───────────────────────────
    model_info['fallback_threshold_win']   = fallback_win
    model_info['fallback_threshold_place'] = fallback_place

    for key, info in per_model.items():
        if key not in model_info['models']:
            continue
        if info.get('fallback'):
            model_info['models'][key]['best_threshold_win']   = fallback_win
            model_info['models'][key]['best_threshold_place'] = fallback_place
            model_info['models'][key]['threshold_source']     = 'fallback'
        else:
            model_info['models'][key]['best_threshold_win']   = info['best_threshold_win']
            model_info['models'][key]['best_threshold_place'] = info['best_threshold_place']
            model_info['models'][key]['threshold_source']     = 'model'

    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    print(f"閾値を model_info.json に保存しました。")

    # ── 分布サマリー ─────────────────────────────────────────
    if not all_df.empty:
        print(f"\n{'='*65}")
        print(f" 全体 偏差値の差 × 的中率（参考）")
        print(f"{'='*65}")
        print(f"{'閾値':>6}  {'対象馬数':>7}  {'単勝的中率':>9}  {'複勝的中率':>9}  {'対ランダム比':>11}")
        print(f"{'-'*65}")
        random_win = 1 / avg_horses
        for t in THRESHOLDS:
            subset = all_df[all_df['偏差値の差'] >= t]
            if len(subset) < MIN_SAMPLES:
                continue
            wr = subset['target_win'].mean()
            pr = subset['target_place'].mean()
            print(f"  {t:>+4}以上  {len(subset):>7,}頭  {wr:>9.1%}  {pr:>9.1%}  {wr/random_win:>10.1f}倍")
        print(f"{'-'*65}")
        print(f"  ランダム: 単勝 {random_win:.1%} / 複勝 {3/avg_horses:.1%}（平均{avg_horses:.0f}頭）")

if __name__ == "__main__":
    main()

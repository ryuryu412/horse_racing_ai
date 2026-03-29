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

def extract_venue(kaikai):
    m = re.search(r'\d+([^\d]+)', str(kaikai))
    return m.group(1) if m else str(kaikai)

def deviation_score(values, mean, std):
    if std == 0:
        return pd.Series([50.0] * len(values), index=values.index)
    return 50 + 10 * (values - mean) / std

def calc_roi(bets_df, payout_col='単勝配当'):
    """単勝購入のROIを計算（払戻金は100円単位を想定）"""
    total_bets = len(bets_df)
    if total_bets == 0:
        return 0.0, 0, 0
    wins = bets_df[bets_df['target_win'] == 1]
    total_paid = wins[payout_col].sum()
    roi = total_paid / (total_bets * 100) - 1.0
    return roi, len(wins), total_bets

def main():
    print("--- 競馬AI モデル評価（バイナリ + ランカー ROIバックテスト）---")

    base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if 'src' in os.path.abspath(__file__) else '.'
    input_file = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.csv')
    model_dir  = os.path.join(base_dir, 'models')
    ranker_dir = os.path.join(model_dir, 'ranker')

    # モデル情報
    with open(os.path.join(model_dir, 'model_info.json'), 'r', encoding='utf-8') as f:
        model_info = json.load(f)
    features       = model_info['features']
    trained_models = model_info['models']

    # ランカー情報
    ranker_info_path = os.path.join(ranker_dir, 'ranker_info.json')
    trained_rankers = {}
    if os.path.exists(ranker_info_path):
        with open(ranker_info_path, 'r', encoding='utf-8') as f:
            ranker_info = json.load(f)
        trained_rankers = ranker_info.get('rankers', {})
    print(f"バイナリモデル: {len(trained_models)}グループ / ランカー: {len(trained_rankers)}グループ")

    # データ読み込み
    print("データを読み込んでいます...")
    df = pd.read_csv(input_file, low_memory=False)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    if '着順_num' in df.columns:
        df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    else:
        df['着順_num'] = (df['着順'].astype(str)
            .str.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
            .pipe(pd.to_numeric, errors='coerce'))
    df = df.dropna(subset=['日付_num', '着順_num'])
    df['target_win']   = (df['着順_num'] == 1).astype(int)
    df['target_place'] = (df['着順_num'] <= 3).astype(int)
    df['会場']      = df['開催'].apply(extract_venue)
    df['model_key'] = df['会場'] + '_' + df['距離'].astype(str)

    # 単勝オッズ（各馬の個別オッズ）を優先、なければ単勝配当で代替
    if '単勝オッズ' in df.columns:
        df['単勝オッズ'] = pd.to_numeric(df['単勝オッズ'], errors='coerce')
        has_payout = df['単勝オッズ'].notna().sum() > 1000
        if has_payout:
            # EV計算用: 単勝配当（円）に統一（単勝オッズ×100）
            df['単勝配当'] = df['単勝オッズ'] * 100
            print(f"単勝オッズあり（{df['単勝オッズ'].notna().sum():,}件）→ EV計算に使用")
    elif '単勝配当' in df.columns:
        df['単勝配当'] = pd.to_numeric(df['単勝配当'], errors='coerce')
        # 単勝配当は勝ち馬のみ値が入るため、各馬の個別オッズとしては使えない
        # → winner's payout を各馬のオッズ代替として使うため注意
        has_payout = df['単勝配当'].notna().sum() > 1000
        print(f"単勝配当データあり（{df['単勝配当'].notna().sum():,}件）※各馬個別オッズではない")
    else:
        has_payout = False
        print("※ オッズデータなし → 的中率のみ集計")

    for col in features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ────────────────────────────────────────────────
    # テストデータ（各モデルの後ろ20%）を収集
    # ────────────────────────────────────────────────
    all_test_rows = []

    for key in trained_models:
        g = df[df['model_key'] == key].sort_values('日付_num').reset_index(drop=True)
        n = len(g)
        split = int(n * 0.8)
        test = g.iloc[split:].copy()
        if len(test) < 50:
            continue

        # バイナリモデルで確率計算
        win_path = os.path.join(model_dir, trained_models[key]['win'])
        plc_path = os.path.join(model_dir, trained_models[key]['place'])
        if not os.path.exists(win_path):
            continue
        with open(win_path, 'rb') as f:
            m_win = pickle.load(f)
        wf = m_win.booster_.feature_name()
        test['prob_win'] = m_win.predict_proba(test[wf])[:, 1]

        # 偏差値計算
        stats = trained_models[key].get('stats', {})
        w_mean = stats.get('win_mean', test['prob_win'].mean())
        w_std  = stats.get('win_std',  test['prob_win'].std())
        test['コース偏差値'] = deviation_score(test['prob_win'], w_mean, w_std).values

        test['レース内偏差値'] = np.nan
        for (_, _, _), rdf in test.groupby(['日付_num', '開催', 'レース名']):
            if len(rdf) < 2:
                continue
            r_mean = rdf['prob_win'].mean()
            r_std  = rdf['prob_win'].std()
            test.loc[rdf.index, 'レース内偏差値'] = deviation_score(
                rdf['prob_win'], r_mean, r_std if r_std > 0 else 1
            ).values
        test['偏差値の差'] = test['レース内偏差値'] - test['コース偏差値']

        # ランカーモデルでスコア計算
        test['ランカースコア'] = np.nan
        if key in trained_rankers:
            rpath = os.path.join(ranker_dir, trained_rankers[key])
            if os.path.exists(rpath):
                with open(rpath, 'rb') as f:
                    ranker = pickle.load(f)
                test['ランカースコア'] = ranker.predict(test[features])

        # レース内ランカー順位（1位=最強予測）
        test['ランカー順位'] = np.nan
        for (_, _, _), rdf in test.groupby(['日付_num', '開催', 'レース名']):
            if rdf['ランカースコア'].isna().all():
                continue
            ranked = rdf['ランカースコア'].rank(ascending=False, method='min').astype(int)
            test.loc[rdf.index, 'ランカー順位'] = ranked

        test['model_key'] = key
        all_test_rows.append(test)

    if not all_test_rows:
        print("テストデータが空です。")
        return

    all_test = pd.concat(all_test_rows, ignore_index=True)
    print(f"\nテストデータ: {len(all_test):,}件（{all_test['model_key'].nunique()}グループ）\n")

    # ────────────────────────────────────────────────
    # 1. バイナリモデル：偏差値の差 × ROI
    # ────────────────────────────────────────────────
    print(f"{'='*70}")
    print(f" [1] バイナリモデル：偏差値の差 × 単勝ROI")
    print(f"{'='*70}")
    print(f"{'閾値':>6}  {'対象馬':>7}  {'的中数':>6}  {'的中率':>8}", end='')
    print(f"  {'ROI':>8}" if has_payout else "")
    print(f"{'-'*70}")

    data_nodiff = all_test.dropna(subset=['偏差値の差'])
    for thr in [-5, 0, 5, 10, 15, 20, 25]:
        sub = data_nodiff[data_nodiff['偏差値の差'] >= thr]
        if len(sub) < 30:
            continue
        hits = sub['target_win'].sum()
        rate = sub['target_win'].mean()
        line = f"  {thr:>+4}以上  {len(sub):>7,}頭  {hits:>6}  {rate:>8.1%}"
        if has_payout:
            sub_pay = sub.dropna(subset=['単勝配当'])
            roi, w, n = calc_roi(sub_pay)
            line += f"  {roi:>+8.1%}"
        print(line)

    # ────────────────────────────────────────────────
    # 2. ランカーモデル：ランカー上位N位 × ROI
    # ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f" [2] ランカーモデル：ランカー上位N位に賭けた場合")
    print(f"{'='*70}")
    print(f"{'対象':>10}  {'対象馬':>7}  {'的中数':>6}  {'的中率':>8}", end='')
    print(f"  {'ROI':>8}" if has_payout else "")
    print(f"{'-'*70}")

    data_ranked = all_test.dropna(subset=['ランカー順位'])
    for top_n in [1, 2, 3]:
        sub = data_ranked[data_ranked['ランカー順位'] <= top_n]
        if len(sub) < 30:
            continue
        hits = sub['target_win'].sum()
        rate = sub['target_win'].mean()
        line = f"  上位{top_n}位以内  {len(sub):>7,}頭  {hits:>6}  {rate:>8.1%}"
        if has_payout:
            sub_pay = sub.dropna(subset=['単勝配当'])
            roi, w, n = calc_roi(sub_pay)
            line += f"  {roi:>+8.1%}"
        print(line)

    # ────────────────────────────────────────────────
    # 3. 組み合わせ：ランカー1位 × 偏差値の差
    # ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f" [3] 組み合わせ：ランカー1位 & 偏差値の差 ≥ 閾値")
    print(f"{'='*70}")
    print(f"{'閾値':>6}  {'対象馬':>7}  {'的中数':>6}  {'的中率':>8}", end='')
    print(f"  {'ROI':>8}" if has_payout else "")
    print(f"{'-'*70}")

    data_combo = all_test.dropna(subset=['ランカー順位', '偏差値の差'])
    data_combo = data_combo[data_combo['ランカー順位'] == 1]
    for thr in [-5, 0, 5, 10, 15, 20]:
        sub = data_combo[data_combo['偏差値の差'] >= thr]
        if len(sub) < 20:
            continue
        hits = sub['target_win'].sum()
        rate = sub['target_win'].mean()
        line = f"  {thr:>+4}以上  {len(sub):>7,}頭  {hits:>6}  {rate:>8.1%}"
        if has_payout:
            sub_pay = sub.dropna(subset=['単勝配当'])
            roi, w, n = calc_roi(sub_pay)
            line += f"  {roi:>+8.1%}"
        print(line)

    # ────────────────────────────────────────────────
    # 4. ランカー1位の的中率（会場別）
    # ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f" [4] ランカー1位的中率 TOP10（グループ別）")
    print(f"{'='*70}")
    rank1 = data_ranked[data_ranked['ランカー順位'] == 1].copy()
    grp_stats = rank1.groupby('model_key').agg(
        対象=('target_win', 'count'),
        的中=('target_win', 'sum'),
    )
    grp_stats['的中率'] = grp_stats['的中'] / grp_stats['対象']
    if has_payout:
        pay_stats = rank1.dropna(subset=['単勝配当']).groupby('model_key').apply(
            lambda x: (x['単勝配当'].where(x['target_win']==1, 0).sum()) / (len(x)*100) - 1
        )
        grp_stats['ROI'] = pay_stats

    top10 = grp_stats[grp_stats['対象'] >= 20].sort_values('的中率', ascending=False).head(10)
    print(top10.to_string())

    # ────────────────────────────────────────────────
    # 5. 確信度帯別 期待値分析
    # 単勝配当は勝ち馬のみに値が入るため、モデルのprob_winで馬を区分する
    # ────────────────────────────────────────────────
    def ev_by_prob(subset, label):
        """prob_win 帯別に的中率・ROI・平均配当を集計"""
        print(f"\n{'='*70}")
        print(f" {label}")
        print(f"{'='*70}")
        print(f"  {'確信度帯':16s}  {'頭数':>6}  {'的中':>5}  {'的中率':>7}  {'勝時平均配当':>10}  {'ROI':>8}")
        print(f"  {'-'*65}")
        prob_bins   = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 1.01]
        prob_labels = ['~5%', '~10%', '~15%', '~20%', '~30%', '30%~']
        sub2 = subset.copy()
        sub2['確信度帯'] = pd.cut(sub2['prob_win'], bins=prob_bins, labels=prob_labels)
        for band in prob_labels:
            s = sub2[sub2['確信度帯'] == band]
            if len(s) < 20: continue
            hits    = s['target_win'].sum()
            rate    = s['target_win'].mean()
            if has_payout:
                avg_pay = s.loc[s['target_win']==1, '単勝配当'].replace(0, np.nan).mean()
                roi     = s['単勝配当'].where(s['target_win']==1, 0).sum() / (len(s)*100) - 1
                print(f"  {band:16s}  {len(s):>6,}  {hits:>5}  {rate:>7.1%}  {avg_pay:>10.0f}円  {roi:>+8.1%}")
            else:
                print(f"  {band:16s}  {len(s):>6,}  {hits:>5}  {rate:>7.1%}")

    ev_by_prob(all_test.dropna(subset=['prob_win']),
               "[5] 確信度帯別 期待値（全馬）")
    ev_by_prob(rank1.dropna(subset=['prob_win']),
               "[6] 確信度帯別 期待値（ランカー1位）")
    ev_by_prob(all_test[all_test['偏差値の差'] >= 10].dropna(subset=['prob_win']),
               "[7] 確信度帯別 期待値（偏差値の差 ≥ 10）")

    # ────────────────────────────────────────────────
    # 8. 全体サマリー
    # ────────────────────────────────────────────────
    avg_horses = all_test.groupby(['日付_num', '開催', 'レース名']).size().mean()
    print(f"\n{'='*70}")
    print(f" サマリー")
    print(f"{'='*70}")
    print(f"  平均頭数       : {avg_horses:.1f}頭")
    print(f"  単純ランダム単勝: {1/avg_horses:.1%}")
    print(f"  ランカー1位的中 : {rank1['target_win'].mean():.1%}（{rank1['target_win'].sum()}/{len(rank1)}）")
    data_15 = data_nodiff[data_nodiff['偏差値の差'] >= 15]
    if len(data_15) > 0:
        print(f"  偏差値の差≥15  : {data_15['target_win'].mean():.1%}（{data_15['target_win'].sum()}/{len(data_15)}）")

if __name__ == "__main__":
    main()

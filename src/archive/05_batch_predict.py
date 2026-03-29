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

def extract_venue(kaikai):
    m = re.search(r'\d+([^\d]+)', str(kaikai))
    return m.group(1) if m else str(kaikai)

def deviation_score(values, mean, std):
    if std == 0:
        return pd.Series([50.0] * len(values), index=values.index)
    return 50 + 10 * (values - mean) / std

def normalize_probs(probs, n_expected):
    total = probs.sum()
    if total == 0:
        return probs
    return probs / total * n_expected

def predict_race(df_race, features, trained_models, model_dir, trained_rankers=None):
    """1レース分の予測を実行してDataFrameを返す"""
    df = df_race.copy()
    for col in features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['単勝確率_raw'] = np.nan
    df['複勝確率_raw'] = np.nan
    df['使用モデル']    = '未対応'

    for key, group in df.groupby('model_key'):
        if key not in trained_models:
            continue
        paths = {
            'win':   os.path.join(model_dir, trained_models[key]['win']),
            'place': os.path.join(model_dir, trained_models[key]['place'])
        }
        for label, path in paths.items():
            if not os.path.exists(path):
                continue
            with open(path, 'rb') as f:
                model = pickle.load(f)
            model_features = model.booster_.feature_name()
            prob = model.predict_proba(group[model_features])[:, 1]
            col  = '単勝確率_raw' if label == 'win' else '複勝確率_raw'
            df.loc[group.index, col] = prob
        df.loc[group.index, '使用モデル'] = key

    # 偏差値
    df['コース偏差値']   = np.nan
    df['レース内偏差値'] = np.nan
    for key, group in df.groupby('使用モデル'):
        if key == '未対応' or key not in trained_models:
            continue
        stats = trained_models[key].get('stats', {})
        if stats.get('win_std', 0) > 0:
            df.loc[group.index, 'コース偏差値'] = deviation_score(
                group['単勝確率_raw'], stats['win_mean'], stats['win_std']
            ).values
        race_mean = group['単勝確率_raw'].mean()
        race_std  = group['単勝確率_raw'].std()
        df.loc[group.index, 'レース内偏差値'] = deviation_score(
            group['単勝確率_raw'], race_mean, race_std if race_std > 0 else 1
        ).values
    df['偏差値の差'] = (df['レース内偏差値'] - df['コース偏差値']).round(1)

    # ランカーモデルで予測
    df['ランカースコア'] = np.nan
    df['ランカー順位']   = np.nan
    if trained_rankers:
        ranker_dir = os.path.join(model_dir, 'ranker')
        for key, group in df.groupby('model_key'):
            if key not in trained_rankers:
                continue
            ranker_path = os.path.join(ranker_dir, trained_rankers[key])
            if not os.path.exists(ranker_path):
                continue
            with open(ranker_path, 'rb') as f:
                ranker = pickle.load(f)
            scores = ranker.predict(group[features])
            df.loc[group.index, 'ランカースコア'] = scores
        for key, group in df.groupby('model_key'):
            if group['ランカースコア'].isna().all():
                continue
            ranked = group['ランカースコア'].rank(ascending=False, method='min').astype(int)
            df.loc[group.index, 'ランカー順位'] = ranked

    # 確率正規化
    n_horses = len(df.dropna(subset=['単勝確率_raw']))
    n_place  = min(3, n_horses)
    df['単勝確率'] = normalize_probs(df['単勝確率_raw'].fillna(0), 1.0)
    df['複勝確率'] = normalize_probs(df['複勝確率_raw'].fillna(0), float(n_place))
    df['単勝確率%'] = (df['単勝確率'] * 100).round(1)
    df['複勝確率%'] = (df['複勝確率'] * 100 / n_place).round(1)

    return df

def main():
    parser = argparse.ArgumentParser(description='全レース一括予測')
    parser.add_argument('--date', type=str, default='',
                        help='予測日付（例: 260308）。省略時は最新日付')
    args = parser.parse_args()

    print("--- 競馬AI 全レース一括予測 ---")

    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if 'src' in os.path.abspath(__file__) else '.'
    input_file  = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.csv')
    model_dir   = os.path.join(base_dir, 'models')
    output_dir  = os.path.join(base_dir, 'predictions')
    info_path   = os.path.join(model_dir, 'model_info.json')
    os.makedirs(output_dir, exist_ok=True)

    # モデル情報
    if not os.path.exists(info_path):
        print("エラー: model_info.json が見つかりません。先に 02_train_model.py を実行してください。")
        return
    with open(info_path, 'r', encoding='utf-8') as f:
        model_info = json.load(f)
    features       = model_info['features']
    trained_models = model_info['models']

    # ランカーモデル情報
    ranker_info_path = os.path.join(model_dir, 'ranker', 'ranker_info.json')
    trained_rankers = {}
    if os.path.exists(ranker_info_path):
        with open(ranker_info_path, 'r', encoding='utf-8') as f:
            ranker_info = json.load(f)
        trained_rankers = ranker_info.get('rankers', {})
        print(f"ランカーモデル: {len(trained_rankers)}グループ")

    # データ読み込み
    print("データを読み込んでいます...")
    df = pd.read_csv(input_file, low_memory=False)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df = df.dropna(subset=['日付_num'])
    df['会場']      = df['開催'].apply(extract_venue)
    df['model_key'] = df['会場'] + '_' + df['距離'].astype(str)

    # 日付選択
    available_dates = sorted(df['日付_num'].unique().astype(int), reverse=True)
    if args.date:
        target_date = int(args.date)
    else:
        print(f"最新10件の利用可能日付: {available_dates[:10]}")
        target_date = available_dates[0]

    df_day = df[df['日付_num'] == target_date].copy()
    if df_day.empty:
        print(f"日付 {target_date} のデータが見つかりません。")
        return

    races = df_day['レース名'].dropna().unique() if 'レース名' in df_day.columns else []
    print(f"\n日付: {target_date} / レース数: {len(races)}\n")

    # 全レース一括予測
    all_results = []
    for race_name in races:
        df_race = df_day[df_day['レース名'] == race_name].copy()
        key     = df_race['model_key'].iloc[0] if len(df_race) > 0 else '不明'
        supported = key in trained_models

        result = predict_race(df_race, features, trained_models, model_dir, trained_rankers)
        # 追加指標の整形
        if '近走_改善トレンド' in result.columns:
            result['近走トレンド'] = result['近走_改善トレンド'].map(
                lambda x: '↑上昇' if x > 1 else ('↓下降' if x < -1 else '→横ばい') if pd.notna(x) else '-'
            )
        if '騎手変更' in result.columns:
            result['乗替り'] = result['騎手変更'].map(
                lambda x: '○' if x == 1.0 else '-' if pd.notna(x) else '-'
            )
        if '前走着差タイム' in result.columns:
            result['前走着差'] = result['前走着差タイム'].round(1)
        if '同会場_複勝率_近5走' in result.columns:
            result['同会場複勝率'] = (result['同会場_複勝率_近5走'] * 100).round(0).map(
                lambda x: f'{int(x)}%' if pd.notna(x) else '-'
            )
        if '間隔' in result.columns:
            result['間隔(週)'] = result['間隔'].map(
                lambda x: f'{int(x)}週' if pd.notna(x) else '-'
            )

        display_cols = ['馬名S']
        for c in ['着順', '馬体重']:
            if c in result.columns:
                display_cols.append(c)
        display_cols += ['単勝確率%', '複勝確率%', 'コース偏差値', 'レース内偏差値', '偏差値の差']
        if 'ランカー順位' in result.columns and result['ランカー順位'].notna().any():
            display_cols.append('ランカー順位')
        for c in ['近走トレンド', '乗替り', '前走着差', '同会場複勝率', '間隔(週)']:
            if c in result.columns:
                display_cols.append(c)

        result_out = result[display_cols].dropna(subset=['単勝確率%']) \
                                         .sort_values('単勝確率%', ascending=False) \
                                         .reset_index(drop=True)
        result_out.index += 1

        mark = "✓" if supported else "×"
        print(f"[{mark}] {race_name}（{len(df_race)}頭 / {key}）")

        if not result_out.empty:
            print(result_out.to_string())
            # CSV保存
            safe_name = re.sub(r'[\\/:*?"<>|]', '_', race_name)
            out_path = os.path.join(output_dir, f"prediction_{target_date}_{safe_name}.csv")
            result_out.to_csv(out_path, index=True, encoding='utf-8-sig')
            all_results.append(result_out.assign(レース名=race_name))
        else:
            print("  （予測データなし）")
        print()

    # 全レース統合CSV
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined_path = os.path.join(output_dir, f"prediction_{target_date}_ALL.csv")
        combined.to_csv(combined_path, index=False, encoding='utf-8-sig')
        print(f"{'='*60}")
        print(f"全レース統合ファイル保存: {combined_path}")
        print(f"予測済みレース数: {len(races)} / うち対応モデルあり: {sum(1 for r in races if df_day[df_day['レース名']==r]['model_key'].iloc[0] in trained_models)}")

if __name__ == "__main__":
    main()

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
    """偏差値を計算（平均50・標準偏差10のスケール）"""
    if std == 0:
        return pd.Series([50.0] * len(values), index=values.index)
    return 50 + 10 * (values - mean) / std

def normalize_probs(probs, n_expected):
    """
    レース内で確率を正規化する。
    - 単勝: n_expected=1  → 全馬の確率の合計が1.0になる（1頭だけ勝つ）
    - 複勝: n_expected=3  → 全馬の確率の合計が3.0になる（3頭が3着以内）
    """
    total = probs.sum()
    if total == 0:
        return probs
    return probs / total * n_expected

def main():
    print("--- 競馬AI レース予測（単勝・複勝・期待値）---")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if 'src' in os.path.abspath(__file__) else '.'
    input_file = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.csv')
    model_dir  = os.path.join(base_dir, 'models')
    output_dir = os.path.join(base_dir, 'predictions')
    info_path  = os.path.join(model_dir, 'model_info.json')
    os.makedirs(output_dir, exist_ok=True)

    # 1. モデル情報の読み込み
    if not os.path.exists(info_path):
        print("エラー: model_info.json が見つかりません。先に 02_train_model.py を実行してください。")
        return
    with open(info_path, 'r', encoding='utf-8') as f:
        model_info = json.load(f)
    features       = model_info['features']
    trained_models = model_info['models']
    print(f"利用可能なモデル: {len(trained_models)}グループ × 単勝/複勝 の2種")

    # ランカーモデル情報
    ranker_dir  = os.path.join(model_dir, 'ranker')
    ranker_info_path = os.path.join(ranker_dir, 'ranker_info.json')
    trained_rankers = {}
    if os.path.exists(ranker_info_path):
        with open(ranker_info_path, 'r', encoding='utf-8') as f:
            ranker_info = json.load(f)
        trained_rankers = ranker_info.get('rankers', {})
        print(f"ランカーモデル: {len(trained_rankers)}グループ")
    else:
        print("※ ランカーモデルなし（07_train_ranker.py を実行すると追加されます）")

    # 2. データ読み込み
    print("加工済みデータを読み込んでいます...")
    df = pd.read_csv(input_file, low_memory=False)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df = df.dropna(subset=['日付_num'])
    df['会場']      = df['開催'].apply(extract_venue)
    df['model_key'] = df['会場'] + '_' + df['距離'].astype(str)
    print(f"データ読み込み完了: {len(df):,}件")

    # 3. 日付の選択
    available_dates = sorted(df['日付_num'].unique().astype(int), reverse=True)
    print(f"\n利用可能な最新の日付（最大10件）:")
    for i, d in enumerate(available_dates[:10]):
        print(f"  {i+1}. {d}")
    print("\n予測したい日付を入力してください（例: 260301）。Enterで最新日付。")
    user_input  = input("日付 > ").strip()
    target_date = int(available_dates[0]) if user_input == "" else int(user_input)

    df_day = df[df['日付_num'] == target_date].copy()
    if df_day.empty:
        print(f"日付 {target_date} のデータが見つかりません。")
        return

    # 4. レースの選択
    races = df_day['レース名'].dropna().unique() if 'レース名' in df_day.columns else []
    target_race = None
    if len(races) > 1:
        print(f"\n日付 {target_date} のレース一覧:")
        for i, r in enumerate(races):
            count = len(df_day[df_day['レース名'] == r])
            key   = df_day[df_day['レース名'] == r]['model_key'].iloc[0]
            mark  = "✓" if key in trained_models else "×"
            print(f"  {i+1:2d}. [{mark}] {r}（{count}頭 / {key}）")
        print("\n予測するレース番号を入力してください。")
        race_input = input("番号 > ").strip()
        try:
            target_race = races[int(race_input) - 1]
            df_day = df_day[df_day['レース名'] == target_race].copy()
        except (ValueError, IndexError):
            print("無効な入力のため全レースを対象にします。")
    elif len(races) == 1:
        target_race = races[0]
        print(f"レース: {target_race}")

    # 5. 予測実行
    print(f"\n対象馬数: {len(df_day)}頭")
    for col in features:
        if col in df_day.columns:
            df_day[col] = pd.to_numeric(df_day[col], errors='coerce')

    df_day['単勝確率_raw']  = np.nan
    df_day['複勝確率_raw']  = np.nan
    df_day['使用モデル']     = '未対応'

    for key, group in df_day.groupby('model_key'):
        if key not in trained_models:
            print(f"  ※ {key} のモデルが見つかりません（データ不足で未学習）")
            continue

        paths = {
            'win':   os.path.join(model_dir, trained_models[key]['win']),
            'place': os.path.join(model_dir, trained_models[key]['place'])
        }
        for label, path in paths.items():
            if not os.path.exists(path):
                print(f"  ※ {key}_{label} ファイルが見つかりません")
                continue
            with open(path, 'rb') as f:
                model = pickle.load(f)
            model_features = model.booster_.feature_name()
            prob = model.predict_proba(group[model_features])[:, 1]
            col  = '単勝確率_raw' if label == 'win' else '複勝確率_raw'
            df_day.loc[group.index, col] = prob

        df_day.loc[group.index, '使用モデル'] = key
        print(f"  ✓ {key} モデルで予測（{len(group)}頭）")

    # 5.5. ランカーモデルで予測
    df_day['ランカースコア'] = np.nan
    for col in features:
        if col in df_day.columns:
            df_day[col] = pd.to_numeric(df_day[col], errors='coerce')
    for key, group in df_day.groupby('model_key'):
        if key not in trained_rankers:
            continue
        ranker_path = os.path.join(ranker_dir, trained_rankers[key])
        if not os.path.exists(ranker_path):
            continue
        with open(ranker_path, 'rb') as f:
            ranker = pickle.load(f)
        scores = ranker.predict(group[features])
        df_day.loc[group.index, 'ランカースコア'] = scores
    # スコアを順位に変換（スコア降順 → 1位が最強）
    df_day['ランカー順位'] = np.nan
    for key, group in df_day.groupby('model_key'):
        if group['ランカースコア'].isna().all():
            continue
        ranked = group['ランカースコア'].rank(ascending=False, method='min').astype(int)
        df_day.loc[group.index, 'ランカー順位'] = ranked

    # 6. 偏差値の計算
    df_day['コース偏差値'] = np.nan
    df_day['レース内偏差値'] = np.nan

    for key, group in df_day.groupby('使用モデル'):
        if key == '未対応' or key not in trained_models:
            continue
        stats = trained_models[key].get('stats', {})

        # コース偏差値：このコース全体の歴史的分布との比較
        if stats.get('win_std', 0) > 0:
            df_day.loc[group.index, 'コース偏差値'] = deviation_score(
                group['単勝確率_raw'], stats['win_mean'], stats['win_std']
            ).values

        # レース内偏差値：同じレース内の馬との比較
        race_mean = group['単勝確率_raw'].mean()
        race_std  = group['単勝確率_raw'].std()
        df_day.loc[group.index, 'レース内偏差値'] = deviation_score(
            group['単勝確率_raw'], race_mean, race_std if race_std > 0 else 1
        ).values

    df_day['偏差値の差'] = (df_day['レース内偏差値'] - df_day['コース偏差値']).round(1)

    # 7. レース内で確率を正規化（合計が意味のある数字になる）
    # 単勝：1頭しか勝てないので合計=1.0
    # 複勝：3頭が3着以内なので合計=3.0（ただし頭数が少ない場合は調整）
    n_horses = len(df_day.dropna(subset=['単勝確率_raw']))
    n_place  = min(3, n_horses)  # 出走頭数が3頭未満なら全頭

    df_day['単勝確率'] = normalize_probs(df_day['単勝確率_raw'].fillna(0), 1.0)
    df_day['複勝確率'] = normalize_probs(df_day['複勝確率_raw'].fillna(0), float(n_place))

    # 単勝確率をパーセント表示用に変換（レース内での相対的な強さ）
    df_day['単勝確率%'] = (df_day['単勝確率'] * 100).round(1)
    df_day['複勝確率%'] = (df_day['複勝確率'] * 100 / n_place).round(1)  # 1頭あたりの確率

    # 8. 追加指標の整形
    # 近走トレンド → ↑上昇 / ↓下降 / → 横ばい
    if '近走_改善トレンド' in df_day.columns:
        df_day['近走トレンド'] = df_day['近走_改善トレンド'].map(
            lambda x: '↑上昇' if x > 1 else ('↓下降' if x < -1 else '→横ばい') if pd.notna(x) else '-'
        )
    # 騎手変更 → ○ / -
    if '騎手変更' in df_day.columns:
        df_day['乗替り'] = df_day['騎手変更'].map(
            lambda x: '○' if x == 1.0 else '-' if pd.notna(x) else '-'
        )
    # 前走着差タイム → 数値のまま（小さいほど僅差）
    if '前走着差タイム' in df_day.columns:
        df_day['前走着差'] = df_day['前走着差タイム'].round(1)
    # 同会場複勝率 → %表示
    if '同会場_複勝率_近5走' in df_day.columns:
        df_day['同会場複勝率'] = (df_day['同会場_複勝率_近5走'] * 100).round(0).map(
            lambda x: f'{int(x)}%' if pd.notna(x) else '-'
        )
    # 間隔 → そのまま（週）
    if '間隔' in df_day.columns:
        df_day['間隔(週)'] = df_day['間隔'].map(
            lambda x: f'{int(x)}週' if pd.notna(x) else '-'
        )

    # 9. 表示する列を組み立て
    display_cols = ['馬名S']
    for c in ['着順', '馬体重']:
        if c in df_day.columns:
            display_cols.append(c)
    display_cols += ['単勝確率%', '複勝確率%', 'コース偏差値', 'レース内偏差値', '偏差値の差']
    if df_day['ランカー順位'].notna().any():
        display_cols.append('ランカー順位')
    for c in ['近走トレンド', '乗替り', '前走着差', '同会場複勝率', '間隔(週)']:
        if c in df_day.columns:
            display_cols.append(c)

    result = df_day[display_cols].dropna(subset=['単勝確率%']) \
                                  .sort_values('単勝確率%', ascending=False) \
                                  .reset_index(drop=True)
    result.index += 1

    race_label = f"_{target_race}" if target_race else ""
    print(f"\n{'='*80}")
    print(f" 予測結果（{target_date}{race_label}）")
    print(f"{'='*80}")
    print(result.to_string())
    print(f"\n  偏差値の差  : プラス＝このレース特有の穴馬候補（≥15で単勝買い目安）")
    print(f"  近走トレンド: 3走前→1走前の着順変化")
    print(f"  乗替り     : ○＝前走から騎手変更あり")
    print(f"  前走着差   : 勝ち馬との秒差（小さいほど僅差負け）")
    print(f"  同会場複勝率: このコースでの近5走複勝率")
    print(f"  間隔(週)   : 前走からの間隔")
    print(f"{'='*80}")

    # 9. CSV保存
    output_file = os.path.join(output_dir, f"prediction_{target_date}{race_label}.csv")
    result.to_csv(output_file, index=True, encoding='utf-8-sig')
    print(f"\n予測結果を保存しました: {output_file}")

if __name__ == "__main__":
    main()

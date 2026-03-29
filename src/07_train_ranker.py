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
import lightgbm as lgb
from sklearn.metrics import ndcg_score

def extract_venue(kaikai):
    m = re.search(r'\d+([^\d]+)', str(kaikai))
    return m.group(1) if m else str(kaikai)

def main():
    print("--- 競馬AI ランキングモデル学習（LGBMRanker / 会場×コース別）---")

    base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if 'src' in os.path.abspath(__file__) else '.'
    input_file = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.csv')
    model_dir  = os.path.join(base_dir, 'models', 'ranker')
    os.makedirs(model_dir, exist_ok=True)

    # 既存モデルの特徴量リストを流用
    info_path = os.path.join(base_dir, 'models', 'model_info.json')
    with open(info_path, 'r', encoding='utf-8') as f:
        model_info = json.load(f)
    features = model_info['features']

    # データ読み込み
    print("データを読み込んでいます...")
    df = pd.read_csv(input_file, low_memory=False)
    df['日付_num']  = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num']  = pd.to_numeric(df['着順_num'], errors='coerce')
    df['頭数_num']  = pd.to_numeric(df['頭数'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num'])

    # ランキングラベル：1着=最高点、着外(99)=0
    # label = max(0, 頭数 - 着順 + 1) → 1着が最大、最下位が1、着外が0
    df['rank_label'] = np.where(
        df['着順_num'] >= 99,
        0,
        np.maximum(0, df['頭数_num'] - df['着順_num'] + 1).fillna(0)
    ).astype(int)

    df['会場']      = df['開催'].apply(extract_venue)
    df['model_key'] = df['会場'] + '_' + df['距離'].astype(str)

    # 特徴量を数値化
    for col in features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # レースIDを作成（グループキー用）
    df['race_id'] = df['日付_num'].astype(str) + '_' + df['開催'].astype(str) + '_' + df['レース名'].astype(str)

    MIN_SAMPLES = 300
    group_counts = df['model_key'].value_counts()
    target_groups = group_counts[group_counts >= MIN_SAMPLES].index.tolist()
    print(f"学習対象グループ: {len(target_groups)}種類\n")

    trained_rankers = {}
    results = []

    for i, key in enumerate(target_groups, 1):
        df_g = df[df['model_key'] == key].sort_values('日付_num').reset_index(drop=True)
        n = len(df_g)
        split = int(n * 0.8)

        df_train = df_g.iloc[:split].copy()
        df_test  = df_g.iloc[split:].copy()
        if len(df_test) < 50:
            continue

        # グループサイズ（1レースあたりの頭数）を計算
        train_groups = df_train.groupby('race_id').size().values
        test_groups  = df_test.groupby('race_id').size().values

        # グループ順にソート（LGBMRankerの要件）
        df_train = df_train.sort_values('race_id')
        df_test  = df_test.sort_values('race_id')

        X_train = df_train[features]
        y_train = df_train['rank_label']
        X_test  = df_test[features]
        y_test  = df_test['rank_label']

        # LGBMRanker 学習
        ranker = lgb.LGBMRanker(
            n_estimators=300,
            random_state=42,
            verbosity=-1,
            objective='lambdarank',
            metric='ndcg',
            ndcg_eval_at=[1, 3],
        )
        ranker.fit(
            X_train, y_train,
            group=train_groups,
            eval_set=[(X_test, y_test)],
            eval_group=[test_groups],
            callbacks=[
                lgb.early_stopping(stopping_rounds=20, verbose=False),
                lgb.log_evaluation(period=-1),
            ]
        )

        # 評価：テストデータで1着馬を何位に予測できているか
        scores = ranker.predict(X_test)
        df_test = df_test.copy()
        df_test['score'] = scores

        # レースごとに「1着馬の予測順位」を計算
        rank1_positions = []
        for _, grp in df_test.groupby('race_id'):
            if len(grp) < 2:
                continue
            sorted_grp = grp.sort_values('score', ascending=False).reset_index(drop=True)
            pos = sorted_grp[sorted_grp['着順_num'] == 1].index
            if len(pos) > 0:
                rank1_positions.append(pos[0] + 1)  # 1-indexed

        avg_pos = np.mean(rank1_positions) if rank1_positions else 99
        top1_rate = np.mean([p == 1 for p in rank1_positions]) if rank1_positions else 0

        fn = f'ranker_{key}.pkl'
        with open(os.path.join(model_dir, fn), 'wb') as f:
            pickle.dump(ranker, f)

        trained_rankers[key] = fn
        results.append({
            '会場_コース': key, 'サンプル数': n,
            '1着予測平均順位': f'{avg_pos:.2f}位',
            '1着を1位予測': f'{top1_rate:.1%}'
        })
        print(f"[{i:3d}/{len(target_groups)}] {key:12s} {n:6,}件  1着平均予測順位:{avg_pos:.2f}  1位的中:{top1_rate:.1%}")

    # ランカー情報を保存
    ranker_info = {
        'features': features,
        'rankers': trained_rankers,
    }
    ranker_info_path = os.path.join(model_dir, 'ranker_info.json')
    with open(ranker_info_path, 'w', encoding='utf-8') as f:
        json.dump(ranker_info, f, ensure_ascii=False, indent=2)

    print(f"\n=== 学習完了 ===")
    print(f"ランキングモデル数: {len(trained_rankers)}グループ")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nランカー情報を保存しました: {ranker_info_path}")

if __name__ == "__main__":
    main()

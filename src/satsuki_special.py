"""
皐月賞特別予想モデル
- 過去の皐月賞データ（2013〜）でLightGBM Rankerを学習
- 前哨戦タイム指数・クラス・上り3F指数など皐月賞固有の重要特徴を重視
- 既存モデルスコアと並べてHTML出力
"""
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent

# ---------- 特徴量定義 ----------
FEATURES = [
    "1走前_タイム指数",       # 前哨戦スピード指数
    "2走前_タイム指数",       # 2走前スピード指数
    "近5走_タイム指数平均",    # 中長期スピード水準
    "1走前_クラス_rank",      # 前哨戦のクラス（G1=9, G2=8, G3=7...）
    "1走前_着順_num",         # 前哨戦着順
    "1走前_クラス調整着順",    # クラス補正済み着順
    "1走前_上り3F_指数",      # 前哨戦上り指数（差し・末脚）
    "2走前_上り3F_指数",
    "近5走_上り3F指数平均",   # 末脚の安定性
    "馬体重",
    "馬体重増減",
    "1走前_クラス差",         # クラスアップ/ダウンの度合い
]


def load_train_data() -> pd.DataFrame:
    """皐月賞の過去データをparquetから取得"""
    df = pd.read_parquet(ROOT / "data/processed/all_venues_features.parquet")
    satsuki = df[df["レース名"] == "皐月賞G1"].copy()
    satsuki["target"] = (satsuki["着順_num"] == 1).astype(int)
    return satsuki


def load_current_horses() -> pd.DataFrame:
    """キャッシュから本日の中山11R（皐月賞）を取得"""
    cache_path = ROOT / "data/raw/cache/出馬表形式4月19日オッズcsv.cache.pkl"
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    result = cache["result"]
    r11 = result[(result["会場"] == "中") & (result["Ｒ"] == 11)].copy()
    return r11


def train_ranker(train: pd.DataFrame) -> lgb.LGBMRanker:
    """皐月賞データでLambdaMARTランカーを学習"""
    # 年ごとにグループ分け（1レース=1グループ）
    train = train.dropna(subset=["着順_num"]).sort_values("日付")
    groups = train.groupby("日付").size().values

    # 使える特徴量だけ抽出（欠損率50%未満）
    feat_ok = [f for f in FEATURES if f in train.columns and train[f].isnull().mean() < 0.5]
    X = train[feat_ok].fillna(train[feat_ok].median())
    y = train["着順_num"].clip(upper=18).astype(int)  # 着順をlabelに使う

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[1, 3],
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=16,
        min_child_samples=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    # relevance: 1着=18点、18着=1点（LambdaRankはnon-negative必須）
    relevance = (19 - y).clip(lower=1)
    model.fit(X, relevance, group=groups)
    model.feature_name_used_ = feat_ok
    return model


def score_current(model: lgb.LGBMRanker, horses: pd.DataFrame) -> pd.DataFrame:
    """現在の出走馬をスコアリング"""
    feat_ok = model.feature_name_used_
    X = horses[feat_ok].fillna(horses[feat_ok].median())
    horses = horses.copy()
    horses["satsuki_score"] = model.predict(X)
    horses["satsuki_rank"] = horses["satsuki_score"].rank(ascending=False).astype(int)
    return horses


def feature_importance_text(model: lgb.LGBMRanker) -> str:
    feat_ok = model.feature_name_used_
    fi = sorted(zip(feat_ok, model.feature_importances_), key=lambda x: -x[1])
    lines = []
    total = sum(v for _, v in fi)
    for name, val in fi:
        lines.append(f"  {name:<22} {val/total*100:4.1f}%")
    return "\n".join(lines)


def generate_html(horses: pd.DataFrame, fi_text: str) -> str:
    rows_html = ""
    sorted_horses = horses.sort_values("satsuki_rank")
    for _, h in sorted_horses.iterrows():
        rank = int(h["satsuki_rank"])
        cur_rank = int(h.get("cur_ランカー順位", 99))
        sub_rank = int(h.get("sub_ランカー順位", 99))
        odds = h.get("単勝オッズ", "-")
        odds_str = f"{odds:.1f}" if isinstance(odds, (int, float)) and not np.isnan(odds) else "-"

        # 順位色
        if rank == 1:
            rank_color = "#e74c3c"
            rank_bg = "background:#e74c3c;color:white;"
        elif rank <= 3:
            rank_color = "#e67e22"
            rank_bg = "background:#e67e22;color:white;"
        elif rank <= 6:
            rank_color = "#27ae60"
            rank_bg = ""
        else:
            rank_color = "#555"
            rank_bg = ""

        # 既存モデルとの比較
        diff_cur = cur_rank - rank   # 正=既存より皐月特別が高評価
        diff_arrow = ""
        if diff_cur >= 3:
            diff_arrow = '<span style="color:#e74c3c;font-weight:bold">↑↑</span>'
        elif diff_cur >= 1:
            diff_arrow = '<span style="color:#e67e22">↑</span>'
        elif diff_cur <= -3:
            diff_arrow = '<span style="color:#3498db;font-weight:bold">↓↓</span>'
        elif diff_cur <= -1:
            diff_arrow = '<span style="color:#3498db">↓</span>'

        ti1 = h.get("1走前_タイム指数", float("nan"))
        ur1 = h.get("1走前_上り3F_指数", float("nan"))
        cls1 = h.get("1走前_クラス_rank", float("nan"))
        cls_label = {9: "G1", 8: "G2", 7: "G3", 6: "OP", 5: "3勝", 4: "2勝", 3: "1勝"}.get(
            int(cls1) if not np.isnan(cls1) else -1, "-"
        )

        rows_html += f"""
        <tr>
          <td style="{rank_bg}text-align:center;font-weight:bold;font-size:16px;color:{'white' if rank<=3 else rank_color}">{rank}</td>
          <td style="font-weight:bold;font-size:14px">{h['馬名']}</td>
          <td style="color:#aaa">{h.get('騎手','-')}</td>
          <td style="color:#f0a500;font-weight:bold">{odds_str}</td>
          <td style="text-align:center">{cls_label}</td>
          <td style="text-align:center">{f'{ti1:.1f}' if not np.isnan(ti1) else '-'}</td>
          <td style="text-align:center">{f'{ur1:.1f}' if not np.isnan(ur1) else '-'}</td>
          <td style="text-align:center;color:#8b949e">{cur_rank}</td>
          <td style="text-align:center;color:#8b949e">{sub_rank}</td>
          <td style="text-align:center">{diff_arrow}</td>
        </tr>"""

    now = datetime.now().strftime("%Y/%m/%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>皐月賞 特別予想モデル 2026</title>
<style>
  body {{ background:#0d1117; color:#c9d1d9; font-family:'Noto Sans JP',sans-serif; margin:0; padding:20px; }}
  .container {{ max-width:900px; margin:0 auto; }}
  h1 {{ color:#f0a500; border-bottom:2px solid #f0a500; padding-bottom:8px; }}
  .subtitle {{ color:#8b949e; margin-bottom:20px; font-size:13px; }}
  table {{ width:100%; border-collapse:collapse; background:#161b22; border-radius:8px; overflow:hidden; }}
  th {{ background:#21262d; color:#8b949e; padding:10px 8px; font-size:12px; text-align:center; }}
  td {{ padding:10px 8px; border-bottom:1px solid #21262d; font-size:13px; }}
  tr:hover {{ background:#1c2128; }}
  .fi-box {{ background:#161b22; border-radius:8px; padding:16px; margin-top:20px; }}
  .fi-box h3 {{ color:#8b949e; font-size:13px; margin:0 0 10px; }}
  .fi-box pre {{ color:#c9d1d9; font-size:12px; margin:0; }}
  .note {{ color:#8b949e; font-size:11px; margin-top:16px; }}
</style>
</head>
<body>
<div class="container">
  <h1>🏆 皐月賞 特別予想モデル</h1>
  <div class="subtitle">
    2026年4月19日（日）中山11R 芝2000m G1 ／ 生成: {now}<br>
    訓練データ: 過去の皐月賞（2013〜2026年） ／ モデル: LightGBM LambdaMART（前哨戦特化）
  </div>
  <table>
    <thead>
      <tr>
        <th>皐月Rnk</th><th>馬名</th><th>騎手</th><th>オッズ</th>
        <th>前走<br>クラス</th><th>前走<br>タイム指数</th><th>前走<br>上り指数</th>
        <th>通常<br>距離Rnk</th><th>通常<br>クラスRnk</th><th>評価<br>変化</th>
      </tr>
    </thead>
    <tbody>
{rows_html}
    </tbody>
  </table>

  <div class="fi-box">
    <h3>特徴量重要度（皐月賞モデル）</h3>
    <pre>{fi_text}</pre>
  </div>

  <div class="note">
    ※「評価変化」は通常モデル（距離Rnk）との比較。↑↑=皐月特別モデルで大幅に高評価、↓↓=低評価。<br>
    ※ このモデルは前哨戦の質とタイム指数を重視。通常モデルは会場×距離の累積成績ベース。両方見て判断推奨。
  </div>
</div>
</body>
</html>"""


def main():
    print("皐月賞特別モデル: 訓練データ読み込み中...")
    train = load_train_data()
    print(f"  皐月賞データ: {len(train)}件 ({train['日付'].nunique()}年分)")

    print("モデル学習中...")
    model = train_ranker(train)
    print(f"  使用特徴量: {model.feature_name_used_}")

    print("現在の出走馬スコアリング中...")
    horses = load_current_horses()
    horses = score_current(model, horses)

    print("\n===== 皐月賞 特別予想ランキング =====")
    display_cols = ["馬名", "騎手", "単勝オッズ", "satsuki_rank", "cur_ランカー順位", "sub_ランカー順位",
                    "1走前_タイム指数", "1走前_クラス_rank", "1走前_上り3F_指数"]
    display_cols = [c for c in display_cols if c in horses.columns]
    print(horses[display_cols].sort_values("satsuki_rank").to_string(index=False))

    print("\n--- 特徴量重要度 ---")
    fi_text = feature_importance_text(model)
    print(fi_text)

    # HTML出力
    out_path = ROOT / "docs/satsuki_2026.html"
    html = generate_html(horses, fi_text)
    out_path.write_text(html, encoding="utf-8")
    print(f"\nHTML出力: {out_path}")


if __name__ == "__main__":
    main()

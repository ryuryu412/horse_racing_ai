#!/bin/bash
# ============================================================
# ROI週次更新スクリプト
# 使い方:
#   bash update_roi.sh             # 通常更新（Git push あり）
#   bash update_roi.sh --no-push   # push なし（動作確認用）
#
# 事前準備:
#   - data/tohyo/ に YYYYMMDD_tohyo.csv を配置
#   - data/raw/results/ に 結果CSV を配置
#   - data/raw/cache/ に .cache.pkl を配置（予想時点ROI用）
# ============================================================

set -e
cd "$(dirname "$0")"
PYTHON=".venv_new/Scripts/python.exe"
NO_PUSH=false
[[ "$1" == "--no-push" ]] && NO_PUSH=true

# all_tohyo.csv が空 / 存在しない場合は git から復元
TOHYO="data/tohyo/all_tohyo.csv"
TOHYO_LINES=$(wc -l < "$TOHYO" 2>/dev/null || echo 0)
if [ "$TOHYO_LINES" -le 1 ]; then
    echo "[復元] all_tohyo.csv が見つかりません。git から復元します..."
    git show 2397b9c:data/tohyo/all_tohyo.csv > "$TOHYO"
    echo "       復元完了（1/4〜3/29）"
fi

echo ""
echo "=== [1/4] 馬券データ マージ ==="
$PYTHON data/merge_tohyo.py

echo ""
echo "=== [2/4] 予測ROI HTML 更新（最終オッズ） ==="
$PYTHON src/_daily_roi_2026.py

echo ""
echo "=== [3/4] 予測ROI HTML 更新（予想時点オッズ） ==="
$PYTHON src/_predict_time_roi_2026.py

echo ""
echo "=== [4/4] 実馬券ROI HTML 更新 ==="
$PYTHON data/update_roi_html.py

echo ""
echo "=== Git コミット & プッシュ ==="
DATE=$(date '+%Y%m%d')
git add docs/daily_roi_2026.html docs/predict_time_roi_2026.html docs/actual_bet_roi.html
if git diff --cached --quiet; then
    echo "変更なし。コミットをスキップします。"
else
    git commit -m "ROI更新 $DATE

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    if [ "$NO_PUSH" = false ]; then
        git push origin main
        echo ""
        echo "=== 完了 ==="
        echo "GitHub Pages: https://keiba-dragon.github.io/horse_racing_ai/"
    else
        echo "(--no-push のため push をスキップ)"
    fi
fi

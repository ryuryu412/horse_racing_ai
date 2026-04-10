#!/bin/bash
cd "G:/マイドライブ/horse_racing_ai"
LOG="data/overnight_pipeline.log"
echo "=============================" >> $LOG
echo "Pipeline start: $(date)" >> $LOG
echo "=============================" >> $LOG

# グリッドサーチ完了を待つ
echo "[$(date)] グリッドサーチ完了待機..." >> $LOG
wait $(pgrep -f _grid_search.py) 2>/dev/null
echo "[$(date)] グリッドサーチ完了" >> $LOG

# 01 特徴量生成
echo "[$(date)] 01_make_features.py 開始..." >> $LOG
python src/01_make_features.py >> $LOG 2>&1
echo "[$(date)] 01完了 (exit $?)" >> $LOG

# 02 距離モデル学習
echo "[$(date)] 02_train_model.py 開始..." >> $LOG
python src/02_train_model.py >> $LOG 2>&1
echo "[$(date)] 02完了 (exit $?)" >> $LOG

# 07 距離ランカー学習
echo "[$(date)] 07_train_ranker.py 開始..." >> $LOG
python src/07_train_ranker.py >> $LOG 2>&1
echo "[$(date)] 07完了 (exit $?)" >> $LOG

# 09 クラスモデル学習
echo "[$(date)] 09_train_submodel.py 開始..." >> $LOG
python src/09_train_submodel.py >> $LOG 2>&1
echo "[$(date)] 09完了 (exit $?)" >> $LOG

# 11 クラスランカー学習
echo "[$(date)] 11_train_class_ranker.py 開始..." >> $LOG
python src/11_train_class_ranker.py >> $LOG 2>&1
echo "[$(date)] 11完了 (exit $?)" >> $LOG

# 新モデルでグリッドサーチ再実行
echo "[$(date)] 新モデル グリッドサーチ開始..." >> $LOG
python src/_grid_search.py > data/grid_search_results_v2.txt 2>&1
echo "[$(date)] グリッドサーチv2完了 (exit $?)" >> $LOG

echo "=============================" >> $LOG
echo "Pipeline 全完了: $(date)" >> $LOG
echo "=============================" >> $LOG

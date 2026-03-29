"""
ステップ1: データ分割のみ（メモリ解放してから学習ステップへ）
実行後すぐにPythonプロセスが終了するので、メモリが完全に解放される
"""
import sys, io, os, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
proc_dir  = os.path.join(base_dir, 'data', 'processed')
src_csv   = os.path.join(proc_dir, 'all_venues_features.csv')
train_csv = os.path.join(proc_dir, 'all_venues_features_2025train.csv')
test_csv  = os.path.join(proc_dir, 'all_venues_features_2026test.csv')

print("データ分割中...")
df = pd.read_csv(src_csv, low_memory=False)
df['日付_num'] = pd.to_numeric(df['日付_num'], errors='coerce')
train_df = df[df['日付_num'] <= 251231].copy()
test_df  = df[df['日付_num'] >= 260101].copy()
del df

train_df.to_csv(train_csv, index=False)
del train_df
test_df.to_csv(test_csv, index=False)
del test_df

print(f"完了:")
print(f"  学習用 → {os.path.basename(train_csv)}")
print(f"  テスト用 → {os.path.basename(test_csv)}")
print("\n次: python src/_train_2025_step2_train.py")

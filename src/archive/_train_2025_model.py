"""
ROI検証用モデル学習パイプライン
・学習データ: 2013〜2025年末
・テストデータ: 2026年1〜3月
・保存先: models_2025/ （現行 models/ は最後に完全復元）

実行順:
  python src/_train_2025_model.py
  python src/_validate_2026_roi.py
"""
import sys, io, os, subprocess, shutil, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
proc_dir  = os.path.join(base_dir, 'data', 'processed')
src_csv   = os.path.join(proc_dir, 'all_venues_features.csv')
model_dir = os.path.join(base_dir, 'models')
model_bak = os.path.join(base_dir, 'models_orig_backup')
model_2025 = os.path.join(base_dir, 'models_2025')
src_dir   = os.path.join(base_dir, 'src')

train_csv = os.path.join(proc_dir, 'all_venues_features_2025train.csv')
test_csv  = os.path.join(proc_dir, 'all_venues_features_2026test.csv')

# ── ① データ分割 ──
print("="*60)
print("① データ分割")
print("="*60)
print("読み込み中...")
df = pd.read_csv(src_csv, low_memory=False)
df['日付_num'] = pd.to_numeric(df['日付_num'], errors='coerce')
train_df = df[df['日付_num'] <= 251231].copy()
test_df  = df[df['日付_num'] >= 260101].copy()
train_df.to_csv(train_csv, index=False)
test_df.to_csv(test_csv,   index=False)
print(f"学習用: {len(train_df):,}行 (〜2025年末)")
print(f"テスト用: {len(test_df):,}行 (2026年1〜3月)")

# ── ② 現行モデル・データを完全バックアップ ──
print("\n" + "="*60)
print("② 現行モデルをバックアップ")
print("="*60)
if os.path.exists(model_bak):
    shutil.rmtree(model_bak)
shutil.copytree(model_dir, model_bak)
print(f"  models/ → models_orig_backup/ にコピー完了")

# 学習用データに差し替え
csv_bak = src_csv + '.orig_bak'
shutil.copy2(src_csv, csv_bak)
shutil.copy2(train_csv, src_csv)
print(f"  all_venues_features.csv を2025年学習用に差し替え")

# ── ③ 学習実行 ──
def run(name, script):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    r = subprocess.run([sys.executable, script], capture_output=True, text=True, encoding='utf-8')
    elapsed = time.time() - t0
    lines = (r.stdout + r.stderr).strip().split('\n')
    for line in lines[-8:]:
        if line.strip(): print(' ', line)
    status = "完了" if r.returncode == 0 else "エラー"
    print(f"  → {status} ({elapsed:.0f}秒)")
    return r.returncode == 0

scripts = [
    ("02: 距離モデル",   '02_train_model.py'),
    ("07: 距離ランカー", '07_train_ranker.py'),
    ("09: クラスモデル", '09_train_submodel.py'),
    ("11: クラスランカー",'11_train_class_ranker.py'),
]

all_ok = True
for name, fname in scripts:
    ok = run(name, os.path.join(src_dir, fname))
    if not ok:
        all_ok = False
        print(f"  !! {name} が失敗しました。処理を中断します。")
        break

# ── ④ 2025モデルを保存 ──
if all_ok:
    print("\n" + "="*60)
    print("④ 2025年モデルを models_2025/ に保存")
    print("="*60)
    if os.path.exists(model_2025):
        shutil.rmtree(model_2025)
    shutil.copytree(model_dir, model_2025)
    print(f"  models/ → models_2025/ にコピー完了")

# ── ⑤ 現行モデル・データを完全復元 ──
print("\n" + "="*60)
print("⑤ 現行モデル・データを復元")
print("="*60)
shutil.rmtree(model_dir)
shutil.copytree(model_bak, model_dir)
shutil.rmtree(model_bak)
print("  models/ を復元完了")

shutil.copy2(csv_bak, src_csv)
os.remove(csv_bak)
print("  all_venues_features.csv を復元完了")

if all_ok:
    print("\n✓ 完了！models_2025/ に2025年までのモデルが保存されました。")
    print("次のステップ: python src/_validate_2026_roi.py")
else:
    print("\n✗ 学習途中でエラーが発生しました。現行モデルは復元済みです。")

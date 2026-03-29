"""
ステップ2: 2025年モデル学習＆復元
前提: _train_2025_step1_split.py を先に実行済みであること
"""
import sys, io, os, subprocess, shutil, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

base_dir   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
proc_dir   = os.path.join(base_dir, 'data', 'processed')
src_csv    = os.path.join(proc_dir, 'all_venues_features.csv')
train_csv  = os.path.join(proc_dir, 'all_venues_features_2025train.csv')
model_dir  = os.path.join(base_dir, 'models')
model_bak  = os.path.join(base_dir, 'models_orig_backup')
model_2025 = os.path.join(base_dir, 'models_2025')
src_dir    = os.path.join(base_dir, 'src')

# 確認
if not os.path.exists(train_csv):
    print("エラー: 先に _train_2025_step1_split.py を実行してください")
    sys.exit(1)

# ── 現行モデルをバックアップ ──
print("現行モデルをバックアップ中...")
if os.path.exists(model_bak): shutil.rmtree(model_bak)
shutil.copytree(model_dir, model_bak)
print("  models/ → models_orig_backup/ 完了")

# ── 学習用CSVに差し替え ──
csv_bak = src_csv + '.orig_bak'
shutil.copy2(src_csv, csv_bak)
shutil.copy2(train_csv, src_csv)
print("  all_venues_features.csv を2025年学習用に差し替え")

def run(name, script):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    t0 = time.time()
    r = subprocess.run([sys.executable, script], capture_output=True, text=True, encoding='utf-8')
    elapsed = time.time() - t0
    lines = (r.stdout + r.stderr).strip().split('\n')
    for line in lines[-8:]:
        if line.strip(): print(' ', line)
    status = "完了" if r.returncode == 0 else f"エラー(code={r.returncode})"
    print(f"  → {status} ({elapsed:.0f}秒)")
    return r.returncode == 0

scripts = [
    ("02: 距離モデル",    '02_train_model.py'),
    ("07: 距離ランカー",  '07_train_ranker.py'),
    ("09: クラスモデル",  '09_train_submodel.py'),
    ("11: クラスランカー",'11_train_class_ranker.py'),
]

all_ok = True
for name, fname in scripts:
    ok = run(name, os.path.join(src_dir, fname))
    if not ok:
        all_ok = False
        print(f"\n!! {name} が失敗。以降スキップして復元します。")
        break

# ── 2025モデルを保存 ──
if all_ok:
    print(f"\n{'='*55}")
    print("  2025モデルを models_2025/ に保存")
    print(f"{'='*55}")
    if os.path.exists(model_2025): shutil.rmtree(model_2025)
    shutil.copytree(model_dir, model_2025)
    print("  完了")

# ── 必ず復元 ──
print(f"\n{'='*55}")
print("  現行モデル・データを復元")
print(f"{'='*55}")
shutil.rmtree(model_dir)
shutil.copytree(model_bak, model_dir)
shutil.rmtree(model_bak)
print("  models/ 復元完了")
shutil.copy2(csv_bak, src_csv)
os.remove(csv_bak)
print("  all_venues_features.csv 復元完了")

if all_ok:
    print("\n✓ 完了！models_2025/ に2025年モデルが保存されました。")
    print("次: python src/_validate_2026_roi.py")
else:
    print("\n✗ エラーがありましたが現行モデルは復元済みです。")

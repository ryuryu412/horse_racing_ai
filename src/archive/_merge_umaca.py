import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import pandas as pd, os, re

def read_csv_auto(path):
    for enc in ['utf-8-sig', 'utf-8', 'cp932']:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except:
            continue
    raise ValueError(f"読み込み失敗: {path}")

def conv_date(s):
    m = re.search(r'(\d+)月(\d+)日', str(s))
    if m: return int(f'2026{int(m.group(1)):02d}{int(m.group(2)):02d}')
    return None

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_dir = os.path.join(base_dir, 'data', 'tohyo')

umaca_path = 'G:/マイドライブ/keiiba2026/2026年1月/2026即バット以外.csv'
df_umaca = read_csv_auto(umaca_path)
df_umaca = df_umaca[df_umaca['通番'].astype(str) != '合計'].copy()
df_umaca['日付'] = df_umaca['日付'].apply(conv_date)
df_umaca['返還金額'] = 0
if '購入場所' in df_umaca.columns:
    df_umaca = df_umaca.drop(columns=['購入場所'])

for date, grp in df_umaca.groupby('日付'):
    fname = os.path.join(out_dir, f'{int(date)}_tohyo.csv')
    if os.path.exists(fname):
        existing = read_csv_auto(fname)
        merged = pd.concat([existing, grp], ignore_index=True)
        merged.to_csv(fname, index=False, encoding='utf-8-sig')
        print(f'{int(date)}: {len(existing)}行 + {len(grp)}行(UMACA) → {len(merged)}行')
    else:
        grp.to_csv(fname, index=False, encoding='utf-8-sig')
        print(f'{int(date)}: 新規 {len(grp)}行')

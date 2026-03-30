import pandas as pd
import glob, os, re, shutil

def parse_amount(val):
    s = str(val).strip()
    if s in ('', 'nan', '―', '-', '×', '\\', '¥', '＼'):
        return 0
    if '／' in s or '/' in s:
        parts = re.split(r'[／/]', s)
        for p in reversed(parts):
            p = p.strip().replace(',', '')
            if re.match(r'^\d+$', p):
                return int(p)
        return 0
    try:
        return float(str(s).replace(',', ''))
    except:
        return 0

tohyo_dir = 'C:/Users/tsuch/Desktop/horse_racing_ai/data/tohyo'
archive_dir = os.path.join(tohyo_dir, 'archive')
os.makedirs(archive_dir, exist_ok=True)

# 日付別CSVを読み込んで結合（all_tohyo.csv は除外）
files = sorted([
    f for f in glob.glob(os.path.join(tohyo_dir, '*_tohyo.csv'))
    if os.path.basename(f) != 'all_tohyo.csv'
])
dfs = []
for f in files:
    fname = os.path.basename(f)
    if fname == 'video_tohyo_data.csv':
        continue
    for enc in ['cp932', 'utf-8-sig', 'utf-8']:
        try:
            df = pd.read_csv(f, encoding=enc)
            dfs.append(df)
            break
        except:
            pass

all_df = pd.concat(dfs, ignore_index=True)

# 合計行除外
all_df = all_df[all_df['日付'].notna()].copy()

# 金額を数値に統一
for col in ['購入金額', '払戻金額', '返還金額']:
    all_df[col] = all_df[col].apply(parse_amount)

# 日付を整数に統一
all_df['日付'] = all_df['日付'].apply(
    lambda x: int(float(str(x))) if str(x).replace('.','').isdigit() else None)
all_df = all_df[all_df['日付'].notna()].copy()
all_df['日付'] = all_df['日付'].astype(int)

# 日付・受付番号・通番でソート
all_df = all_df.sort_values(['日付', '受付番号', '通番']).reset_index(drop=True)

# 保存
out_path = os.path.join(tohyo_dir, 'all_tohyo.csv')
all_df.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f'保存: {out_path}  ({len(all_df)}行)')

# アーカイブ：日付別CSV をアーカイブへ移動（all_tohyo.csv は移動しない）
archived = []
for f in glob.glob(os.path.join(tohyo_dir, '*_tohyo.csv')):
    fname = os.path.basename(f)
    if fname == 'all_tohyo.csv':
        continue
    dest = os.path.join(archive_dir, fname)
    shutil.move(f, dest)
    archived.append(fname)

print(f'\nアーカイブ済み ({len(archived)}件):')
for name in archived:
    print(f'  {name}')
print(f'\nアーカイブ先: {archive_dir}')

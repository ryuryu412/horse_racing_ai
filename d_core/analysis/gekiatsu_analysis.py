# coding: utf-8
"""激熱条件探索 - cur_cs × sub_cs × gap_ratio"""
import sys, io, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from d_metric import load_models, calc_d
from config import PARQUET_PATH, PAY_CSV_PATH

print("モデル読み込み中...")
cur_info, sub_info = load_models()
print("parquet 読み込み中...")
raw = pd.read_parquet(PARQUET_PATH)
dnum_col = '日付_num' if '日付_num' in raw.columns else '日付'
raw['_dnum'] = pd.to_numeric(raw[dnum_col], errors='coerce')
raw = raw[raw['_dnum'] >= 230715].reset_index(drop=True)
print("D指標計算中...")
raw = calc_d(raw, cur_info, sub_info)

def zen_to_num(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    m = re.search(r'\d+', s)
    return int(m.group()) if m else np.nan

try:    pay = pd.read_csv(PAY_CSV_PATH, encoding='cp932', low_memory=False)
except: pay = pd.read_csv(PAY_CSV_PATH, encoding='utf-8',  low_memory=False)

pay['_dnum'] = pd.to_numeric(pay['日付'], errors='coerce')
pay['_tan']  = pd.to_numeric(pay['単勝配当'], errors='coerce')
pay['_fuku'] = pd.to_numeric(pay['複勝配当'], errors='coerce')
pay['_atch'] = pay['着順'].apply(zen_to_num)
pay = pay[pay['_dnum'] >= 230715].copy()
pay['_R'] = pd.to_numeric(pay['Ｒ'], errors='coerce')

win_rows = pay[pay['_atch'] == 1].drop_duplicates(['_dnum', '開催', '_R']).copy()
win_rows['_rk'] = win_rows['_dnum'].astype(str) + '_' + win_rows['開催'].astype(str) + '_' + win_rows['_R'].astype(str)
tan_map = win_rows.set_index('_rk')['_tan'].to_dict()

pay['_rk']      = pay['_dnum'].astype(str) + '_' + pay['開催'].astype(str) + '_' + pay['_R'].astype(str)
pay['_join_key'] = pay['_rk'] + '_' + pay['馬名S'].astype(str)
fuku_map = pay.set_index('_join_key')['_fuku'].to_dict()
atch_map  = pay.set_index('_join_key')['_atch'].to_dict()

raw['_pay_rk']   = raw['_dnum'].astype(str) + '_' + raw['開催'].astype(str) + '_' + pd.to_numeric(raw['Ｒ'], errors='coerce').astype(str)
raw['_join_key'] = raw['_pay_rk'] + '_' + raw['馬名S'].astype(str)
raw['_atch'] = raw['_join_key'].map(atch_map)
raw['_fuku'] = raw['_join_key'].map(fuku_map)
raw['_tan']  = raw['_pay_rk'].map(tan_map)
raw['odds']  = pd.to_numeric(raw['単勝オッズ'], errors='coerce')

df = raw[raw['D'].notna() & raw['_atch'].notna()].copy()
df['D_rank'] = df.groupby('_pay_rk')['D'].rank(ascending=False, method='min')
df['D_mean'] = df.groupby('_pay_rk')['D'].transform('mean').clip(lower=1)
df['D_pct']  = (df['D'] - df['D_mean']) / df['D_mean'] * 100

def get_gap(g):
    g2 = g.sort_values('D', ascending=False).reset_index(drop=True)
    d1 = g2.iloc[0]['D'] if len(g2) >= 1 else np.nan
    d2 = g2.iloc[1]['D'] if len(g2) >= 2 else np.nan
    return pd.Series({'gap_ratio': d1/d2 if pd.notna(d2) and d2 > 0 else np.nan})

gap_df = df.groupby('_pay_rk').apply(get_gap)
top1 = df[df['D_rank'] == 1].copy()
top1 = top1.merge(gap_df, left_on='_pay_rk', right_index=True, how='left')
print(f"D1位馬: {len(top1)}頭\n")

def show(sub, label):
    n = len(sub)
    if n < 30: return
    wr = sub['_atch'].eq(1).mean()
    pr = (sub['_atch'] <= 3).mean()
    sub_t = sub.dropna(subset=['_tan'])
    roi_t = (sub_t[sub_t['_atch']==1]['_tan'].sum()/100 - len(sub_t)) / len(sub_t) if len(sub_t) > 0 else np.nan
    placed = sub[sub['_atch'] <= 3].dropna(subset=['_fuku'])
    roi_f = (placed['_fuku'].sum()/100 - n) / n if n > 0 else np.nan
    rt = f"{roi_t:>+8.1%}" if pd.notna(roi_t) else "       -"
    rf = f"{roi_f:>+8.1%}" if pd.notna(roi_f) else "       -"
    print(f"  {label:<40}  {n:>5}  {wr:>6.1%}  {pr:>6.1%}  {rt}  {rf}")

hdr = f"  {'条件':<40}  {'N':>5}  {'勝率':>6}  {'複勝率':>6}  {'単ROI':>8}  {'複ROI':>8}"
sep = "  " + "-" * 82

# ── cur_cs 分布確認 ──
print("cur_cs 分布（D1位馬）:")
for p in [50, 60, 70, 75, 80, 90]:
    n = (top1['cur_cs'] >= p).sum()
    print(f"  cur_cs >= {p}: {n}頭 ({n/len(top1):.1%})")

print("\nsub_cs 分布（D1位馬）:")
for p in [50, 60, 70, 75, 80, 90]:
    n = (top1['sub_cs'] >= p).sum()
    print(f"  sub_cs >= {p}: {n}頭 ({n/len(top1):.1%})")

# ── cur_cs 単体 ──
print(f"\n{'='*84}")
print("【cur_cs 閾値別】"); print(hdr); print(sep)
for thr in [55, 60, 65, 70, 75]:
    show(top1[top1['cur_cs'] >= thr], f"cur_cs≥{thr}")

# ── sub_cs 単体 ──
print(f"\n{'='*84}")
print("【sub_cs 閾値別】"); print(hdr); print(sep)
for thr in [55, 60, 65, 70, 75]:
    show(top1[top1['sub_cs'] >= thr], f"sub_cs≥{thr}")

# ── cur_cs × sub_cs ──
print(f"\n{'='*84}")
print("【cur_cs × sub_cs 組み合わせ】"); print(hdr); print(sep)
for c in [60, 65, 70]:
    for s in [60, 65, 70]:
        show(top1[(top1['cur_cs'] >= c) & (top1['sub_cs'] >= s)],
             f"cur_cs≥{c} & sub_cs≥{s}")
    print(sep)

# ── cur_cs × sub_cs × gap ──
print(f"\n{'='*84}")
print("【cur_cs × sub_cs × gap_ratio 三重条件】"); print(hdr); print(sep)
for c in [60, 65]:
    for s in [60, 65]:
        for g in [2, 3, 5]:
            show(top1[(top1['cur_cs'] >= c) & (top1['sub_cs'] >= s) & (top1['gap_ratio'] >= g)],
                 f"cur≥{c} & sub≥{s} & gap≥{g}x")
        print(sep)

# ── 激熱候補まとめ（N≥100かつ両ROI高い） ──
print(f"\n{'='*84}")
print("【激熱候補 - cur_cs×sub_cs×gap 全組み合わせ ROI降順】"); print(hdr); print(sep)

results = []
for c in [55, 60, 65, 70]:
    for s in [55, 60, 65, 70]:
        for g in [1.5, 2, 3, 5]:
            sub = top1[(top1['cur_cs'] >= c) & (top1['sub_cs'] >= s) & (top1['gap_ratio'] >= g)]
            n = len(sub)
            if n < 100: continue
            sub_t = sub.dropna(subset=['_tan'])
            roi_t = (sub_t[sub_t['_atch']==1]['_tan'].sum()/100 - len(sub_t)) / len(sub_t) if len(sub_t) > 0 else np.nan
            placed = sub[sub['_atch'] <= 3].dropna(subset=['_fuku'])
            roi_f = (placed['_fuku'].sum()/100 - n) / n if n > 0 else np.nan
            results.append({'cur': c, 'sub': s, 'gap': g, 'N': n, '単ROI': roi_t, '複ROI': roi_f,
                            'wr': sub['_atch'].eq(1).mean(), 'pr': (sub['_atch']<=3).mean()})

df_r = pd.DataFrame(results).sort_values('単ROI', ascending=False).head(20)
for _, row in df_r.iterrows():
    label = f"cur≥{int(row['cur'])} & sub≥{int(row['sub'])} & gap≥{row['gap']}x"
    rt = f"{row['単ROI']:>+8.1%}" if pd.notna(row['単ROI']) else "       -"
    rf = f"{row['複ROI']:>+8.1%}" if pd.notna(row['複ROI']) else "       -"
    print(f"  {label:<40}  {int(row['N']):>5}  {row['wr']:>6.1%}  {row['pr']:>6.1%}  {rt}  {rf}")

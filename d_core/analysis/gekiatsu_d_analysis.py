# coding: utf-8
"""激熱条件探索 - D値ベース（D_pct × gap_ratio）"""
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

pay['_rk']       = pay['_dnum'].astype(str) + '_' + pay['開催'].astype(str) + '_' + pay['_R'].astype(str)
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
df['_n_qual'] = df.groupby('_pay_rk')['D_pct'].transform(lambda x: (x > 200).sum())

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
    print(f"  {label:<38}  {n:>5}  {wr:>6.1%}  {pr:>6.1%}  {rt}  {rf}")

hdr = f"  {'条件':<38}  {'N':>5}  {'勝率':>6}  {'複勝率':>6}  {'単ROI':>8}  {'複ROI':>8}"
sep = "  " + "-" * 80

# ── D_pct 分布確認 ──
print("D_pct 分布（D1位馬）:")
for p in [200, 300, 500, 1000, 2000, 5000]:
    n = (top1['D_pct'] > p).sum()
    print(f"  D_pct > {p:>5}%: {n:>5}頭 ({n/len(top1):.1%})")

print("\ngap_ratio 分布（D1位馬）:")
for g in [2, 3, 5, 8, 10, 20]:
    n = (top1['gap_ratio'] > g).sum()
    print(f"  gap_ratio > {g:>2}x: {n:>5}頭 ({n/len(top1):.1%})")

# ── D_pct 単体 ──
print(f"\n{'='*82}")
print("【D_pct 閾値別（D1位が平均からどれだけ離れているか）】")
print(hdr); print(sep)
for thr in [200, 300, 500, 1000, 2000]:
    show(top1[top1['D_pct'] > thr], f"D_pct>{thr}%")

# ── gap_ratio 単体 ──
print(f"\n{'='*82}")
print("【gap_ratio 閾値別（D1/D2の倍率）】")
print(hdr); print(sep)
for g in [3, 5, 8, 10, 20]:
    show(top1[top1['gap_ratio'] > g], f"gap_ratio>{g}x")

# ── D_pct × gap_ratio ──
print(f"\n{'='*82}")
print("【D_pct × gap_ratio 組み合わせ】")
print(hdr); print(sep)
for pct in [300, 500, 1000]:
    for g in [3, 5, 8, 10]:
        show(top1[(top1['D_pct'] > pct) & (top1['gap_ratio'] > g)],
             f"D_pct>{pct}% & gap>{g}x")
    print(sep)

# ── 1頭抜け × D_pct ──
print(f"\n{'='*82}")
print("【1頭抜け（n_qual==1）× D_pct】")
print(hdr); print(sep)
for pct in [200, 300, 500, 1000]:
    show(top1[(top1['D_pct'] > pct) & (top1['_n_qual'] == 1)],
         f"1頭抜け & D_pct>{pct}%")

# ── 全グリッド N≥100 ROI降順 ──
print(f"\n{'='*82}")
print("【全グリッド N≥100 単ROI降順 Top20】")
print(hdr); print(sep)
results = []
for pct in [200, 300, 500, 1000, 2000]:
    for g in [1, 2, 3, 5, 8, 10]:
        sub = top1[(top1['D_pct'] > pct) & (top1['gap_ratio'] > g)]
        n = len(sub)
        if n < 100: continue
        sub_t = sub.dropna(subset=['_tan'])
        roi_t = (sub_t[sub_t['_atch']==1]['_tan'].sum()/100 - len(sub_t)) / len(sub_t) if len(sub_t) > 0 else np.nan
        placed = sub[sub['_atch'] <= 3].dropna(subset=['_fuku'])
        roi_f = (placed['_fuku'].sum()/100 - n) / n if n > 0 else np.nan
        results.append({'D_pct': pct, 'gap': g, 'N': n, '単ROI': roi_t, '複ROI': roi_f,
                        'wr': sub['_atch'].eq(1).mean(), 'pr': (sub['_atch']<=3).mean()})

df_r = pd.DataFrame(results).sort_values('単ROI', ascending=False).head(20)
for _, row in df_r.iterrows():
    label = f"D_pct>{int(row['D_pct'])}% & gap>{row['gap']}x"
    rt = f"{row['単ROI']:>+8.1%}" if pd.notna(row['単ROI']) else "       -"
    rf = f"{row['複ROI']:>+8.1%}" if pd.notna(row['複ROI']) else "       -"
    print(f"  {label:<38}  {int(row['N']):>5}  {row['wr']:>6.1%}  {row['pr']:>6.1%}  {rt}  {rf}")

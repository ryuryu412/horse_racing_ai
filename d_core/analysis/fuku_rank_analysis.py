# coding: utf-8
"""D2位・D3位の複勝ROI分析"""
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
pay['_fuku'] = pd.to_numeric(pay['複勝配当'], errors='coerce')
pay['_atch'] = pay['着順'].apply(zen_to_num)
pay = pay[pay['_dnum'] >= 230715].copy()
pay['_R'] = pd.to_numeric(pay['Ｒ'], errors='coerce')
pay['_rk'] = pay['_dnum'].astype(str) + '_' + pay['開催'].astype(str) + '_' + pay['_R'].astype(str)
pay['_join_key'] = pay['_rk'] + '_' + pay['馬名S'].astype(str)
fuku_map = pay.set_index('_join_key')['_fuku'].to_dict()
atch_map  = pay.set_index('_join_key')['_atch'].to_dict()

raw['_pay_rk']   = raw['_dnum'].astype(str) + '_' + raw['開催'].astype(str) + '_' + pd.to_numeric(raw['Ｒ'], errors='coerce').astype(str)
raw['_join_key'] = raw['_pay_rk'] + '_' + raw['馬名S'].astype(str)
raw['_atch'] = raw['_join_key'].map(atch_map)
raw['_fuku'] = raw['_join_key'].map(fuku_map)
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
df = df.merge(gap_df, left_on='_pay_rk', right_index=True, how='left')

print(f"分析対象: {len(df)}頭 / {df['_pay_rk'].nunique()}レース\n")

def show_fuku(sub, label):
    n = len(sub)
    if n < 30: return
    pr = (sub['_atch'] <= 3).mean()
    placed = sub[sub['_atch'] <= 3].dropna(subset=['_fuku'])
    roi_f = (placed['_fuku'].sum()/100 - n) / n if n > 0 else np.nan
    ao = sub['odds'].mean()
    rf = f"{roi_f:>+8.1%}" if pd.notna(roi_f) else "       -"
    print(f"  {label:<35}  {n:>6}  {pr:>7.1%}  {rf}  {ao:>6.1f}倍")

hdr = f"  {'条件':<35}  {'N':>6}  {'複勝率':>7}  {'複ROI':>8}  {'平均OD':>6}"
sep = "  " + "-" * 72

# ── D順位別ベースライン ──
print("=" * 80)
print("【D順位別ベースライン】")
print(hdr); print(sep)
for rank in [1, 2, 3, 4, 5]:
    show_fuku(df[df['D_rank'] == rank], f"D{rank}位（全体）")

# ── D2位・D3位 × OD条件 ──
print()
print("【D2位 × OD条件】")
print(hdr); print(sep)
d2 = df[df['D_rank'] == 2]
for od in [3, 4, 5, 6, 8]:
    show_fuku(d2[d2['odds'] > od], f"D2位 & OD>{od}")

print()
print("【D3位 × OD条件】")
print(hdr); print(sep)
d3 = df[df['D_rank'] == 3]
for od in [3, 4, 5, 6, 8]:
    show_fuku(d3[d3['odds'] > od], f"D3位 & OD>{od}")

# ── D2位・D3位 × D_pct条件 ──
print()
print("【D2位 × D_pct条件（平均からの距離）】")
print(hdr); print(sep)
for thr in [0, 50, 100, 200]:
    show_fuku(d2[d2['D_pct'] > thr], f"D2位 & D_pct>{thr}%")
for thr in [-50, -70, -90]:
    show_fuku(d2[d2['D_pct'] < thr], f"D2位 & D_pct<{thr}%（消し候補）")

print()
print("【D3位 × D_pct条件】")
print(hdr); print(sep)
for thr in [0, 50, 100]:
    show_fuku(d3[d3['D_pct'] > thr], f"D3位 & D_pct>{thr}%")
for thr in [-50, -70, -90]:
    show_fuku(d3[d3['D_pct'] < thr], f"D3位 & D_pct<{thr}%（消し候補）")

# ── D2位・D3位 × OD × D_pct組み合わせ ──
print()
print("【D2位 × OD × D_pct組み合わせ】")
print(hdr); print(sep)
for od in [4, 5, 6]:
    for pct in [50, 100, 200]:
        show_fuku(d2[(d2['odds'] > od) & (d2['D_pct'] > pct)], f"D2位 & OD>{od} & D_pct>{pct}%")

print()
print("【D3位 × OD × D_pct組み合わせ】")
print(hdr); print(sep)
for od in [4, 5, 6]:
    for pct in [50, 100]:
        show_fuku(d3[(d3['odds'] > od) & (d3['D_pct'] > pct)], f"D3位 & OD>{od} & D_pct>{pct}%")

# ── gap × D2・D3 ──
print()
print("【gap≥3x レースのD2位・D3位】")
print(hdr); print(sep)
show_fuku(d2[d2['gap_ratio'] >= 3], "D2位 & gap≥3x（D1が飛び抜けてる）")
show_fuku(d3[d3['gap_ratio'] >= 3], "D3位 & gap≥3x")
show_fuku(d2[d2['gap_ratio'] < 3],  "D2位 & gap<3x（接戦レース）")
show_fuku(d3[d3['gap_ratio'] < 3],  "D3位 & gap<3x（接戦レース）")

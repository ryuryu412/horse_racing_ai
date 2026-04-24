# coding: utf-8
"""D指標 ROI分析（parquet × 配当マスターCSV）"""
import sys, io, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from d_metric import load_models, calc_d, add_gap
from config import PARQUET_PATH, PAY_CSV_PATH

# ── データ読み込み ──
print("モデル読み込み中...")
cur_info, sub_info = load_models()

print("parquet 読み込み中...")
raw = pd.read_parquet(PARQUET_PATH)
dnum_col = '日付_num' if '日付_num' in raw.columns else '日付'
raw['_dnum'] = pd.to_numeric(raw[dnum_col], errors='coerce')
raw = raw[raw['_dnum'] >= 230715].reset_index(drop=True)
print(f"テストデータ: {len(raw)}行")

print("D指標計算中...")
raw = calc_d(raw, cur_info, sub_info)
print(f"D指標: {raw['D'].notna().sum()}頭 / {raw[raw['D'].notna()]['race_key'].nunique()}レース")

# ── 配当マスター読み込み ──
print("\n配当マスター読み込み中...")
def zen_to_num(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    m = re.search(r'\d+', s)
    return int(m.group()) if m else np.nan

try:    pay = pd.read_csv(PAY_CSV_PATH, encoding='cp932', low_memory=False)
except: pay = pd.read_csv(PAY_CSV_PATH, encoding='utf-8',  low_memory=False)

pay['_dnum']  = pd.to_numeric(pay['日付'], errors='coerce')
pay['_tan']   = pd.to_numeric(pay['単勝配当'], errors='coerce')
pay['_fuku']  = pd.to_numeric(pay['複勝配当'], errors='coerce')
pay['_atch']  = pay['着順'].apply(zen_to_num)
pay['_ninki'] = pay['人気'].apply(zen_to_num) if '人気' in pay.columns else np.nan
pay = pay[pay['_dnum'] >= 230715].copy()
pay['_R'] = pd.to_numeric(pay['Ｒ'], errors='coerce')
print(f"配当データ（2023-07〜）: {len(pay)}行 / {pay['_dnum'].nunique()}日")

# レース単位の単勝配当マップ（1着馬から）
win_rows = pay[pay['_atch'] == 1].drop_duplicates(['_dnum', '開催', '_R']).copy()
win_rows['_rk'] = win_rows['_dnum'].astype(str) + '_' + win_rows['開催'].astype(str) + '_' + win_rows['_R'].astype(str)
tan_map = win_rows.set_index('_rk')['_tan'].to_dict()

pay['_rk']      = pay['_dnum'].astype(str) + '_' + pay['開催'].astype(str) + '_' + pay['_R'].astype(str)
pay['_join_key'] = pay['_rk'] + '_' + pay['馬名S'].astype(str)
fuku_map  = pay.set_index('_join_key')['_fuku'].to_dict()
atch_map  = pay.set_index('_join_key')['_atch'].to_dict()
ninki_map = pay.set_index('_join_key')['_ninki'].to_dict()

# ── parquetと配当をマッチング ──
raw['_pay_rk']   = raw['_dnum'].astype(str) + '_' + raw['開催'].astype(str) + '_' + pd.to_numeric(raw['Ｒ'], errors='coerce').astype(str)
raw['_join_key'] = raw['_pay_rk'] + '_' + raw['馬名S'].astype(str)
raw['_atch']  = raw['_join_key'].map(atch_map)
raw['_fuku']  = raw['_join_key'].map(fuku_map)
raw['_tan']   = raw['_pay_rk'].map(tan_map)
raw['_ninki'] = raw['_join_key'].map(ninki_map)
raw['odds']   = pd.to_numeric(raw['単勝オッズ'], errors='coerce')

matched = raw['_atch'].notna().sum()
print(f"着順マッチング: {matched}行 ({matched/len(raw):.1%})")

# ── 分析用データ作成 ──
df = raw[raw['D'].notna() & raw['_atch'].notna()].copy()
df['D_rank']  = df.groupby('_pay_rk')['D'].rank(ascending=False, method='min')
df['D_mean']  = df.groupby('_pay_rk')['D'].transform('mean').clip(lower=1)
df['D_pct']   = (df['D'] - df['D_mean']) / df['D_mean'] * 100
df['_n_qual'] = df.groupby('_pay_rk')['D_pct'].transform(lambda x: (x > 200).sum())

def get_gap(g):
    g2 = g.sort_values('D', ascending=False).reset_index(drop=True)
    d1 = g2.iloc[0]['D'] if len(g2) >= 1 else np.nan
    d2 = g2.iloc[1]['D'] if len(g2) >= 2 else np.nan
    return pd.Series({
        'gap_ratio': d1/d2 if pd.notna(d2) and d2 > 0 else np.nan,
        'gap_pct':   (d1-d2)/d2*100 if pd.notna(d2) and d2 > 0 else np.nan,
    })

print("gap計算中...")
gap_df = df.groupby('_pay_rk').apply(get_gap)
top1 = df[df['D_rank'] == 1].copy()
top1 = top1.merge(gap_df, left_on='_pay_rk', right_index=True, how='left')

n_races = df['_pay_rk'].nunique()
print(f"\n分析対象: {len(df)}頭 / {n_races}レース / D1位馬: {len(top1)}頭\n")

# ── 1番人気ベースライン ──
fav = df.sort_values('_ninki').groupby('_pay_rk').first()
def _roi_t(sub): return (sub[sub['_atch']==1]['_tan'].sum()/100 - len(sub)) / len(sub) if len(sub) > 0 else np.nan
def _roi_f(sub, n): return (sub[sub['_atch']<=3]['_fuku'].sum()/100 - n) / n if n > 0 else np.nan
fav_t = fav.dropna(subset=['_tan'])
fav_p = fav[fav['_atch'] <= 3].dropna(subset=['_fuku'])
print(f"1番人気: 勝率{fav['_atch'].eq(1).mean():.1%} / 複勝率{(fav['_atch']<=3).mean():.1%} / "
      f"単勝ROI{_roi_t(fav_t):+.1%} / 複勝ROI{_roi_f(fav_p, len(fav)):+.1%}\n")

# ── 表示関数 ──
def show(sub, label):
    n = len(sub)
    if n < 10: return
    wr = sub['_atch'].eq(1).mean()
    pr = (sub['_atch'] <= 3).mean()
    sub_t = sub.dropna(subset=['_tan'])
    roi_t = (sub_t[sub_t['_atch']==1]['_tan'].sum()/100 - len(sub_t)) / len(sub_t) if len(sub_t) > 0 else np.nan
    placed = sub[sub['_atch'] <= 3].dropna(subset=['_fuku'])
    roi_f  = (placed['_fuku'].sum()/100 - n) / n if n > 0 else np.nan
    ao     = sub['odds'].mean() if 'odds' in sub.columns else np.nan
    rt = f"{roi_t:>+7.1%}" if pd.notna(roi_t) else "      -"
    rf = f"{roi_f:>+7.1%}" if pd.notna(roi_f) else "      -"
    aod = f"{ao:>6.1f}倍" if pd.notna(ao) else "     -"
    print(f"  {label:<30}  {n:>6}  {wr:>7.1%}  {pr:>7.1%}  {rt}  {rf}  {aod}")

hdr = f"  {'条件':<30}  {'N':>6}  {'勝率':>7}  {'複勝率':>7}  {'単勝ROI':>7}  {'複勝ROI':>7}  {'平均OD':>6}"
sep = "  " + "-" * 84

# ── gap_pct ──
print("── gap_pct（D1-D2の%差）閾値別 ──"); print(hdr); print(sep)
for thr in [0, 50, 100, 200, 300, 500, 1000]:
    show(top1[top1['gap_pct'] > thr], f"gap_pct>{thr}%")

print()
print("── gap_pct × OD>6 ──"); print(hdr); print(sep)
for thr in [0, 50, 100, 200, 300, 500]:
    show(top1[(top1['gap_pct'] > thr) & (top1['odds'] > 6)], f"gap>{thr}% & OD>6")

# ── gap_ratio ──
print()
print("── gap_ratio（D1/D2倍率）閾値別 ──"); print(hdr); print(sep)
for thr in [1.5, 2, 3, 5, 8, 10]:
    show(top1[top1['gap_ratio'] > thr], f"ratio>{thr}x")

print()
print("── gap_ratio × OD>6 ──"); print(hdr); print(sep)
for thr in [1.5, 2, 3, 5, 8]:
    show(top1[(top1['gap_ratio'] > thr) & (top1['odds'] > 6)], f"ratio>{thr}x & OD>6")

# ── D_pct × 1頭抜け ──
print()
print("── D_pct>200% × 1頭抜け vs 複数抜け ──"); print(hdr); print(sep)
one_q   = df[(df['D_pct'] > 200) & (df['_n_qual'] == 1)]
multi_q = df[(df['D_pct'] > 200) & (df['_n_qual'] >= 2)]
multi_t1= df[(df['D_pct'] > 200) & (df['_n_qual'] >= 2) & (df['D_rank'] == 1)]
show(one_q,   "D_pct>200% 1頭のみ")
show(multi_q, "D_pct>200% 2頭以上（全員）")
show(multi_t1,"D_pct>200% 2頭以上のD1位")

print()
print("── D_pct>200% × OD>6 ──"); print(hdr); print(sep)
show(one_q[one_q['odds'] > 6],         "1頭抜け & OD>6")
show(multi_q[multi_q['odds'] > 6],     "2頭以上抜け & OD>6（全員）")
show(multi_t1[multi_t1['odds'] > 6],   "2頭以上のD1位 & OD>6")

# ── D1位 × OD閾値 ──
print()
print("── D1位 × OD閾値 ──"); print(hdr); print(sep)
show(top1, "D1位（全体）")
for od in [3, 4, 5, 6, 8, 10]:
    show(top1[top1['odds'] > od], f"D1位 & OD>{od}")

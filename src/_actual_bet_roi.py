# coding: utf-8
"""実際の馬券購入ROIレポート -> G:/マイドライブ/競馬AI/actual_bet_roi.html
data/tohyo/ 以下の投票CSVを集計して日別・式別ROIを表示する。
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import os, glob

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── CSVロード ──────────────────────────────────────────────────
def read_tohyo(f):
    for enc in ['utf-8-sig', 'cp932', 'utf-8']:
        try: return pd.read_csv(f, encoding=enc)
        except: pass
    return None

files = sorted(glob.glob(os.path.join(base_dir, 'data', 'tohyo', '*.csv')))
dfs = [read_tohyo(f) for f in files]
df = pd.concat([d for d in dfs if d is not None], ignore_index=True)

# 合計行（式別がNaN）を除外
df = df[df['式別'].notna()].copy()

def parse_amount(s):
    """300／900 形式は合計（後ろ）を使う、通常数値はそのまま"""
    s = str(s).strip()
    if '／' in s:
        return float(s.split('／')[-1])
    try: return float(s)
    except: return 0.0

df['購入金額'] = df['購入金額'].apply(parse_amount)
df['払戻金額'] = pd.to_numeric(df['払戻金額'], errors='coerce').fillna(0)
df['返還金額'] = pd.to_numeric(df['返還金額'], errors='coerce').fillna(0)
df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
# 返還分は投資から除く（未的中でも返還 → 実質投資 = 購入 - 返還）
df['実質投資'] = df['購入金額'] - df['返還金額']
df['損益']    = df['払戻金額'] - df['実質投資']

# 式別グループ（表示用に整理）
def shiki_group(s):
    s = str(s)
    if '単勝' in s: return '単勝'
    if '複勝' in s: return '複勝'
    if 'ワイド' in s: return 'ワイド'
    if '馬連' in s: return '馬連'
    if '馬単' in s: return '馬単'
    if '枠連' in s: return '枠連'
    if '3連複' in s or '３連複' in s: return '3連複'
    if '3連単' in s or '３連単' in s: return '3連単'
    return 'その他'

df['式別G'] = df['式別'].apply(shiki_group)

d = str(int(df['日付_num'].max()))
last_date = f"{d[:4]}/{d[4:6]}/{d[6:8]}"

print(f"読込: {len(files)}ファイル / {len(df)}行")
print(f"日付範囲: {df['日付_num'].min():.0f} ~ {df['日付_num'].max():.0f}")

# ── 日別集計 ───────────────────────────────────────────────────
daily_rows = []
cum_pf = 0
for dnum, grp in df.groupby('日付_num'):
    invest = int(grp['実質投資'].sum())
    ret    = int(grp['払戻金額'].sum())
    pf     = ret - invest
    roi    = ret / invest - 1.0 if invest > 0 else 0
    cum_pf += pf

    # 式別内訳
    shiki_detail = {}
    for sg, sg_grp in grp.groupby('式別G'):
        si = int(sg_grp['実質投資'].sum())
        sr = int(sg_grp['払戻金額'].sum())
        if si > 0:
            shiki_detail[sg] = {'投資': si, '回収': sr, 'pf': sr-si, 'roi': sr/si-1}

    d_str = str(int(dnum))
    date_str = f"{d_str[:4]}/{d_str[4:6]}/{d_str[6:8]}"
    daily_rows.append({
        '日付': date_str, '日付_num': dnum,
        '投資': invest, '回収': ret, '損益': pf, 'ROI': roi,
        '累計損益': cum_pf,
        '式別': shiki_detail,
    })
    sign = '+' if pf >= 0 else ''
    print(f"{date_str}  投資{invest:,}円  回収{ret:,}円  {sign}{pf:,}円 ({roi:+.1%})")

# ── 式別累計集計 ───────────────────────────────────────────────
shiki_total = {}
for sg, sg_grp in df.groupby('式別G'):
    si = int(sg_grp['実質投資'].sum())
    sr = int(sg_grp['払戻金額'].sum())
    hits = int((sg_grp['払戻金額'] > 0).sum())
    bets = int((sg_grp['実質投資'] > 0).sum())
    if si > 0:
        shiki_total[sg] = {'投資': si, '回収': sr, 'pf': sr-si, 'roi': sr/si-1,
                           'hits': hits, 'bets': bets}

total_invest = sum(r['投資'] for r in daily_rows)
total_ret    = sum(r['回収'] for r in daily_rows)
total_pf     = total_ret - total_invest
total_roi    = total_ret / total_invest - 1.0 if total_invest > 0 else 0
plus_days    = sum(1 for r in daily_rows if r['損益'] >= 0)
total_days   = len(daily_rows)

print(f"\n累計: {('+' if total_pf>=0 else '')}{total_pf:,}円  ROI{total_roi:+.1%}  {plus_days}/{total_days}日プラス")

# ── HTML生成 ────────────────────────────────────────────────────
SHIKI_ORDER = ['単勝','複勝','ワイド','馬連','枠連','馬単','3連複','3連単','その他']
col_total = '#2d862d' if total_pf >= 0 else '#c0392b'

def pf_td(pf, roi):
    sign = '+' if pf >= 0 else ''
    col  = '#2d862d' if pf >= 0 else '#c0392b'
    return f'<td style="color:{col};font-weight:bold;text-align:right">{sign}{pf:,}円<br><small>({roi:+.1%})</small></td>'

def cum_td(pf):
    col = '#2d862d' if pf >= 0 else '#c0392b'
    sign = '+' if pf >= 0 else ''
    return f'<td style="color:{col};font-weight:bold;text-align:right">{sign}{pf:,}円</td>'

# 式別ミニバッジ
def shiki_badges(detail):
    badges = []
    for sg in SHIKI_ORDER:
        if sg not in detail: continue
        d = detail[sg]
        col = '#2d862d' if d['pf'] >= 0 else '#c0392b'
        sign = '+' if d['pf'] >= 0 else ''
        badges.append(
            f'<span style="display:inline-block;margin:1px 2px;padding:1px 5px;'
            f'border-radius:3px;font-size:9px;background:#2a2a4a;color:{col}">'
            f'{sg} {sign}{d["pf"]:,}</span>'
        )
    return ''.join(badges)

# 日別行
rows_html = ''
for r in daily_rows:
    cum_col = '#2d862d' if r['累計損益'] >= 0 else '#c0392b'
    sign = '+' if r['累計損益'] >= 0 else ''
    rows_html += f'''<tr>
<td style="text-align:center;white-space:nowrap">{r["日付"]}</td>
<td style="text-align:right">{r["投資"]:,}円</td>
<td style="text-align:right">{r["回収"]:,}円</td>
{pf_td(r["損益"], r["ROI"])}
<td style="text-align:left;font-size:9px">{shiki_badges(r["式別"])}</td>
{cum_td(r["累計損益"])}
</tr>'''

# 式別サマリー行
shiki_rows = ''
for sg in SHIKI_ORDER:
    if sg not in shiki_total: continue
    s = shiki_total[sg]
    hit_rate = f'{s["hits"]}/{s["bets"]}' if s["bets"] > 0 else '-'
    col = '#2d862d' if s['pf'] >= 0 else '#c0392b'
    sign = '+' if s['pf'] >= 0 else ''
    shiki_rows += f'''<tr>
<td style="font-weight:bold">{sg}</td>
<td style="text-align:right">{s["投資"]:,}円</td>
<td style="text-align:right">{s["回収"]:,}円</td>
<td style="color:{col};font-weight:bold;text-align:right">{sign}{s["pf"]:,}円</td>
<td style="color:{col};font-weight:bold;text-align:right">{s["roi"]:+.1%}</td>
<td style="text-align:center">{hit_rate}</td>
</tr>'''

html = f'''<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8">
<title>実際の馬券ROI 2026</title>
<style>
body{{font-family:"Hiragino Kaku Gothic Pro",Meiryo,sans-serif;background:#1a1a2e;color:#e0e0e0;padding:20px}}
h2{{color:#f0c040;margin-top:28px}}
h3{{color:#aaddff;margin-top:20px}}
.summary{{display:flex;gap:16px;justify-content:center;margin:10px 0 20px;flex-wrap:wrap}}
.card{{background:#16213e;border-radius:8px;padding:12px 20px;text-align:center;min-width:110px}}
.card .val{{font-size:1.5em;font-weight:bold}}
table{{width:100%;border-collapse:collapse;font-size:0.85em;margin-bottom:20px}}
th{{background:#16213e;color:#f0c040;padding:6px 8px;text-align:center;position:sticky;top:0}}
td{{padding:5px 8px;border-bottom:1px solid #2a2a4a}}
tr:nth-child(even){{background:#16213e88}}
tr:hover{{background:#1a3a5a}}
</style></head><body>
<h2>実際の馬券ROI　（最終更新: {last_date}）</h2>
<div class="summary">
  <div class="card"><div>累計損益</div><div class="val" style="color:{col_total}">{("+" if total_pf>=0 else "")}{total_pf:,}円</div></div>
  <div class="card"><div>累計ROI</div><div class="val" style="color:{col_total}">{total_roi:+.1%}</div></div>
  <div class="card"><div>総投資</div><div class="val">{total_invest:,}円</div></div>
  <div class="card"><div>総回収</div><div class="val">{total_ret:,}円</div></div>
  <div class="card"><div>プラス日数</div><div class="val">{plus_days}/{total_days}日</div></div>
</div>

<h3>式別ROI</h3>
<table>
<thead><tr>
<th>式別</th><th>投資</th><th>回収</th><th>損益</th><th>ROI</th><th>的中/買い目数</th>
</tr></thead><tbody>
{shiki_rows}
</tbody></table>

<h3>日別損益</h3>
<table>
<thead><tr>
<th>日付</th><th>投資</th><th>回収</th><th>損益</th><th>式別内訳</th><th>累計損益</th>
</tr></thead><tbody>
{rows_html}
</tbody></table>
</body></html>'''

out = r'G:\マイドライブ\競馬AI\actual_bet_roi.html'
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"\n出力: {out}")

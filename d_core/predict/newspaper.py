# coding: utf-8
"""D指標競馬新聞 HTML出力
使い方:
  python newspaper.py           # 最新の cache pkl
  python newspaper.py 260419    # 日付指定
"""
import sys, io, os, glob, pickle, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from config import CACHE_DIR, OUTPUT_DIR

parser = argparse.ArgumentParser()
parser.add_argument('date', nargs='?', default=None)
args = parser.parse_args()

# ── cache pkl 特定 ──
pkl_files = sorted(glob.glob(os.path.join(CACHE_DIR, '*.cache.pkl')))
if not pkl_files:
    print("cache pkl が見つかりません"); sys.exit(1)

date_map = {}
for p in pkl_files:
    try:
        with open(p, 'rb') as f: c = pickle.load(f)
        d = str(c.get('target_date', ''))
        if d: date_map[d] = p
    except: pass

if args.date:
    if args.date not in date_map:
        print(f"{args.date} に対応する cache pkl が見つかりません")
        print(f"利用可能: {sorted(date_map.keys())[-5:]}"); sys.exit(1)
    pkl_path = date_map[args.date]
else:
    pkl_path = date_map[max(date_map.keys())]

with open(pkl_path, 'rb') as f:
    cache = pickle.load(f)

target_date = cache.get('target_date', '?')
res = cache['result']
print(f"読み込み: {target_date}  ({os.path.basename(pkl_path)})")

# ── D指標計算 ──
def gs(col):
    return pd.to_numeric(
        pd.Series(res[col].tolist() if col in res.columns else [np.nan]*len(res)),
        errors='coerce')

sub_cs = gs('sub_コース偏差値')
sub_ri = gs('sub_レース内偏差値')
cur_r  = gs('cur_ランカー順位')
sub_r  = gs('sub_ランカー順位')
odds   = gs('単勝オッズ')
venue  = pd.Series((res['会場'] if '会場' in res.columns else res['開催']).tolist())
rnum   = pd.Series(res['Ｒ'].tolist())
uma    = pd.Series(res['馬名S'].tolist())
banum  = pd.Series(res['馬番'].tolist()) if '馬番' in res.columns else pd.Series(['-']*len(res))
rname  = pd.Series(res['レース名'].tolist()) if 'レース名' in res.columns else pd.Series(['']*len(res))

prod_r = (cur_r * sub_r).clip(lower=0.25)
D = sub_cs * sub_ri / prod_r

df = pd.DataFrame({
    'venue': venue, 'R': rnum, 'race_name': rname,
    'uma': uma, 'banum': banum, 'odds': odds,
    'D': D, 'sub_cs': sub_cs, 'sub_ri': sub_ri,
    'cur_r': cur_r, 'sub_r': sub_r,
})
df = df.dropna(subset=['D'])
df['race_key'] = df['venue'].astype(str) + '_' + df['R'].astype(str)

df['D_rank']  = df.groupby('race_key')['D'].rank(ascending=False, method='min')
df['D_mean']  = df.groupby('race_key')['D'].transform('mean').clip(lower=1)
df['D_pct']   = (df['D'] - df['D_mean']) / df['D_mean'] * 100
df['_n_qual'] = df.groupby('race_key')['D_pct'].transform(lambda x: (x > 200).sum())

def _gap(g):
    g2 = g.sort_values('D', ascending=False).reset_index(drop=True)
    d1 = g2.iloc[0]['D'] if len(g2) >= 1 else np.nan
    d2 = g2.iloc[1]['D'] if len(g2) >= 2 else np.nan
    return pd.Series({'gap_ratio': d1/d2 if pd.notna(d2) and d2 > 0 else np.nan})

gap_df = df.groupby('race_key').apply(_gap)
df = df.merge(gap_df, left_on='race_key', right_index=True, how='left')

# ── 印判定 ──
def gekiatsu_label(gap):
    if pd.isna(gap): return ''
    if gap > 20: return '超激熱'
    if gap > 10: return '激熱'
    return ''

def race_mark(rank, dpct):
    if dpct < -90:        return 'keshi'
    if -90 <= dpct < -70: return 'keshi2'
    return {1:'honmei', 2:'taikou', 3:'tanki', 4:'renpuku', 5:'oshi'}.get(int(rank), '')

def tan_atsu(row):
    od, gap, dpct, nq = row['odds'], row['gap_ratio'], row['D_pct'], row['_n_qual']
    is_ippon = (dpct > 200) and (nq == 1)
    if pd.notna(od) and od > 8 and is_ippon:                   return 'tan_honmei'
    if pd.notna(od) and od > 6 and pd.notna(gap) and gap >= 3: return 'tan_taikou'
    if pd.notna(od) and od > 6:                                 return 'tan_tanki'
    return ''

def fuku_atsu(row):
    od, gap = row['odds'], row['gap_ratio']
    if pd.notna(od) and od > 6 and pd.notna(gap) and gap >= 3: return 'fuku_honmei'
    if pd.notna(od) and od > 5 and pd.notna(gap) and gap >= 3: return 'fuku_taikou'
    if pd.notna(od) and od > 5:                                 return 'fuku_tanki'
    return ''

MARK_LABEL = {
    'honmei':'◎', 'taikou':'○', 'tanki':'▲', 'renpuku':'△', 'oshi':'×',
    'keshi':'消し', 'keshi2':'消候',
    'tan_honmei':'◎単', 'tan_taikou':'○単', 'tan_tanki':'▲単',
    'fuku_honmei':'◎複', 'fuku_taikou':'○複', 'fuku_tanki':'▲複',
}
MARK_COLOR = {
    'honmei':'#c0392b', 'taikou':'#27ae60', 'tanki':'#8e44ad',
    'renpuku':'#1f618d', 'oshi':'#7f8c8d',
    'keshi':'#555', 'keshi2':'#888',
    'tan_honmei':'#c0392b', 'tan_taikou':'#e67e22', 'tan_tanki':'#f39c12',
    'fuku_honmei':'#2471a3', 'fuku_taikou':'#1a9ed4', 'fuku_tanki':'#85c1e9',
}

# ── サマリー収集 ──
sum_gekiatsu, sum_tan, sum_fuku = [], [], []

for rk in df['race_key'].unique():
    sub = df[df['race_key'] == rk].sort_values('D', ascending=False).reset_index(drop=True)
    d1 = sub.iloc[0]
    gap = d1['gap_ratio']
    geki = gekiatsu_label(gap)
    tm = tan_atsu(d1)
    fm = fuku_atsu(d1)
    od_s = f"{d1['odds']:.1f}" if pd.notna(d1['odds']) else '-'
    gap_s = f"{gap:.1f}x" if pd.notna(gap) else '-'
    info = (str(d1['venue']), int(d1['R']), str(d1['race_name']), str(d1['uma']), od_s, gap_s)
    if geki: sum_gekiatsu.append((geki, *info))
    if tm in ('tan_honmei', 'tan_taikou'): sum_tan.append((MARK_LABEL[tm], *info))
    if fm in ('fuku_honmei', 'fuku_taikou'): sum_fuku.append((MARK_LABEL[fm], *info))

# ── HTML生成 ──
CSS = """
@page { size: A4 landscape; margin: 8mm; }
body { font-family:'Yu Gothic','Hiragino Sans',sans-serif; font-size:11px; margin:6px; background:#f0f0f0; }
h1 { font-size:18px; margin:4px 0; color:#1a1a2e; letter-spacing:2px; }
h2 { font-size:13px; color:white; padding:5px 10px; margin:12px 0 0; border-radius:4px 4px 0 0; }
.subtitle { font-size:9px; color:#666; margin:2px 0 6px; padding:0 4px; }
.page { page-break-after:always; break-after:page; padding:4px; }
.page:last-child { page-break-after:avoid; }
table { border-collapse:collapse; width:100%; background:white; margin-bottom:4px; }
th { background:#2c3e50; color:white; padding:3px 6px; font-size:9px; text-align:center; border:1px solid #1a252f; white-space:nowrap; }
td { padding:2px 5px; text-align:center; font-size:10px; border:1px solid #ccc; white-space:nowrap; }
td.left { text-align:left; }
.mark { font-size:15px; font-weight:bold; min-width:32px; display:inline-block; text-align:center; }
/* 激熱 */
.geki-super { background:#4a0000; color:#ffd700; font-weight:bold; padding:2px 8px; border-radius:4px; font-size:12px; }
.geki       { background:#8b0000; color:#ff9; font-weight:bold; padding:2px 8px; border-radius:4px; font-size:12px; }
/* 行ハイライト */
.row-geki-super td { background:#fff3cd !important; }
.row-geki       td { background:#fff9f0 !important; }
.row-keshi      td { background:#f5f5f5 !important; color:#aaa; }
/* サマリーカード */
.sum-grid { display:flex; flex-wrap:wrap; gap:8px; padding:6px; background:white; }
.sum-card { border-radius:6px; padding:6px 12px; min-width:200px; }
.sum-card .race  { font-size:9px; color:#666; }
.sum-card .horse { font-size:14px; font-weight:bold; margin:2px 0; }
.sum-card .info  { font-size:9px; color:#888; }
/* レース内印色 */
.mk-honmei  { color:#c0392b; }
.mk-taikou  { color:#27ae60; }
.mk-tanki   { color:#8e44ad; }
.mk-renpuku { color:#1f618d; }
.mk-oshi    { color:#7f8c8d; }
.mk-keshi   { color:#aaa; }
/* 熱さバッジ */
.badge { display:inline-block; font-size:8px; font-weight:bold; border-radius:3px; padding:1px 4px; margin:1px; }
.b-tan  { background:#c0392b; color:white; }
.b-fuku { background:#2471a3; color:white; }
/* D_pct色 */
.pct-hi  { color:#c0392b; font-weight:bold; }
.pct-mid { color:#e67e22; }
.pct-lo  { color:#aaa; }
@media print { body { margin:0; } }
"""

def pct_class(v):
    if v >= 200: return 'pct-hi'
    if v >= 50:  return 'pct-mid'
    if v < -70:  return 'pct-lo'
    return ''

def sum_card(items, title, color, badge_cls):
    if not items:
        return f'<p style="color:#aaa;padding:6px">該当なし</p>'
    html = '<div class="sum-grid">'
    for mk, venue, r, rname, uma, od, gap in items:
        html += f'''
        <div class="sum-card" style="border:2px solid {color}; background:{color}11">
          <div class="race">{venue}{r}R　{rname}</div>
          <div class="horse">{uma}</div>
          <div class="info">
            <span class="badge {badge_cls}">{mk}</span>
            OD={od}倍　gap={gap}
          </div>
        </div>'''
    html += '</div>'
    return html

pages = []

# ── Page 1: 激熱まとめ ──
geki_body = ''
if sum_gekiatsu:
    rows = ''
    for mk, venue, r, rname, uma, od, gap in sum_gekiatsu:
        badge = '<span class="geki-super">◆◆超激熱</span>' if mk == '超激熱' else '<span class="geki">◆激熱</span>'
        rows += f'<tr><td class="left">{badge}</td><td class="left"><b>{venue}{r}R</b>　{rname}</td><td class="left" style="font-size:13px;font-weight:bold">{uma}</td><td>{od}倍</td><td>{gap}</td></tr>'
    geki_body = f'<table><thead><tr style="background:#4a0000"><th>種別</th><th style="text-align:left">レース</th><th style="text-align:left">馬名</th><th>オッズ</th><th>gap</th></tr></thead><tbody>{rows}</tbody></table>'
else:
    geki_body = '<p style="color:#aaa;padding:8px">該当なし</p>'

tan_body  = sum_card(sum_tan,  '単勝', '#c0392b', 'b-tan')
fuku_body = sum_card(sum_fuku, '複勝', '#2471a3', 'b-fuku')

pages.append(f'''
<div class="page" style="background:#fff8f0">
  <h1>D 指 標 競 馬 新 聞　{target_date}</h1>
  <p class="subtitle">D指標: sub_cs × sub_ri ÷ (cur_r × sub_r)　|　分析期間 2023-07〜 / 8796レース実績</p>

  <h2 style="background:#4a0000; margin-top:12px">◆ 激熱まとめ　gap&gt;10x（勝率58% 単ROI+66%）/ gap&gt;20x（勝率71% 単ROI+107%）</h2>
  {geki_body}

  <h2 style="background:#922b21; margin-top:12px">単勝まとめ　◎単(OD&gt;8+1頭抜け 単ROI+515%) / ○単(OD&gt;6+gap≥3x 単ROI+354%)</h2>
  {tan_body}

  <h2 style="background:#1f618d; margin-top:12px">複勝まとめ　◎複(OD&gt;6+gap≥3x 複ROI+89%) / ○複(OD&gt;5+gap≥3x 複ROI+62%)</h2>
  {fuku_body}

  <div style="margin-top:16px;font-size:9px;color:#888;border-top:1px solid #ddd;padding-top:6px">
    凡例:
    <b>レース内印</b>: ◎=D1位 ○=D2位 ▲=D3位 △=D4位 ×=D5位 | 消し=D_pct&lt;-90%(複勝率7%) 消候=D_pct -90〜-70%(19%)<br>
    <b>熱さ印</b>: ◎単(OD&gt;8+1頭抜け) ○単(OD&gt;6+gap≥3x) ▲単(OD&gt;6) / ◎複(OD&gt;6+gap≥3x) ○複(OD&gt;5+gap≥3x) ▲複(OD&gt;5)<br>
    <b>激熱</b>: ◆激熱(gap&gt;10x) ◆◆超激熱(gap&gt;20x) ※オッズ無関係・D指標のみで判定
  </div>
</div>
''')

# ── Page 2〜: レース別 ──
for rk in df['race_key'].unique():
    sub = df[df['race_key'] == rk].sort_values('D', ascending=False).reset_index(drop=True)
    d1  = sub.iloc[0]
    gap = d1['gap_ratio']
    geki = gekiatsu_label(gap)
    tm = tan_atsu(d1)
    fm = fuku_atsu(d1)
    gap_s = f"{gap:.1f}x" if pd.notna(gap) else '-'

    # ヘッダーバッジ
    badges = ''
    if geki == '超激熱': badges += '<span class="geki-super">◆◆超激熱</span> '
    elif geki == '激熱':  badges += '<span class="geki">◆激熱</span> '
    if tm: badges += f'<span class="badge b-tan">{MARK_LABEL[tm]}</span> '
    if fm: badges += f'<span class="badge b-fuku">{MARK_LABEL[fm]}</span> '

    # レースヘッダー色
    h2_bg = '#4a0000' if geki == '超激熱' else ('#8b0000' if geki == '激熱' else '#2c3e50')

    # 馬テーブル
    rows = ''
    for i, row in sub.iterrows():
        mk  = race_mark(row['D_rank'], row['D_pct'])
        is_keshi = mk in ('keshi', 'keshi2')
        row_cls = 'row-keshi' if is_keshi else (
            'row-geki-super' if (i == 0 and geki == '超激熱') else
            'row-geki' if (i == 0 and geki == '激熱') else '')

        mark_html = f'<span class="mark mk-{mk}">{MARK_LABEL.get(mk,"")}</span>'
        od_html = f'{row["odds"]:.1f}' if pd.notna(row['odds']) else '-'
        d_html  = f'{row["D"]:,.0f}'
        pct_v   = row['D_pct']
        pct_html = f'<span class="{pct_class(pct_v)}">{pct_v:+.0f}%</span>'
        ba = f'{int(row["banum"])}' if str(row['banum']).isdigit() else str(row['banum'])

        # D2位 参考複勝
        ref = ''
        if row['D_rank'] == 2 and pd.notna(row['odds']) and row['odds'] > 6 and row['D_pct'] > 100:
            ref = '<span class="badge b-fuku" style="font-size:7px">参考複</span>'

        cs_html = f'{row["sub_cs"]:.1f}' if pd.notna(row['sub_cs']) else '-'
        ri_html = f'{row["sub_ri"]:.1f}' if pd.notna(row['sub_ri']) else '-'

        rows += f'''<tr class="{row_cls}">
          <td>{ba}</td>
          <td style="font-size:14px;font-weight:bold;text-align:left">{mark_html} {str(row["uma"])}{ref}</td>
          <td>{d_html}</td>
          <td>{pct_html}</td>
          <td>{cs_html}</td>
          <td>{ri_html}</td>
          <td style="color:#e67e22;font-weight:{'bold' if pd.notna(row['odds']) and row['odds']>6 else 'normal'}">{od_html}倍</td>
        </tr>'''

    pages.append(f'''
<div class="page">
  <h2 style="background:{h2_bg}">
    【{d1['venue']}{int(d1['R'])}R】{d1['race_name']}
    <span style="font-size:10px;font-weight:normal;margin-left:12px">gap: {gap_s}</span>
    <span style="margin-left:8px">{badges}</span>
  </h2>
  <div class="subtitle">
    D1位: <b>{d1['uma']}</b>　D={d1['D']:,.0f}　D_pct={d1['D_pct']:+.0f}%　OD={f"{d1['odds']:.1f}倍" if pd.notna(d1['odds']) else '-'}
  </div>
  <table>
    <thead><tr>
      <th>馬番</th><th style="text-align:left">印　馬名</th>
      <th>D値</th><th>D_pct</th><th>sub_cs</th><th>sub_ri</th><th>オッズ</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>
''')

HTML = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>D指標競馬新聞 {target_date}</title>
  <style>{CSS}</style>
</head>
<body>
{''.join(pages)}
</body>
</html>"""

os.makedirs(OUTPUT_DIR, exist_ok=True)
ts = datetime.datetime.now().strftime('%Y%m%d_%H%M')
out_path = os.path.join(OUTPUT_DIR, f'd_newspaper_{target_date}_{ts}.html')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(HTML)

print(f"出力: {out_path}")

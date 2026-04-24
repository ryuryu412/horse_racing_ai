# coding: utf-8
"""D指標競馬新聞 HTML出力（サイト型）
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
def gekiatsu_level(gap):
    if pd.isna(gap): return 0
    if gap > 20: return 2
    if gap > 10: return 1
    return 0

def race_mark(rank, dpct):
    if dpct < -90:        return 'keshi'
    if -90 <= dpct < -70: return 'keshi2'
    return {1:'◎', 2:'○', 3:'▲', 4:'△', 5:'×'}.get(int(rank), '')

def tan_level(row):
    od, gap, dpct, nq = row['odds'], row['gap_ratio'], row['D_pct'], row['_n_qual']
    is_ippon = (dpct > 200) and (nq == 1)
    if pd.notna(od) and od > 8 and is_ippon:                   return 3
    if pd.notna(od) and od > 6 and pd.notna(gap) and gap >= 3: return 2
    if pd.notna(od) and od > 6:                                 return 1
    return 0

def fuku_level(row):
    od, gap = row['odds'], row['gap_ratio']
    if pd.notna(od) and od > 6 and pd.notna(gap) and gap >= 3: return 3
    if pd.notna(od) and od > 5 and pd.notna(gap) and gap >= 3: return 2
    if pd.notna(od) and od > 5:                                 return 1
    return 0

TAN_LABEL  = {3:'◎単', 2:'○単', 1:'▲単', 0:''}
FUKU_LABEL = {3:'◎複', 2:'○複', 1:'▲複', 0:''}
TAN_COLOR  = {3:'#e74c3c', 2:'#e67e22', 1:'#f39c12'}
FUKU_COLOR = {3:'#2471a3', 2:'#1a9ed4', 1:'#5dade2'}

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Hiragino Sans','Yu Gothic',sans-serif;background:#0d1117;color:#e6edf3;font-size:13px;line-height:1.5}
.top-bar{background:linear-gradient(135deg,#1a1f3a 0%,#2d1b4e 100%);padding:20px 16px;border-bottom:4px solid #e8b400}
.top-bar h1{font-size:24px;color:#e8b400;letter-spacing:2px}
.top-bar .date{font-size:15px;color:#fff;margin-top:6px;font-weight:bold}
.top-bar .subtitle{color:#aaa;font-size:12px;margin-top:4px}
.top-nav{position:sticky;top:0;z-index:100;background:#161b22;border-bottom:2px solid #30363d;padding:10px 12px;display:flex;gap:8px;overflow-x:auto;flex-wrap:wrap}
.top-nav a{padding:5px 14px;background:#21262d;border-radius:14px;text-decoration:none;font-size:13px;font-weight:bold;white-space:nowrap;border:1px solid transparent;color:#e6edf3}
.top-nav a:hover{background:#30363d}
.summary-box{margin:14px;padding:14px 16px;background:linear-gradient(135deg,#1a0a0a,#2d1010);border:2px solid #c0392b;border-radius:10px}
.summary-title{font-size:16px;font-weight:bold;color:#e74c3c;margin-bottom:10px}
.summary-fuku{margin:14px;padding:14px 16px;background:linear-gradient(135deg,#0a0a1a,#10102d);border:2px solid #2471a3;border-radius:10px}
.summary-fuku .summary-title{color:#2471a3}
.summary-geki{margin:14px;padding:14px 16px;background:linear-gradient(135deg,#1a0000,#3d0000);border:2px solid #8b0000;border-radius:10px}
.summary-geki .summary-title{color:#ff6b6b}
.picks-table{width:100%;border-collapse:collapse}
.picks-table th{background:rgba(255,255,255,0.05);color:#8b949e;padding:6px 10px;text-align:center;font-size:11px;border-bottom:1px solid #30363d}
.picks-table td{padding:7px 10px;text-align:center;border-bottom:1px solid rgba(255,255,255,0.04)}
.venue-section{margin:16px 0}
.venue-header{margin:0 12px;padding:12px 16px;background:linear-gradient(90deg,#1f2937,#161b22);border-left:5px solid #e8b400;border-radius:6px 6px 0 0;display:flex;align-items:center;gap:12px}
.venue-name{font-size:20px;font-weight:900;letter-spacing:1px}
.venue-count{font-size:12px;color:#aaa;background:#21262d;padding:2px 8px;border-radius:10px}
.venue-nav{margin:0 12px;padding:8px 10px;background:#161b22;border-bottom:1px solid #30363d;display:flex;gap:6px;overflow-x:auto;flex-wrap:wrap}
.venue-nav a{padding:3px 10px;background:#21262d;border-radius:10px;color:#58a6ff;text-decoration:none;font-size:12px;font-weight:bold;white-space:nowrap}
.race-card{margin:12px;background:#161b22;border-radius:10px;border:1px solid #30363d;overflow:hidden}
.race-header{padding:12px 14px;background:linear-gradient(90deg,#1f2937,#1a2035);display:flex;align-items:flex-start;gap:12px;flex-wrap:wrap}
.race-number{font-size:32px;font-weight:900;color:#e8b400;min-width:52px;line-height:1}
.race-info{flex:1;min-width:200px}
.race-name{font-size:17px;font-weight:bold;color:#fff}
.race-badges{display:flex;gap:6px;margin-top:6px;flex-wrap:wrap;align-items:center}
.badge-geki-s{background:#8b0000;color:#ffd700;font-weight:bold;padding:3px 10px;border-radius:12px;font-size:12px;border:1px solid #ffd700}
.badge-geki{background:#3d0000;color:#ff9;font-weight:bold;padding:3px 10px;border-radius:12px;font-size:12px;border:1px solid #c0392b}
.badge-tan{background:#c0392b;color:white;font-weight:bold;padding:3px 10px;border-radius:12px;font-size:12px}
.badge-fuku{background:#2471a3;color:white;font-weight:bold;padding:3px 10px;border-radius:12px;font-size:12px}
.gap-info{color:#aaa;font-size:12px;margin-top:4px}
.table-wrap{overflow-x:auto}
.horse-table{width:100%;border-collapse:collapse;min-width:600px}
.horse-table thead tr{background:#21262d}
.horse-table th{padding:6px 8px;text-align:center;color:#8b949e;font-size:11px;border-bottom:1px solid #30363d;white-space:nowrap}
.horse-table td{padding:7px 8px;text-align:center;border-bottom:1px solid #1c2128;vertical-align:middle}
.horse-table tr:last-child td{border-bottom:none}
.horse-table tr:hover{background:rgba(88,166,255,0.05)}
.row-d1{background:rgba(232,180,0,0.07)!important}
.row-keshi{background:rgba(255,255,255,0.02)!important;opacity:0.5}
.row-geki-s{background:rgba(139,0,0,0.2)!important}
.row-geki{background:rgba(139,0,0,0.1)!important}
.mark-honmei{color:#e74c3c;font-size:16px;font-weight:bold}
.mark-taikou{color:#2ecc71;font-size:16px;font-weight:bold}
.mark-tanki{color:#9b59b6;font-size:16px;font-weight:bold}
.mark-renpuku{color:#3498db;font-size:16px;font-weight:bold}
.mark-oshi{color:#7f8c8d;font-size:15px}
.mark-keshi{color:#555;font-size:11px}
.d-hi{color:#e74c3c;font-weight:bold}
.d-mid{color:#e67e22}
.d-lo{color:#555}
.ref-badge{background:#1a5276;color:#85c1e9;font-size:9px;padding:1px 5px;border-radius:8px;margin-left:4px}
footer{text-align:center;color:#555;font-size:11px;padding:24px;border-top:1px solid #21262d;margin-top:24px}
"""

# ── データ整理 ──
venues = df['venue'].unique().tolist()
venue_colors = {}
colors = ['#e8b400','#58a6ff','#f85149','#3fb950','#d2a8ff']
for i, v in enumerate(venues):
    venue_colors[v] = colors[i % len(colors)]

# サマリー収集
sum_geki, sum_tan, sum_fuku = [], [], []
for rk in df['race_key'].unique():
    sub = df[df['race_key'] == rk].sort_values('D', ascending=False).reset_index(drop=True)
    d1  = sub.iloc[0]
    gap = d1['gap_ratio']
    gl  = gekiatsu_level(gap)
    tl  = tan_level(d1)
    fl  = fuku_level(d1)
    od  = f"{d1['odds']:.1f}" if pd.notna(d1['odds']) else '-'
    gap_s = f"{gap:.1f}x" if pd.notna(gap) else '-'
    v, r, rn, u = str(d1['venue']), int(d1['R']), str(d1['race_name']), str(d1['uma'])
    if gl >= 1: sum_geki.append((gl, v, r, rn, u, od, gap_s))
    if tl >= 2: sum_tan.append((tl, v, r, rn, u, od, gap_s))
    if fl >= 2: sum_fuku.append((fl, v, r, rn, u, od, gap_s))

# ── HTML構築 ──
now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
total_races = df['race_key'].nunique()
venues_str = '・'.join(venues) + f' 計{total_races}レース'

# トップナビ
nav_links = ''
for v in venues:
    c = venue_colors[v]
    nav_links += f'<a href="#venue-{v}" style="border-color:{c};color:{c}">{v}</a>'

# サマリー：激熱
def geki_rows():
    if not sum_geki: return '<tr><td colspan="5" style="color:#555;padding:12px">該当なし</td></tr>'
    rows = ''
    for gl, v, r, rn, u, od, gap_s in sum_geki:
        badge = '<span class="badge-geki-s">◆◆超激熱</span>' if gl==2 else '<span class="badge-geki">◆激熱</span>'
        rows += f'''<tr>
          <td>{badge}</td>
          <td style="color:#aaa">{v}</td>
          <td style="font-weight:bold;color:#e8b400">{r}R</td>
          <td style="font-weight:bold;color:#fff;font-size:14px">{u}</td>
          <td style="color:#f0a500">{od}倍</td>
          <td style="color:#aaa">{gap_s}</td>
        </tr>'''
    return rows

def tan_rows():
    if not sum_tan: return '<tr><td colspan="5" style="color:#555;padding:12px">該当なし</td></tr>'
    rows = ''
    for tl, v, r, rn, u, od, gap_s in sum_tan:
        mk = TAN_LABEL[tl]
        c  = TAN_COLOR[tl]
        rows += f'''<tr>
          <td style="font-weight:bold;color:{c};font-size:15px">{mk}</td>
          <td style="color:#aaa">{v}</td>
          <td style="font-weight:bold;color:#e8b400">{r}R</td>
          <td style="font-weight:bold;color:#fff;font-size:14px">{u}</td>
          <td style="color:#f0a500">{od}倍</td>
          <td style="color:#aaa">{gap_s}</td>
        </tr>'''
    return rows

def fuku_rows():
    if not sum_fuku: return '<tr><td colspan="5" style="color:#555;padding:12px">該当なし</td></tr>'
    rows = ''
    for fl, v, r, rn, u, od, gap_s in sum_fuku:
        mk = FUKU_LABEL[fl]
        c  = FUKU_COLOR[fl]
        rows += f'''<tr>
          <td style="font-weight:bold;color:{c};font-size:15px">{mk}</td>
          <td style="color:#aaa">{v}</td>
          <td style="font-weight:bold;color:#e8b400">{r}R</td>
          <td style="font-weight:bold;color:#fff;font-size:14px">{u}</td>
          <td style="color:#f0a500">{od}倍</td>
          <td style="color:#aaa">{gap_s}</td>
        </tr>'''
    return rows

summary_html = f'''
<div class="summary-geki">
  <div class="summary-title">◆ 激熱まとめ　gap&gt;10x 勝率58% / gap&gt;20x 勝率71%</div>
  <table class="picks-table">
    <thead><tr><th>種別</th><th>会場</th><th>R</th><th>馬名</th><th>オッズ</th><th>gap</th></tr></thead>
    <tbody>{geki_rows()}</tbody>
  </table>
</div>
<div class="summary-box">
  <div class="summary-title">単勝まとめ　◎単(OD&gt;8+1頭抜け +515%) / ○単(OD&gt;6+gap≥3x +354%)</div>
  <table class="picks-table">
    <thead><tr><th>印</th><th>会場</th><th>R</th><th>馬名</th><th>オッズ</th><th>gap</th></tr></thead>
    <tbody>{tan_rows()}</tbody>
  </table>
</div>
<div class="summary-fuku">
  <div class="summary-title">複勝まとめ　◎複(OD&gt;6+gap≥3x +89%) / ○複(OD&gt;5+gap≥3x +62%)</div>
  <table class="picks-table">
    <thead><tr><th>印</th><th>会場</th><th>R</th><th>馬名</th><th>オッズ</th><th>gap</th></tr></thead>
    <tbody>{fuku_rows()}</tbody>
  </table>
</div>
'''

# 会場別レースカード
venue_html = ''
for v in venues:
    vc = venue_colors[v]
    vdf = df[df['venue'] == v]
    race_keys = vdf['race_key'].unique().tolist()
    r_nums = sorted([int(df[df['race_key']==rk].iloc[0]['R']) for rk in race_keys])

    nav_items = ''.join(f'<a href="#v{v}r{r}">{r}R</a>' for r in r_nums)

    cards = ''
    for rk in sorted(race_keys, key=lambda k: int(df[df['race_key']==k].iloc[0]['R'])):
        sub = df[df['race_key'] == rk].sort_values('D', ascending=False).reset_index(drop=True)
        d1  = sub.iloc[0]
        r   = int(d1['R'])
        gap = d1['gap_ratio']
        gl  = gekiatsu_level(gap)
        tl  = tan_level(d1)
        fl  = fuku_level(d1)
        gap_s = f"{gap:.1f}x" if pd.notna(gap) else '-'

        # ヘッダーバッジ
        badges = ''
        if gl == 2: badges += '<span class="badge-geki-s">◆◆超激熱</span>'
        elif gl==1: badges += '<span class="badge-geki">◆激熱</span>'
        if tl >= 1: badges += f'<span class="badge-tan">{TAN_LABEL[tl]}</span>'
        if fl >= 1: badges += f'<span class="badge-fuku">{FUKU_LABEL[fl]}</span>'

        # カードヘッダー背景
        hdr_bg = 'linear-gradient(90deg,#3d0000,#1a2035)' if gl>=1 else 'linear-gradient(90deg,#1f2937,#1a2035)'

        # 馬テーブル
        horse_rows = ''
        for i, row in sub.iterrows():
            mk   = race_mark(row['D_rank'], row['D_pct'])
            is_keshi = mk in ('keshi', 'keshi2')
            row_cls = 'row-keshi' if is_keshi else (
                'row-geki-s' if (i==0 and gl==2) else
                'row-geki'   if (i==0 and gl==1) else
                'row-d1'     if i==0 else '')

            # 印HTML
            mk_cls = {'◎':'mark-honmei','○':'mark-taikou','▲':'mark-tanki',
                      '△':'mark-renpuku','×':'mark-oshi',
                      'keshi':'mark-keshi','keshi2':'mark-keshi'}.get(mk,'')
            mk_txt = {'keshi':'消し','keshi2':'消候'}.get(mk, mk)
            mk_html = f'<span class="{mk_cls}">{mk_txt}</span>'

            ba   = str(int(row['banum'])) if str(row['banum']).isdigit() else str(row['banum'])
            od_s = f'{row["odds"]:.1f}' if pd.notna(row['odds']) else '-'
            od_c = '#f0a500' if (pd.notna(row['odds']) and row['odds'] > 6) else '#8b949e'
            d_s  = f'{row["D"]:,.0f}'
            pct  = row['D_pct']
            pct_c = '#e74c3c' if pct>=200 else ('#e67e22' if pct>=50 else ('#555' if pct<-70 else '#8b949e'))
            pct_s = f'{pct:+.0f}%'
            cs_s = f'{row["sub_cs"]:.1f}' if pd.notna(row['sub_cs']) else '-'
            ri_s = f'{row["sub_ri"]:.1f}' if pd.notna(row['sub_ri']) else '-'

            ref = ''
            if row['D_rank']==2 and pd.notna(row['odds']) and row['odds']>6 and row['D_pct']>100:
                ref = '<span class="ref-badge">参考複</span>'

            horse_rows += f'''<tr class="{row_cls}">
              <td>{mk_html}</td>
              <td style="color:#8b949e">{ba}</td>
              <td style="text-align:left;font-weight:bold;font-size:13px">{row['uma']}{ref}</td>
              <td style="color:{od_c};font-weight:bold">{od_s}倍</td>
              <td style="font-weight:bold">{d_s}</td>
              <td style="color:{pct_c};font-weight:bold">{pct_s}</td>
              <td style="color:#555;font-size:11px">{cs_s}</td>
              <td style="color:#555;font-size:11px">{ri_s}</td>
            </tr>'''

        cards += f'''
<div class="race-card" id="v{v}r{r}">
  <div class="race-header" style="background:{hdr_bg}">
    <div class="race-number" style="color:{vc}">{r}R</div>
    <div class="race-info">
      <div class="race-name">{d1['race_name']}</div>
      <div class="race-badges">{badges}</div>
      <div class="gap-info">gap: {gap_s}　D1位: <b>{d1['uma']}</b>　D={d1['D']:,.0f}　OD={f"{d1['odds']:.1f}倍" if pd.notna(d1['odds']) else '-'}</div>
    </div>
  </div>
  <div class="table-wrap">
    <table class="horse-table">
      <thead><tr>
        <th>印</th><th>馬番</th><th style="text-align:left">馬名</th>
        <th>オッズ</th><th>D値</th><th>D_pct</th>
        <th style="font-size:9px;color:#555">sub_cs</th><th style="font-size:9px;color:#555">sub_ri</th>
      </tr></thead>
      <tbody>{horse_rows}</tbody>
    </table>
  </div>
</div>'''

    venue_html += f'''
<div class="venue-section" id="venue-{v}">
  <div class="venue-header" style="border-left-color:{vc}">
    <span class="venue-name" style="color:{vc}">{v}競馬場</span>
    <span class="venue-count">{len(r_nums)}レース</span>
  </div>
  <div class="venue-nav">{nav_items}</div>
  {cards}
</div>'''

# 凡例
legend = '''
<div style="margin:14px;padding:12px 16px;background:#161b22;border-radius:8px;border:1px solid #30363d;font-size:11px;color:#8b949e;line-height:1.8">
  <b style="color:#e6edf3">印の見方</b><br>
  レース内印: <span style="color:#e74c3c">◎</span>D1位  <span style="color:#2ecc71">○</span>D2位  <span style="color:#9b59b6">▲</span>D3位  <span style="color:#3498db">△</span>D4位  <span style="color:#7f8c8d">×</span>D5位  消し(D_pct&lt;-90% 複勝率7%)  消候(D_pct -90〜-70% 複勝率19%)<br>
  熱さ印(D1位のみ): <span style="color:#e74c3c">◎単</span>(OD&gt;8+1頭抜け 単ROI+515%)  <span style="color:#e67e22">○単</span>(OD&gt;6+gap≥3x +354%)  <span style="color:#2471a3">◎複</span>(OD&gt;6+gap≥3x 複ROI+89%)  <span style="color:#1a9ed4">○複</span>(OD&gt;5+gap≥3x +62%)<br>
  激熱: <span style="color:#ff9">◆激熱</span>(gap&gt;10x 勝率58%)  <span style="color:#ffd700">◆◆超激熱</span>(gap&gt;20x 勝率71%) ※オッズ無関係・D指標のみ判定<br>
  <span style="color:#85c1e9">参考複</span>: D2位 & OD&gt;6 & D_pct&gt;100% (複ROI+13%)
</div>
'''

HTML = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>D指標競馬新聞 {target_date}</title>
<style>{CSS}</style>
</head>
<body>
<div class="top-bar">
  <h1>D指標競馬新聞</h1>
  <div class="date">{target_date} — {venues_str}</div>
  <div class="subtitle">D = sub_cs × sub_ri ÷ (cur_r × sub_r)　実績: 8796レース / 2023-07〜</div>
  <div style="color:#666;font-size:11px;margin-top:4px">更新: {now_str}</div>
</div>
<nav class="top-nav">{nav_links}</nav>
{legend}
{summary_html}
{venue_html}
<footer>D指標競馬新聞 / keiba-dragon | <a href="https://github.com/keiba-dragon/horse_racing_ai" style="color:#4a9eff">GitHub</a></footer>
</body>
</html>"""

os.makedirs(OUTPUT_DIR, exist_ok=True)
ts = datetime.datetime.now().strftime('%Y%m%d_%H%M')
out_path = os.path.join(OUTPUT_DIR, f'd_newspaper_{target_date}_{ts}.html')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(HTML)

print(f"出力: {out_path}")

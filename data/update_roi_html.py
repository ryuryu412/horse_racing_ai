import pandas as pd
import os, re
from datetime import datetime

def normalize_shikibetsu(s):
    s = str(s).strip()
    if '3連単' in s or '３連単' in s: return '3連単'
    if '3連複' in s or '３連複' in s: return '3連複'
    if 'ワイド' in s: return 'ワイド'
    if '馬連' in s: return '馬連'
    if '馬単' in s: return '馬単'
    if '枠連' in s: return '枠連'
    if '単勝' in s: return '単勝'
    if '複勝' in s: return '複勝'
    return s

# all_tohyo.csv 読み込み（数値済み）
df = pd.read_csv('C:/Users/tsuch/Desktop/horse_racing_ai/data/tohyo/all_tohyo.csv', encoding='utf-8-sig')
df['式別_norm'] = df['式別'].apply(normalize_shikibetsu)
df['日付'] = df['日付'].astype(str)

# 全期間集計
total_buy = df['購入金額'].sum()
total_pay = df['払戻金額'].sum()
total_ret = df['返還金額'].sum()
net = total_pay + total_ret - total_buy
roi = net / total_buy * 100 if total_buy > 0 else 0

# 日付別
daily = df.groupby('日付').agg({'購入金額':'sum','払戻金額':'sum','返還金額':'sum'}).reset_index()
daily['損益'] = daily['払戻金額'] + daily['返還金額'] - daily['購入金額']
daily['ROI'] = (daily['損益'] / daily['購入金額'].replace(0, float('nan')) * 100).round(1)
daily = daily.sort_values('日付').reset_index(drop=True)
daily['累計損益'] = daily['損益'].cumsum()
plus_days = int((daily['損益'] > 0).sum())
total_days = len(daily)

# 式別集計
by_type = df.groupby('式別_norm').agg({'購入金額':'sum','払戻金額':'sum','返還金額':'sum'}).reset_index()
by_type['損益'] = by_type['払戻金額'] + by_type['返還金額'] - by_type['購入金額']
by_type['ROI'] = (by_type['損益'] / by_type['購入金額'].replace(0, float('nan')) * 100).round(1)
by_type = by_type[by_type['購入金額'] > 0].sort_values('購入金額', ascending=False)

today = datetime.now().strftime('%Y/%m/%d')

def fmt(v):
    return f'+{v:,.0f}円' if v >= 0 else f'{v:,.0f}円'
def col(v):
    return '#2d862d' if v >= 0 else '#c0392b'
def roi_str(v):
    if pd.isna(v): return '-'
    return f'+{v:.1f}%' if v >= 0 else f'{v:.1f}%'

# 式別テーブル行
type_rows = ''
for _, r in by_type.iterrows():
    roi_val = r['ROI'] if pd.notna(r['ROI']) else 0
    type_rows += f'''<tr>
<td style="font-weight:bold">{r["式別_norm"]}</td>
<td style="text-align:right">{r["購入金額"]:,.0f}円</td>
<td style="text-align:right">{r["払戻金額"]:,.0f}円</td>
<td style="color:{col(r["損益"])};font-weight:bold;text-align:right">{fmt(r["損益"])}</td>
<td style="color:{col(roi_val)};font-weight:bold;text-align:right">{roi_str(roi_val)}</td>
</tr>'''

# 日付別テーブル行
daily_rows = ''
for _, r in daily.iterrows():
    d = str(r['日付'])
    ds = f"{d[:4]}/{d[4:6]}/{d[6:8]}"
    roi_val = r['ROI'] if pd.notna(r['ROI']) else 0
    daily_rows += f'''<tr>
<td style="text-align:center">{ds}</td>
<td style="text-align:right">{r["購入金額"]:,.0f}円</td>
<td style="text-align:right">{r["払戻金額"]:,.0f}円</td>
<td style="text-align:right">{r["返還金額"]:,.0f}円</td>
<td style="color:{col(r["損益"])};font-weight:bold;text-align:right">{fmt(r["損益"])}</td>
<td style="color:{col(roi_val)};font-weight:bold;text-align:right">{roi_str(roi_val)}</td>
<td style="color:{col(r["累計損益"])};font-weight:bold;text-align:right">{fmt(r["累計損益"])}</td>
</tr>'''

# グラフ用データ
chart_labels = [f"{str(r['日付'])[4:6]}/{str(r['日付'])[6:8]}" for _, r in daily.iterrows()]
chart_daily  = [int(r['損益']) for _, r in daily.iterrows()]
chart_cumul  = [int(r['累計損益']) for _, r in daily.iterrows()]
chart_colors = ['"#2d862d"' if v >= 0 else '"#c0392b"' for v in chart_daily]

labels_js  = '[' + ','.join(f'"{l}"' for l in chart_labels) + ']'
daily_js   = '[' + ','.join(str(v) for v in chart_daily) + ']'
cumul_js   = '[' + ','.join(str(v) for v in chart_cumul) + ']'
colors_js  = '[' + ','.join(chart_colors) + ']'

html = f'''<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8">
<title>実際の馬券ROI 2026</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
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
.chart-wrap{{background:#16213e;border-radius:8px;padding:16px;margin-bottom:24px}}
</style></head><body>
<h2>実際の馬券ROI　（最終更新: {today}）</h2>
<div class="summary">
  <div class="card"><div>累計損益</div><div class="val" style="color:{col(net)}">{fmt(net)}</div></div>
  <div class="card"><div>累計ROI</div><div class="val" style="color:{col(roi)}">{roi_str(roi)}</div></div>
  <div class="card"><div>総投資</div><div class="val">{total_buy:,.0f}円</div></div>
  <div class="card"><div>総回収</div><div class="val">{total_pay+total_ret:,.0f}円</div></div>
  <div class="card"><div>プラス日数</div><div class="val">{plus_days}/{total_days}日</div></div>
</div>

<h3>日別損益グラフ</h3>
<div class="chart-wrap">
  <canvas id="dailyChart" height="90"></canvas>
</div>
<h3>累計損益グラフ</h3>
<div class="chart-wrap">
  <canvas id="cumulChart" height="80"></canvas>
</div>

<script>
const labels = {labels_js};
const dailyData = {daily_js};
const cumulData = {cumul_js};
const barColors = {colors_js};

new Chart(document.getElementById('dailyChart'), {{
  type: 'bar',
  data: {{
    labels: labels,
    datasets: [{{
      label: '日別損益（円）',
      data: dailyData,
      backgroundColor: barColors,
      borderRadius: 3
    }}]
  }},
  options: {{
    plugins: {{
      legend: {{ labels: {{ color: '#e0e0e0' }} }},
      tooltip: {{ callbacks: {{ label: ctx => (ctx.raw >= 0 ? '+' : '') + ctx.raw.toLocaleString() + '円' }} }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#aaa', maxRotation: 45 }}, grid: {{ color: '#2a2a4a' }} }},
      y: {{ ticks: {{ color: '#aaa', callback: v => (v>=0?'+':'')+v.toLocaleString()+'円' }}, grid: {{ color: '#2a2a4a' }} }}
    }}
  }}
}});

new Chart(document.getElementById('cumulChart'), {{
  type: 'line',
  data: {{
    labels: labels,
    datasets: [{{
      label: '累計損益（円）',
      data: cumulData,
      borderColor: '#f0c040',
      backgroundColor: 'rgba(240,192,64,0.1)',
      borderWidth: 2,
      pointRadius: 3,
      fill: true,
      tension: 0.3
    }}]
  }},
  options: {{
    plugins: {{
      legend: {{ labels: {{ color: '#e0e0e0' }} }},
      tooltip: {{ callbacks: {{ label: ctx => (ctx.raw >= 0 ? '+' : '') + ctx.raw.toLocaleString() + '円' }} }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#aaa', maxRotation: 45 }}, grid: {{ color: '#2a2a4a' }} }},
      y: {{ ticks: {{ color: '#aaa', callback: v => (v>=0?'+':'')+v.toLocaleString()+'円' }}, grid: {{ color: '#2a2a4a' }} }}
    }}
  }}
}});
</script>

<h3>式別ROI</h3>
<table><thead><tr>
<th>式別</th><th>投資</th><th>払戻</th><th>損益</th><th>ROI</th>
</tr></thead><tbody>
{type_rows}
</tbody></table>

<h3>日付別損益</h3>
<table><thead><tr>
<th>日付</th><th>購入金額</th><th>払戻金額</th><th>返還金額</th><th>日計</th><th>ROI</th><th>累計損益</th>
</tr></thead><tbody>
{daily_rows}
</tbody></table>
</body></html>'''

out = 'G:/マイドライブ/競馬AI/ROI/actual_bet_roi.html'
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'更新完了: {out}')
print(f'損益: {fmt(net)} / ROI: {roi_str(roi)} / {plus_days}/{total_days}日')

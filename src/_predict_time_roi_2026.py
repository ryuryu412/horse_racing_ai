# coding: utf-8
"""予想時点ROI集計 -> G:/マイドライブ/競馬AI/predict_time_roi_2026.html
cache.pkl（06_predict_from_card.py 実行時に保存）から予想時点の印を取得し、
結果確認CSVで勝敗判定。モデル再計算なし。
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import os, pickle, glob, re

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _zen(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    m = re.search(r'\d+', s)
    return int(m.group()) if m else np.nan

def _dnum_from_str(s):
    """'2026.3.28' や '2026/3/28' → 260328"""
    parts = str(s).replace('/', '.').split('.')
    if len(parts) == 3:
        return (int(parts[0]) - 2000) * 10000 + int(parts[1]) * 100 + int(parts[2])
    return None

# ── キャッシュ一覧 ──────────────────────────────────────────────
cache_files = sorted(glob.glob(os.path.join(base_dir, 'data', 'raw', 'cache', '*.cache.pkl')))
if not cache_files:
    print("cache.pkl が見つかりません。06_predict_from_card.py を先に実行してください。")
    sys.exit(0)

print(f"cache.pkl: {len(cache_files)}件")

# ── 結果確認CSV 一覧（日付 → ファイルパス）──────────────────────
result_csvs = sorted(glob.glob(os.path.join(base_dir, 'data', 'raw', 'results', '出馬表形式*結果確認.csv')))
result_csv_by_dnum = {}
for rf in result_csvs:
    try:    tmp = pd.read_csv(rf, encoding='cp932', low_memory=False, nrows=1)
    except: tmp = pd.read_csv(rf, encoding='utf-8',  low_memory=False, nrows=1)
    if '日付S' not in tmp.columns: continue
    dn = _dnum_from_str(tmp['日付S'].iloc[0])
    if dn: result_csv_by_dnum[dn] = rf

print(f"結果確認CSV: {sorted(result_csv_by_dnum.keys())}")

# ── 各キャッシュを処理 ─────────────────────────────────────────
daily_rows = []

for cache_path in cache_files:
    with open(cache_path, 'rb') as f:
        cached = pickle.load(f)

    result       = cached['result']      # predict_date()の出力（_印なし）
    card_df      = cached['card_df']
    dnum         = cached['target_date']
    predicted_at = cached.get('predicted_at', '')  # 予想実行時刻

    d = str(int(dnum))
    date_str = f"20{d[:2]}/{d[2:4]}/{d[4:6]}"
    if predicted_at:
        date_str += f"<br><small style='color:#aaa'>{predicted_at}予想</small>"

    # card_dfから朝オッズをマージして予想時点の印を再計算
    card_map = {'枠番':'dc_枠番','馬番':'dc_馬番','単勝オッズ':'dc_単勝オッズ'}
    card_cols = ['馬名S'] + [c for c in card_map if c in card_df.columns]
    card_disp = card_df[card_cols].drop_duplicates('馬名S').rename(
        columns={k: v for k, v in card_map.items() if k in card_df.columns})
    res = result.merge(card_disp, on='馬名S', how='left')

    # 予想時点の印を計算（06_predict_from_card.py と同じロジック）
    cur_diff = pd.to_numeric(res.get('cur_偏差値の差'), errors='coerce')
    sub_diff = pd.to_numeric(res.get('sub_偏差値の差'), errors='coerce')
    cur_r    = pd.to_numeric(res.get('cur_ランカー順位'), errors='coerce')
    sub_r    = pd.to_numeric(res.get('sub_ランカー順位'), errors='coerce')
    both_r1  = (cur_r == 1) & (sub_r == 1)
    star     = (cur_r <= 3) & (sub_r <= 3) & ~both_r1
    _odds    = pd.to_numeric(res.get('dc_単勝オッズ'), errors='coerce')
    if _odds.isna().all() and '単勝オッズ' in res.columns:
        _odds = pd.to_numeric(res['単勝オッズ'], errors='coerce')
    ok3 = _odds.isna() | (_odds >= 3)
    ok5 = _odds.isna() | (_odds >= 5)
    res['_印'] = ''
    res.loc[star & ~((cur_r<=2)&(sub_r<=2)) & (sub_diff>=10) & ok5, '_印'] = '☆'
    res.loc[(cur_r<=2)&(sub_r<=2)&~both_r1 & (sub_diff>=10) & ok5, '_印'] = '▲'
    res.loc[both_r1 & (sub_diff>=10) & ok3, '_印'] = '〇'
    res.loc[both_r1 & (cur_diff>=10) & (sub_diff>=10) & ok5, '_印'] = '激熱'

    res['朝オッズ'] = _odds

    # 結果確認CSVがない場合
    if dnum not in result_csv_by_dnum:
        print(f"{date_str}  結果確認CSV なし → 印のみ集計")
        has_result = False
        n_g = int((res['_印'] == '激熱').sum())
        n_o = int((res['_印'] == '〇').sum())
        n_d = int((res['_印'] == '▲').sum())
        n_s = int((res['_印'] == '☆').sum())
        daily_rows.append({
            '日付': date_str, '日付_num': dnum, 'has_result': False,
            '激熱_n': n_g, '激熱_w': 0, '激熱_pf': 0, '激熱_roi': 0,
            '〇_n': n_o,   '〇_w': 0,   '〇_pf': 0,   '〇_roi': 0,
            '▲_n': n_d,   '▲_w': 0,   '▲_pf': 0,   '▲_roi': 0,
            '☆_n': n_s,   '☆_w': 0,   '☆_pf': 0,   '☆_roi': 0,
            '計_pf': 0, '計_roi': 0,
            '計_tb': n_g*1000 + n_o*300 + n_d*500 + n_s*200,
        })
        continue

    # 結果確認CSV 読み込み
    rf = result_csv_by_dnum[dnum]
    try:    dfr = pd.read_csv(rf, encoding='cp932', low_memory=False)
    except: dfr = pd.read_csv(rf, encoding='utf-8',  low_memory=False)

    dfr['着_num'] = dfr['着'].apply(_zen)
    dfr['_tan']   = pd.to_numeric(dfr['単勝'],    errors='coerce')
    dfr['_fuku']  = pd.to_numeric(dfr.get('複勝', pd.Series(np.nan, index=dfr.index)), errors='coerce')

    # 馬名Sで結果とマージ
    merged = res.merge(dfr[['馬名S', '着_num', '_tan', '_fuku']].drop_duplicates('馬名S'),
                       on='馬名S', how='left')

    # レースキーを作成して単勝配当をレース全馬に展開
    race_key_cols = [c for c in ['開催', 'Ｒ'] if c in merged.columns]
    if race_key_cols:
        merged['_race_key'] = merged[race_key_cols].astype(str).agg('_'.join, axis=1)
    else:
        merged['_race_key'] = '0'

    win_pay = (merged[merged['着_num'] == 1]
               .drop_duplicates('_race_key')
               .set_index('_race_key')['_tan'])
    merged['_tansho'] = merged['_race_key'].map(win_pay)

    fuku_pay = merged[merged['着_num'] <= 3].groupby('_race_key')['_fuku'].mean()
    merged['_fukusho'] = merged['_race_key'].map(fuku_pay)

    has_result = merged['_tansho'].notna().any()

    def _roi(mask, bet):
        b = merged[mask]
        n = len(b)
        if n == 0: return 0, 0, 0, 0
        tb = n * bet
        if not has_result: return n, None, None, None
        hits = int((b['着_num'] == 1).sum())
        ret  = b[b['着_num'] == 1]['_tansho'].sum() * bet / 100
        pf   = int(ret - tb)
        roi  = ret / tb - 1.0 if tb > 0 else 0
        return n, pf, roi, hits

    n_g, pf_g, roi_g, w_g = _roi(merged['_印'] == '激熱', 1000)
    n_o, pf_o, roi_o, w_o = _roi(merged['_印'] == '〇',    300)
    n_d, pf_d, roi_d, w_d = _roi(merged['_印'] == '▲',    500)
    n_s, pf_s, roi_s, w_s = _roi(merged['_印'] == '☆',    200)

    if has_result:
        total_tb = n_g*1000 + n_o*300 + n_d*500 + n_s*200
        total_ret = 0
        for mask, bet in [(merged['_印']=='激熱',1000),(merged['_印']=='〇',300),
                          (merged['_印']=='▲',500),(merged['_印']=='☆',200)]:
            b = merged[mask]
            total_ret += b[b['着_num']==1]['_tansho'].sum() * bet / 100
        total_pf  = int(total_ret - total_tb)
        total_roi = total_ret / total_tb - 1.0 if total_tb > 0 else 0
        sign = '+' if total_pf >= 0 else ''
        roi_str = f"{sign}{total_pf:,}円 ({total_roi:+.1%})"
    else:
        total_pf = 0; total_roi = 0; roi_str = "結果未格納"

    print(f"{date_str}  激熱{n_g}/{w_g}  〇{n_o}/{w_o}  ▲{n_d}/{w_d}  ☆{n_s}/{w_s}  計{roi_str}")

    daily_rows.append({
        '日付': date_str, '日付_num': dnum, 'has_result': has_result,
        '激熱_n': n_g, '激熱_w': w_g or 0, '激熱_pf': pf_g or 0, '激熱_roi': roi_g or 0,
        '〇_n': n_o,   '〇_w': w_o or 0,   '〇_pf': pf_o or 0,   '〇_roi': roi_o or 0,
        '▲_n': n_d,   '▲_w': w_d or 0,   '▲_pf': pf_d or 0,   '▲_roi': roi_d or 0,
        '☆_n': n_s,   '☆_w': w_s or 0,   '☆_pf': pf_s or 0,   '☆_roi': roi_s or 0,
        '計_pf': total_pf, '計_roi': total_roi,
        '計_tb': n_g*1000 + n_o*300 + n_d*500 + n_s*200,
    })

if not daily_rows:
    print("集計データなし")
    sys.exit(0)

df_daily = pd.DataFrame(daily_rows)
df_res_only = df_daily[df_daily['has_result']]

# 累計
cum_pf = 0; cum_tb = 0; cum_ret = 0
for i, r in df_daily.iterrows():
    if r['has_result']:
        cum_tb  += r['計_tb']
        cum_ret += r['計_tb'] * (r['計_roi'] + 1)
        cum_pf  += r['計_pf']
    df_daily.loc[i, '累計_pf'] = int(cum_pf)
cum_roi_final = cum_ret / cum_tb - 1.0 if cum_tb > 0 else 0

# ── HTML生成 ────────────────────────────────────────────────────
def pf_cell(pf, roi, has_res=True):
    if not has_res:
        return '<td style="text-align:center;color:#888">-</td>'
    if pf is None: return '<td>-</td>'
    sign = '+' if pf >= 0 else ''
    col = '#2d862d' if pf >= 0 else '#c0392b'
    return f'<td style="color:{col};font-weight:bold;text-align:right">{sign}{pf:,}円<br><small>({roi:+.1%})</small></td>'

def mark_cell(n, w, pf, roi, has_res):
    if n == 0: return '<td style="text-align:center;color:#555">-</td>'
    col = '#2d862d' if (pf or 0) >= 0 else '#c0392b'
    pf_str = f'<br><small style="color:{col}">{("+" if (pf or 0)>=0 else "")}{int(pf or 0):,}円</small>' if has_res else ''
    return f'<td style="text-align:center">{n}頭/{w}的{pf_str}</td>'

rows_html = ''
for _, r in df_daily.iterrows():
    hr = bool(r['has_result'])
    cum_col = '#2d862d' if r['累計_pf'] >= 0 else '#c0392b'
    rows_html += f'''<tr>
<td style="text-align:center">{r["日付"]}</td>
{mark_cell(r["激熱_n"], r["激熱_w"], r["激熱_pf"], r["激熱_roi"], hr)}
{mark_cell(r["〇_n"],    r["〇_w"],    r["〇_pf"],    r["〇_roi"],    hr)}
{mark_cell(r["▲_n"],    r["▲_w"],    r["▲_pf"],    r["▲_roi"],    hr)}
{mark_cell(r["☆_n"],    r["☆_w"],    r["☆_pf"],    r["☆_roi"],    hr)}
{pf_cell(r["計_pf"] if hr else None, r["計_roi"] if hr else None, hr)}
<td style="color:{cum_col};font-weight:bold;text-align:right">{("+" if r["累計_pf"]>=0 else "")}{int(r["累計_pf"]):,}円</td>
</tr>'''

total_days = len(df_res_only)
plus_days  = int((df_res_only['計_pf'] >= 0).sum()) if total_days > 0 else 0
col_all    = '#2d862d' if cum_pf >= 0 else '#c0392b'

html = f'''<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8">
<title>2026年 予想時点ROI</title>
<style>
body{{font-family:"Hiragino Kaku Gothic Pro",Meiryo,sans-serif;background:#1a1a2e;color:#e0e0e0;padding:20px}}
h2{{color:#f0c040;text-align:center}}
.note{{text-align:center;font-size:0.85em;color:#aaa;margin:-8px 0 16px}}
.summary{{display:flex;gap:20px;justify-content:center;margin:10px 0 20px;flex-wrap:wrap}}
.card{{background:#16213e;border-radius:8px;padding:12px 20px;text-align:center;min-width:120px}}
.card .val{{font-size:1.6em;font-weight:bold}}
table{{width:100%;border-collapse:collapse;font-size:0.85em}}
th{{background:#16213e;color:#f0c040;padding:6px 8px;text-align:center;position:sticky;top:0}}
td{{padding:5px 8px;border-bottom:1px solid #2a2a4a}}
tr:nth-child(even){{background:#16213e88}}
tr:hover{{background:#1a3a5a}}
</style></head><body>
<h2>2026年 予想時点ROI　（最終更新: {df_res_only["日付"].iloc[-1] if total_days > 0 else "-"}）</h2>
<p class="note">予想実行時点のオッズ・印を使用（最終オッズによる印変動なし）</p>
<div class="summary">
  <div class="card"><div>累計損益</div><div class="val" style="color:{col_all}">{("+" if cum_pf>=0 else "")}{cum_pf:,}円</div></div>
  <div class="card"><div>累計ROI</div><div class="val" style="color:{col_all}">{cum_roi_final:+.1%}</div></div>
  <div class="card"><div>プラス日数</div><div class="val">{plus_days}/{total_days}日</div></div>
  <div class="card"><div>激熱的中率</div><div class="val">{int(df_res_only["激熱_w"].sum()) if total_days>0 else 0}/{int(df_res_only["激熱_n"].sum()) if total_days>0 else 0}頭</div></div>
  <div class="card"><div>▲的中率</div><div class="val">{int(df_res_only["▲_w"].sum()) if total_days>0 else 0}/{int(df_res_only["▲_n"].sum()) if total_days>0 else 0}頭</div></div>
</div>
<table><thead><tr>
<th>日付</th>
<th>激熱<br>単1000円</th>
<th>〇<br>単300円</th>
<th>▲<br>単500円</th>
<th>☆<br>単200円</th>
<th>日計</th><th>累計損益</th>
</tr></thead><tbody>
{rows_html}
</tbody></table></body></html>'''

out = r'G:\マイドライブ\競馬AI\predict_time_roi_2026.html'
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"\n出力: {out}")

docs_out = 'C:/Users/tsuch/Desktop/horse_racing_ai/docs/predict_time_roi_2026.html'
os.makedirs(os.path.dirname(docs_out), exist_ok=True)
with open(docs_out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"出力: {docs_out}")
if total_days > 0:
    print(f"累計: {('+' if cum_pf>=0 else '')}{cum_pf:,}円  ROI{cum_roi_final:+.1%}  {plus_days}/{total_days}日プラス")
else:
    print("結果確認CSV付きのデータなし（今後追加されると集計開始）")

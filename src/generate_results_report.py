"""
結果照合レポート生成スクリプト
予測と実績を並べたHTML/PDFレポートを出力する
Usage: python src/generate_results_report.py <結果CSV> [--date YYMMDD] [--out <path>]
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import os, pickle, json, re, time, argparse, subprocess

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, 'models_2025')

# ── ヘルパー ──────────────────────────────────────────────────────
def get_distance_band(dist):
    m = re.search(r'\d+', str(dist))
    if not m: return None
    d = int(m.group())
    if d <= 1400:   return '短距離'
    elif d <= 1800: return 'マイル'
    elif d <= 2200: return '中距離'
    else:           return '長距離'

def get_class_group(class_rank):
    try:
        r = float(class_rank)
    except: return '3勝以上'
    if np.isnan(r): return '3勝以上'
    r = int(r)
    if r == 1:   return '新馬'
    elif r == 2: return '未勝利'
    elif r == 3: return '1勝'
    elif r == 4: return '2勝'
    elif r >= 5: return '3勝以上'
    return '3勝以上'

def zen_to_num(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    m = re.search(r'\d+', s)
    return int(m.group()) if m else np.nan

VENUE_MAP = {
    '中山': '中', '東京': '東', '阪神': '阪', '中京': '名',
    '京都': '京', '函館': '函', '新潟': '新', '小倉': '小',
    '札幌': '札', '福島': '福',
}

# ── 引数 ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('result_csv', nargs='?',
    default=os.path.join(base_dir, 'data', 'raw', '出馬表形式3月21日結果確認.csv'))
parser.add_argument('--date', default=None, help='日付 YYMMDD (例: 260321)')
parser.add_argument('--out', default=None, help='出力HTMLパス')
args = parser.parse_args()

res_path = args.result_csv

# ── 結果CSV読み込み ──────────────────────────────────────────────
print(f"結果CSV: {res_path}")
try:
    df_res = pd.read_csv(res_path, encoding='cp932', low_memory=False)
except Exception:
    df_res = pd.read_csv(res_path, encoding='utf-8', low_memory=False)

df_res['着_num'] = df_res['着'].apply(zen_to_num)
df_res['target_win']   = (df_res['着_num'] == 1).astype(int)
df_res['target_place'] = (df_res['着_num'] <= 3).astype(int)
df_res['単勝_num'] = pd.to_numeric(df_res['単勝'], errors='coerce')
df_res['複勝_num'] = pd.to_numeric(df_res['複勝'], errors='coerce')
df_res['単オッズ_num'] = pd.to_numeric(df_res['単オッズ'], errors='coerce')

# 単勝配当をレース全馬にブロードキャスト
race_w = df_res[df_res['target_win']==1][['場所','Ｒ','単勝_num']].rename(columns={'単勝_num':'tansho'})
df_res = df_res.merge(race_w, on=['場所','Ｒ'], how='left')

# 日付を推定
if args.date:
    date_num = int(args.date)
elif '日付S' in df_res.columns:
    ds = str(df_res['日付S'].iloc[0])
    parts = ds.replace('/', '.').split('.')
    date_num = (int(parts[0])-2000)*10000 + int(parts[1])*100 + int(parts[2])
else:
    date_num = 260321
date_str = f"20{str(date_num)[0:2]}/{str(date_num)[2:4]}/{str(date_num)[4:6]}"

n_races = df_res[['場所','Ｒ']].drop_duplicates().__len__()
print(f"日付: {date_str} / {len(df_res)}頭 / {n_races}レース")

# ── コースキー設定 ──────────────────────────────────────────────
df_res['会場'] = df_res['場所'].astype(str).map(VENUE_MAP).fillna(df_res['場所'].astype(str))
if 'コースマーク' in df_res.columns:
    cm = df_res['コースマーク'].astype(str).str.strip()
    df_res['会場'] = df_res['会場'] + cm.where(cm.isin(['A','B','C']), '')
df_res['_surface'] = df_res['芝ダ'].astype(str).str.strip()
df_res['cur_key'] = df_res['会場'] + '_' + df_res['_surface'] + df_res['距離'].astype(str)
df_res['_dist_band'] = df_res['距離'].apply(get_distance_band)
mask = (df_res['_surface'] == 'ダ') & (df_res['_dist_band'].isin(['中距離', '長距離']))
df_res.loc[mask, '_dist_band'] = '中長距離'
if 'クラス_rank' in df_res.columns:
    df_res['_cls_group'] = df_res['クラス_rank'].apply(get_class_group)
elif 'クラス' in df_res.columns:
    _cls_map = {'新馬': 1, '未勝利': 2, '1勝': 3, '2勝': 4}
    def _cls_to_rank(v):
        s = str(v).strip()
        for k, r in _cls_map.items():
            if k in s:
                return r
        return 5
    df_res['クラス_rank'] = df_res['クラス'].apply(_cls_to_rank)
    df_res['_cls_group'] = df_res['クラス_rank'].apply(get_class_group)
else:
    df_res['_cls_group'] = '3勝以上'
df_res['sub_key'] = df_res['_surface'] + '_' + df_res['_dist_band'].astype(str) + '_' + df_res['_cls_group'].astype(str)

# ── モデル読み込み ──────────────────────────────────────────────
with open(os.path.join(model_dir, 'model_info.json'), encoding='utf-8') as f:
    cur_info = json.load(f)
cur_features = cur_info['features']
cur_models   = cur_info['models']
with open(os.path.join(model_dir, 'ranker', 'ranker_info.json'), encoding='utf-8') as f:
    cur_rankers = json.load(f).get('rankers', {})
with open(os.path.join(model_dir, 'submodel', 'submodel_info.json'), encoding='utf-8') as f:
    sub_info = json.load(f)
sub_features = sub_info['features']
sub_models   = sub_info['models']
with open(os.path.join(model_dir, 'submodel_ranker', 'class_ranker_info.json'), encoding='utf-8') as f:
    sub_rankers = json.load(f).get('rankers', {})

# ── 特徴量マージ（Parquetから最新）─────────────────────────────
print("Parquet読み込み中...")
t0 = time.time()
df_feat = pd.read_parquet(os.path.join(base_dir, 'data', 'processed', 'all_venues_features.parquet'))
print(f"完了: {time.time()-t0:.1f}秒")
feat_name_col = '馬名S'
df_latest = df_feat.sort_values('日付').groupby(feat_name_col).last().reset_index()

all_feats = list(set(cur_features + sub_features))
feat_subset = [feat_name_col] + [c for c in all_feats if c in df_latest.columns]
# 間隔・前距離の計算に使う列も取得
for _ec in ['日付', '距離']:
    if _ec in df_latest.columns and _ec not in feat_subset:
        feat_subset.append(_ec)
df_merge = df_res.merge(df_latest[feat_subset], on=feat_name_col, how='left', suffixes=('', '_f'))

# ── 間隔・前距離・性別_numを正しく計算 ──
df_merge['性別_num'] = df_merge['性別'].map({'牡': 0, '牝': 1, 'セ': 2}).astype(float)
if '距離_f' in df_merge.columns:
    df_merge['前距離'] = df_merge['距離_f'].astype(str).str.extract(r'(\d+)').iloc[:, 0].astype(float)
if '日付_f' in df_merge.columns:
    def _yymmdd(v):
        try:
            v = int(v)
            return pd.Timestamp(2000 + v // 10000, (v // 100) % 100, v % 100)
        except:
            return pd.NaT
    _cur_date = pd.Timestamp(int('20' + str(date_num)[:2]), int(str(date_num)[2:4]), int(str(date_num)[4:6]))
    df_merge['間隔'] = ((_cur_date - df_merge['日付_f'].apply(_yymmdd)).dt.days / 7).round(0)

for col in all_feats:
    if col in df_merge.columns:
        df_merge[col] = pd.to_numeric(df_merge[col], errors='coerce')

# ── モデルキャッシュ ─────────────────────────────────────────────
cur_model_cache = {}; cur_ranker_cache = {}
sub_model_cache = {}; sub_ranker_cache = {}
for ck in df_merge['cur_key'].dropna().unique():
    if ck in cur_models:
        p = os.path.join(model_dir, cur_models[ck]['win'])
        if os.path.exists(p):
            with open(p,'rb') as f: m = pickle.load(f)
            cur_model_cache[ck] = (m, m.booster_.feature_name())
    if ck in cur_rankers:
        p = os.path.join(model_dir, 'ranker', cur_rankers[ck])
        if os.path.exists(p):
            with open(p,'rb') as f: cur_ranker_cache[ck] = pickle.load(f)
for sk in df_merge['sub_key'].dropna().unique():
    if sk in sub_models:
        p = os.path.join(model_dir, 'submodel', sub_models[sk]['win'])
        if os.path.exists(p):
            with open(p,'rb') as f: m = pickle.load(f)
            sub_model_cache[sk] = (m, m.booster_.feature_name())
    if sk in sub_rankers:
        p = os.path.join(model_dir, 'submodel_ranker', sub_rankers[sk])
        if os.path.exists(p):
            with open(p,'rb') as f: sub_ranker_cache[sk] = pickle.load(f)
print(f"モデル: 距離{len(cur_model_cache)} クラス{len(sub_model_cache)}")

# ── レース別予測 ─────────────────────────────────────────────────
df_merge['cur_diff'] = np.nan; df_merge['cur_rank'] = np.nan; df_merge['cur_cs'] = np.nan
df_merge['sub_diff'] = np.nan; df_merge['sub_rank'] = np.nan; df_merge['sub_cs'] = np.nan

all_rows = []
for gk, idx in df_merge.groupby(['場所','Ｒ'], sort=False).groups.items():
    sub = df_merge.loc[idx].copy()
    ck = sub['cur_key'].iloc[0]
    sk = sub['sub_key'].iloc[0]
    if ck in cur_model_cache:
        m, wf = cur_model_cache[ck]
        for c in wf:
            if c not in sub.columns: sub[c] = np.nan
        prob = m.predict_proba(sub[wf])[:, 1]
        st = cur_models[ck].get('stats', {})
        wm = st.get('win_mean', prob.mean()); ws = st.get('win_std', prob.std())
        cs = 50 + 10*(prob - wm)/(ws if ws > 0 else 1)
        rm = prob.mean(); rs = prob.std()
        rs_val = 50 + 10*(prob - rm)/(rs if rs > 0 else 1)
        sub['cur_cs']   = cs
        sub['cur_diff'] = rs_val - cs
        if ck in cur_ranker_cache:
            scores = cur_ranker_cache[ck].predict(sub[cur_features])
            sub['cur_rank'] = pd.Series(scores, index=sub.index).rank(ascending=False, method='min').astype(int)
    if sk in sub_model_cache:
        m, wf = sub_model_cache[sk]
        for c in wf:
            if c not in sub.columns: sub[c] = np.nan
        prob = m.predict_proba(sub[wf])[:, 1]
        st = sub_models[sk].get('stats', {})
        wm = st.get('win_mean', prob.mean()); ws = st.get('win_std', prob.std())
        cs = 50 + 10*(prob - wm)/(ws if ws > 0 else 1)
        rm = prob.mean(); rs = prob.std()
        rs_val = 50 + 10*(prob - rm)/(rs if rs > 0 else 1)
        sub['sub_cs']   = cs
        sub['sub_diff'] = rs_val - cs
        if sk in sub_ranker_cache:
            scores = sub_ranker_cache[sk].predict(sub[wf])
            sub['sub_rank'] = pd.Series(scores, index=sub.index).rank(ascending=False, method='min').astype(int)
    all_rows.append(sub)

result = pd.concat(all_rows, ignore_index=True)
print(f"予測完了: {len(result)}頭")

# ── combo_gap計算（コース偏差値の1位-2位差の合計 / 06と同一ロジック）─
result['cur_gap'] = np.nan
result['sub_gap'] = np.nan
for gk, idx in result.groupby(['場所', 'Ｒ'], sort=False).groups.items():
    s = result.loc[idx]
    for score_col, gap_col in [('cur_cs', 'cur_gap'), ('sub_cs', 'sub_gap')]:
        sc2 = s[score_col].dropna().sort_values(ascending=False).values
        result.loc[idx, gap_col] = (sc2[0] - sc2[1]) if len(sc2) >= 2 else np.nan
result['combo_gap'] = result['cur_gap'].fillna(0) + result['sub_gap'].fillna(0)

cd = pd.to_numeric(result['cur_diff'], errors='coerce')
sd = pd.to_numeric(result['sub_diff'], errors='coerce')
cr = result['cur_rank']
sr = result['sub_rank']
combo_gap = result['combo_gap']
odds_col = (pd.to_numeric(result['単オッズ_num'], errors='coerce')
            if '単オッズ_num' in result.columns
            else pd.Series(np.nan, index=result.index))

# ── 新印体系 ────────────────────────────────────────────────────
both_r1  = (cr == 1) & (sr == 1)
star     = (cr <= 3) & (sr <= 3) & ~both_r1
odds_ok3 = odds_col.isna() | (odds_col >= 3)
odds_ok5 = odds_col.isna() | (odds_col >= 5)

mask_gekiatu = both_r1 & (combo_gap >= 15) & (sd >= 10) & odds_ok3
mask_maru    = both_r1 & (combo_gap >= 10) & odds_ok3 & ~mask_gekiatu
mask_maru2   = both_r1 & (combo_gap <  10) & odds_ok3
mask_hoshi   = star & odds_ok5

result['_印'] = ''
result.loc[mask_hoshi,   '_印'] = '☆'
result.loc[mask_maru2,   '_印'] = '〇'
result.loc[mask_maru,    '_印'] = '◎'
result.loc[mask_gekiatu, '_印'] = '激熱'
n_marks = (result['_印'] != '').sum()
print(f"印付き: 激熱{(result['_印']=='激熱').sum()} ◎{(result['_印']=='◎').sum()} 〇{(result['_印']=='〇').sum()} ☆{(result['_印']=='☆').sum()} / 計{n_marks}頭")

# ── HTML生成 ─────────────────────────────────────────────────────
def fmt(val, f='+.1f', default='-'):
    try: return format(float(val), f) if pd.notna(val) else default
    except: return default

def diff_badge(val, prefix=''):
    try:
        v = float(val)
        if v >= 20: return f'<span style="color:#c0392b;font-weight:bold">{prefix}{v:+.1f}</span>'
        if v >= 15: return f'<span style="color:#e67e22;font-weight:bold">{prefix}{v:+.1f}</span>'
        if v >= 10: return f'<span style="color:#2980b9">{prefix}{v:+.1f}</span>'
        return f'{prefix}{v:+.1f}'
    except: return '-'

css = """<style>
  @page { size: A4 landscape; margin: 8mm; }
  body { font-family:'Yu Gothic','Hiragino Sans',sans-serif; font-size:11px; margin:6px; background:#f0f0f0; }
  h1 { font-size:16px; margin:6px 0; color:#1a252f; }
  h2 { font-size:13px; background:#2c3e50; color:white; padding:5px 10px; margin:14px 0 0; border-radius:4px 4px 0 0; }
  .page { page-break-after:always; break-after:page; padding:4px; }
  .page:last-child { page-break-after:avoid; }
  table { border-collapse:collapse; width:100%; background:white; margin-bottom:6px; }
  th { background:#2c3e50; color:white; padding:3px 6px; font-size:9px; text-align:center; white-space:nowrap; border:1px solid #1a252f; }
  td { padding:3px 5px; text-align:center; white-space:nowrap; font-size:10px; border:1px solid #ccc; }
  td.name { text-align:left; font-weight:bold; font-size:12px; }
  /* 結果ハイライト */
  tr.win  td { background:#fff3cd !important; }
  tr.place td { background:#d4edda !important; }
  /* 印バッジ */
  .mark-cur  { display:inline-block; background:#27ae60; color:white; font-size:8px;
               font-weight:bold; border-radius:3px; padding:1px 3px; margin-right:2px; }
  .mark-sub  { display:inline-block; background:#2980b9; color:white; font-size:8px;
               font-weight:bold; border-radius:3px; padding:1px 3px; margin-right:2px; }
  .mark-both { display:inline-block; background:#8e44ad; color:white; font-size:8px;
               font-weight:bold; border-radius:3px; padding:1px 3px; margin-right:2px; }
  /* サマリーページ */
  .summary-table { border-collapse:collapse; width:100%; background:white; }
  .summary-table th { background:#34495e; color:white; padding:6px 10px; font-size:11px; }
  .summary-table td { padding:5px 10px; border:1px solid #ccc; font-size:11px; }
  .hit-row td { background:#e8f8e8; font-weight:bold; }
  .miss-row td { background:#fff; color:#888; }
  /* ランカーヒット */
  .cur-hit { background:#eafaf1; }
  .sub-hit { background:#eaf4fb; }
  .both-hit { background:#fdf5ff; }
  /* 着順 */
  .ato-1 { font-size:14px; font-weight:bold; color:#c0392b; }
  .ato-23 { font-weight:bold; color:#27ae60; }
  .ato-n { color:#aaa; }
  /* 新印バッジ */
  .mark-gekiatu { display:inline-block; background:#c0392b; color:white; font-size:9px;
                  font-weight:bold; border-radius:3px; padding:1px 4px; margin-right:3px; }
  .mark-maru    { display:inline-block; background:#e67e22; color:white; font-size:9px;
                  font-weight:bold; border-radius:3px; padding:1px 4px; margin-right:3px; }
  .mark-maru2   { display:inline-block; background:#27ae60; color:white; font-size:9px;
                  font-weight:bold; border-radius:3px; padding:1px 4px; margin-right:3px; }
  .mark-hoshi   { display:inline-block; background:#2980b9; color:white; font-size:9px;
                  font-weight:bold; border-radius:3px; padding:1px 4px; margin-right:3px; }
  tr.row-geki td { background:#fff0f0 !important; }
  @media print { body { margin:0; } }
</style>"""

# ── サマリーページ ──────────────────────────────────────────────
def mark_strategy_row(label, mask, tan_bet, fuku_bet=0):
    bets = result[mask]
    n = len(bets)
    if n == 0:
        return (f'<tr class="miss-row"><td>{label}</td>'
                f'<td colspan="7">-</td></tr>')
    # 単勝
    tan_wins   = int(bets['target_win'].sum())
    tansho_sum = bets[bets['target_win']==1]['tansho'].sum()
    tan_tb     = n * tan_bet
    tan_ret    = tansho_sum * tan_bet / 100
    tan_roi    = tan_ret / tan_tb - 1.0 if tan_tb > 0 else 0
    tan_pf     = int(tan_ret - tan_tb)
    # 複勝
    places  = int(bets['target_place'].sum())
    p_rate  = places / n
    if fuku_bet > 0:
        fuku_sum = bets[bets['target_place']==1]['複勝_num'].sum()
        fuku_tb  = n * fuku_bet
        fuku_ret = fuku_sum * fuku_bet / 100
        fuku_roi = fuku_ret / fuku_tb - 1.0 if fuku_tb > 0 else 0
        fuku_pf  = int(fuku_ret - fuku_tb)
        fuku_cell = f'{places}/{n}頭 {fuku_roi:+.1%} {fuku_pf:+,}円'
    else:
        fuku_tb = fuku_pf = 0
        fuku_cell = f'({p_rate:.0%})'
    # 合計
    total_tb = tan_tb + fuku_tb
    total_pf = tan_pf + fuku_pf
    cls = 'hit-row' if tan_wins > 0 or (fuku_bet > 0 and places > 0) else 'miss-row'
    winners = []
    for _, r in bets[bets['target_win']==1].iterrows():
        ninki = f"{int(r['人気'])}人気" if pd.notna(r.get('人気')) else ''
        winners.append(f"{r['場所']}{int(r['Ｒ'])}R {r['馬名']}({ninki} 単{int(r['tansho'])}円)")
    winners_str = '　'.join(winners) if winners else '—'
    return (f'<tr class="{cls}"><td>{label}</td>'
            f'<td>{tan_wins}/{n}頭</td>'
            f'<td style="text-align:right;font-weight:bold">{tan_roi:+.1%}</td>'
            f'<td style="text-align:right">{tan_pf:+,}円</td>'
            f'<td>{fuku_cell}</td>'
            f'<td style="font-weight:bold;text-align:right">{total_pf:+,}円</td>'
            f'<td style="text-align:right;font-size:9px">{total_tb:,}円</td>'
            f'<td style="text-align:left;font-size:9px">{winners_str}</td></tr>')

# 本日合計ROI計算（単勝+複勝）
def _daily_total():
    # (mask, tan_bet, fuku_bet)
    rows = [
        (result['_印']=='激熱', 2000, 0),
        (result['_印']=='◎',   1000, 0),
        (result['_印']=='〇',    500, 0),
        (result['_印']=='☆',    300, 200),
    ]
    total_tb = total_ret = 0
    for mask, tb, fb in rows:
        bets = result[mask]
        n = len(bets)
        total_tb  += n * tb + n * fb
        total_ret += bets[bets['target_win']==1]['tansho'].sum() * tb / 100
        if fb > 0:
            total_ret += bets[bets['target_place']==1]['複勝_num'].sum() * fb / 100
    roi    = total_ret / total_tb - 1.0 if total_tb > 0 else 0
    profit = int(total_ret - total_tb)
    return total_tb, profit, roi

_tb, _pf, _roi = _daily_total()
_daily_str = f"本日合計　投資: {_tb:,}円　収支: {_pf:+,}円　ROI: {_roi:+.1%}"

summary_html = f"""
<table class="summary-table">
  <thead><tr>
    <th style="text-align:left;min-width:190px">印・買い目</th>
    <th>単勝的中</th><th>単勝ROI</th><th>単勝収支</th>
    <th>複勝（的中率/ROI/収支）</th>
    <th>合計収支</th><th>投資額</th>
    <th style="text-align:left">単勝的中馬</th>
  </tr></thead>
  <tbody>
    <tr><td colspan="8" style="background:#fde8e8;font-weight:bold;color:#c0392b">■ 推奨買い目（新印体系）</td></tr>
    {mark_strategy_row('激熱　単2000円',      result["_印"]=="激熱", 2000)}
    {mark_strategy_row('◎　単1000円',         result["_印"]=="◎",   1000)}
    {mark_strategy_row('〇　単500円',          result["_印"]=="〇",    500)}
    {mark_strategy_row('☆　単300+複200円',    result["_印"]=="☆",    300, 200)}
    <tr><td colspan="8" style="background:#f0f0f0;font-size:11px;font-weight:bold;padding:6px 10px">
      {_daily_str}
    </td></tr>
  </tbody>
</table>"""

page_summary = f"""<div class="page">
  <h1>競馬AI 結果照合レポート　{date_str}</h1>
  <h2>推奨買い目 的中・ROI サマリー</h2>
  {summary_html}
</div>"""

# ── レース別詳細 ─────────────────────────────────────────────────
race_pages = []
venues_order = ['中山', '阪神', '中京']
sorted_races = []
for venue in venues_order:
    vdf = result[result['場所']==venue]
    for r_num in sorted(vdf['Ｒ'].dropna().unique()):
        sorted_races.append((venue, r_num))

for (venue, r_num) in sorted_races:
    sub = result[(result['場所']==venue) & (result['Ｒ']==r_num)].copy()
    sub = sub.sort_values('着_num', na_position='last')
    race_name = sub['レース名'].iloc[0] if len(sub) > 0 else ''
    tansho_val = sub[sub['target_win']==1]['tansho'].values
    tansho_str = f"{int(tansho_val[0])}円" if len(tansho_val) > 0 and pd.notna(tansho_val[0]) else "-"
    jotai = sub['芝ダ'].iloc[0] if '芝ダ' in sub.columns else ''
    kyori = int(sub['距離'].iloc[0]) if '距離' in sub.columns else ''
    head_n = len(sub)

    rows_html = []
    for _, r in sub.iterrows():
        ato = r['着_num']
        if pd.notna(ato):
            ato_int = int(ato)
            if ato_int == 1:
                ato_str = f'<span class="ato-1">1着</span>'
                row_cls = 'win'
            elif ato_int <= 3:
                ato_str = f'<span class="ato-23">{ato_int}着</span>'
                row_cls = 'place'
            else:
                ato_str = f'<span class="ato-n">{ato_int}着</span>'
                row_cls = ''
        else:
            ato_str = '<span class="ato-n">-</span>'
            row_cls = ''

        banum = r.get('馬番', '-')
        try: banum = int(float(banum))
        except: pass

        horse = r.get('馬名', r.get('馬名S', ''))
        jockey = r.get('騎手', '')
        odds = r.get('単オッズ_num', np.nan)
        odds_str = fmt(odds, '.1f') if pd.notna(odds) else '-'
        fuku = r.get('複勝_num', np.nan)
        tansho_self = r.get('tansho', np.nan)
        tan_str  = f'<b>{int(tansho_self)}円</b>' if pd.notna(tansho_self) and r['target_win']==1 else '-'
        fuku_str = f'<b>{int(fuku)}円</b>'        if pd.notna(fuku)        and r['target_place']==1 else '-'

        # 印
        ink = r.get('_印', '')
        _MARK_HTML = {
            '激熱': '<span class="mark-gekiatu">激熱</span>',
            '◎':   '<span class="mark-maru">◎</span>',
            '〇':   '<span class="mark-maru2">〇</span>',
            '☆':   '<span class="mark-hoshi">☆</span>',
        }
        ink_html = _MARK_HTML.get(ink, '')
        row_extra = ' row-geki' if ink == '激熱' else ''

        cur_r = r.get('cur_rank')
        sub_r = r.get('sub_rank')
        cur_d = r.get('cur_diff')
        sub_d = r.get('sub_diff')
        cgap  = r.get('combo_gap', np.nan)

        rows_html.append(
            f'<tr class="{row_cls}{row_extra}">'
            f'<td>{ato_str}</td>'
            f'<td>{banum}</td>'
            f'<td style="text-align:center;min-width:36px">{ink_html}</td>'
            f'<td class="name">{horse}</td>'
            f'<td style="font-size:9px">{jockey}</td>'
            f'<td style="font-size:9px">{fmt(cur_r, ".0f", "-")}位</td>'
            f'<td style="font-size:9px">{fmt(sub_r, ".0f", "-")}位</td>'
            f'<td>{diff_badge(cgap)}</td>'
            f'<td>{diff_badge(sub_d)}</td>'
            f'<td style="color:#e67e22">{odds_str}</td>'
            f'<td style="color:#c0392b">{tan_str}</td>'
            f'<td style="color:#27ae60">{fuku_str}</td>'
            f'</tr>'
        )

    race_pages.append(f"""<div class="page">
  <h2>【{venue} {int(r_num)}R】{race_name}　{jotai}{kyori}m　{head_n}頭立　単勝{tansho_str}</h2>
  <table>
    <thead><tr>
      <th>着順</th><th>馬番</th><th>印</th><th style="text-align:left;min-width:130px">馬名</th>
      <th>騎手</th>
      <th>距離Rnk</th><th>クラスRnk</th>
      <th>combo_gap</th><th>sub_diff</th>
      <th>単オッズ</th><th style="color:#ffcccc">単勝配当</th><th style="color:#ccffcc">複勝配当</th>
    </tr></thead>
    <tbody>{"".join(rows_html)}</tbody>
  </table>
</div>""")

# ── 出力 ─────────────────────────────────────────────────────────
html_content = f"""<!DOCTYPE html>
<html lang="ja"><head>
<meta charset="UTF-8">
<title>競馬AI 結果照合 {date_str}</title>
{css}
</head><body>
{page_summary}
{"".join(race_pages)}
</body></html>"""

# 出力先
if args.out:
    out_path = args.out
else:
    out_dir = r'G:\マイドライブ\競馬AI'
    if not os.path.exists(out_dir):
        out_dir = os.path.join(base_dir, 'output')
        os.makedirs(out_dir, exist_ok=True)
    fname = f"result_{str(date_num)}.html"
    out_path = os.path.join(out_dir, fname)

with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html_content)
print(f"\nHTML出力: {out_path}")

# ── Chrome PDF ──────────────────────────────────────────────────
pdf_path = out_path.replace('.html', '.pdf')
chrome_candidates = [
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
]
chrome = next((c for c in chrome_candidates if os.path.exists(c)), None)
if chrome:
    try:
        r_pdf = subprocess.run([
            chrome, '--headless=new', '--disable-gpu', '--no-sandbox',
            '--no-pdf-header-footer',
            f'--print-to-pdf={pdf_path}',
            f'file:///{out_path.replace(chr(92), "/")}',
        ], capture_output=True, timeout=60)
        if os.path.exists(pdf_path):
            print(f"PDF出力: {pdf_path}")
        else:
            print(f"PDF生成失敗: {r_pdf.stderr.decode('utf-8','ignore')[-200:]}")
    except Exception as e:
        print(f"PDF生成エラー: {e}")
else:
    print("Chrome未検出のためPDF生成スキップ")

print("\n完了")

"""
2012 / 2026 両データで印4段階条件をグリッドサーチ
目的:
  - 頭数順: 激熱 < 〇 < ▲ < ☆
  - ROI順:  激熱 > 〇 > ▲ > ☆
  - 激熱: 2012で週0.5以上
  - 2012は全印プラス（最低限）
  - 2026はデータ少量なので参考程度
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import pandas as pd, numpy as np, os, pickle, json, re, time, itertools

base_dir  = r'C:\Users\tsuch\Desktop\horse_racing_ai'
model_dir = os.path.join(base_dir, 'models_2025')

def get_distance_band(dist):
    m = re.search(r'\d+', str(dist))
    if not m: return None
    d = int(m.group())
    if d <= 1400: return '短距離'
    elif d <= 1800: return 'マイル'
    elif d <= 2200: return '中距離'
    return '長距離'

def get_class_group(r):
    try: r = int(float(r))
    except: return '3勝以上'
    if r == 1: return '新馬'
    elif r == 2: return '未勝利'
    elif r == 3: return '1勝'
    elif r == 4: return '2勝'
    return '3勝以上'

def extract_venue(kaikai):
    m = re.search(r'\d+([^\d]+)', str(kaikai))
    return m.group(1) if m else str(kaikai)

# ── モデル読み込み ──────────────────────────────────────────
print("モデル読み込み中...")
with open(os.path.join(model_dir,'model_info.json'),encoding='utf-8') as f: ci=json.load(f)
cur_features=ci['features']; cur_models=ci['models']
with open(os.path.join(model_dir,'ranker','ranker_info.json'),encoding='utf-8') as f:
    cur_rankers=json.load(f).get('rankers',{})
with open(os.path.join(model_dir,'submodel','submodel_info.json'),encoding='utf-8') as f: si=json.load(f)
sub_features=si['features']; sub_models=si['models']
with open(os.path.join(model_dir,'submodel_ranker','class_ranker_info.json'),encoding='utf-8') as f:
    sub_rankers=json.load(f).get('rankers',{})
all_feats=list(set(cur_features+sub_features))

def load_and_predict(csv_path, enc='utf-8'):
    print(f"\nデータ読み込み: {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path, low_memory=False, encoding=enc)
    df['着順_num']   = pd.to_numeric(df['着順_num'],   errors='coerce')
    df['単勝配当']   = pd.to_numeric(df['単勝配当'],   errors='coerce')
    df['単勝オッズ'] = pd.to_numeric(df['単勝オッズ'], errors='coerce')
    df['複勝配当']   = pd.to_numeric(df['複勝配当'],   errors='coerce')
    df = df.dropna(subset=['着順_num','単勝配当'])
    df['target_win']   = (df['着順_num'] == 1).astype(int)
    df['target_place'] = (df['着順_num'] <= 3).astype(int)
    df['会場']       = df['開催'].apply(extract_venue)
    df['_surface']   = df['芝・ダ'].astype(str).str.strip()
    df['cur_key']    = df['会場'] + '_' + df['距離'].astype(str)
    df['_dist_band'] = df['距離'].apply(get_distance_band)
    mask_da = (df['_surface']=='ダ') & (df['_dist_band'].isin(['中距離','長距離']))
    df.loc[mask_da,'_dist_band'] = '中長距離'
    df['_cls_group'] = df['クラス_rank'].apply(get_class_group) if 'クラス_rank' in df.columns else '3勝以上'
    df['sub_key']    = df['_surface'] + '_' + df['_dist_band'] + '_' + df['_cls_group']
    df['日付_num']   = pd.to_numeric(df['日付'], errors='coerce')
    for col in all_feats:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"  {len(df):,}頭  期間: {int(df['日付_num'].min())}〜{int(df['日付_num'].max())}")

    # モデルキャッシュ
    mc={}; rc={}; smc={}; src={}
    for ck in df['cur_key'].dropna().unique():
        if ck in cur_models:
            p=os.path.join(model_dir,cur_models[ck]['win'])
            if os.path.exists(p):
                with open(p,'rb') as f2: m=pickle.load(f2); mc[ck]=(m,m.booster_.feature_name())
        if ck in cur_rankers:
            p=os.path.join(model_dir,'ranker',cur_rankers[ck])
            if os.path.exists(p):
                with open(p,'rb') as f2: rc[ck]=pickle.load(f2)
    for sk in df['sub_key'].dropna().unique():
        if sk in sub_models:
            p=os.path.join(model_dir,'submodel',sub_models[sk]['win'])
            if os.path.exists(p):
                with open(p,'rb') as f2: m=pickle.load(f2); smc[sk]=(m,m.booster_.feature_name())
        if sk in sub_rankers:
            p=os.path.join(model_dir,'submodel_ranker',sub_rankers[sk])
            if os.path.exists(p):
                with open(p,'rb') as f2: src[sk]=pickle.load(f2)

    race_keys = [c for c in ['開催','Ｒ'] if c in df.columns]
    t0=time.time(); all_rows=[]
    for gk,idx in df.groupby(race_keys,sort=False).groups.items():
        s=df.loc[idx].copy()
        ck=s['cur_key'].iloc[0]; sk=s['sub_key'].iloc[0]
        s['cur_rank']=np.nan; s['cur_diff']=np.nan; s['cur_score']=np.nan
        s['sub_rank']=np.nan; s['sub_diff']=np.nan; s['sub_score']=np.nan
        if ck in mc:
            m,wf=mc[ck]
            for c in wf:
                if c not in s.columns: s[c]=np.nan
            prob=m.predict_proba(s[wf])[:,1]
            st=cur_models[ck].get('stats',{}); wm=st.get('win_mean',prob.mean()); ws=st.get('win_std',prob.std())
            cs=50+10*(prob-wm)/(ws if ws>0 else 1); rm=prob.mean(); rs=prob.std()
            s['cur_score']=cs; s['cur_diff']=50+10*(prob-rm)/(rs if rs>0 else 1)-cs
            if ck in rc:
                sc=rc[ck].predict(s[cur_features])
                s['cur_rank']=pd.Series(sc,index=s.index).rank(ascending=False,method='min').astype(int)
        if sk in smc:
            m,wf=smc[sk]
            for c in wf:
                if c not in s.columns: s[c]=np.nan
            prob=m.predict_proba(s[wf])[:,1]
            st=sub_models[sk].get('stats',{}); wm=st.get('win_mean',prob.mean()); ws=st.get('win_std',prob.std())
            cs=50+10*(prob-wm)/(ws if ws>0 else 1); rm=prob.mean(); rs=prob.std()
            s['sub_score']=cs; s['sub_diff']=50+10*(prob-rm)/(rs if rs>0 else 1)-cs
            if sk in src:
                sc=src[sk].predict(s[wf])
                s['sub_rank']=pd.Series(sc,index=s.index).rank(ascending=False,method='min').astype(int)
        for sc2,gc in [('cur_score','cur_gap'),('sub_score','sub_gap')]:
            vals=s[sc2].dropna().sort_values(ascending=False).values
            s[gc]=(vals[0]-vals[1]) if len(vals)>=2 else np.nan
        s['combo_gap']=s.get('cur_gap',pd.Series(0,index=s.index)).fillna(0)+\
                       s.get('sub_gap',pd.Series(0,index=s.index)).fillna(0)
        all_rows.append(s)

    result=pd.concat(all_rows,ignore_index=True)
    print(f"  予測完了: {len(result):,}頭 ({time.time()-t0:.0f}秒)")
    return result

# 両データ予測
res12 = load_and_predict(
    os.path.join(base_dir,'data','processed','features_2012_test.csv'), enc='utf-8-sig')
res26 = load_and_predict(
    os.path.join(base_dir,'data','processed','all_venues_features_2026test.csv'))

def compute_vectors(res):
    cr = pd.to_numeric(res['cur_rank'], errors='coerce')
    sr = pd.to_numeric(res['sub_rank'], errors='coerce')
    cd = pd.to_numeric(res['cur_diff'], errors='coerce')
    sd = pd.to_numeric(res['sub_diff'], errors='coerce')
    cg = pd.to_numeric(res['combo_gap'], errors='coerce')
    od = pd.to_numeric(res['単勝オッズ'], errors='coerce')
    both_r1 = (cr==1) & (sr==1)
    r2_excl = (cr<=2) & (sr<=2) & ~both_r1   # ▲プール
    star    = (cr<=3) & (sr<=3) & ~((cr<=2) & (sr<=2))  # ☆プール
    return cr, sr, cd, sd, cg, od, both_r1, r2_excl, star

vecs12 = compute_vectors(res12)
vecs26 = compute_vectors(res26)

def calc_roi(res, mask):
    s = res[mask]
    n = len(s)
    if n == 0: return 0.0, 0, 0
    wins = int(s['target_win'].sum())
    pay  = s.loc[s['target_win']==1,'単勝配当'].sum()
    roi  = (pay/(n*100)-1)*100
    return roi, n, wins

# 週数
dates12 = res12['日付_num'].dropna().astype(int)
# 日付からYYYYWW（年+週番号）を計算して一意の週を数える
def count_weeks(date_series):
    dates = pd.to_datetime(date_series.astype(str), format='%y%m%d', errors='coerce')
    if dates.isna().all():
        # try YYYYMMDD
        dates = pd.to_datetime(date_series.astype(str), format='%Y%m%d', errors='coerce')
    weeks = dates.dt.isocalendar().week.astype(str) + '_' + dates.dt.isocalendar().year.astype(str)
    return weeks.nunique()

weeks12 = count_weeks(dates12)
weeks26 = count_weeks(res26['日付_num'].dropna().astype(int))
print(f"\n2012: {weeks12}週  2026: {weeks26}週")

# ── グリッドサーチ（2026メイン・2012参考）──────────────────
# 激熱: both_r1 & cd≥X & sd≥Y & gap≥Z & odds≥W
# 〇:   both_r1 & sd≥A & odds≥B & ~激熱
# ▲:   r2_excl & sd≥C & odds≥D
# ☆:   star    & sd≥E & odds≥F

geki_cd_list      = [0, 5, 10, 15]
geki_sd_list      = [5, 10, 15]
geki_gap_list     = [0, 5, 10, 15]    # combo_gap
geki_odds_list    = [3, 5]
maru_sd_list      = [0, 5, 10]
maru_odds_list    = [2, 3]
sankaku_sd_list   = [0, 5, 10]
sankaku_odds_list = [3, 5]
hoshi_sd_list     = [0, 5, 10]
hoshi_odds_list   = [3, 5]

total = (len(geki_cd_list)*len(geki_sd_list)*len(geki_gap_list)*len(geki_odds_list)*
         len(maru_sd_list)*len(maru_odds_list)*
         len(sankaku_sd_list)*len(sankaku_odds_list)*
         len(hoshi_sd_list)*len(hoshi_odds_list))
print(f"\nグリッドサーチ開始（2026メイン）: {total:,}通り...")

cr12, sr12, cd12, sd12, cg12, od12, br1_12, r2_12, star12 = vecs12
cr26, sr26, cd26, sd26, cg26, od26, br1_26, r2_26, star26 = vecs26

MIN_GEKI_26 = max(1, int(weeks26 * 0.5))   # 激熱 2026で最低0.5/週
results = []
t0 = time.time()

for geki_cd, geki_sd, geki_gap, geki_odds, maru_sd, maru_odds, sankaku_sd, sankaku_odds, hoshi_sd, hoshi_odds in itertools.product(
        geki_cd_list, geki_sd_list, geki_gap_list, geki_odds_list,
        maru_sd_list, maru_odds_list,
        sankaku_sd_list, sankaku_odds_list,
        hoshi_sd_list, hoshi_odds_list):

    # 2026マスク
    geki26  = br1_26 & (cd26>=geki_cd) & (sd26>=geki_sd) & (cg26>=geki_gap) & (od26>=geki_odds)
    maru26  = br1_26 & (sd26>=maru_sd) & (od26>=maru_odds) & ~geki26
    san26   = r2_26  & (sd26>=sankaku_sd) & (od26>=sankaku_odds)
    hoshi26 = star26 & (sd26>=hoshi_sd) & (od26>=hoshi_odds)

    roi_g26, n_g26, _ = calc_roi(res26, geki26)
    roi_m26, n_m26, _ = calc_roi(res26, maru26)
    roi_s26, n_s26, _ = calc_roi(res26, san26)
    roi_h26, n_h26, _ = calc_roi(res26, hoshi26)

    # 激熱 2026で0.5/週以上
    if n_g26 < MIN_GEKI_26:
        continue

    # 2026 頭数順チェック
    if not (n_g26 < n_m26 and n_m26 < n_s26 and n_s26 < n_h26):
        continue

    # スコア（2026メイン）
    score = 0
    if roi_g26 > roi_m26: score += 5
    if roi_m26 > roi_s26: score += 3
    if roi_s26 > roi_h26: score += 2
    if roi_g26 > roi_m26 > roi_s26 > roi_h26: score += 5   # 完全順序ボーナス
    all_plus_26 = all(r > 0 for r in [roi_g26, roi_m26, roi_s26, roi_h26])
    if all_plus_26: score += 8
    else:
        if roi_g26 > 0: score += 2
        if roi_m26 > 0: score += 1
        if roi_s26 > 0: score += 1

    # 2012参考
    geki12  = br1_12 & (cd12>=geki_cd) & (sd12>=geki_sd) & (cg12>=geki_gap) & (od12>=geki_odds)
    maru12  = br1_12 & (sd12>=maru_sd) & (od12>=maru_odds) & ~geki12
    san12   = r2_12  & (sd12>=sankaku_sd) & (od12>=sankaku_odds)
    hoshi12 = star12 & (sd12>=hoshi_sd) & (od12>=hoshi_odds)

    roi_g12, n_g12, _ = calc_roi(res12, geki12)
    roi_m12, n_m12, _ = calc_roi(res12, maru12)
    roi_s12, n_s12, _ = calc_roi(res12, san12)
    roi_h12, n_h12, _ = calc_roi(res12, hoshi12)

    all_plus_12 = all(r > 0 for r in [roi_g12, roi_m12, roi_s12, roi_h12] if
                      (n_g12+n_m12+n_s12+n_h12) > 0)
    if all_plus_12: score += 3
    if roi_g12 > 0: score += 1

    results.append({
        'score': score,
        'geki_cd': geki_cd, 'geki_sd': geki_sd, 'geki_gap': geki_gap, 'geki_odds': geki_odds,
        'maru_sd': maru_sd, 'maru_odds': maru_odds,
        'sankaku_sd': sankaku_sd, 'sankaku_odds': sankaku_odds,
        'hoshi_sd': hoshi_sd, 'hoshi_odds': hoshi_odds,
        'n_g26': n_g26, 'roi_g26': roi_g26,
        'n_m26': n_m26, 'roi_m26': roi_m26,
        'n_s26': n_s26, 'roi_s26': roi_s26,
        'n_h26': n_h26, 'roi_h26': roi_h26,
        'all_plus_26': all_plus_26,
        'n_g12': n_g12, 'roi_g12': roi_g12,
        'n_m12': n_m12, 'roi_m12': roi_m12,
        'n_s12': n_s12, 'roi_s12': roi_s12,
        'n_h12': n_h12, 'roi_h12': roi_h12,
        'all_plus_12': all_plus_12,
    })

print(f"サーチ完了: {len(results):,}件 ({time.time()-t0:.1f}秒)")

results.sort(key=lambda x: -x['score'])

# ── 結果表示 ──────────────────────────────────────────────
print()
print('='*90)
print('  TOP20 条件 (2026メイン: 頭数順＋ROI順 両立)')
print('='*90)

for r in results[:20]:
    cond = (f"激熱: cd≥{r['geki_cd']} sd≥{r['geki_sd']} gap≥{r['geki_gap']} odds≥{r['geki_odds']} | "
            f"〇: sd≥{r['maru_sd']} odds≥{r['maru_odds']} | "
            f"▲: sd≥{r['sankaku_sd']} odds≥{r['sankaku_odds']} | "
            f"☆: sd≥{r['hoshi_sd']} odds≥{r['hoshi_odds']}")
    roi26 = (f"激{r['roi_g26']:+.0f}%({r['n_g26']}) "
             f"〇{r['roi_m26']:+.0f}%({r['n_m26']}) "
             f"▲{r['roi_s26']:+.0f}%({r['n_s26']}) "
             f"☆{r['roi_h26']:+.0f}%({r['n_h26']})")
    roi12 = (f"激{r['roi_g12']:+.0f}%({r['n_g12']}) "
             f"〇{r['roi_m12']:+.0f}%({r['n_m12']}) "
             f"▲{r['roi_s12']:+.0f}%({r['n_s12']}) "
             f"☆{r['roi_h12']:+.0f}%({r['n_h12']})")
    plus26 = '★2026全+' if r['all_plus_26'] else ''
    plus12 = '★2012全+' if r['all_plus_12'] else ''
    print(f"  score={r['score']:2d} {plus26} {plus12}")
    print(f"    条件: {cond}")
    print(f"    2026: {roi26}  激{r['n_g26']/weeks26:.1f}/週")
    print(f"    2012: {roi12}  激{r['n_g12']/weeks12:.1f}/週")
    print()

# 2026全プラス条件
all_plus26 = [r for r in results if r['all_plus_26']]
print(f"\n2026全印プラス条件: {len(all_plus26)}件")
if all_plus26:
    print('\n【2026全印プラス TOP10】')
    for r in all_plus26[:10]:
        print(f"  2026: 激{r['roi_g26']:+.0f}%({r['n_g26']}) 〇{r['roi_m26']:+.0f}%({r['n_m26']}) ▲{r['roi_s26']:+.0f}%({r['n_s26']}) ☆{r['roi_h26']:+.0f}%({r['n_h26']})  /週{r['n_g26']/weeks26:.1f}")
        print(f"  2012: 激{r['roi_g12']:+.0f}%({r['n_g12']}) 〇{r['roi_m12']:+.0f}%({r['n_m12']}) ▲{r['roi_s12']:+.0f}%({r['n_s12']}) ☆{r['roi_h12']:+.0f}%({r['n_h12']})")
        print(f"    cd≥{r['geki_cd']} sd≥{r['geki_sd']} gap≥{r['geki_gap']} odds≥{r['geki_odds']} | "
              f"m_sd≥{r['maru_sd']} m_odds≥{r['maru_odds']} | "
              f"s_sd≥{r['sankaku_sd']} s_odds≥{r['sankaku_odds']} | "
              f"h_sd≥{r['hoshi_sd']} h_odds≥{r['hoshi_odds']}")
        print()

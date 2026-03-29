"""
新ルール適用チェック: combo_rank=1 & sub_diff≥20 & 単勝オッズ≥5.0
3月21日・3月22日の該当馬と結果を表示
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import pandas as pd, numpy as np, os, pickle, json, re, time

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, 'models')

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

def zen_to_num(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    m = re.search(r'\d+', s)
    return int(m.group()) if m else np.nan

venue_map = {
    '中山':'中','東京':'東','阪神':'阪','中京':'名',
    '京都':'京','函館':'函','新潟':'新','小倉':'小','札幌':'札','福島':'福',
}

# ── モデル読み込み ──
with open(os.path.join(model_dir, 'model_info.json'), encoding='utf-8') as f: cur_info = json.load(f)
cur_features = cur_info['features']; cur_models = cur_info['models']
with open(os.path.join(model_dir, 'ranker', 'ranker_info.json'), encoding='utf-8') as f:
    cur_rankers = json.load(f).get('rankers', {})
with open(os.path.join(model_dir, 'submodel', 'submodel_info.json'), encoding='utf-8') as f: sub_info = json.load(f)
sub_features = sub_info['features']; sub_models = sub_info['models']
with open(os.path.join(model_dir, 'submodel_ranker', 'class_ranker_info.json'), encoding='utf-8') as f:
    sub_rankers = json.load(f).get('rankers', {})
all_feats = list(set(cur_features + sub_features))

# ── Parquet ──
print("特徴量読み込み中...")
df_feat = pd.read_parquet(os.path.join(base_dir, 'data', 'processed', 'all_venues_features.parquet'))
df_latest = df_feat.sort_values('日付').groupby('馬名S').last().reset_index()

def run_date(res_file, date_str, cur_date):
    print(f"\n{'='*65}")
    print(f"  {date_str} 新ルール適用結果")
    print(f"  条件: combo_rank=1 & sub_diff≥20 & 単勝オッズ≥5.0")
    print(f"{'='*65}")

    df_res = pd.read_csv(res_file, encoding='cp932', low_memory=False)
    df_res['着_num']      = df_res['着'].apply(zen_to_num)
    df_res['target_win']  = (df_res['着_num'] == 1).astype(int)
    df_res['target_place']= (df_res['着_num'] <= 3).astype(int)
    df_res['単勝_num']    = pd.to_numeric(df_res['単勝'], errors='coerce')
    df_res['単オッズ_num'] = pd.to_numeric(df_res.get('単オッズ', df_res.get('単勝オッズ', np.nan)), errors='coerce')

    race_w = df_res[df_res['target_win']==1][['場所','Ｒ','単勝_num']].rename(columns={'単勝_num':'tansho'})
    df_res = df_res.merge(race_w, on=['場所','Ｒ'], how='left')

    df_res['会場']       = df_res['場所'].astype(str).map(venue_map).fillna(df_res['場所'].astype(str))
    df_res['_surface']   = df_res['芝ダ'].astype(str).str.strip()
    df_res['cur_key']    = df_res['会場'] + '_' + df_res['_surface'] + df_res['距離'].astype(str)
    df_res['_dist_band'] = df_res['距離'].apply(get_distance_band)
    mask = (df_res['_surface'] == 'ダ') & (df_res['_dist_band'].isin(['中距離','長距離']))
    df_res.loc[mask, '_dist_band'] = '中長距離'
    df_res['_cls_group'] = df_res['クラス_rank'].apply(get_class_group) if 'クラス_rank' in df_res.columns else '3勝以上'
    df_res['sub_key']    = df_res['_surface'] + '_' + df_res['_dist_band'] + '_' + df_res['_cls_group']

    feat_subset = ['馬名S'] + [c for c in all_feats + ['日付','距離'] if c in df_latest.columns]
    df_merge = df_res.merge(df_latest[list(set(feat_subset))], on='馬名S', how='left', suffixes=('','_f'))
    df_merge['性別_num'] = df_merge['性別'].map({'牡':0,'牝':1,'セ':2}).astype(float)
    if '距離_f' in df_merge.columns:
        df_merge['前距離'] = df_merge['距離_f'].astype(str).str.extract(r'(\d+)').iloc[:,0].astype(float)
    if '日付_f' in df_merge.columns:
        def _yymmdd(v):
            try:
                v = int(v); return pd.Timestamp(2000+v//10000,(v//100)%100,v%100)
            except: return pd.NaT
        df_merge['間隔'] = ((cur_date - df_merge['日付_f'].apply(_yymmdd)).dt.days/7).round(0)
    for col in all_feats:
        if col in df_merge.columns: df_merge[col] = pd.to_numeric(df_merge[col], errors='coerce')

    # ── モデルキャッシュ ──
    cur_mc={}; cur_rc={}; sub_mc={}; sub_rc={}
    for ck in df_merge['cur_key'].dropna().unique():
        if ck in cur_models:
            p = os.path.join(model_dir, cur_models[ck]['win'])
            if os.path.exists(p):
                with open(p,'rb') as f: m=pickle.load(f)
                cur_mc[ck]=(m,m.booster_.feature_name())
        if ck in cur_rankers:
            p = os.path.join(model_dir,'ranker',cur_rankers[ck])
            if os.path.exists(p):
                with open(p,'rb') as f: cur_rc[ck]=pickle.load(f)
    for sk in df_merge['sub_key'].dropna().unique():
        if sk in sub_models:
            p = os.path.join(model_dir,'submodel',sub_models[sk]['win'])
            if os.path.exists(p):
                with open(p,'rb') as f: m=pickle.load(f)
                sub_mc[sk]=(m,m.booster_.feature_name())
        if sk in sub_rankers:
            p = os.path.join(model_dir,'submodel_ranker',sub_rankers[sk])
            if os.path.exists(p):
                with open(p,'rb') as f: sub_rc[sk]=pickle.load(f)

    # ── 予測 ──
    race_keys=['場所','Ｒ']
    all_rows=[]
    for gk, idx in df_merge.groupby(race_keys, sort=False).groups.items():
        sub = df_merge.loc[idx].copy()
        ck=sub['cur_key'].iloc[0]; sk=sub['sub_key'].iloc[0]
        sub['cur_score']=np.nan; sub['sub_score']=np.nan
        sub['cur_diff']=np.nan;  sub['cur_rank']=np.nan
        sub['sub_diff']=np.nan;  sub['sub_rank']=np.nan
        if ck in cur_mc:
            m,wf=cur_mc[ck]
            for c in wf:
                if c not in sub.columns: sub[c]=np.nan
            prob=m.predict_proba(sub[wf])[:,1]
            st=cur_models[ck].get('stats',{}); wm=st.get('win_mean',prob.mean()); ws=st.get('win_std',prob.std())
            cs=50+10*(prob-wm)/(ws if ws>0 else 1)
            rm=prob.mean(); rs=prob.std()
            sub['cur_score']=cs; sub['cur_diff']=50+10*(prob-rm)/(rs if rs>0 else 1)-cs
            if ck in cur_rc:
                sc=cur_rc[ck].predict(sub[cur_features])
                sub['cur_rank']=pd.Series(sc,index=sub.index).rank(ascending=False,method='min').astype(int)
        if sk in sub_mc:
            m,wf=sub_mc[sk]
            for c in wf:
                if c not in sub.columns: sub[c]=np.nan
            prob=m.predict_proba(sub[wf])[:,1]
            st=sub_models[sk].get('stats',{}); wm=st.get('win_mean',prob.mean()); ws=st.get('win_std',prob.std())
            cs=50+10*(prob-wm)/(ws if ws>0 else 1)
            rm=prob.mean(); rs=prob.std()
            sub['sub_score']=cs; sub['sub_diff']=50+10*(prob-rm)/(rs if rs>0 else 1)-cs
            if sk in sub_rc:
                sc=sub_rc[sk].predict(sub[wf])
                sub['sub_rank']=pd.Series(sc,index=sub.index).rank(ascending=False,method='min').astype(int)
        all_rows.append(sub)

    result = pd.concat(all_rows, ignore_index=True)

    cd=pd.to_numeric(result['cur_diff'],errors='coerce')
    sd=pd.to_numeric(result['sub_diff'],errors='coerce')
    cr=pd.to_numeric(result['cur_rank'],errors='coerce')
    sr=pd.to_numeric(result['sub_rank'],errors='coerce')
    cs_cur=pd.to_numeric(result['cur_score'],errors='coerce')
    cs_sub=pd.to_numeric(result['sub_score'],errors='coerce')

    result['combo_diff'] = cd.fillna(0)+sd.fillna(0)
    n_h = result.groupby(race_keys)['cur_rank'].transform('count')
    result['rank_sum']   = cr.fillna(n_h+1)+sr.fillna(n_h+1)
    result['combo_rank'] = result.groupby(race_keys)['combo_diff'].rank(ascending=False,method='min')

    # cur_gap / sub_gap
    for key, grp in result.groupby(race_keys):
        idx=grp.index
        for col, gcol in [('cur_score','cur_gap'),('sub_score','sub_gap')]:
            scores=grp[col].dropna().sort_values(ascending=False).values
            gap = scores[0]-scores[1] if len(scores)>=2 else np.nan
            result.loc[idx,gcol]=gap

    # オッズ
    result['odds'] = pd.to_numeric(result.get('単オッズ', result.get('単勝オッズ', np.nan)), errors='coerce')

    # ── 新ルール適用 ──
    mask_new = (result['combo_rank']==1) & (sd>=20) & (result['odds']>=5.0)
    mask_ref  = (result['combo_rank']==1) & (sd>=20)

    print(f"\n【参考】combo_rank=1 & sd≥20 のみ（オッズ制限なし）")
    ref = result[mask_ref]
    n=len(ref); wins=int(ref['target_win'].sum())
    roi=ref[ref['target_win']==1]['tansho'].sum()/(n*100)-1 if n>0 else 0
    print(f"  該当: {n}頭  的中: {wins}頭  単勝ROI: {roi:+.1%}")
    for _,r in ref.iterrows():
        hit='✓' if r['target_win']==1 else '✗'
        odds_s=f"{r['odds']:.1f}倍" if pd.notna(r['odds']) else "?倍"
        print(f"  {hit} {r['場所']}{int(r['Ｒ'])}R {r['馬名']:<16} cd={cd[r.name]:.1f} sd={sd[r.name]:.1f} gap_c={r.get('cur_gap',np.nan):.1f} gap_s={r.get('sub_gap',np.nan):.1f} {odds_s}  着={int(r['着_num']) if pd.notna(r['着_num']) else '?'}着  tansho={int(r['tansho']) if pd.notna(r.get('tansho')) else '-'}円")

    print(f"\n【新ルール】combo_rank=1 & sd≥20 & オッズ≥5.0")
    new = result[mask_new]
    n=len(new); wins=int(new['target_win'].sum())
    roi=new[new['target_win']==1]['tansho'].sum()/(n*100)-1 if n>0 else 0
    print(f"  該当: {n}頭  的中: {wins}頭  単勝ROI: {roi:+.1%}")
    if n==0:
        print("  → 該当なし")
    for _,r in new.iterrows():
        hit='✓' if r['target_win']==1 else '✗'
        odds_s=f"{r['odds']:.1f}倍" if pd.notna(r['odds']) else "?倍"
        tansho_s=f"{int(r['tansho'])}円" if pd.notna(r.get('tansho')) else '-'
        print(f"  {hit} {r['場所']}{int(r['Ｒ'])}R {r['馬名']:<16} sd={sd[r.name]:.1f} {odds_s}  {int(r['着_num']) if pd.notna(r['着_num']) else '?'}着  {tansho_s}")

# ── 実行 ──
run_date(
    os.path.join(base_dir,'data','raw','出馬表形式3月21日結果確認.csv'),
    '3月21日（土）', pd.Timestamp(2026,3,21)
)
run_date(
    os.path.join(base_dir,'data','raw','出馬表形式3月22日結果確認.csv'),
    '3月22日（日）', pd.Timestamp(2026,3,22)
)

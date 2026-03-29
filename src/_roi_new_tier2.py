"""
重複除いて4ティアROI再計算
激熱 / [中間] / 〇 / ☆ の完全非重複版
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import pandas as pd, numpy as np, os, pickle, json, re, time

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

def extract_venue(k):
    m = re.search(r'\d+([^\d]+)', str(k))
    return m.group(1) if m else str(k)

with open(os.path.join(model_dir,'model_info.json'),encoding='utf-8') as f: ci=json.load(f)
cur_features=ci['features']; cur_models=ci['models']
with open(os.path.join(model_dir,'ranker','ranker_info.json'),encoding='utf-8') as f:
    cur_rankers=json.load(f).get('rankers',{})
with open(os.path.join(model_dir,'submodel','submodel_info.json'),encoding='utf-8') as f: si=json.load(f)
sub_features=si['features']; sub_models=si['models']
with open(os.path.join(model_dir,'submodel_ranker','class_ranker_info.json'),encoding='utf-8') as f:
    sub_rankers=json.load(f).get('rankers',{})
all_feats = list(set(cur_features + sub_features))

def load_and_predict(path, enc='utf-8'):
    df = pd.read_csv(path, low_memory=False, encoding=enc)
    df['着順_num']   = pd.to_numeric(df['着順_num'],   errors='coerce')
    df['単勝配当']   = pd.to_numeric(df['単勝配当'],   errors='coerce')
    df['単勝オッズ'] = pd.to_numeric(df['単勝オッズ'], errors='coerce')
    df = df.dropna(subset=['着順_num','単勝配当'])
    df['target_win']   = (df['着順_num'] == 1).astype(int)
    df['会場']       = df['開催'].apply(extract_venue)
    df['_surface']   = df['芝・ダ'].astype(str).str.strip()
    df['cur_key']    = df['会場'] + '_' + df['距離'].astype(str)
    df['_dist_band'] = df['距離'].apply(get_distance_band)
    mask_da = (df['_surface']=='ダ') & (df['_dist_band'].isin(['中距離','長距離']))
    df.loc[mask_da,'_dist_band'] = '中長距離'
    df['_cls_group'] = df['クラス_rank'].apply(get_class_group) if 'クラス_rank' in df.columns else '3勝以上'
    df['sub_key']    = df['_surface'] + '_' + df['_dist_band'] + '_' + df['_cls_group']
    for col in all_feats:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

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
    all_rows = []
    for gk, idx in df.groupby(race_keys, sort=False).groups.items():
        s = df.loc[idx].copy()
        ck = s['cur_key'].iloc[0]; sk = s['sub_key'].iloc[0]
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
    return pd.concat(all_rows, ignore_index=True)

print("予測中...")
res26 = load_and_predict(os.path.join(base_dir,'data','processed','all_venues_features_2026test.csv'))
res12 = load_and_predict(os.path.join(base_dir,'data','processed','features_2012_test.csv'), enc='utf-8-sig')
print("完了")

def analyze(res, label, weeks):
    cr = pd.to_numeric(res['cur_rank'], errors='coerce')
    sr = pd.to_numeric(res['sub_rank'], errors='coerce')
    cd = pd.to_numeric(res['cur_diff'], errors='coerce')
    sd = pd.to_numeric(res['sub_diff'], errors='coerce')
    cg = pd.to_numeric(res['combo_gap'], errors='coerce')
    od = pd.to_numeric(res['単勝オッズ'], errors='coerce')

    both_r1  = (cr==1) & (sr==1)
    # 片方r1・片方r2 (厳密: 一方が1位、他方が2位)
    one1one2 = ((cr==1)&(sr==2)) | ((cr==2)&(sr==1))
    # ☆ = 両Rnk≤3 & ~両Rnk1 & ~one1one2 (重複なし)
    hoshi_pool = (cr<=3)&(sr<=3)&~both_r1&~one1one2

    def roi(mask):
        s=res[mask]; n=len(s)
        if n==0: return 0.0,0,0
        w=int(s['target_win'].sum())
        p=s.loc[s['target_win']==1,'単勝配当'].sum()
        return (p/(n*100)-1)*100, n, w

    print(f'\n{"="*65}')
    print(f'  {label}  【4ティア 完全非重複】')
    print(f'{"="*65}')
    print(f'  {"印":<6} {"条件":<40} {"ROI":>7}  {"頭数":>5}  {"/週":>5}')
    print(f'  {"-"*60}')

    # ── パターン1: 激熱=cur_diff & sub_diff ──
    print(f'  ▼ 激熱条件A: cur_diff≥10 & sub_diff≥10 & odds≥5')
    geki_A = both_r1 & (cd>=10) & (sd>=10) & (od>=5)
    maru_A = both_r1 & ~geki_A & (sd>=10) & (od>=3)

    for odds_mid in [3, 5]:
        mid_A  = one1one2 & (sd>=10) & (od>=odds_mid)
        hoshi_A = hoshi_pool & (sd>=10) & (od>=odds_mid)

        rg,ng,wg = roi(geki_A)
        rm,nm,wm_ = roi(maru_A)
        ri,ni,wi = roi(mid_A)
        rh,nh,wh = roi(hoshi_A)

        print(f'  [odds_mid≥{odds_mid}]')
        print(f'  激熱  both_r1 cd≥10 sd≥10 odds≥5         {rg:>+7.0f}%  {ng:>5}  {ng/weeks:>5.1f}')
        print(f'  中間  片方r1片方r2 sd≥10 odds≥{odds_mid}           {ri:>+7.0f}%  {ni:>5}  {ni/weeks:>5.1f}')
        print(f'  〇    both_r1 ~激熱 sd≥10 odds≥3          {rm:>+7.0f}%  {nm:>5}  {nm/weeks:>5.1f}')
        print(f'  ☆    両Rnk≤3 ~上記 sd≥10 odds≥{odds_mid}         {rh:>+7.0f}%  {nh:>5}  {nh/weeks:>5.1f}')

        # 重複チェック
        overlap = int((geki_A & mid_A).sum()) + int((geki_A & maru_A & mid_A).sum())
        print(f'  重複: {overlap}頭  ROI順: 激熱{rg:+.0f}> 中間{ri:+.0f}> 〇{rm:+.0f}> ☆{rh:+.0f} → {"OK" if rg>ri>rm>rh else "NG"}')
        print()

analyze(res26, '2026', 11)
analyze(res12, '2012', 259)

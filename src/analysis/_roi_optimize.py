"""ROI最大化グリッドサーチ"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import pandas as pd, numpy as np, os, pickle, json, re, time

base_dir  = r'G:\マイドライブ\horse_racing_ai'
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

# ── データ・モデル読み込み ──────────────────────────────────
with open(os.path.join(model_dir,'model_info.json'),encoding='utf-8') as f: ci=json.load(f)
cur_features=ci['features']; cur_models=ci['models']
with open(os.path.join(model_dir,'ranker','ranker_info.json'),encoding='utf-8') as f:
    cur_rankers=json.load(f).get('rankers',{})
with open(os.path.join(model_dir,'submodel','submodel_info.json'),encoding='utf-8') as f: si=json.load(f)
sub_features=si['features']; sub_models=si['models']
with open(os.path.join(model_dir,'submodel_ranker','class_ranker_info.json'),encoding='utf-8') as f:
    sub_rankers=json.load(f).get('rankers',{})
all_feats=list(set(cur_features+sub_features))

df=pd.read_csv(os.path.join(base_dir,'data','processed','all_venues_features_2026test.csv'),low_memory=False)
df['着順_num']=pd.to_numeric(df['着順_num'],errors='coerce')
df['単勝配当']=pd.to_numeric(df['単勝配当'],errors='coerce')
df['単勝オッズ']=pd.to_numeric(df['単勝オッズ'],errors='coerce')
df['複勝配当']=pd.to_numeric(df['複勝配当'],errors='coerce')
df=df.dropna(subset=['着順_num','単勝配当'])
df['target_win']=(df['着順_num']==1).astype(int)
df['target_place']=(df['着順_num']<=3).astype(int)
df['会場']=df['開催'].apply(extract_venue)
df['_surface']=df['芝・ダ'].astype(str).str.strip()
df['cur_key']=df['会場']+'_'+df['距離'].astype(str)
df['_dist_band']=df['距離'].apply(get_distance_band)
mask_da=(df['_surface']=='ダ')&(df['_dist_band'].isin(['中距離','長距離']))
df.loc[mask_da,'_dist_band']='中長距離'
df['_cls_group']=df['クラス_rank'].apply(get_class_group) if 'クラス_rank' in df.columns else '3勝以上'
df['sub_key']=df['_surface']+'_'+df['_dist_band']+'_'+df['_cls_group']
for col in all_feats:
    if col in df.columns: df[col]=pd.to_numeric(df[col],errors='coerce')
print(f'データ: {len(df):,}頭')

cur_mc={}; cur_rc={}; sub_mc={}; sub_rc={}
for ck in df['cur_key'].dropna().unique():
    if ck in cur_models:
        p=os.path.join(model_dir,cur_models[ck]['win'])
        if os.path.exists(p):
            with open(p,'rb') as f: m=pickle.load(f); cur_mc[ck]=(m,m.booster_.feature_name())
    if ck in cur_rankers:
        p=os.path.join(model_dir,'ranker',cur_rankers[ck])
        if os.path.exists(p):
            with open(p,'rb') as f: cur_rc[ck]=pickle.load(f)
for sk in df['sub_key'].dropna().unique():
    if sk in sub_models:
        p=os.path.join(model_dir,'submodel',sub_models[sk]['win'])
        if os.path.exists(p):
            with open(p,'rb') as f: m=pickle.load(f); sub_mc[sk]=(m,m.booster_.feature_name())
    if sk in sub_rankers:
        p=os.path.join(model_dir,'submodel_ranker',sub_rankers[sk])
        if os.path.exists(p):
            with open(p,'rb') as f: sub_rc[sk]=pickle.load(f)

race_keys=[c for c in ['開催','Ｒ'] if c in df.columns]
t0=time.time(); all_rows=[]
for gk,idx in df.groupby(race_keys,sort=False).groups.items():
    sub=df.loc[idx].copy()
    ck=sub['cur_key'].iloc[0]; sk=sub['sub_key'].iloc[0]
    sub['cur_rank']=np.nan; sub['cur_diff']=np.nan; sub['cur_score']=np.nan
    sub['sub_rank']=np.nan; sub['sub_diff']=np.nan; sub['sub_score']=np.nan
    if ck in cur_mc:
        m,wf=cur_mc[ck]
        for c in wf:
            if c not in sub.columns: sub[c]=np.nan
        prob=m.predict_proba(sub[wf])[:,1]
        st=cur_models[ck].get('stats',{}); wm=st.get('win_mean',prob.mean()); ws=st.get('win_std',prob.std())
        cs=50+10*(prob-wm)/(ws if ws>0 else 1); rm=prob.mean(); rs=prob.std()
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
        cs=50+10*(prob-wm)/(ws if ws>0 else 1); rm=prob.mean(); rs=prob.std()
        sub['sub_score']=cs; sub['sub_diff']=50+10*(prob-rm)/(rs if rs>0 else 1)-cs
        if sk in sub_rc:
            sc=sub_rc[sk].predict(sub[wf])
            sub['sub_rank']=pd.Series(sc,index=sub.index).rank(ascending=False,method='min').astype(int)
    for sc2,gc in [('cur_score','cur_gap'),('sub_score','sub_gap')]:
        vals=sub[sc2].dropna().sort_values(ascending=False).values
        sub[gc]=(vals[0]-vals[1]) if len(vals)>=2 else np.nan
    sub['combo_gap']=sub.get('cur_gap',pd.Series(0,index=sub.index)).fillna(0)+\
                     sub.get('sub_gap',pd.Series(0,index=sub.index)).fillna(0)
    all_rows.append(sub)

result=pd.concat(all_rows,ignore_index=True)
print(f'予測完了: {len(result):,}頭 ({time.time()-t0:.0f}秒)')

cr=pd.to_numeric(result['cur_rank'],errors='coerce')
sr=pd.to_numeric(result['sub_rank'],errors='coerce')
sd=pd.to_numeric(result['sub_diff'],errors='coerce')
cd=pd.to_numeric(result['cur_diff'],errors='coerce')
cg=pd.to_numeric(result['combo_gap'],errors='coerce')
od=pd.to_numeric(result['単勝オッズ'],errors='coerce')
both_r1=(cr==1)&(sr==1)
star=(cr<=3)&(sr<=3)&~both_r1

# ── グリッドサーチ ─────────────────────────────────────────
print()
print('='*72)
print('  条件グリッドサーチ（N≥10・ROI降順TOP25）')
print('='*72)
print(f"  {'条件':<46} {'N':>4}  {'的中':>4}  {'単勝ROI':>8}  {'週N':>5}")
print('-'*72)

rl=[]
weeks=11

# 両Rnk=1系
for gap_th in [0,5,10,15,20]:
    for sd_th in [0,5,10,15,20]:
        for od_lo in [2,3,5]:
            mask=both_r1&(cg>=gap_th)&(sd>=sd_th)&(od>=od_lo)
            sub=result[mask]; n=len(sub)
            if n<10: continue
            wins=int(sub['target_win'].sum())
            pay=sub.loc[sub['target_win']==1,'単勝配当'].sum()
            roi=(pay/(n*100)-1)*100
            rl.append((f'両Rnk=1 gap≥{gap_th} sd≥{sd_th} odds≥{od_lo}',n,wins,roi))

# star系
for sd_th in [5,10,15,20]:
    for od_lo in [3,5,8]:
        for gap_th in [0,5,10,15]:
            mask=star&(sd>=sd_th)&(od>=od_lo)&(cg>=gap_th)
            sub=result[mask]; n=len(sub)
            if n<10: continue
            wins=int(sub['target_win'].sum())
            pay=sub.loc[sub['target_win']==1,'単勝配当'].sum()
            roi=(pay/(n*100)-1)*100
            rl.append((f'star sd≥{sd_th} odds≥{od_lo} gap≥{gap_th}',n,wins,roi))

# cur≤2 & sub≤2系
for sd_th in [0,5,10]:
    for od_lo in [3,5]:
        mask=(cr<=2)&(sr<=2)&(sd>=sd_th)&(od>=od_lo)
        sub=result[mask]; n=len(sub)
        if n<10: continue
        wins=int(sub['target_win'].sum())
        pay=sub.loc[sub['target_win']==1,'単勝配当'].sum()
        roi=(pay/(n*100)-1)*100
        rl.append((f'両Rnk≤2 sd≥{sd_th} odds≥{od_lo}',n,wins,roi))

rl.sort(key=lambda x:-x[3])
for label,n,wins,roi in rl[:25]:
    print(f'  {label:<46} {n:>4}  {wins:>4}  {roi:>+7.1f}%  {n/weeks:>4.1f}')

# ── 推奨条件の単複傾斜シミュレーション ────────────────────
print()
print('='*72)
print('  推奨条件 単複傾斜シミュレーション（1〜3月累計）')
print('='*72)

best = [
    ('【A】両Rnk=1 gap≥10 sd≥10 odds≥3 （激熱候補）',
     both_r1&(cg>=10)&(sd>=10)&(od>=3)),
    ('【B】両Rnk=1 gap<10 sd≥10 odds≥3 （〇候補）',
     both_r1&(cg<10)&(sd>=10)&(od>=3)),
    ('【C】両Rnk=1 gap≥10 sd≥0 odds≥3 （◎全体）',
     both_r1&(cg>=10)&(od>=3)),
    ('【D】star sd≥10 odds≥5 （☆新条件）',
     star&(sd>=10)&(od>=5)),
    ('【E】star sd≥20 odds≥5 （☆厳選）',
     star&(sd>=20)&(od>=5)),
]

for cond_name, mask in best:
    sub=result[mask]; n=len(sub)
    if n==0: continue
    wins=int(sub['target_win'].sum()); places=int(sub['target_place'].sum())
    win_pay=sub.loc[sub['target_win']==1,'単勝配当'].sum()
    place_pay=sub.loc[sub['target_place']==1,'複勝配当'].sum()
    avg_od=sub['単勝オッズ'].mean()
    print(f'\n{cond_name}')
    print(f'  {n}頭 ({n/weeks:.1f}/週)  単勝率{wins/n:.1%}  複勝率{places/n:.1%}  平均{avg_od:.1f}倍')
    print(f'  {"単勝額":>6}  {"単ROI":>8}  + {"複200円":>8}  {"複ROI":>8}  = {"合計ROI":>8}')
    for bet in [100,200,300,500,1000]:
        tinv=n*bet; tpay=win_pay*(bet/100)
        troi=(tpay-tinv)/tinv*100
        finv=n*200; fpay=place_pay*2 if place_pay>0 else 0
        croi=((tpay+fpay)-(tinv+finv))/(tinv+finv)*100
        print(f'  {bet:>6,}円  {troi:>+7.1f}%    複200:  {(fpay-finv)/finv*100 if finv else 0:>+7.1f}%    {croi:>+7.1f}%')

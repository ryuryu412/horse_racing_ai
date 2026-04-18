"""複勝ROI分析"""
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
for c in ['着順_num','単勝配当','単勝オッズ','複勝配当']:
    df[c]=pd.to_numeric(df[c],errors='coerce')
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
all_rows=[]
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
    sub['combo_gap']=sub.get('cur_gap',pd.Series(0,index=sub.index)).fillna(0)+sub.get('sub_gap',pd.Series(0,index=sub.index)).fillna(0)
    all_rows.append(sub)

result=pd.concat(all_rows,ignore_index=True)
cr=pd.to_numeric(result['cur_rank'],errors='coerce')
sr=pd.to_numeric(result['sub_rank'],errors='coerce')
sd=pd.to_numeric(result['sub_diff'],errors='coerce')
cd=pd.to_numeric(result['cur_diff'],errors='coerce')
od=pd.to_numeric(result['単勝オッズ'],errors='coerce')
both_r1=(cr==1)&(sr==1)
ok3=od.isna()|(od>=3); ok5=od.isna()|(od>=5)
weeks=11

# 複勝配当の分布確認
fuku = pd.to_numeric(result['複勝配当'], errors='coerce')
print(f'複勝配当データあり: {fuku.notna().sum()}/{len(result)}頭')
hit3 = result['target_place']==1
print(f'複勝配当(3着以内)の分布:')
print(f'  平均: {fuku[hit3].mean():.0f}円  中央値: {fuku[hit3].median():.0f}円')
print(f'  100円台(1倍台): {(fuku[hit3]<200).sum()}頭 ({(fuku[hit3]<200).mean():.1%})')
print(f'  200-299円(2倍台): {((fuku[hit3]>=200)&(fuku[hit3]<300)).sum()}頭 ({((fuku[hit3]>=200)&(fuku[hit3]<300)).mean():.1%})')
print(f'  300円以上(3倍以上): {(fuku[hit3]>=300).sum()}頭 ({(fuku[hit3]>=300).mean():.1%})')

print()
print('='*62)
print('  条件別 単勝 vs 複勝 ROI比較（2026年1-3月）')
print('='*62)

mask_g   = both_r1 & (cd>=10) & (sd>=10) & ok5
mask_o   = both_r1 & (sd>=10) & ok3 & ~mask_g
mask_tri = (cr<=2)&(sr<=2)&~both_r1 & (sd>=10) & ok5
mask_h   = (cr<=3)&(sr<=3)&~both_r1 & ~((cr<=2)&(sr<=2)) & (sd>=10) & ok5

for label, mask in [('激熱',mask_g),('〇',mask_o),('▲',mask_tri),('☆',mask_h)]:
    sub=result[mask]; n=len(sub)
    if n==0: continue
    places=int(sub['target_place'].sum())
    tan_pay=sub.loc[sub['target_win']==1,'単勝配当'].sum()
    tan_roi=(tan_pay/(n*100)-1)*100
    # 複勝: 3着以内かつ複勝配当がある馬だけ
    fuku_sub = sub.loc[sub['target_place']==1, '複勝配当'].dropna()
    avg_fuku = fuku_sub.mean() if len(fuku_sub)>0 else 0
    # 全馬ベースのROI（全馬に100円賭けた場合）
    fuku_roi_all = (fuku_sub.sum()/(n*100)-1)*100 if n>0 else 0
    print(f'\n【{label}】{n}頭  複勝率{places/n:.1%}  平均複勝配当{avg_fuku:.0f}円')
    print(f'  単勝ROI(全馬100円): {tan_roi:+.1f}%')
    print(f'  複勝ROI(全馬100円): {fuku_roi_all:+.1f}%  ← 平均{avg_fuku:.0f}円×{places}頭÷{n}頭×100円')
    # オッズ帯別
    print(f'  オッズ帯別:')
    for od_lo, od_hi, label2 in [(3,5,'3-5倍'),(5,8,'5-8倍'),(8,15,'8-15倍'),(15,999,'15倍以上')]:
        sub2=result[mask & (od>=od_lo) & (od<od_hi)]
        n2=len(sub2)
        if n2<3: continue
        f2=sub2.loc[sub2['target_place']==1,'複勝配当'].dropna()
        fpl=int((sub2['target_place']==1).sum())
        avg2=f2.mean() if len(f2)>0 else 0
        roi2=(f2.sum()/(n2*100)-1)*100
        tan2_pay=sub2.loc[sub2['target_win']==1,'単勝配当'].sum()
        tan2_roi=(tan2_pay/(n2*100)-1)*100
        print(f'    odds{label2}: {n2}頭 複勝率{fpl/n2:.1%} 平均配当{avg2:.0f}円 複勝ROI{roi2:+.1f}% / 単勝ROI{tan2_roi:+.1f}%')

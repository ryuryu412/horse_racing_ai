"""
両Rnk≤3 評価
・両Rnk=1 vs 両Rnk≤3（combo1位のみ）vs 両Rnk≤3（全頭）を比較
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import pandas as pd, numpy as np, os, pickle, json, re

base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

with open(os.path.join(model_dir, 'model_info.json'), encoding='utf-8') as f: cur_info = json.load(f)
cur_features = cur_info['features']; cur_models = cur_info['models']
with open(os.path.join(model_dir, 'ranker', 'ranker_info.json'), encoding='utf-8') as f:
    cur_rankers = json.load(f).get('rankers', {})
with open(os.path.join(model_dir, 'submodel', 'submodel_info.json'), encoding='utf-8') as f: sub_info = json.load(f)
sub_features = sub_info['features']; sub_models = sub_info['models']
with open(os.path.join(model_dir, 'submodel_ranker', 'class_ranker_info.json'), encoding='utf-8') as f:
    sub_rankers = json.load(f).get('rankers', {})
all_feats = list(set(cur_features + sub_features))

test_path = os.path.join(base_dir, 'data', 'processed', 'all_venues_features_2026test.csv')
df = pd.read_csv(test_path, low_memory=False)
df['着順_num']   = pd.to_numeric(df['着順_num'],   errors='coerce')
df['単勝配当']   = pd.to_numeric(df['単勝配当'],   errors='coerce')
df['単勝オッズ'] = pd.to_numeric(df['単勝オッズ'], errors='coerce')
df['複勝配当']   = pd.to_numeric(df['複勝配当'],   errors='coerce')
df = df.dropna(subset=['着順_num', '単勝配当'])
df['target_win']   = (df['着順_num'] == 1).astype(int)
df['target_place'] = (df['着順_num'] <= 3).astype(int)
df['会場']       = df['開催'].apply(extract_venue)
df['_surface']   = df['芝・ダ'].astype(str).str.strip()
df['cur_key']    = df['会場'] + '_' + df['距離'].astype(str)
df['_dist_band'] = df['距離'].apply(get_distance_band)
mask_da = (df['_surface'] == 'ダ') & (df['_dist_band'].isin(['中距離','長距離']))
df.loc[mask_da, '_dist_band'] = '中長距離'
df['_cls_group'] = df['クラス_rank'].apply(get_class_group) if 'クラス_rank' in df.columns else '3勝以上'
df['sub_key']    = df['_surface'] + '_' + df['_dist_band'] + '_' + df['_cls_group']
for col in all_feats:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

cur_mc={}; cur_rc={}; sub_mc={}; sub_rc={}
for ck in df['cur_key'].dropna().unique():
    if ck in cur_models:
        p = os.path.join(model_dir, cur_models[ck]['win'])
        if os.path.exists(p):
            with open(p,'rb') as f: m=pickle.load(f)
            cur_mc[ck]=(m,m.booster_.feature_name())
    if ck in cur_rankers:
        p = os.path.join(model_dir,'ranker',cur_rankers[ck])
        if os.path.exists(p):
            with open(p,'rb') as f: cur_rc[ck]=pickle.load(f)
for sk in df['sub_key'].dropna().unique():
    if sk in sub_models:
        p = os.path.join(model_dir,'submodel',sub_models[sk]['win'])
        if os.path.exists(p):
            with open(p,'rb') as f: m=pickle.load(f)
            sub_mc[sk]=(m,m.booster_.feature_name())
    if sk in sub_rankers:
        p = os.path.join(model_dir,'submodel_ranker',sub_rankers[sk])
        if os.path.exists(p):
            with open(p,'rb') as f: sub_rc[sk]=pickle.load(f)

race_keys = [c for c in ['開催','Ｒ'] if c in df.columns]
all_rows=[]
for gk, idx in df.groupby(race_keys, sort=False).groups.items():
    sub = df.loc[idx].copy()
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
cr  = pd.to_numeric(result['cur_rank'],  errors='coerce')
sr  = pd.to_numeric(result['sub_rank'],  errors='coerce')
sd  = pd.to_numeric(result['sub_diff'],  errors='coerce')
cd  = pd.to_numeric(result['cur_diff'],  errors='coerce')
result['combo_diff'] = cd.fillna(0)+sd.fillna(0)
result['combo_rank'] = result.groupby(race_keys)['combo_diff'].rank(ascending=False, method='min')
result['odds']  = pd.to_numeric(result['単勝オッズ'], errors='coerce')
result['sd_val']= sd
result['cr_val']= cr
result['sr_val']= sr

def stats(label, s):
    n=len(s)
    if n==0: print(f'  {label}: 該当なし'); return
    wr=s['target_win'].mean()
    hr=s['target_place'].mean()
    ret_t=s.loc[s['target_win']==1,'単勝配当'].sum()
    ret_f=s.loc[s['target_place']==1,'複勝配当'].sum()
    roi_t=ret_t/(n*100)-1
    roi_f=ret_f/(n*100)-1
    ao=s['odds'].mean()
    print(f'  {label:<42} N={n:>4}  勝率{wr:.1%}  複勝率{hr:.1%}  平均{ao:.1f}倍  単勝ROI{roi_t:>+7.1%}  複勝ROI{roi_f:>+6.1%}')

print('='*90)
print('  ランク条件別 基本比較（odds≥3, 一律100円）')
print('='*90)

# パターン1: 両Rnk=1
r1 = result[(cr==1)&(sr==1)&(result['odds']>=3)]
stats('① 両Rnk=1', r1)

# パターン2: 両Rnk≤3 の中でcombo_rank=1
r2 = result[(cr<=3)&(sr<=3)&(result['combo_rank']==1)&(result['odds']>=3)]
stats('② 両Rnk≤3 & combo_rank=1（1頭絞り）', r2)

# パターン3: 両Rnk≤3 全頭
r3 = result[(cr<=3)&(sr<=3)&(result['odds']>=3)]
stats('③ 両Rnk≤3 全頭', r3)

# パターン4: 両Rnk≤2 の中でcombo_rank=1
r4 = result[(cr<=2)&(sr<=2)&(result['combo_rank']==1)&(result['odds']>=3)]
stats('④ 両Rnk≤2 & combo_rank=1（1頭絞り）', r4)

# パターン5: 片方=1 もう片方≤3 でcombo=1
r5 = result[((cr==1)|(sr==1))&(cr<=3)&(sr<=3)&(result['combo_rank']==1)&(result['odds']>=3)]
stats('⑤ どちらか1位・両方3位以内 & combo=1', r5)

print()
print('='*90)
print('  sd≥10 フィルター追加')
print('='*90)
stats('① 両Rnk=1 & sd≥10', result[(cr==1)&(sr==1)&(result['odds']>=3)&(result['sd_val']>=10)])
stats('② 両Rnk≤3 & combo=1 & sd≥10', result[(cr<=3)&(sr<=3)&(result['combo_rank']==1)&(result['odds']>=3)&(result['sd_val']>=10)])
stats('③ 両Rnk≤3 全頭 & sd≥10', result[(cr<=3)&(sr<=3)&(result['odds']>=3)&(result['sd_val']>=10)])
stats('⑤ どちらか1位・3位以内 & combo=1 & sd≥10', result[((cr==1)|(sr==1))&(cr<=3)&(sr<=3)&(result['combo_rank']==1)&(result['odds']>=3)&(result['sd_val']>=10)])

print()
print('='*90)
print('  1レースあたりの平均賭け頭数（両Rnk≤3 全頭の場合）')
print('='*90)
r3_full = result[(cr<=3)&(sr<=3)&(result['odds']>=3)]
per_race = r3_full.groupby(race_keys).size()
print(f'  平均 {per_race.mean():.1f}頭/レース  (最小{per_race.min()}頭 / 最大{per_race.max()}頭)')
print(f'  1頭だけ該当: {(per_race==1).sum()}レース ({(per_race==1).mean():.0%})')
print(f'  2頭該当:     {(per_race==2).sum()}レース ({(per_race==2).mean():.0%})')
print(f'  3頭以上該当: {(per_race>=3).sum()}レース ({(per_race>=3).mean():.0%})')

print()
print('='*90)
print('  【参考】ランク別ROI内訳（両Rnk≤3 & combo=1 & odds≥3）')
print('='*90)
for cv, sv in [(1,1),(1,2),(1,3),(2,1),(3,1),(2,2),(2,3),(3,2),(3,3)]:
    s = result[(cr==cv)&(sr==sv)&(result['combo_rank']==1)&(result['odds']>=3)]
    if len(s)<5: continue
    stats(f'cur_rank={cv} & sub_rank={sv}', s)

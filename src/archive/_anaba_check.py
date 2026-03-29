"""
穴馬探し: 両Rnk 2〜3位 × 高オッズ
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
result['odds']   = pd.to_numeric(result['単勝オッズ'], errors='coerce')
result['sd_val'] = sd
result['cr_val'] = cr
result['sr_val'] = sr

def show(label, s):
    n=len(s)
    if n==0: return
    wr=s['target_win'].mean(); hr=s['target_place'].mean()
    ret_t=s.loc[s['target_win']==1,'単勝配当'].sum()
    ret_f=s.loc[s['target_place']==1,'複勝配当'].sum()
    roi_t=ret_t/(n*100)-1; roi_f=ret_f/(n*100)-1
    ao=s['odds'].mean()
    print(f'  {label:<44} N={n:>3}  平均{ao:.1f}倍  単勝率{wr:.1%} ROI{roi_t:>+7.1%}  複勝率{hr:.1%} ROI{roi_f:>+6.1%}')

# ── ① 両方2〜3位 × オッズ帯 ──
print('='*85)
print('  【穴馬候補】両Rnk 2〜3位 × オッズ帯別')
print('='*85)
base_23 = result[(cr>=2)&(cr<=3)&(sr>=2)&(sr<=3)]
show('両Rnk 2〜3位 全体',              base_23)
show('  odds≥5',                       base_23[base_23['odds']>=5])
show('  odds≥7',                       base_23[base_23['odds']>=7])
show('  odds≥10',                      base_23[base_23['odds']>=10])
show('  odds≥5 & sd≥0（sdプラス）',    base_23[(base_23['odds']>=5)&(base_23['sd_val']>=0)])
show('  odds≥5 & sd≥5',               base_23[(base_23['odds']>=5)&(base_23['sd_val']>=5)])
show('  odds≥5 & sd≥10',              base_23[(base_23['odds']>=5)&(base_23['sd_val']>=10)])
show('  odds≥5 & combo_rank≤3',       base_23[(base_23['odds']>=5)&(result['combo_rank']<=3)])
show('  odds≥5 & combo_rank=2',       base_23[(base_23['odds']>=5)&(result['combo_rank']==2)])

print()
# ── ② ランク組み合わせ別（高オッズ） ──
print('='*85)
print('  【ランク組み合わせ別】odds≥5 / combo_rank≤3')
print('='*85)
for cv in [2,3]:
    for sv in [2,3]:
        s = result[(cr==cv)&(sr==sv)&(result['odds']>=5)&(result['combo_rank']<=3)]
        show(f'cur={cv} & sub={sv} & combo≤3 & odds≥5', s)

print()
# ── ③ combo_rank=2 の穴馬 ──
print('='*85)
print('  【combo_rank=2 穴馬】両Rnk≤3 & odds≥5')
print('  ※1位の次点、モデル的には準推奨')
print('='*85)
c2 = result[(result['combo_rank']==2)&(cr<=3)&(sr<=3)&(result['odds']>=5)]
show('combo_rank=2 & 両Rnk≤3 & odds≥5',      c2)
show('  sd≥0',                                 c2[c2['sd_val']>=0])
show('  sd≥5',                                 c2[c2['sd_val']>=5])
show('  sd≥10',                                c2[c2['sd_val']>=10])
show('  cr=2 & sr=2',                          c2[(c2['cr_val']==2)&(c2['sr_val']==2)])
show('  cr=2 & sr=3',                          c2[(c2['cr_val']==2)&(c2['sr_val']==3)])
show('  cr=3 & sr=2',                          c2[(c2['cr_val']==3)&(c2['sr_val']==2)])
show('  cr=3 & sr=3',                          c2[(c2['cr_val']==3)&(c2['sr_val']==3)])

print()
# ── ④ 本命+穴馬 セット戦略 ──
print('='*85)
print('  【戦略案】本命(両Rnk=1 & odds≥3 & sd≥10) + 穴馬セット')
print('='*85)
honmei = result[(cr==1)&(sr==1)&(result['odds']>=3)&(result['sd_val']>=10)]
anaba_best = result[(cr>=2)&(cr<=3)&(sr>=2)&(sr<=3)&(result['odds']>=5)&(result['sd_val']>=5)&(result['combo_rank']<=3)]

print(f'\n  本命: 単勝200円  N={len(honmei)}')
inv_h=len(honmei)*200; ret_h=honmei.loc[honmei['target_win']==1,'単勝配当'].mul(2).sum()
print(f'    投資{inv_h:,}円  回収{int(ret_h):,}円  損益{int(ret_h-inv_h):+,}円  ROI{ret_h/inv_h-1:>+7.1%}')

print(f'\n  穴馬: 複勝100円  N={len(anaba_best)}')
inv_a=len(anaba_best)*100; ret_a=anaba_best.loc[anaba_best['target_place']==1,'複勝配当'].sum()
print(f'    投資{inv_a:,}円  回収{int(ret_a):,}円  損益{int(ret_a-inv_a):+,}円  ROI{ret_a/inv_a-1:>+7.1%}')

print(f'\n  合計:')
print(f'    投資{inv_h+inv_a:,}円  回収{int(ret_h+ret_a):,}円  損益{int(ret_h+ret_a-inv_h-inv_a):+,}円  ROI{(ret_h+ret_a)/(inv_h+inv_a)-1:>+7.1%}')

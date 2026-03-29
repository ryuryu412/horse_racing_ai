"""
両Rnk=1 オッズフィルター比較（1〜2倍見送り）
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
cr=pd.to_numeric(result['cur_rank'],errors='coerce')
sr=pd.to_numeric(result['sub_rank'],errors='coerce')
sd=pd.to_numeric(result['sub_diff'],errors='coerce')

base_all = result[(cr==1)&(sr==1)].copy()
base_all['sd_val'] = sd[base_all.index]
base_all['odds']   = pd.to_numeric(base_all['単勝オッズ'], errors='coerce')

def show_tan(label, s):
    n = len(s)
    if n == 0:
        print(f'  {label}: 該当なし'); return
    wins = int(s['target_win'].sum())
    ret  = s.loc[s['target_win']==1,'単勝配当'].sum()
    roi  = ret/(n*100)-1
    ao   = s['odds'].mean()
    print(f'  {label:<32} N={n:>3}  勝率{wins/n:.1%}  平均{ao:.1f}倍  損益{int(ret-n*100):>+7,}円  ROI{roi:>+7.1%}')

def show_fuku(label, s):
    n = len(s)
    if n == 0:
        print(f'  {label}: 該当なし'); return
    hits = int(s['target_place'].sum())
    ret  = s.loc[s['target_place']==1,'複勝配当'].sum()
    roi  = ret/(n*100)-1
    print(f'  {label:<32} N={n:>3}  複勝率{hits/n:.1%}  損益{int(ret-n*100):>+7,}円  ROI{roi:>+7.1%}')

# ── 単勝 ──
print('='*70)
print('  【単勝】両Rnk=1 オッズフィルター比較')
print('='*70)
show_tan('全部(フィルターなし)',             base_all)
show_tan('odds≥3（1〜2倍見送り）',           base_all[base_all['odds']>=3])
show_tan('odds≥3 & sd≥10',                  base_all[(base_all['odds']>=3)&(base_all['sd_val']>=10)])
show_tan('odds 3〜9倍',                      base_all[(base_all['odds']>=3)&(base_all['odds']<10)])
show_tan('odds 3〜9倍 & sd≥10',              base_all[(base_all['odds']>=3)&(base_all['odds']<10)&(base_all['sd_val']>=10)])

print()
print('  変動ベット(単勝): odds≥3 & sd≥10→200円 / それ以外→100円')
filt = base_all[base_all['odds']>=3].copy()
filt['bet'] = filt['sd_val'].apply(lambda x: 200 if x>=10 else 100)
invest = filt['bet'].sum()
ret    = filt.apply(lambda r: r['単勝配当']*(r['bet']/100) if r['target_win']==1 else 0, axis=1).sum()
roi    = ret/invest-1
print(f'  投資{int(invest):,}円  回収{int(ret):,}円  損益{int(ret-invest):+,}円  ROI{roi:>+7.1%}')

# ── 複勝 ──
print()
print('='*70)
print('  【複勝】両Rnk=1 オッズフィルター比較')
print('='*70)
show_fuku('全部(フィルターなし)',             base_all)
show_fuku('odds≥3（1〜2倍見送り）',           base_all[base_all['odds']>=3])
show_fuku('odds≥3 & sd≥10',                  base_all[(base_all['odds']>=3)&(base_all['sd_val']>=10)])
show_fuku('odds 3〜9倍',                      base_all[(base_all['odds']>=3)&(base_all['odds']<10)])
show_fuku('odds 3〜9倍 & sd≥10',              base_all[(base_all['odds']>=3)&(base_all['odds']<10)&(base_all['sd_val']>=10)])
show_fuku('sd≥20',                            base_all[base_all['sd_val']>=20])
show_fuku('odds≥3 & sd≥20',                  base_all[(base_all['odds']>=3)&(base_all['sd_val']>=20)])

print()
print('  変動ベット(複勝): odds≥3 & sd≥10→200円 / それ以外→100円')
filt2 = base_all[base_all['odds']>=3].copy()
filt2['bet'] = filt2['sd_val'].apply(lambda x: 200 if x>=10 else 100)
invest2 = filt2['bet'].sum()
ret2    = filt2.apply(lambda r: r['複勝配当']*(r['bet']/100) if r['target_place']==1 else 0, axis=1).sum()
roi2    = ret2/invest2-1
print(f'  投資{int(invest2):,}円  回収{int(ret2):,}円  損益{int(ret2-invest2):+,}円  ROI{roi2:>+7.1%}')

# ── sd帯別 複勝ROI ──
print()
print('  【複勝 sub_diff帯別 詳細】')
for lo, hi, label in [(-99,0,'sd<0'),(0,10,'0≤sd<10'),(10,20,'10≤sd<20'),(20,99,'sd≥20')]:
    s = base_all[(base_all['sd_val']>=lo)&(base_all['sd_val']<hi)&(base_all['odds']>=3)]
    if len(s)==0: continue
    n=len(s); hits=int(s['target_place'].sum())
    ret=s.loc[s['target_place']==1,'複勝配当'].sum()
    roi=ret/(n*100)-1
    avg_fuku = s.loc[s['target_place']==1,'複勝配当'].mean() if hits>0 else 0
    print(f'  {label:<15} N={n:>3}  複勝率{hits/n:.1%}  平均複勝配当{avg_fuku:.0f}円  損益{int(ret-n*100):>+6,}円  ROI{roi:>+7.1%}')

# ── combo_gap 計算 ──
for key, grp in result.groupby(race_keys):
    idx=grp.index
    for col, gcol in [('cur_score','cur_gap'),('sub_score','sub_gap')]:
        scores=grp[col].dropna().sort_values(ascending=False).values
        gap = scores[0]-scores[1] if len(scores)>=2 else np.nan
        result.loc[idx,gcol]=gap

base_all = result[(cr==1)&(sr==1)].copy()
base_all['sd_val']   = sd[base_all.index]
base_all['odds']     = pd.to_numeric(base_all['単勝オッズ'], errors='coerce')
base_all['cur_gap_v']= pd.to_numeric(base_all['cur_gap'], errors='coerce')
base_all['sub_gap_v']= pd.to_numeric(base_all['sub_gap'], errors='coerce')
base_all['combo_gap']= base_all['cur_gap_v'].fillna(0)+base_all['sub_gap_v'].fillna(0)

def fuku_row(label, s):
    n=len(s)
    if n==0: return
    hits=int(s['target_place'].sum())
    ret=s.loc[s['target_place']==1,'複勝配当'].sum()
    roi=ret/(n*100)-1
    avg_f=s.loc[s['target_place']==1,'複勝配当'].mean() if hits>0 else 0
    print(f'  {label:<38} N={n:>3}  複勝率{hits/n:.1%}  平均配当{avg_f:.0f}円  損益{int(ret-n*100):>+6,}円  ROI{roi:>+7.1%}')

print()
print('='*70)
print('  【複勝 条件絞り込み探索】両Rnk=1 & odds≥3 ベース')
print('='*70)
b = base_all[base_all['odds']>=3]
fuku_row('ベース(odds≥3)',                         b)
fuku_row('sd≥10',                                   b[b['sd_val']>=10])
fuku_row('sd≥10 & combo_gap≥10',                   b[(b['sd_val']>=10)&(b['combo_gap']>=10)])
fuku_row('sd≥10 & combo_gap≥15',                   b[(b['sd_val']>=10)&(b['combo_gap']>=15)])
fuku_row('sd≥10 & combo_gap≥20',                   b[(b['sd_val']>=10)&(b['combo_gap']>=20)])
fuku_row('sd≥10 & sub_gap≥8',                      b[(b['sd_val']>=10)&(b['sub_gap_v']>=8)])
fuku_row('sd≥10 & sub_gap≥10',                     b[(b['sd_val']>=10)&(b['sub_gap_v']>=10)])
fuku_row('sd≥10 & odds 3〜7倍',                    b[(b['sd_val']>=10)&(b['odds']<7)])
fuku_row('combo_gap≥20',                            b[b['combo_gap']>=20])
fuku_row('combo_gap≥25',                            b[b['combo_gap']>=25])
fuku_row('sd≥10 & combo_gap≥20 変動200円相当',     b[(b['sd_val']>=10)&(b['combo_gap']>=20)])

print()
print('  【変動ベット案: 複勝 sd≥10 & combo_gap≥15 → 300円 / sd≥10 → 200円 / それ以外 → 100円】')
def assign_fuku_bet(row):
    if row['sd_val']>=10 and row['combo_gap']>=15: return 300
    if row['sd_val']>=10: return 200
    return 100
b2 = base_all[base_all['odds']>=3].copy()
b2['bet'] = b2.apply(assign_fuku_bet, axis=1)
invest3 = b2['bet'].sum()
ret3    = b2.apply(lambda r: r['複勝配当']*(r['bet']/100) if r['target_place']==1 else 0, axis=1).sum()
roi3    = ret3/invest3-1
print(f'  投資{int(invest3):,}円  回収{int(ret3):,}円  損益{int(ret3-invest3):+,}円  ROI{roi3:>+7.1%}')
for tier in [100,200,300]:
    s=b2[b2['bet']==tier]; n=len(s)
    if n==0: continue
    hits=int(s['target_place'].sum())
    ret_t=s.loc[s['target_place']==1,'複勝配当'].mul(tier/100).sum()
    print(f'    {tier}円ティア: N={n:>3}  複勝率{hits/n:.1%}  損益{int(ret_t-n*tier):>+6,}円  ROI{ret_t/(n*tier)-1:>+7.1%}')

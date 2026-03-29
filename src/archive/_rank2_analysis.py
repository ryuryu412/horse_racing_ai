"""
cur=2 & sub=1 の馬がいるレースで、実際に誰が勝っているか分析
→ 単勝向きか複勝向きかを判定
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

# cur=2 & sub=1 & combo=1 のレースに絞る
target = result[(cr==2)&(sr==1)&(result['combo_rank']==1)&(result['odds']>=3)].copy()
race_ids = target[race_keys].drop_duplicates()
# そのレース全体を取得
race_full = result.merge(race_ids, on=race_keys)

print('='*70)
print(f'  cur=2 & sub=1 & combo=1 & odds≥3 の馬がいるレース: {len(target)}レース')
print('='*70)

# そのレースでcur=1の馬の成績
cur1_in_race = race_full[race_full['cr_val']==1]
print(f'\n  同レース内 cur_rank=1 の馬（距離モデル1位）:')
n1=len(cur1_in_race)
w1=int(cur1_in_race['target_win'].sum())
h1=int(cur1_in_race['target_place'].sum())
print(f'    N={n1}  勝率{w1/n1:.1%}  複勝率{h1/n1:.1%}  ← これが単勝を取ってしまう頻度')

# cur=2&sub=1馬自体の単勝・複勝
print(f'\n  cur=2 & sub=1 の馬自体:')
nt=len(target)
wt=int(target['target_win'].sum())
ht=int(target['target_place'].sum())
ret_t=target.loc[target['target_win']==1,'単勝配当'].sum()
ret_f=target.loc[target['target_place']==1,'複勝配当'].sum()
print(f'    単勝: N={nt}  勝率{wt/nt:.1%}  ROI{ret_t/(nt*100)-1:>+7.1%}')
print(f'    複勝: N={nt}  複勝率{ht/nt:.1%}  ROI{ret_f/(nt*100)-1:>+7.1%}')

# 「cur=1が勝った」「cur=2&sub=1が勝った」「その他が勝った」の内訳
print(f'\n  このレースの勝ち馬内訳:')
won_by_cur1  = race_full[(race_full['target_win']==1)&(race_full['cr_val']==1)]
won_by_target= target[target['target_win']==1]
total_races  = len(target)
print(f'    cur=1の馬が勝利:         {len(won_by_cur1)}レース ({len(won_by_cur1)/total_races:.1%})')
print(f'    cur=2&sub=1が勝利:       {len(won_by_target)}レース ({len(won_by_target)/total_races:.1%})')
other = total_races - len(won_by_cur1) - len(won_by_target)
print(f'    それ以外が勝利:          {other}レース ({other/total_races:.1%})')

# sd≥10 で絞った場合
print()
print('='*70)
print('  sd≥10 フィルター追加時')
print('='*70)
tsd = target[target['sd_val']>=10]
if len(tsd)>0:
    wsd=int(tsd['target_win'].sum()); hsd=int(tsd['target_place'].sum())
    ret_tsd=tsd.loc[tsd['target_win']==1,'単勝配当'].sum()
    ret_fsd=tsd.loc[tsd['target_place']==1,'複勝配当'].sum()
    print(f'  N={len(tsd)}  単勝率{wsd/len(tsd):.1%} ROI{ret_tsd/(len(tsd)*100)-1:>+7.1%}  複勝率{hsd/len(tsd):.1%} ROI{ret_fsd/(len(tsd)*100)-1:>+7.1%}')

# 提案: cur=1&sub=1 → 単勝, cur=2&sub=1 → 複勝 の組み合わせ戦略
print()
print('='*70)
print('  【提案戦略】ランクで券種を使い分け（odds≥3 & sd≥10）')
print('='*70)
s11 = result[(cr==1)&(sr==1)&(result['odds']>=3)&(result['sd_val']>=10)]
s21 = result[(cr==2)&(sr==1)&(result['combo_rank']==1)&(result['odds']>=3)&(result['sd_val']>=10)]

invest_tan = len(s11)*200
ret_tan    = s11.loc[s11['target_win']==1,'単勝配当'].mul(2).sum()
invest_fuku= len(s21)*200
ret_fuku   = s21.loc[s21['target_place']==1,'複勝配当'].mul(2).sum()
total_inv  = invest_tan+invest_fuku
total_ret  = ret_tan+ret_fuku

print(f'  cur=1&sub=1 → 単勝200円')
print(f'    N={len(s11)}  勝率{s11["target_win"].mean():.1%}  投資{invest_tan:,}円  回収{int(ret_tan):,}円  損益{int(ret_tan-invest_tan):+,}円  ROI{ret_tan/invest_tan-1:>+7.1%}')
print(f'  cur=2&sub=1(combo=1) → 複勝200円')
print(f'    N={len(s21)}  複勝率{s21["target_place"].mean():.1%}  投資{invest_fuku:,}円  回収{int(ret_fuku):,}円  損益{int(ret_fuku-invest_fuku):+,}円  ROI{ret_fuku/invest_fuku-1:>+7.1%}')
print(f'  合計: 投資{total_inv:,}円  回収{int(total_ret):,}円  損益{int(total_ret-total_inv):+,}円  ROI{total_ret/total_inv-1:>+7.1%}')

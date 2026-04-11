import pickle, pandas as pd, numpy as np, os, sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open('data/raw/cache/出馬表形式4月12日.cache.pkl', 'rb') as f:
    cached = pickle.load(f)
df = cached['result']
card_df = cached['card_df']

hans = df[df['開催'].astype(str).str.contains('阪', na=False)].copy()

# ── 発走時刻マップ ──
time_map = {}
if '発走時刻' in card_df.columns:
    c = card_df[card_df['開催'].astype(str).str.contains('阪', na=False)]
    for _, row in c[['Ｒ','発走時刻']].drop_duplicates().iterrows():
        try: time_map[int(float(row['Ｒ']))] = str(row['発走時刻'])
        except: pass

# ═══════════════════��══════════════════════════
# コース特性データベース
# ══════════════════════════════════════════════
COURSE_INFO = {
    ('ダ', '1200'): {
        'title': '阪神ダート1200m',
        'shape': '向正面2コーナー出口付近からスタート。コーナー1つを経て直線350mへ。実質ほぼ一直線に近いスプリントコース。',
        'points': [
            '🥇 逃げ・先行が圧倒的優位。出遅れはほぼ致命傷、スタートダッシュが全て',
            '🎰 内枠有利。外枠は砂が深いコース外側を長く走る分タイムロス大',
            '⚡ 前半から速いラップが刻まれる。スタミナより純粋なパワー×スピードの勝負',
            '🌧️ 雨で時計がかかるほどパワー型がさらに台頭。前残り傾向は変わらない',
            '👀 前走で砂を被った経験のある馬・ゲートが良い馬を評価',
        ],
        'blood': '🧬 ゴールドアリュール系・フレンチデピュティ系・ストームキャット系など砂スプリント適性の高い血統が主役。母父にサンデーサイレンス系を持つ馬は芝寄りになりすぎて砂で苦労するケースも。先行力が遺伝されているかを父母双方から確認したい。',
        'color': '#e67e22',
    },
    ('ダ', '1400'): {
        'title': '阪神ダート1400m',
        'shape': '2コーナーポケット奥からスタートし、3・4コーナーを経て直線350mへ。阪神ダの中でバランスが良く最も使われる距離。',
        'points': [
            '🥇 先行〜好位が有利だが1200mほど極端ではなく差しも届く',
            '🎰 外枠はコーナーで外を回らされロスが大きい。1〜4枠を重視',
            '📊 ペースが緩みやすく前が残る展開が多い。ただしハイペースなら差し台頭',
            '💪 末脚より道中の機動力・コーナリングのうまさが問われる',
            '👀 前走ダ1200mで先行して粘った馬は距離延長で更に脚を残せる可能性大',
        ],
        'blood': '🧬 ゴールドアリュール・ヘニーヒューズ・カジノドライブ系が安定。マイル〜中距離ダートをこなすスタミナも必要なので、父系だけでなく母系のスタミナ源も確認を。',
        'color': '#e67e22',
    },
    ('ダ', '1800'): {
        'title': '阪神ダート1800m',
        'shape': 'スタンド前からスタートし、1〜4コーナーを大回りして直線350mへ。前半から3コーナーまでの距離が長くスタミナが問われる。',
        'points': [
            '⚖️ 先行〜中団まで幅広く馬券に絡む。差し・追い込みも末脚次第で届く',
            '📊 前半ペースが上がりにくく、4コーナーからの加速戦になりやすい',
            '💪 阪神ダートの中で最もスタミナが必要。底力がある馬が浮上',
            '🌧️ 重馬場になるとパワー型がさらに有利。時計のかかる消耗戦に',
            '👀 前走1200mや1400mで早め先頭に立って失速した馬は距離延長でむしろ良化することも',
        ],
        'blood': '🧬 シニスターミニスター・ホッコータルマエ・ゴールドアリュール系など持久力型ダート血統が主役。欧州系のスタミナ血統を持つ馬も侮れない。父×母父でダート中距離の実績が双方ある馬を最重視。',
        'color': '#e67e22',
    },
    ('ダ', '2000'): {
        'title': '阪神ダート2000m',
        'shape': '内回りコースを使用。スタートから最初のコーナーまで短く、道中に緩急が生まれやすい長距離ダート戦。',
        'points': [
            '🏋️ 阪神ダート最長距離。純粋なスタミナと底力が問われる消耗戦',
            '🎰 内回りのためコーナーで包まれないポジション取りが重要',
            '📊 前半は淡々と流れ、ロングスパート戦になりやすい。瞬発力より持続力',
            '💪 重い馬場でのパフォーマンスを過去走で確認。渋馬場の経験値が問われる',
            '👀 ダート長距離は頭数が少なく実績馬が有利。格上げ初戦の馬は割引',
        ],
        'blood': '🧬 ゴールドアリュール系・クロフネ系など長距離ダートで実績のある血統。欧州スタミナ血統（ガリレオ・サドラーズウェルズ系）を持つ馬もしぶとく走れる。父がダート中距離G1勝ち馬かどうかが最重要チェックポイント。',
        'color': '#e67e22',
    },
    ('芝', '1600'): {
        'title': '阪神芝1600m（外回り）',
        'shape': '2コーナー奥のポケットからスタートし、外回り3・4コーナーを経て国内屈指の長い直線473mへ。高低差3.1m、最後の急坂がある。',
        'points': [
            '🎯 日本最長クラスの直線で差し・追い込みが猛威を振るう名コース',
            '⚖️ 枠の有利不利が少なく純粋な能力差が出やすい。人気通りに決まりやすい',
            '⚡ 上り3Fが勝負の鍵。末脚の絶対値が高い馬を重視',
            '📊 桜花賞・NHKマイルCが行われる最高峰の舞台。高速決着になりやすい',
            '👀 前走で追い込んで届かなかった馬も阪神外回りなら直線の長さで浮上できる',
            '⛰️ 残り200mの急坂でバテる馬続出。最後まで脚が続く持続型末脚が理想',
        ],
        'blood': '🧬 ディープインパクト・キズナ・シルバーステート系の瞬発力血統が主役。外回りマイルはとにかく末脚の切れが問われるため、サンデーサイレンス系の血が色濃い馬が有利。ノーザンダンサー系でも末脚タイプなら十分。逆にロベルト系など先行粘り込みタイプは急坂で止まりやすい。',
        'color': '#27ae60',
    },
    ('芝', '2000'): {
        'title': '阪神芝2000m（内回り）',
        'shape': 'スタンド前からスタートし、内回り1〜4コーナーを経て直線356mへ。コーナーが4つある小回りコースで機動力が問われる。',
        'points': [
            '🥇 内回り小回りコースで先行・好位が断然有利。差しが届きにくい',
            '🎰 内枠から好位を取れるかが最重要。外枠の差し馬は厳しい展開になりがち',
            '💪 大阪杯・宝塚記念も行われる重賞コース。スタミナと機動力の融合が必要',
            '📊 ペースが上がりやすく消耗戦になることも。道中のポジション取りが命',
            '👀 前走外回りコース(東京・阪神外回り)で追い込んだ馬は内回りで評価下げ。逆に中山経験がある馬は内回りコーナリングが上手いケースも',
        ],
        'blood': '🧬 ロベルト系（エピファネイア・スクリーンヒーロー系）・ステイゴールド系のタフなスタミナ血統が内回りでは台頭しやすい。コーナリングの巧さが遺伝されやすくエンジンのかかりが早いタイプを重視。ディープ系でも先行できる機動力型なら問題なし。',
        'color': '#27ae60',
    },
}

def get_course_info(surf, dist):
    surf_key = '芝' if '芝' in str(surf) else 'ダ'
    dist_key = str(dist).replace('m','').strip()
    return COURSE_INFO.get((surf_key, dist_key), None)

# ═══════════════════════════════════���══════════
# 種牡馬特性データベース
# ══════════════════════════════════════════════
SIRE_DB = {
    # ── ダート系 ──
    'ゴールドアリュール':  {'type':'ダ', 'dist':'中', 'comment':'ダートの帝王。産駒はコパノリッキー・エスポワールシチーなど歴代ダート王を輩出。中距離ダートで特に輝く。', 'icon':'💎'},
    'ホッコータルマエ':    {'type':'ダ', 'dist':'中長', 'comment':'ダートチャンピオン。産駒はスタミナ豊富でダート長距離〜中距離向き。馬力型で渋い馬場も苦にしない。', 'icon':'🏋️'},
    'エスポワールシチー':  {'type':'ダ', 'dist':'中', 'comment':'ゴールドアリュール直子でダート実績豊富。産駒もダート適性高く粘り強いタイプ多い。', 'icon':'⚡'},
    'シニスターミニスター':{'type':'ダ', 'dist':'中長', 'comment':'ダート中長距離に強い米国系。パワーとスタミナを兼ね備えた産駒が多く、時計のかかる馬場で真価発揮。', 'icon':'🦾'},
    'カフジテイク':        {'type':'ダ', 'dist':'短中', 'comment':'ダート短〜中距離で活躍。スピード型の産駒が多く先行して粘る競馬が得意。', 'icon':'⚡'},
    'ドレフォン':          {'type':'ダ芝', 'dist':'短中', 'comment':'BCスプリント勝ち馬。産駒はスピード豊富でダート短距離に高適性。芝もこなせる万能型多い。', 'icon':'🚀'},
    'パイロ':              {'type':'ダ', 'dist':'短中', 'comment':'ダートスプリンターとして実績。産駒は先行型スピード馬が多く、砂のスプリント戦で本領発揮。', 'icon':'🔥'},
    'ヘニーヒューズ':      {'type':'ダ', 'dist':'短中', 'comment':'ダートで圧倒的実績を持つ米国産。産駒はスピード×パワーの砂巧者が多く、ダート短〜中距離は鉄板。', 'icon':'💪'},
    'ダノンスマッシュ':    {'type':'芝ダ', 'dist':'短', 'comment':'スプリントG1馬。産駒はスピード型が多く、芝・ダート問わず短距離で活躍が期待できる。', 'icon':'⚡'},
    'マクフィ':            {'type':'ダ', 'dist':'短中', 'comment':'欧州マイルG1馬。産駒はダート適性が高く、先行してしぶとく粘るタイプが多い。', 'icon':'🎯'},
    'ミッキーアイル':      {'type':'芝', 'dist':'短マイル', 'comment':'NHKマイルC勝ち馬。産駒はスプリント〜マイルに適性。芝の短距離戦で末脚を活かすタイプが多い。', 'icon':'🏃'},
    'クロベール':          {'type':'ダ', 'dist':'中', 'comment':'ダート中距離で活躍した種牡馬。産駒はスタミナと粘り強さを持ち、道悪でも力を発揮する。', 'icon':'💪'},

    # ── 芝系・万能系 ──
    'ディープインパクト':  {'type':'芝', 'dist':'マイル長', 'comment':'史上最高の種牡馬。産駒は瞬発力の塊で、特に外回り芝・良馬場の末脚勝負で圧倒的強さを誇る。', 'icon':'👑'},
    'キズナ':              {'type':'芝', 'dist':'マイル中', 'comment':'ディープインパクト直子。産駒は芝マイル〜中距離で高適性。上り3F勝負に強く外回り向き。', 'icon':'💫'},
    'エピファネイア':      {'type':'芝', 'dist':'中長', 'comment':'菊花賞・JC馬。産駒はスタミナ豊富で芝中距離からクラシック路線が主戦場。タフな競馬も苦にしない。', 'icon':'👑'},
    'ハービンジャー':      {'type':'芝', 'dist':'中長', 'comment':'欧州最強クラスの競走馬。産駒はスタミナ型で芝2000m前後に高適性。重い馬場でも走れる欧州色が強い。', 'icon':'🏆'},
    'サートゥルナーリア':  {'type':'芝', 'dist':'マイル中', 'comment':'皐月賞・ホープフルS馬。産駒はスピードとスタミナのバランスが良く、芝マイル〜中距離向き。', 'icon':'⭐'},
    'キタサンブラック':    {'type':'芝', 'dist':'中長', 'comment':'三冠馬。産駒は先行力とスタミナを兼備。内回りコースでの先行競馬で真価を発揮するタイプ多い。', 'icon':'🥇'},
    'ダノンプレミアム':    {'type':'芝', 'dist':'マイル', 'comment':'マイルG1馬。産駒は切れる末脚を持つマイラー傾向。芝1600m外回りで輝くタイプが多い。', 'icon':'💎'},
    'シルバーステート':    {'type':'芝', 'dist':'マイル中', 'comment':'ディープ直子で種牡馬としても大成。産駒は芝マイル〜中距離で高適性、外回り向きの末脚タイプ。', 'icon':'🌟'},
    'モーリス':            {'type':'芝', 'dist':'マイル中', 'comment':'G1を6勝した名馬。産駒は芝マイル〜中距離向きで、持続する末脚を持つ。内外問わず活躍。', 'icon':'👑'},
    'サトノクラウン':      {'type':'芝', 'dist':'中長', 'comment':'宝塚記念・香港ヴァーズ馬。産駒は芝中距離向きで独特のしぶとさを持つ。重馬場での粘りが光る。', 'icon':'👑'},
    'ジャスタウェイ':      {'type':'芝', 'dist':'マイル中', 'comment':'天皇賞秋・ドバイDF馬。産駒は芝マイル〜中距離。ロングスパートが得意で末脚の持続力が武器。', 'icon':'💫'},
    'ロードアスコット':    {'type':'芝', 'dist':'中', 'comment':'芝中距離で活躍。産駒はスタミナ型でコーナリングが上手く、内回りコースに高適性を示す。', 'icon':'⭐'},
    'ドゥラメンテ':        {'type':'芝', 'dist':'中長', 'comment':'クラシック二冠馬。産駒は芝中距離向きでクラシック路線に多数送り込む。スタミナと瞬発力を兼備。', 'icon':'👑'},
    'ライデオン':          {'type':'芝', 'dist':'短マイル', 'comment':'芝短距離〜マイルで活躍した種牡馬。産駒はスピード型が多く、早熟傾向あり。', 'icon':'⚡'},
    'ホッコーサラスター':  {'type':'芝ダ', 'dist':'マイル中', 'comment':'芝・ダート両用タイプの産駒が多い。適応力が高く、コース・馬場を問わず安定したパフォーマンス。', 'icon':'🎯'},
    'スローターネック':    {'type':'ダ', 'dist':'中', 'comment':'ダート中距離で活躍。産駒は粘り強いスタミナ型が多く、重い馬場でも崩れにくい。', 'icon':'💪'},
    'ダノンスプレンダー':  {'type':'芝ダ', 'dist':'短中', 'comment':'万能型の血統傾向。産駒は芝・ダート問わず適応力が高く、先行して粘る競馬を得意とする。', 'icon':'🎯'},
    'フランケル':          {'type':'芝', 'dist':'マイル中', 'comment':'無敗で競走生活を終えた欧州最強馬。産駒は芝マイル〜中距離向きで、高い能力を持つ馬が多い。', 'icon':'🌟'},
    'サトノダイヤモンド':  {'type':'芝', 'dist':'中長', 'comment':'有馬記念・菊花賞馬。産駒はスタミナ型で芝中長距離に高適性。ディープの父系を受け継ぐ。', 'icon':'💎'},
    'タリスマニック':      {'type':'芝ダ', 'dist':'中長', 'comment':'BCターフ馬。欧州系のスタミナ血統で産駒はタフな競馬が得意。芝・ダート両用タイプも出る。', 'icon':'🏆'},
    'ミッキーロケット':    {'type':'芝', 'dist':'中長', 'comment':'宝塚記念馬。産駒は芝中距離向きで先行力があり、持続する末脚を持つ。', 'icon':'🚀'},
    'オルフェーヴル':      {'type':'芝', 'dist':'中長', 'comment':'三冠馬。産駒は芝中長距離で圧倒的な末脚を発揮する。斬れ味鋭く、直線勝負で真価を見せる。', 'icon':'⚡'},
    'ルーラーシップ':      {'type':'芝', 'dist':'中長', 'comment':'クイーンエリザベス2世C馬。産駒は芝中長距離向きで先行してスタミナを活かす競馬が得意。', 'icon':'👑'},
    'スワーヴリチャード':  {'type':'芝', 'dist':'中', 'comment':'大阪杯馬。産駒は芝中距離向きでスピードとスタミナのバランスが良い。阪神芝との相性も◎。', 'icon':'⭐'},
    'ダービーフィズ':      {'type':'芝', 'dist':'中', 'comment':'芝中距離で活躍した種牡馬。産駒はスタミナ型でコーナリングが巧み。', 'icon':'🎯'},
    'イスラボニータ':      {'type':'芝', 'dist':'マイル中', 'comment':'皐月賞馬。産駒はフットワークが軽く芝マイル〜中距離向き。速いペースにも対応できる。', 'icon':'💫'},
    'アドマイヤマーズ':    {'type':'芝', 'dist':'短マイル', 'comment':'NHKマイルC・香港マイル馬。産駒はスプリント〜マイル向きの切れ者が多い。', 'icon':'⚡'},
    'リアルスティール':    {'type':'芝', 'dist':'マイル中', 'comment':'ディープ直子でUAEダービー馬。産駒は芝マイル〜中距離向きで末脚が武器。', 'icon':'💫'},
    'スターズオンアース': {'type':'芝', 'dist':'マイル中', 'comment':'牝馬クラシック二冠馬。産駒データは少ないが血統的に芝マイル〜中距離の適性が高い。', 'icon':'⭐'},
    'ブリックスアンドモルタル': {'type':'芝', 'dist':'中', 'comment':'BCターフ馬。産駒は芝中距離向きで安定したパフォーマンスを示す。外回りコースが得意。', 'icon':'🏆'},
    'マインドユアビスケッツ': {'type':'ダ', 'dist':'短中', 'comment':'BCダートスプリント馬。産駒はダート短〜中距離向きのスピード型が多い。', 'icon':'⚡'},
}

def get_sire_comment(sire_name):
    if pd.isna(sire_name): return None
    sire = str(sire_name).strip()
    return SIRE_DB.get(sire, None)

# ═════════════��════════════��═══════════════════
# ヘルパー関数
# ═══════════════════════════════════���══════════
def fmt_odds(v):
    try:
        f = float(v)
        if f > 0: return f'{f:.1f}'
    except: pass
    return '-'

def mark_bg(m):
    d = {'◎':'#c0392b','〇':'#27ae60','▲':'#2471a3','△':'#95a5a6','×':'#bbb','注':'#e67e22'}
    return d.get(m, 'transparent')

def diff_bar(v, max_v=30):
    if pd.isna(v): return '<span style="color:#555">-</span>'
    v = float(v)
    w = min(abs(v) / max_v * 100, 100)
    color = '#27ae60' if v >= 0 else '#e74c3c'
    sign = '+' if v >= 0 else ''
    return f'<div style="display:flex;align-items:center;gap:4px"><div style="width:{w:.0f}px;height:8px;background:{color};border-radius:3px;min-width:2px;max-width:80px"></div><span style="font-size:11px;color:{color};font-weight:bold">{sign}{v:.1f}</span></div>'

def ai_comment(grp):
    if 'cur_偏差値の差' not in grp.columns: return ''
    double = grp[
        (grp['cur_偏差値の差'].fillna(-99) >= 15) &
        (grp['sub_偏差値の差'].fillna(-99) >= 15)
    ] if 'sub_偏差値の差' in grp.columns else pd.DataFrame()
    top1_cur = grp.nsmallest(1,'cur_ランカー順位')['馬名S'].iloc[0] if 'cur_ランカー順位' in grp.columns and len(grp)>0 else ''
    top1_sub = grp.nsmallest(1,'sub_ランカー順位')['馬名S'].iloc[0] if 'sub_ランカー順位' in grp.columns and len(grp)>0 else ''
    if len(double) > 0:
        names = '・'.join(double['馬名S'].tolist())
        return f'<span style="color:#e74c3c;font-weight:bold">🔥 {names} が両モデル高評価</span>'
    if top1_cur == top1_sub and top1_cur:
        return f'<span style="color:#27ae60;font-weight:bold">✓ {top1_cur} が距離・クラス両モデル1位</span>'
    parts = []
    if top1_cur: parts.append(f'距離1位: <b>{top1_cur}</b>')
    if top1_sub and top1_sub != top1_cur: parts.append(f'クラス1位: <b>{top1_sub}</b>')
    return ' / '.join(parts)

# ── レースデータ収集 ──
races_data = []
for r_num, grp in hans.sort_values('Ｒ').groupby('Ｒ'):
    r = int(r_num)
    name = grp['レース名'].iloc[0] if 'レース名' in grp.columns else f'{r}R'
    dist = str(grp['距離'].iloc[0]) if '距離' in grp.columns else ''
    cls  = grp['クラス'].iloc[0] if 'クラス' in grp.columns else ''
    surf = grp['芝・ダ'].iloc[0] if '芝・ダ' in grp.columns else ''
    t    = time_map.get(r, '')
    grp_s = grp.sort_values('馬番', na_position='last') if '馬番' in grp.columns else grp
    races_data.append((r, name, dist, cls, surf, t, grp_s))

# ── 激熱ピック ──
best_picks = []
for r, name, dist, cls, surf, t, grp in races_data:
    if 'cur_偏差値の差' not in grp.columns: continue
    double = grp[
        (grp['cur_偏差値の差'].fillna(-99) >= 15) &
        (grp['sub_偏差値の差'].fillna(-99) >= 15)
    ] if 'sub_偏差値の差' in grp.columns else pd.DataFrame()
    for _, h in double.iterrows():
        odds = fmt_odds(h.get('単勝オッズ', h.get('dc_単勝オッズ','')))
        best_picks.append((r, name, h['馬名S'], odds,
                           h.get('cur_偏差値の差', np.nan),
                           h.get('sub_偏差値の差', np.nan)))

picks_html = ''
if best_picks:
    rows = ''
    for r, rname, hname, odds, cd, sd in best_picks:
        rows += f'<tr><td style="font-weight:bold;color:#e8b400">{r}R</td><td style="text-align:left">{rname}</td><td style="font-weight:bold;font-size:14px;color:#fff">{hname}</td><td style="color:#f39c12;font-weight:bold">{odds}</td><td style="color:#2ecc71">+{cd:.1f}</td><td style="color:#3498db">+{sd:.1f}</td></tr>'
    picks_html = f'''<div class="picks-box">
      <div class="picks-title">🔥 本日の激熱ピック（両モデル偏差値差+15以上）</div>
      <table class="picks-table"><thead><tr><th>R</th><th>レース名</th><th>馬名</th><th>オッズ</th><th>距離diff</th><th>クラスdiff</th></tr></thead><tbody>{rows}</tbody></table>
    </div>'''

# ═══════════════���══════════════════════════════
# レースカード生成
# ══════════════════════════════════════════════
race_cards = ''
for r, name, dist, cls, surf, t, grp in races_data:
    surf_label = '芝' if '芝' in str(surf) else 'ダ'
    surf_color = '#27ae60' if surf_label == '芝' else '#e67e22'
    course_info = get_course_info(surf, dist)

    # ── コース特性セクション ──
    if course_info:
        points_html = ''.join([f'<li>{p}</li>' for p in course_info['points']])
        course_section = f'''<div class="course-box">
          <div class="course-title" style="border-color:{course_info['color']}">
            📍 {course_info['title']} — コースの特徴
          </div>
          <div class="course-body">
            <div class="course-shape">🗺️ {course_info['shape']}</div>
            <ul class="course-points">{points_html}</ul>
            <div class="course-blood">🧬 血統傾向: {course_info['blood']}</div>
          </div>
        </div>'''
    else:
        course_section = ''

    blood_section = ''  # 血統情報は馬名セル内に統合

    # ── 馬一覧テーブル ──
    rows_html = ''
    for _, h in grp.iterrows():
        mark   = h.get('_印','') or ''
        try: banum = int(float(h['馬番'])) if pd.notna(h.get('馬番')) else '-'
        except: banum = '-'
        horse  = h.get('馬名S','')
        jockey = h.get('dc_騎手', h.get('騎手',''))
        odds   = fmt_odds(h.get('単勝オッズ', h.get('dc_単勝オッズ','')))
        cur_r  = int(h['cur_ランカー順位']) if 'cur_ランカー順位' in grp.columns and pd.notna(h.get('cur_ランカー順位')) else '-'
        sub_r  = int(h['sub_ランカー順位']) if 'sub_ランカー順位' in grp.columns and pd.notna(h.get('sub_ランカー順位')) else '-'
        cur_d  = h.get('cur_偏差値の差', np.nan)
        sub_d  = h.get('sub_偏差値の差', np.nan)
        # 1走前_着順_num を優先（fallback shift 修正済み）、なければ前走着順_num
        _z = h.get('1走前_着順_num') if pd.notna(h.get('1走前_着順_num', float('nan'))) else h.get('前走着順_num')
        zenso = int(_z) if _z is not None and pd.notna(_z) else '-'
        seibetsu = h.get('性齢', h.get('性', ''))
        kinryo = h.get('斤量','')
        try: kinryo = f"{float(kinryo):.1f}"
        except: pass
        sire_name = str(h.get('種牡馬','')).strip() if not pd.isna(h.get('種牡馬','')) else ''

        # 強さPT
        cur_score = h.get('cur_コース偏差値', np.nan)
        sub_score = h.get('sub_コース偏差値', np.nan)
        score_vals = [v for v in [cur_score, sub_score] if not pd.isna(v)]
        total_pt = np.mean(score_vals) if score_vals else np.nan

        mark_style = f'background:{mark_bg(mark)};color:white;font-weight:bold;font-size:14px' if mark else ''
        is_both = (not pd.isna(cur_d) and float(cur_d) >= 15) and (not pd.isna(sub_d) and float(sub_d) >= 15) if not (pd.isna(cur_d) or pd.isna(sub_d)) else False
        is_top  = ((not pd.isna(cur_d) and float(cur_d) >= 10) or (not pd.isna(sub_d) and float(sub_d) >= 10)) if not (pd.isna(cur_d) and pd.isna(sub_d)) else False
        row_class = 'row-both' if is_both else ('row-top' if is_top else '')

        cur_r_style = 'color:#e8b400;font-weight:bold' if cur_r != '-' and cur_r <= 3 else 'color:#8b949e'
        sub_r_style = 'color:#e8b400;font-weight:bold' if sub_r != '-' and sub_r <= 3 else 'color:#8b949e'

        if not pd.isna(total_pt):
            pt_color = '#e74c3c' if total_pt >= 65 else ('#f39c12' if total_pt >= 58 else ('#27ae60' if total_pt >= 52 else '#8b949e'))
            total_pt_html = f'<span style="font-weight:bold;font-size:13px;color:{pt_color}">{total_pt:.1f}</span>'
        else:
            total_pt_html = '<span style="color:#555">-</span>'

        # 血統情報を馬名セル内に統合
        sire_info = get_sire_comment(sire_name)
        bofuha_name = str(h.get('母父馬','')).strip() if not pd.isna(h.get('母父馬','')) else ''
        if sire_info:
            type_color = {'芝':'#27ae60','ダ':'#e67e22','芝ダ':'#3498db','万':'#9b59b6'}.get(sire_info['type'], '#888')
            bofuha_html = f' <span style="color:#666;font-size:10px">母父:{bofuha_name}</span>' if bofuha_name and bofuha_name != 'nan' else ''
            sire_cell = f'''<div style="font-size:10px;color:#777;margin-top:2px">
              {sire_info['icon']} <span style="color:#aaa">{sire_name}</span>{bofuha_html}
              <span style="background:{type_color}22;color:{type_color};border:1px solid {type_color}44;border-radius:4px;padding:0 4px;margin-left:3px;font-size:9px">{sire_info['type']}向き</span>
            </div>
            <div style="font-size:10.5px;color:#7a8a9a;margin-top:2px;line-height:1.4;padding-left:2px">{sire_info['comment']}</div>'''
        else:
            bofuha_html = f' <span style="color:#666;font-size:10px">母父:{bofuha_name}</span>' if bofuha_name and bofuha_name != 'nan' else ''
            sire_cell = f'<div style="font-size:10px;color:#555;margin-top:2px">🐴 {sire_name}{bofuha_html}</div>' if sire_name else ''

        rows_html += f'''<tr class="{row_class}">
          <td style="width:28px;{mark_style}">{mark}</td>
          <td style="width:32px;font-weight:bold;color:#8b949e">{banum}</td>
          <td style="text-align:left"><div style="font-weight:bold;font-size:13px">{horse}</div>{sire_cell}</td>
          <td style="color:#aaa;font-size:12px">{jockey}</td>
          <td style="color:#8b949e;font-size:11px">{seibetsu} {kinryo}kg</td>
          <td style="font-weight:bold;color:#f0a500">{odds}</td>
          <td>{total_pt_html}</td>
          <td style="{cur_r_style};font-size:12px">{cur_r}位</td>
          <td>{diff_bar(cur_d)}</td>
          <td style="{sub_r_style};font-size:12px">{sub_r}位</td>
          <td>{diff_bar(sub_d)}</td>
          <td style="color:#8b949e;font-size:12px">{zenso}</td>
        </tr>'''

    comment = ai_comment(grp)

    race_cards += f'''<div class="race-card" id="r{r}">
      <div class="race-header">
        <div class="race-number">{r}R</div>
        <div class="race-info">
          <div class="race-name">{name}</div>
          <div class="race-meta">
            <span class="surface" style="background:{surf_color}">{surf_label} {dist}m</span>
            <span class="cls-badge">{cls}</span>
            <span class="race-time">🕐 {t}</span>
          </div>
          <div class="race-comment">{comment}</div>
        </div>
      </div>
      {course_section}
      {blood_section}
      <div class="table-wrap">
        <table class="horse-table">
          <thead><tr>
            <th>印</th><th>馬番</th><th>馬名 / 父</th><th>騎手</th><th>性齢/斤量</th><th>オッズ</th>
            <th>強さPT<br><span style="font-size:9px;font-weight:normal">50=平均</span></th>
            <th>距離Rnk</th><th>距離diff</th><th>クラスRnk</th><th>クラスdiff</th>
            <th>前走着順</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
      </div>
    </div>'''

nav_items = ''.join([f'<a href="#r{r}" class="nav-item">{r}R</a>' for r,*_ in races_data])

html = f'''<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>阪神競馬 現地観戦レポート 2026/4/12</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Hiragino Sans','Yu Gothic',sans-serif;background:#0d1117;color:#e6edf3;font-size:13px;line-height:1.5}}
.top-bar{{background:linear-gradient(135deg,#1a1f3a 0%,#2d1b4e 100%);padding:20px 16px;border-bottom:4px solid #e8b400}}
.top-bar h1{{font-size:24px;color:#e8b400;letter-spacing:2px}}
.top-bar .date{{font-size:15px;color:#fff;margin-top:6px;font-weight:bold}}
.top-bar .subtitle{{color:#aaa;font-size:12px;margin-top:4px}}
.sticky-nav{{position:sticky;top:0;z-index:100;background:#161b22;border-bottom:1px solid #30363d;padding:8px 12px;display:flex;gap:6px;overflow-x:auto;flex-wrap:wrap}}
.nav-item{{padding:4px 12px;background:#21262d;border-radius:12px;color:#58a6ff;text-decoration:none;font-size:12px;font-weight:bold;white-space:nowrap}}
.nav-item:hover{{background:#30363d}}

.picks-box{{margin:16px;padding:14px 16px;background:linear-gradient(135deg,#1a0a0a,#2d1010);border:2px solid #c0392b;border-radius:10px}}
.picks-title{{font-size:16px;font-weight:bold;color:#e74c3c;margin-bottom:10px}}
.picks-table{{width:100%;border-collapse:collapse}}
.picks-table th{{background:#3d1010;color:#e74c3c;padding:6px 10px;text-align:center;font-size:12px}}
.picks-table td{{padding:7px 10px;text-align:center;border-bottom:1px solid #3d1010}}

.race-card{{margin:12px;background:#161b22;border-radius:10px;border:1px solid #30363d;overflow:hidden}}
.race-header{{padding:12px 14px;background:linear-gradient(90deg,#1f2937,#1a2035);display:flex;align-items:flex-start;gap:12px;flex-wrap:wrap}}
.race-number{{font-size:32px;font-weight:900;color:#e8b400;min-width:52px;line-height:1}}
.race-info{{flex:1;min-width:200px}}
.race-name{{font-size:17px;font-weight:bold;color:#fff}}
.race-meta{{display:flex;gap:8px;margin-top:5px;align-items:center;flex-wrap:wrap}}
.surface{{padding:3px 10px;border-radius:10px;color:white;font-weight:bold;font-size:12px}}
.cls-badge{{padding:3px 10px;background:#30363d;border-radius:10px;color:#aaa;font-size:12px}}
.race-time{{color:#58a6ff;font-size:12px}}
.race-comment{{font-size:12px;margin-top:6px;padding-top:6px;border-top:1px solid rgba(255,255,255,0.06)}}

/* コース特性 */
.course-box{{margin:0;padding:12px 16px;background:#0f1923;border-bottom:1px solid #1c2a3a}}
.course-title{{font-size:13px;font-weight:bold;color:#58a6ff;padding-bottom:6px;border-bottom:2px solid #1a5276;margin-bottom:8px}}
.course-body{{font-size:12px}}
.course-shape{{color:#aaa;margin-bottom:6px;padding-left:4px}}
.course-points{{padding-left:16px;color:#c8d8e8;margin-bottom:8px}}
.course-points li{{margin-bottom:3px}}
.course-blood{{color:#b8860b;background:rgba(184,134,11,0.1);padding:6px 10px;border-radius:6px;border-left:3px solid #b8860b;font-size:11.5px}}

/* 血統ワンポイント */
.blood-box{{margin:0;padding:12px 16px;background:#0d1a0d;border-bottom:1px solid #1a2e1a}}
.blood-title{{font-size:13px;font-weight:bold;color:#27ae60;margin-bottom:8px}}
.blood-items{{display:flex;flex-wrap:wrap;gap:8px}}
.sire-tip{{display:flex;gap:8px;align-items:flex-start;background:#0f1f0f;border:1px solid #1a3a1a;border-radius:8px;padding:8px 10px;flex:1;min-width:240px;max-width:400px}}
.sire-icon{{font-size:18px;flex-shrink:0;margin-top:1px}}
.sire-content{{flex:1}}
.sire-horse{{font-weight:bold;color:#e6edf3;font-size:13px}}
.sire-name{{font-size:11px;color:#888;margin-left:4px}}
.sire-type{{font-size:10px;padding:1px 6px;border-radius:8px;margin-left:4px;font-weight:bold}}
.sire-芝{{background:#1a4a1a;color:#27ae60}}
.sire-ダ{{background:#4a2a0a;color:#e67e22}}
.sire-万{{background:#1a1a4a;color:#58a6ff}}
.sire-comment{{font-size:11.5px;color:#aaa;margin-top:3px;line-height:1.4}}

.table-wrap{{overflow-x:auto}}
.horse-table{{width:100%;border-collapse:collapse;min-width:650px}}
.horse-table thead tr{{background:#21262d}}
.horse-table th{{padding:6px 8px;text-align:center;color:#8b949e;font-size:11px;border-bottom:1px solid #30363d;white-space:nowrap}}
.horse-table td{{padding:7px 8px;text-align:center;border-bottom:1px solid #1c2128;vertical-align:middle}}
.horse-table tr:last-child td{{border-bottom:none}}
.row-both{{background:rgba(192,57,43,0.18)!important}}
.row-top{{background:rgba(39,174,96,0.07)!important}}
.horse-table tr:hover{{background:rgba(88,166,255,0.05)}}
</style>
</head>
<body>
<div class="top-bar">
  <h1>🏇 阪神競馬 現地観戦レポート</h1>
  <div class="date">2026年4月12日（土）阪神競馬場 全12レース</div>
  <div class="subtitle">距離モデル × クラスモデル AIスコア完全版　コース特性・血統ワンポイント付き</div>
</div>
<nav class="sticky-nav">{nav_items}</nav>
{picks_html}
{race_cards}
<div style="text-align:center;padding:20px;color:#444;font-size:11px">競馬AI 2026-04-12 ／ 強さPT=コース偏差値（50=平均）／ diff=偏差値差</div>
</body>
</html>'''

os.makedirs('output', exist_ok=True)
out = 'output/hanshin_260412_report.html'
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'出力完了: {out}')

# -*- coding: utf-8 -*-
"""
JVLink リアルタイム単勝オッズ取得
使い方: python src/jvlink_rt_odds.py [場コード] [レース番号]
例:    python src/jvlink_rt_odds.py 09 11   (中京11R)
引数なしで全レースの最新オッズを表示
"""
import sys
import pythoncom
pythoncom.CoInitialize()
import win32com.client as wc

TARGET_JYO  = sys.argv[1] if len(sys.argv) > 1 else None  # 例: "09"
TARGET_RACE = sys.argv[2] if len(sys.argv) > 2 else None  # 例: "11"

jv = wc.gencache.EnsureDispatch("JVDTLab.JVLink")
rc_init = jv.JVInit("UNKNOWN")
print(f"JVInit: {rc_init}")
if rc_init != 0:
    print("JVInit失敗。終了します。")
    sys.exit(1)

# リアルタイムオッズ O1 = 単勝・複勝
rc = jv.JVRTOpen("O1", "")
print(f"JVRTOpen: rc={rc}")
if rc < 0:
    print("JVRTOpen失敗。終了します。")
    jv.JVClose()
    sys.exit(1)

buf  = " " * 110000
size = 110000
count = 0

# {(場コード, レース番号): {馬番: 単勝オッズ}}
odds_table = {}

while True:
    try:
        ret, data, sz, fname = jv.JVRead(buf, size, "")
    except Exception as e:
        print(f"JVRead例外: {e}")
        break

    if ret == 0:
        break          # EOF
    if ret == -1:
        continue       # ファイル切り替わり
    if ret == -3:
        continue       # データなし
    if ret < 0:
        print(f"JVReadエラー: {ret}")
        break

    # O1 レコードのパース
    # フォーマット: O1 YYYYMMDD JYO KAI NICHI RACE UMABAN TANSHO_ODDS FUKUSHO_MIN FUKUSHO_MAX ...
    # 固定長フォーマットに従って切り出し
    rec = data.strip()
    if not rec.startswith("O1"):
        count += 1
        continue

    try:
        # O1レコード固定長レイアウト（JVLink仕様書準拠）
        # 位置はすべて0始まり
        rec_type  = rec[0:2]
        kaisai_dt = rec[2:10]   # 開催年月日 YYYYMMDD
        jyo_cd    = rec[10:12]  # 場コード 2桁
        kai       = rec[12:13]  # 回次
        nichi     = rec[13:14]  # 日次
        race_no   = rec[14:16]  # レース番号
        umaban    = rec[16:18]  # 馬番
        tansho    = rec[18:23]  # 単勝オッズ (5桁, 例: "01850" = 18.5倍)

        if TARGET_JYO  and jyo_cd  != TARGET_JYO:
            count += 1
            continue
        if TARGET_RACE and race_no != TARGET_RACE.zfill(2):
            count += 1
            continue

        # オッズ変換: 5桁整数 → 実数 (末尾1桁が小数点以下)
        odds_val = int(tansho) / 10.0 if tansho.strip() else None

        key = (jyo_cd, race_no)
        if key not in odds_table:
            odds_table[key] = {}
        odds_table[key][int(umaban)] = odds_val

    except Exception as e:
        pass

    count += 1

jv.JVClose()
print(f"\n読み取り {count} レコード\n")

# 表示
JYO_NAME = {
    "01":"札幌","02":"函館","03":"福島","04":"新潟","05":"東京",
    "06":"中山","07":"中京","08":"京都","09":"阪神","10":"小倉",
}

if not odds_table:
    print("オッズデータなし（レース開催前またはデータ未配信）")
else:
    for (jyo, race), horses in sorted(odds_table.items()):
        name = JYO_NAME.get(jyo, jyo)
        print(f"=== {name} {int(race)}R ===")
        print(f"{'馬番':>4}  {'単勝オッズ':>8}")
        for uma in sorted(horses):
            o = horses[uma]
            o_str = f"{o:.1f}" if o else "---"
            print(f"{uma:>4}  {o_str:>8}")
        print()

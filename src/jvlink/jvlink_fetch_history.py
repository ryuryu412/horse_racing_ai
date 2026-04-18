# -*- coding: utf-8 -*-
"""
JVLink 過去レース結果一括取得
SE（競走成績）+ HR（払戻金）を取得してCSVに保存する。

使い方:
  python src/jvlink/jvlink_fetch_history.py
  python src/jvlink/jvlink_fetch_history.py --from 20250701  # 開始日指定
  python src/jvlink/jvlink_fetch_history.py --debug          # 生データ確認

出力:
  data/raw/results/jvlink_YYYYMM.csv  (月ごとに分割)

事前に jvlink_probe.py を実行してオフセットを確認してください。
"""
import sys, io, os, re, csv, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pythoncom
pythoncom.CoInitialize()
import win32com.client as wc
from collections import defaultdict

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── 場コード → 場所名 ──
JYO_NAME = {
    "01":"札幌","02":"函館","03":"福島","04":"新潟","05":"東京",
    "06":"中山","07":"中京","08":"京都","09":"阪神","10":"小倉",
}

# ── SE レコード パース ──
# jvlink_probe.py の出力で位置を確認してここを調整する
# JV-Link仕様書 Ver.4.x SE レコードレイアウト（推定値 — probe結果で要確認）
def parse_se(rec):
    """SE レコード → dict  (1頭分の成績)"""
    try:
        if len(rec) < 50: return None
        # ---- 以下オフセットは probe 結果で調整 ----
        kaisai_date = rec[11:19].strip()   # 開催年月日 YYYYMMDD
        jyo_cd      = rec[19:21].strip()   # 場コード
        kai         = rec[21:22].strip()   # 開催回次
        nichi       = rec[22:23].strip()   # 開催日次
        race_no     = rec[23:25].strip()   # レース番号
        umaban      = rec[25:27].strip()   # 馬番
        # 着順は複数候補 — probe で確認して正しい位置に変更
        chakujun    = rec[35:37].strip()   # 着順（数字 or '取' '中' '除' 等）
        # 馬名（全角固定長 or 可変長）— probe で確認
        horse_name  = rec[37:57].strip()   # 馬名（仮: 20バイト）
        # ----------------------------------------
        return {
            'kaisai_date': kaisai_date,
            'jyo_cd':      jyo_cd,
            'kai':         kai,
            'nichi':       nichi,
            'race_no':     race_no,
            'umaban':      umaban,
            'chakujun':    chakujun,
            'horse_name':  horse_name,
        }
    except Exception:
        return None

# ── HR レコード パース ──
# JV-Link仕様書 HR レコードレイアウト（推定値 — probe結果で要確認）
def parse_hr(rec):
    """HR レコード → dict  (1レース分の払戻金)"""
    try:
        if len(rec) < 100: return None
        # ---- 以下オフセットは probe 結果で調整 ----
        kaisai_date = rec[11:19].strip()
        jyo_cd      = rec[19:21].strip()
        kai         = rec[21:22].strip()
        nichi       = rec[22:23].strip()
        race_no     = rec[23:25].strip()
        # 払戻金ブロック（各7桁: 金額, 各2桁: 人気順）
        # 配当順: 単勝 → 複勝×3 → 枠連 → 馬連 → ワイド×3 → 馬単 → 三連複 → 三連単
        # 三連単は末尾付近 — probe で確認
        # 仮オフセット: 三連単 = rec[???:???]
        # ここは probe 結果後に正確な位置を入れる
        tan3_raw = _find_tan3(rec)   # 後述のヘルパーで取得
        # ----------------------------------------
        return {
            'kaisai_date': kaisai_date,
            'jyo_cd':      jyo_cd,
            'kai':         kai,
            'nichi':       nichi,
            'race_no':     race_no,
            'tan3':        tan3_raw,
        }
    except Exception:
        return None

def _find_tan3(rec):
    """
    HR レコードから三連単配当を取得する。
    probe 実行前の暫定実装 — 数字列の末尾付近から大きな値を探す。
    probe 結果が出たら固定オフセットに置き換える。
    """
    # 暫定: 末尾200バイト付近の最大7桁数字列を三連単と見なす
    # （三連単は HR の最後のフィールド）
    candidates = re.findall(r'\d{3,7}', rec[-200:])
    if candidates:
        # 金額として最大のものを選ぶ（三連単は高配当が多い）
        vals = [int(x) for x in candidates if int(x) > 0]
        if vals:
            return max(vals)
    return None

def race_key(d):
    return (d['kaisai_date'], d['jyo_cd'], d['kai'], d['nichi'], d['race_no'])

# ── 日付変換 ──
def kaisai_to_date_s(kaisai_date):
    """20250712 → 2025.7.12"""
    try:
        y = int(kaisai_date[:4])
        m = int(kaisai_date[4:6])
        d = int(kaisai_date[6:8])
        return f"{y}.{m}.{d}"
    except:
        return kaisai_date

# ── メイン ──
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='from_date', default='20250701',
                    help='取得開始日 YYYYMMDD (デフォルト: 20250701)')
    ap.add_argument('--debug', action='store_true',
                    help='生レコードを表示して終了（probe モード）')
    args = ap.parse_args()

    from_dt = args.from_date + "000000"

    jv = wc.gencache.EnsureDispatch("JVDTLab.JVLink")
    rc_init = jv.JVInit("UNKNOWN")
    print(f"JVInit: {rc_init}")
    if rc_init != 0:
        print("JVInit失敗。ターゲットFrontierが起動しているか確認してください。")
        sys.exit(1)

    rc, readcnt, dldcnt, lts = jv.JVOpen("RACE", from_dt, 1, 0, 0, "")
    print(f"JVOpen(RACE): rc={rc} readcnt={readcnt} dldcnt={dldcnt}")
    if rc != 0:
        print(f"JVOpenエラー rc={rc} — 月曜午前はJRA-VANメンテ中の場合があります")
        jv.JVClose()
        sys.exit(1)

    buf = " " * 110000
    size = 110000

    # debugモード: probe相当
    if args.debug:
        se_n = hr_n = 0
        while se_n < 2 or hr_n < 2:
            ret, data, sz, fname = jv.JVRead(buf, size, "")
            if ret == 0: break
            if ret < 0 and ret != -1 and ret != -3: break
            if ret <= 0: continue
            rt = data[:2].strip()
            if rt == "SE" and se_n < 2:
                print(f"\n[SE {se_n+1}] len={ret}")
                for i in range(0, min(ret, 200), 10):
                    print(f"  [{i:03d}:{i+10:03d}] {repr(data[i:i+10])}")
                se_n += 1
            if rt == "HR" and hr_n < 2:
                print(f"\n[HR {hr_n+1}] len={ret}")
                for i in range(0, min(ret, 400), 10):
                    print(f"  [{i:03d}:{i+10:03d}] {repr(data[i:i+10])}")
                hr_n += 1
        jv.JVClose()
        return

    # 本番: SE + HR を収集
    se_records = defaultdict(list)   # race_key → [se_dict, ...]
    hr_records  = {}                 # race_key → hr_dict
    n_se = n_hr = n_skip = 0

    print("読み込み中...")
    while True:
        try:
            ret, data, sz, fname = jv.JVRead(buf, size, "")
        except Exception as e:
            print(f"JVRead例外: {e}")
            break
        if ret == 0:
            print("EOF")
            break
        if ret == -1:
            continue  # ファイル切り替わり
        if ret == -3:
            import time; time.sleep(0.05)
            continue  # 一時的にデータなし
        if ret < 0:
            print(f"JVReadエラー: {ret}")
            break

        rt = data[:2].strip()

        if rt == "SE":
            d = parse_se(data)
            if d:
                se_records[race_key(d)].append(d)
                n_se += 1
        elif rt == "HR":
            d = parse_hr(data)
            if d:
                hr_records[race_key(d)] = d
                n_hr += 1
        else:
            n_skip += 1

    jv.JVClose()
    print(f"SE: {n_se}件 / HR: {n_hr}件 / その他: {n_skip}件")

    if n_se == 0:
        print("SEレコードが取れませんでした。--debug オプションで生データを確認してください。")
        sys.exit(1)

    # ── CSV出力 ──
    out_dir = os.path.join(base_dir, 'data', 'raw', 'results')
    os.makedirs(out_dir, exist_ok=True)

    # 月ごとに集計
    monthly = defaultdict(list)
    for rk, horses in se_records.items():
        kaisai_date = rk[0]  # YYYYMMDD
        ym = kaisai_date[:6]  # YYYYMM
        hr = hr_records.get(rk)
        tan3 = hr['tan3'] if hr else ''
        jyo_cd = rk[1]
        jyo_name = JYO_NAME.get(jyo_cd, jyo_cd)
        race_no = int(rk[4]) if rk[4].isdigit() else rk[4]
        date_s = kaisai_to_date_s(kaisai_date)

        for h in sorted(horses, key=lambda x: x.get('umaban','99')):
            monthly[ym].append({
                '日付S':    date_s,
                '場所':     jyo_name,
                'Ｒ':       race_no,
                '馬名S':    h['horse_name'],
                '着':       h['chakujun'],
                '馬番':     h['umaban'],
                '３連単配当': tan3,
            })

    total_rows = 0
    for ym, rows in sorted(monthly.items()):
        out_path = os.path.join(out_dir, f'jvlink_{ym}.csv')
        with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['日付S','場所','Ｒ','馬名S','着','馬番','３連単配当'])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  → {out_path}  ({len(rows)}行)")
        total_rows += len(rows)

    print(f"\n完了: 合計 {total_rows} 行")
    print("※ 着順・馬名・三連単配当のオフセットが合っているか最初の数行を目視確認してください。")
    print("  ズレている場合は jvlink_probe.py --debug の出力をClaudeに見せてオフセットを修正します。")

if __name__ == '__main__':
    main()

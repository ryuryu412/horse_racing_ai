# -*- coding: utf-8 -*-
"""
JVLink フィールド位置確認用プローブ
SE（成績）・HR（払戻金）レコードの生データを表示してバイト位置を確認する。
実行後、このスクリプトの出力をもとに jvlink_fetch_history.py のオフセットを調整する。

使い方:
  python src/jvlink/jvlink_probe.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pythoncom
pythoncom.CoInitialize()
import win32com.client as wc

jv = wc.gencache.EnsureDispatch("JVDTLab.JVLink")
rc_init = jv.JVInit("UNKNOWN")
print(f"JVInit: {rc_init}")
if rc_init != 0:
    print("JVInit失敗。ターゲットFrontierが起動しているか確認してください。")
    sys.exit(1)

# 直近3週分を取得（調整可）
FROM_DATE = "20260301000000"

rc, readcnt, dldcnt, lts = jv.JVOpen("RACE", FROM_DATE, 1, 0, 0, "")
print(f"JVOpen(RACE): rc={rc} readcnt={readcnt} dldcnt={dldcnt}")
if rc != 0:
    print(f"JVOpenエラー rc={rc}")
    jv.JVClose()
    sys.exit(1)

buf = " " * 110000
size = 110000

se_samples = []
hr_samples = []

while True:
    try:
        ret, data, sz, fname = jv.JVRead(buf, size, "")
    except Exception as e:
        print(f"JVRead例外: {e}")
        break
    if ret == 0:   break
    if ret == -1:  continue
    if ret == -3:  continue
    if ret < 0:
        print(f"JVReadエラー: {ret}")
        break

    rt = data[:2].strip()
    if rt == "SE" and len(se_samples) < 3:
        se_samples.append(data.rstrip())
    if rt == "HR" and len(hr_samples) < 3:
        hr_samples.append(data.rstrip())
    if len(se_samples) >= 3 and len(hr_samples) >= 3:
        break

jv.JVClose()

def dump_record(rec, label):
    print(f"\n{'='*60}")
    print(f"{label}  総長={len(rec)}")
    print(f"{'='*60}")
    print(f"生データ: {repr(rec[:300])}")
    print()
    # 10バイトごとに区切って位置を表示
    print("位置マップ（10バイト単位）:")
    for i in range(0, min(len(rec), 300), 10):
        chunk = rec[i:i+10]
        print(f"  [{i:03d}:{i+10:03d}] {repr(chunk)}")

print("\n" + "="*60)
print("SE レコード（競走成績・1頭分）サンプル")
print("="*60)
for i, r in enumerate(se_samples):
    dump_record(r, f"SE サンプル {i+1}")

print("\n" + "="*60)
print("HR レコード（払戻金）サンプル")
print("="*60)
for i, r in enumerate(hr_samples):
    dump_record(r, f"HR サンプル {i+1}")

print("\n完了。この出力をもとに jvlink_fetch_history.py のオフセットを調整してください。")

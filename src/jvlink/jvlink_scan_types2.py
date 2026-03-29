# -*- coding: utf-8 -*-
"""
SE（成績）スペックで取れるレコードタイプを確認 → 馬体重はここに入る
"""
import pythoncom
pythoncom.CoInitialize()
import win32com.client as wc
from collections import defaultdict

jv = wc.gencache.EnsureDispatch("JVDTLab.JVLink")
print(f"JVInit: {jv.JVInit('UNKNOWN')}")

# SE スペック（競走成績）で今日分
rc, readcnt, dldcnt, lts = jv.JVOpen("SE", "20260320000000", 1, 0, 0, "")
print(f"JVOpen(SE): rc={rc} readcnt={readcnt} dldcnt={dldcnt}")

if rc == 0:
    buf  = " " * 110000
    size = 110000
    samples = defaultdict(list)
    count = 0

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

        rec_type = data[:3].strip()
        count += 1
        if len(samples[rec_type]) < 1:
            samples[rec_type].append(data.strip())

    print(f"\n合計 {count} レコード")
    print("=== レコードタイプ ===")
    for t, recs in sorted(samples.items()):
        print(f"\n[{t}]")
        for r in recs:
            print(f"  {repr(r[:150])}")

jv.JVClose()

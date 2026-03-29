# -*- coding: utf-8 -*-
import pythoncom
pythoncom.CoInitialize()
import win32com.client as wc

jv = wc.gencache.EnsureDispatch("JVDTLab.JVLink")
print(f"JVInit: {jv.JVInit('UNKNOWN')}")

rc, readcnt, dldcnt, lts = jv.JVOpen("RACE", "20260320000000", 1, 0, 0, "")
print(f"JVOpen: rc={rc} readcnt={readcnt} dldcnt={dldcnt}\n")

buf = " " * 110000
size = 110000
records = []

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
        print("--- ファイル切り替わり ---")
        continue
    if ret == -3:
        import time; time.sleep(0.1)
        continue
    if ret < 0:
        print(f"JVReadエラー: {ret}")
        break

    records.append(data.strip())
    print(f"[{len(records)}] type={data[:3]} len={ret} file={fname}")
    print(f"    data={repr(data[:120])}")

jv.JVClose()
print(f"\n合計 {len(records)} レコード")

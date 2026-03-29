# -*- coding: utf-8 -*-
"""
JVLink から馬体重（H15レコード）を取得して表示
"""
import pythoncom
pythoncom.CoInitialize()
import win32com.client as wc

jv = wc.gencache.EnsureDispatch("JVDTLab.JVLink")
print(f"JVInit: {jv.JVInit('UNKNOWN')}")

rc, readcnt, dldcnt, lts = jv.JVOpen("RACE", "20260301000000", 1, 0, 0, "")
print(f"JVOpen(RACE): rc={rc} readcnt={readcnt} dldcnt={dldcnt}\n")

if rc != 0:
    jv.JVClose()
    exit()

buf   = " " * 110000
size  = 110000
h15_samples = []

while True:
    try:
        ret, data, sz, fname = jv.JVRead(buf, size, "")
    except Exception as e:
        print(f"例外: {e}")
        break
    if ret == 0:   break
    if ret == -1:  continue
    if ret == -3:  continue
    if ret < 0:
        print(f"エラー: {ret}")
        break

    if data[:3].strip() == "H15":
        h15_samples.append(data.strip())

jv.JVClose()

print(f"H15レコード {len(h15_samples)} 件\n")
for i, r in enumerate(h15_samples[:5]):
    print(f"[{i+1}] len={len(r)}")
    print(f"     {repr(r)}")
    print()

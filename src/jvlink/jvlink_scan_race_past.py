# -*- coding: utf-8 -*-
"""
過去の RACE データで全レコードタイプを確認（馬体重を探す）
"""
import pythoncom
pythoncom.CoInitialize()
import win32com.client as wc
from collections import defaultdict

jv = wc.gencache.EnsureDispatch("JVDTLab.JVLink")
print(f"JVInit: {jv.JVInit('UNKNOWN')}")

# いくつかのデータスペックを順に試す
for spec in ["RACE", "SE", "HR", "UM", "JG", "RA"]:
    rc, readcnt, dldcnt, lts = jv.JVOpen(spec, "20260301000000", 1, 0, 0, "")
    print(f"JVOpen({spec}): rc={rc} readcnt={readcnt} dldcnt={dldcnt}")
    if rc == 0 and readcnt > 0:
        # 少し読んでレコードタイプ確認
        buf  = " " * 110000
        size = 110000
        types = set()
        for _ in range(200):
            try:
                ret, data, sz, fname = jv.JVRead(buf, size, "")
            except:
                break
            if ret <= 0 and ret != -1 and ret != -3:
                break
            if ret > 0:
                types.add(data[:3].strip())
        print(f"  → レコードタイプ: {sorted(types)}")
    jv.JVClose()

print("\ndone")

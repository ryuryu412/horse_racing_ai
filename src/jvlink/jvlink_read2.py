# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pythoncom
pythoncom.CoInitialize()
import win32com.client as wc

jv = wc.Dispatch("JVDTLab.JVLink")
print(f"JVInit: {jv.JVInit('UNKNOWN')}", flush=True)

rc, readcnt, dldcnt, lts = jv.JVOpen("RACE", "20260320000000", 2, 0, 0, "")
print(f"JVOpen: rc={rc} readcnt={readcnt} dldcnt={dldcnt}", flush=True)

if rc == 0:
    import time
    for i in range(10):
        rc2, buf, filename = jv.JVRead()
        print(f"JVRead[{i}]: rc={rc2} file={filename!r} buflen={len(buf) if buf else 0}", flush=True)
        if rc2 < 0:
            print(f"終了: {rc2}", flush=True)
            break
        if rc2 == 0:
            time.sleep(0.1)

jv.JVClose()
print("done", flush=True)

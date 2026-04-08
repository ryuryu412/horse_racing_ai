# -*- coding: utf-8 -*-
import pythoncom
pythoncom.CoInitializeEx(pythoncom.COINIT_MULTITHREADED)
import win32com.client as wc
import traceback

log = open("G:/マイドライブ/horse_racing_ai/out2.txt", "w", encoding="utf-8")

def p(s):
    print(s)
    log.write(s + "\n")
    log.flush()

jv = wc.Dispatch("JVDTLab.JVLink")
p(f"JVInit: {jv.JVInit('UNKNOWN')}")

rc, readcnt, dldcnt, lts = jv.JVOpen("RACE", "20260320000000", 2, 0, 0, "")
p(f"JVOpen: rc={rc} readcnt={readcnt} dldcnt={dldcnt}")

if rc == 0:
    try:
        rc2, buf, filename = jv.JVRead()
        p(f"JVRead: rc={rc2} filename={repr(filename)} buflen={len(buf) if buf else 0}")
    except Exception as e:
        p(f"JVRead例外: {e}")
        p(traceback.format_exc())

jv.JVClose()
p("done")
log.close()

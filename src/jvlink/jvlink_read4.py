# -*- coding: utf-8 -*-
import pythoncom
pythoncom.CoInitialize()
import win32com.client as wc

# 早期バインディング（型ライブラリ使用）
try:
    jv = wc.gencache.EnsureDispatch("JVDTLab.JVLink")
    print("EnsureDispatch OK")
except Exception as e:
    print(f"EnsureDispatch失敗: {e} → Dispatch使用")
    jv = wc.Dispatch("JVDTLab.JVLink")

print(f"JVInit: {jv.JVInit('UNKNOWN')}")

# mode=1（通常取得）で試す
rc, readcnt, dldcnt, lts = jv.JVOpen("RACE", "20260320000000", 1, 0, 0, "")
print(f"JVOpen(mode=1): rc={rc} readcnt={readcnt} dldcnt={dldcnt}")

if rc == 0 and readcnt > 0:
    import time
    print("JVRead 呼び出し中...")
    # バッファサイズを明示的に渡す
    buf = " " * 110000
    size = 110000
    try:
        result = jv.JVRead(buf, size, "")
        print(f"JVRead result: {result}")
    except Exception as e:
        print(f"JVRead例外: {e}")
        import traceback; traceback.print_exc()

jv.JVClose()
print("done")

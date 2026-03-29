# -*- coding: utf-8 -*-
import win32com.client

SERVICE_KEY = "BUJC-42KD-07XW-0WNF-4"

jv = win32com.client.Dispatch('JVDTLab.JVLink')

ret = jv.JVInit("UNKNOWN")
print(f"JVInit: {ret}")

if ret == 0:
    ret2 = jv.JVSetServiceKey(SERVICE_KEY)
    print(f"JVSetServiceKey: {ret2}")
    if ret2 == 0:
        print("認証成功!")
        ret3 = jv.JVRTOpen("0B41", "")
        print(f"JVRTOpen('0B41', ''): {ret3}")
    else:
        print(f"エラー: {ret2}")

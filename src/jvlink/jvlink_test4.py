# -*- coding: utf-8 -*-
# JVSetServiceKey をスキップしてJVRTOpenを直接試す
import win32com.client

jv = win32com.client.Dispatch('JVDTLab.JVLink')

ret = jv.JVInit("UNKNOWN")
print(f"JVInit: {ret}")

if ret == 0:
    # キーはレジストリに既に保存済みのはず → スキップして直接オープン
    specs = ["0B41", "0B42", "0B51", "0B52"]
    for spec in specs:
        ret2 = jv.JVRTOpen(spec, "")
        print(f"JVRTOpen('{spec}'): {ret2}")
        if ret2 >= 0:
            print(f"  → 成功! spec={spec}")
            break
        jv.JVClose()

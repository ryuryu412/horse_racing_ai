# -*- coding: utf-8 -*-
# JV-Link設定ダイアログを表示する
import win32com.client

jv = win32com.client.Dispatch('JVDTLab.JVLink')

ret = jv.JVInit("UNKNOWN")
print(f"JVInit: {ret}")

if ret == 0:
    # GUI設定ダイアログを開く
    ret2 = jv.JVSetUIProperties()
    print(f"JVSetUIProperties: {ret2}")

    # ダイアログ後にもう一度テスト
    import winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\JRA-VAN Data Lab.\uid_pass") as k:
        sk_raw, _ = winreg.QueryValueEx(k, "servicekey")
    sk = sk_raw.strip()
    sk_h = f"{sk[0:4]}-{sk[4:8]}-{sk[8:12]}-{sk[12:16]}-{sk[16]}"
    ret3 = jv.JVSetServiceKey(sk_h)
    print(f"JVSetServiceKey after GUI: {ret3}")

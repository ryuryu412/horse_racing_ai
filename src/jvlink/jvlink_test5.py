# -*- coding: utf-8 -*-
import subprocess, time, winreg, win32com.client as wc

# 1. JV-Link設定ツール起動
print("=== 1. JV-Link設定ツール起動 ===")
subprocess.Popen([r"C:\Program Files (x86)\JRA-VAN\Data Lab\JV-Link設定.exe"])
time.sleep(3)

# 2. レジストリ確認
print("\n=== 2. レジストリ確認 ===")
try:
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                        r"SOFTWARE\JRA-VAN Data Lab.\uid_pass") as k:
        sk, _ = winreg.QueryValueEx(k, "servicekey")
        uk, _ = winreg.QueryValueEx(k, "ukey")
        print(f"servicekey: {sk}")
        print(f"ukey: {uk}")
except Exception as e:
    print(f"レジストリ読み取りエラー: {e}")

# 3. JVSetServiceKey なしで JVOpen テスト
print("\n=== 3. JVOpen テスト ===")
jv = wc.Dispatch("JVDTLab.JVLink")
print(f"JVInit: {jv.JVInit('UNKNOWN')}")
rc, readcnt, dldcnt, lts = jv.JVOpen("RACE", "20260101000000", 2, 0, 0, "")
print(f"JVOpen(RACE): rc={rc} readcnt={readcnt} dldcnt={dldcnt} lts={lts}")
jv.JVClose()

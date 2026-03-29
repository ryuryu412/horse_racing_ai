# -*- coding: utf-8 -*-
import win32com.client
import winreg

# レジストリからサービスキーを取得
with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\JRA-VAN Data Lab.\uid_pass") as k:
    servicekey_raw, _ = winreg.QueryValueEx(k, "servicekey")
    ukey, _ = winreg.QueryValueEx(k, "ukey")

# ハイフン付きに変換 (XXXX-XXXX-XXXX-XXXX-X)
sk = servicekey_raw.strip()
if '-' not in sk and len(sk) == 17:
    sk_with_hyphen = f"{sk[0:4]}-{sk[4:8]}-{sk[8:12]}-{sk[12:16]}-{sk[16]}"
else:
    sk_with_hyphen = sk

print(f"ServiceKey (raw): {servicekey_raw!r}")
print(f"ServiceKey (hyphen): {sk_with_hyphen!r}")
print(f"ukey: {ukey!r}")

jv = win32com.client.Dispatch('JVDTLab.JVLink')

ret = jv.JVInit("UNKNOWN")
print(f"\nJVInit('UNKNOWN'): {ret}")

if ret == 0:
    ret2 = jv.JVSetServiceKey(sk_with_hyphen)
    print(f"JVSetServiceKey: {ret2}")
    if ret2 == 0:
        print("認証成功!")
        # 試しにオープン
        ret3 = jv.JVRTOpen("0B41", "")
        print(f"JVRTOpen('0B41', ''): {ret3}")
    else:
        codes = {
            -100: "サービスキーエラー/認証失敗",
            -101: "サービスキー未設定",
            -111: "JVLinkAgent未起動",
        }
        print(f"  → {codes.get(ret2, f'不明エラー {ret2}')}")

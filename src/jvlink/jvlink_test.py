# -*- coding: utf-8 -*-
# JVLink test
import win32com.client

SERVICE_KEY = input("サービスキーを入力してください: ").strip()

jv = win32com.client.Dispatch('JVDTLab.JVLink')

# 初期化
ret = jv.JVInit(SERVICE_KEY)
print(f"JVInit: {ret}")

if ret == 0:
    print("接続OK")
    print(f"JVLinkバージョン: {jv.m_JVLinkVersion}")
else:
    codes = {
        -1: "サービスキーが未設定",
        -2: "認証エラー（サービスキーが間違い）",
        -111: "JVLinkが起動していない",
    }
    print(f"エラー: {codes.get(ret, f'コード {ret}')}")

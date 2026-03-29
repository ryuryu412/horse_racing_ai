# -*- coding: utf-8 -*-
import pythoncom
pythoncom.CoInitialize()
import win32com.client as wc

jv = wc.Dispatch("JVDTLab.JVLink")
print(f"JVInit: {jv.JVInit('UNKNOWN')}")

# 今日の出走情報を取得（前回取得時刻以降）
rc, readcnt, dldcnt, lts = jv.JVOpen("RACE", "20260320000000", 2, 0, 0, "")
print(f"JVOpen: rc={rc} readcnt={readcnt} dldcnt={dldcnt} lts={lts}")

if rc == 0:
    records = []
    count = 0
    print("JVRead 開始...")
    while True:
        rc2, buf, filename = jv.JVRead()
        print(f"  JVRead: rc={rc2} filename={filename!r} buflen={len(buf) if buf else 0}", flush=True)
        if rc2 < 0:
            print(f"JVRead終了: rc={rc2}")
            break
        if rc2 == 0:
            import time; time.sleep(0.5)
            continue
        if buf:
            try:
                text = buf.encode('latin-1').decode('cp932', errors='replace')
            except:
                text = str(buf)
            records.append(text)
            if len(records) % 100 == 0:
                print(f"  読み込み: {len(records)} レコード", flush=True)

    print(f"\n取得レコード数: {len(records)}")
    for r in records[:3]:
        print(repr(r[:100]))

jv.JVClose()

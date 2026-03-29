# -*- coding: utf-8 -*-
"""
今日のRACEデータで取れるレコードタイプを一覧表示
"""
import pythoncom
pythoncom.CoInitialize()
import win32com.client as wc
from collections import defaultdict

jv = wc.gencache.EnsureDispatch("JVDTLab.JVLink")
print(f"JVInit: {jv.JVInit('UNKNOWN')}")

# 今日分を取得
rc, readcnt, dldcnt, lts = jv.JVOpen("RACE", "20260321000000", 1, 0, 0, "")
print(f"JVOpen: rc={rc} readcnt={readcnt} dldcnt={dldcnt}\n")

buf  = " " * 110000
size = 110000
samples = defaultdict(list)  # type -> [data]

while True:
    try:
        ret, data, sz, fname = jv.JVRead(buf, size, "")
    except Exception as e:
        print(f"JVRead例外: {e}")
        break

    if ret == 0:   break
    if ret == -1:  continue
    if ret == -3:  continue
    if ret < 0:
        print(f"JVReadエラー: {ret}")
        break

    rec_type = data[:3].strip()
    if len(samples[rec_type]) < 2:
        samples[rec_type].append(data.strip())

print("=== 取得できたレコードタイプ ===")
for t, recs in sorted(samples.items()):
    print(f"\n[{t}]")
    for r in recs:
        print(f"  {repr(r[:120])}")

jv.JVClose()

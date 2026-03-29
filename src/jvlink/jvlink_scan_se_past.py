# -*- coding: utf-8 -*-
"""
先週の SE データでレコードタイプと馬体重の位置を確認
"""
import pythoncom
pythoncom.CoInitialize()
import win32com.client as wc
from collections import defaultdict

jv = wc.gencache.EnsureDispatch("JVDTLab.JVLink")
print(f"JVInit: {jv.JVInit('UNKNOWN')}")

# 先週分（3/15〜3/20）
rc, readcnt, dldcnt, lts = jv.JVOpen("SE", "20260315000000", 1, 0, 0, "")
print(f"JVOpen(SE): rc={rc} readcnt={readcnt} dldcnt={dldcnt}")

if rc != 0:
    jv.JVClose()
    exit()

buf   = " " * 110000
size  = 110000
samples = defaultdict(list)
count = 0

while True:
    try:
        ret, data, sz, fname = jv.JVRead(buf, size, "")
    except Exception as e:
        print(f"例外: {e}")
        break
    if ret == 0:   break
    if ret == -1:  continue
    if ret == -3:  continue
    if ret < 0:
        print(f"エラー: {ret}")
        break

    rec_type = data[:3].strip()
    count += 1
    if len(samples[rec_type]) < 2:
        samples[rec_type].append(data.strip())

    if count > 500:  # 十分なサンプルが取れたら打ち切り
        print("(500件で打ち切り)")
        break

jv.JVClose()
print(f"合計 {count} レコード\n")

for t, recs in sorted(samples.items()):
    print(f"[{t}]")
    for r in recs:
        print(f"  {repr(r[:200])}")
    print()

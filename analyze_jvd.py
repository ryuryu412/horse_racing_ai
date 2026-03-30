import sys, os, zlib
sys.stdout.reconfigure(encoding='utf-8')

# eventフォルダ確認
event_dir = r'C:\ProgramData\JRA-VAN\Data Lab\event'
print("=== event フォルダ ===")
try:
    for f in os.listdir(event_dir):
        fpath = os.path.join(event_dir, f)
        size = os.path.getsize(fpath)
        print(f"  {f} ({size} bytes)")
        if size < 1000:
            with open(fpath, 'rb') as fh:
                raw = fh.read()
            print(f"    hex: {raw.hex()}")
            try:
                print(f"    text: {raw.decode('cp932', errors='replace')}")
            except:
                pass
except Exception as e:
    print(f"error: {e}")

# SEDWファイルのHexDump
print("\n=== SEDW ファイルの構造解析 ===")
cache_dir = r'C:\ProgramData\JRA-VAN\Data Lab\cache'
target = os.path.join(cache_dir, 'SEDW2026032920260328112831.jvd')

with open(target, 'rb') as f:
    raw = f.read()

print(f"ファイルサイズ: {len(raw):,} bytes")
print(f"先頭30bytes hex: {raw[:30].hex()}")
print(f"先頭30bytes ascii: {raw[:30]}")

# 解凍
try:
    dec = zlib.decompress(raw[10:])
    print(f"解凍後サイズ: {len(dec):,} bytes")
    print(f"解凍先頭30bytes hex: {dec[:30].hex()}")
    print(f"解凍先頭100bytes: {dec[:100]}")

    # JV-Linkのレコードフォーマット確認
    # 固定長レコード。先頭2バイト=レコード種別
    print(f"\n解凍後先頭2bytes: {dec[:2]} = {dec[:2].decode('cp932', errors='replace')}")

    # 8バイト目から試す
    for offset in range(0, min(50, len(dec))):
        snippet = dec[offset:offset+2]
        try:
            txt = snippet.decode('cp932')
            if txt.isprintable():
                print(f"  offset {offset}: {txt}")
        except:
            pass

except Exception as e:
    print(f"decompress error: {e}")
    # rawデータ全体を解凍試み
    import struct
    print("\n先頭100bytes hex dump:")
    for i in range(0, min(100, len(raw)), 16):
        chunk = raw[i:i+16]
        hexpart = ' '.join(f'{b:02x}' for b in chunk)
        asciipart = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        print(f"  {i:04x}: {hexpart:<48}  {asciipart}")

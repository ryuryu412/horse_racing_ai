# -*- coding: utf-8 -*-
import winreg
import os

def check_reg(path, hive=winreg.HKEY_CLASSES_ROOT):
    try:
        with winreg.OpenKey(hive, path) as k:
            val, _ = winreg.QueryValueEx(k, "")
            return val
    except Exception as e:
        return f"ERROR: {e}"

# Find CLSID for JVDTLab.JVLink
clsid = check_reg(r"JVDTLab.JVLink\CLSID")
print(f"JVDTLab.JVLink CLSID: {clsid}")

# Check InprocServer32 under WOW6432Node
for root_path in [
    rf"WOW6432Node\CLSID\{clsid}\InprocServer32",
    rf"CLSID\{clsid}\InprocServer32",
]:
    val = check_reg(root_path)
    print(f"  HKCR\\{root_path}: {val}")

# Check JVLink version in registry
for hive, hive_name, path in [
    (winreg.HKEY_LOCAL_MACHINE, "HKLM", r"SOFTWARE\JRA-VAN Data Lab.\JV-Link"),
    (winreg.HKEY_LOCAL_MACHINE, "HKLM", r"SOFTWARE\WOW6432Node\JRA-VAN Data Lab.\JV-Link"),
]:
    try:
        with winreg.OpenKey(hive, path) as k:
            i = 0
            while True:
                try:
                    name, val, _ = winreg.EnumValue(k, i)
                    print(f"  {hive_name}\\{path}\\{name} = {val}")
                    i += 1
                except OSError:
                    break
    except Exception as e:
        print(f"  {hive_name}\\{path}: {e}")

# Check DLL file timestamp
dll_path = r"C:\Windows\SysWow64\JVDTLAB\JVDTLab.dll"
if os.path.exists(dll_path):
    stat = os.stat(dll_path)
    import datetime
    mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
    print(f"\nDLL: {dll_path}")
    print(f"  Modified: {mtime}")
    print(f"  Size: {stat.st_size:,} bytes")
else:
    print(f"\nDLL not found: {dll_path}")

# Check for JVDTLab.dll in Program Files
for search_dir in [r"C:\Program Files (x86)\JRA-VAN", r"C:\Program Files\JRA-VAN"]:
    for root, dirs, files in os.walk(search_dir):
        for f in files:
            if f.lower() == "jvdtlab.dll":
                full = os.path.join(root, f)
                stat = os.stat(full)
                import datetime
                mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
                print(f"\nFound: {full}")
                print(f"  Modified: {mtime}")
                print(f"  Size: {stat.st_size:,} bytes")

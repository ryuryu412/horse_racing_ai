# -*- coding: utf-8 -*-
import winreg

def list_subkeys(hive, path):
    try:
        with winreg.OpenKey(hive, path) as k:
            i = 0
            while True:
                try:
                    name = winreg.EnumKey(k, i)
                    print(f"  {name}")
                    i += 1
                except OSError:
                    break
    except Exception as e:
        print(f"  ERROR: {e}")

def list_values(hive, path):
    try:
        with winreg.OpenKey(hive, path) as k:
            i = 0
            while True:
                try:
                    name, val, _ = winreg.EnumValue(k, i)
                    print(f"  {name!r} = {val!r}")
                    i += 1
                except OSError:
                    break
    except Exception as e:
        print(f"  ERROR: {e}")

print("=== HKLM\\SOFTWARE\\JRA-VAN Data Lab. subkeys ===")
list_subkeys(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\JRA-VAN Data Lab.")

print("\n=== HKLM\\SOFTWARE\\WOW6432Node\\JRA-VAN Data Lab. subkeys ===")
list_subkeys(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\JRA-VAN Data Lab.")

print("\n=== HKLM\\SOFTWARE\\JRA-VAN Data Lab.\\ver values ===")
list_values(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\JRA-VAN Data Lab.")

print("\n=== HKLM\\SOFTWARE\\WOW6432Node\\JRA-VAN Data Lab. values ===")
list_values(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\JRA-VAN Data Lab.")

# Check installed programs
print("\n=== JRA-VAN in Uninstall ===")
for base in [r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
             r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"]:
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as parent:
            i = 0
            while True:
                try:
                    subkey = winreg.EnumKey(parent, i)
                    i += 1
                    try:
                        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base + '\\' + subkey) as k:
                            try:
                                name, _, _ = winreg.QueryValueEx(k, "DisplayName")
                                if "JRA" in str(name) or "JV" in str(name) or "Data Lab" in str(name):
                                    try:
                                        ver, _, _ = winreg.QueryValueEx(k, "DisplayVersion")
                                    except:
                                        ver = "?"
                                    print(f"  {name} v{ver}")
                            except:
                                pass
                    except:
                        pass
                except OSError:
                    break
    except Exception as e:
        print(f"  {base}: {e}")

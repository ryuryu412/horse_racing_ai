import winreg

clsid = '{2AB1774D-0C41-11D7-916F-0003479BEB3F}'

searches = [
    (winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\Classes\CLSID', winreg.KEY_READ),
    (winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\WOW6432Node\Classes\CLSID', winreg.KEY_READ),
    (winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\Classes\CLSID', winreg.KEY_READ | winreg.KEY_WOW64_32KEY),
    (winreg.HKEY_CLASSES_ROOT, r'CLSID', winreg.KEY_READ | winreg.KEY_WOW64_32KEY),
    (winreg.HKEY_CLASSES_ROOT, r'CLSID', winreg.KEY_READ | winreg.KEY_WOW64_64KEY),
]

for hive, base_path, flags in searches:
    full_path = base_path + '\\' + clsid
    hive_name = 'HKLM' if hive == winreg.HKEY_LOCAL_MACHINE else 'HKCR'
    try:
        key = winreg.OpenKey(hive, full_path, 0, flags)
        print(f'Found: {hive_name}\\{full_path}')
        i = 0
        while True:
            try:
                name = winreg.EnumKey(key, i)
                print(f'  [{name}]')
                try:
                    subkey = winreg.OpenKey(key, name)
                    val, _ = winreg.QueryValueEx(subkey, '')
                    print(f'    = {val}')
                except:
                    pass
                i += 1
            except OSError:
                break
    except Exception as e:
        print(f'Not found: {hive_name}\\{full_path}')

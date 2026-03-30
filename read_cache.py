import win32com.client, sys
sys.stdout.reconfigure(encoding='utf-8')

jv = win32com.client.Dispatch('JVDTLab.JVLink')
jv.JVInit('UNKNOWN')

# dataフォルダにあるファイルのプレフィックスから推測したスペック
# H1VM→H1、BNXM→BN、BRXM→BR、BTXM→BT、CHXM→CH、CSXM→CS
# 他にも試す
specs_to_try = [
    'H1  ', 'H6  ', 'O1  ', 'O2  ', 'O3  ', 'O4  ', 'O5  ', 'O6  ',
    'BN  ', 'BR  ', 'BT  ', 'CH  ', 'CS  ',
    'UM  ', 'SK  ', 'JG  ', 'WH  ', 'WE  ', 'AV  ', 'UM  ',
    'HN  ', 'TN  ', 'KS  ', 'HC  ', 'WC  ',
    'SE  ', 'RA  ',
]

print("スペック別JVOpen試行（fromtime=20260101, opt=1）:")
hits = []
for spec in specs_to_try:
    try:
        rc, readcnt, dldcnt, lts = jv.JVOpen(spec.strip(), '20260101000000', 1, 0, 0, '')
        if rc >= 0:
            print(f"  HIT! spec={spec!r} rc={rc} readcnt={readcnt} dldcnt={dldcnt}")
            hits.append(spec)
        jv.JVClose()
    except Exception as e:
        print(f"  {spec!r} 例外: {e}")

print(f"\n成功スペック: {hits}")

# 0B11 (単複速報) も試す
print("\n0B系試行:")
for spec in ['0B11', '0B12', '0B15', '0B13', '0B14']:
    try:
        rc, readcnt, dldcnt, lts = jv.JVOpen(spec, '20260101000000', 1, 0, 0, '')
        print(f"  spec={spec} rc={rc} readcnt={readcnt}")
        jv.JVClose()
    except Exception as e:
        print(f"  {spec} 例外: {e}")

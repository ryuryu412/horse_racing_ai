import win32com.client, sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

jv = win32com.client.Dispatch('JVDTLab.JVLink')
print('JVInit: %d' % jv.JVInit('UNKNOWN'))

# どのdataspecが使えるか総当たり
specs = ['RACE', 'BRFW', 'BNFW', 'SE  ', 'RA  ', '0B11', '0B12', '0B15']
for opt in [1, 2, 3, 4]:
    for spec in specs[:3]:  # まず代表3種類
        try:
            rc, readcnt, dldcnt, lts = jv.JVOpen(spec, '20260101000000', opt, 0, 0, '')
            if rc >= 0:
                print('HIT! spec=%-6s opt=%d rc=%d readcnt=%d dldcnt=%d' % (spec, opt, rc, readcnt, dldcnt))
            else:
                print('     spec=%-6s opt=%d rc=%d' % (spec, opt, rc))
            jv.JVClose()
        except Exception as e:
            print('spec=%-6s opt=%d 例外: %s' % (spec, opt, e))
print('done')

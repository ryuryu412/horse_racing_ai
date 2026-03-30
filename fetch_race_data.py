import win32com.client, sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

jv = win32com.client.Dispatch('JVDTLab.JVLink')
jv.JVInit('UNKNOWN')

# fromtimeを色々変えて試す
test_dates = [
    '20260329000000',  # 昨日
    '20260328000000',  # 一昨日
    '20260325000000',  # 先週
    '20260301000000',  # 3月頭
    '20260201000000',  # 2月頭
    '20260101000000',  # 1月頭
    '19860101000000',  # 全期間
]

for fromtime in test_dates:
    rc, readcnt, dldcnt, lts = jv.JVOpen('RACE', fromtime, 1, 0, 0, '')
    print(f'fromtime={fromtime}: rc={rc} readcnt={readcnt} dldcnt={dldcnt} lts={lts}')
    jv.JVClose()

print('done')

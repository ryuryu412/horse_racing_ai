import win32com.client, sys
sys.stdout.reconfigure(encoding='utf-8')

jv = win32com.client.Dispatch('JVDTLab.JVLink')
rc = jv.JVInit('UNKNOWN')
print(f"JVInit: {rc}")

# プロパティ確認
print(f"m_savepath:  {jv.m_savepath}")
print(f"m_saveflag:  {jv.m_saveflag}")
print(f"m_servicekey:{jv.m_servicekey}")
print(f"m_JVLinkVersion: {jv.m_JVLinkVersion}")
print(f"JVStatus: {jv.JVStatus()}")

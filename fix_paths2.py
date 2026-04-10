import os
import glob

old = r'G:\マイドライブ\horse_racing_ai'
old2 = 'G:/マイドライブ/horse_racing_ai'
new_bs = r'G:\マイドライブ\horse_racing_ai'
new_fs = 'G:/マイドライブ/horse_racing_ai'

base = r'G:\マイドライブ\horse_racing_ai'
patterns = ['**/*.py']
skip_dirs = {'.venv', '.venv32', '.git'}

fixed = []
for pattern in patterns:
    for fpath in glob.glob(os.path.join(base, pattern), recursive=True):
        # skip venv/git
        parts = fpath.replace('\\', '/').split('/')
        if any(s in skip_dirs for s in parts):
            continue
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            continue
        new_content = content.replace(old, new_bs).replace(old2, new_fs)
        if new_content != content:
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            fixed.append(fpath)

print(f'Fixed {len(fixed)} files:')
for f in fixed:
    print(' ', f)

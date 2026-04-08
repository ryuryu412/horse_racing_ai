files = [
    r'G:\マイドライブ\horse_racing_ai\src\_fuku_analysis.py',
    r'G:\マイドライブ\horse_racing_ai\src\_roi_new_tier.py',
    r'G:\マイドライブ\horse_racing_ai\src\_roi_new_tier2.py',
    r'G:\マイドライブ\horse_racing_ai\src\_roi_new_tier3.py',
    r'G:\マイドライブ\horse_racing_ai\src\_roi_optimize.py',
    r'G:\マイドライブ\horse_racing_ai\src\_roi_tier_optimize.py',
]
old = r'G:\マイドライブ\horse_racing_ai'
new = r'G:\マイドライブ\horse_racing_ai'

for f in files:
    with open(f, 'r', encoding='utf-8') as fh:
        content = fh.read()
    if old in content:
        content = content.replace(old, new)
        with open(f, 'w', encoding='utf-8') as fh:
            fh.write(content)
        print('Fixed:', f)
    else:
        print('No match:', f)

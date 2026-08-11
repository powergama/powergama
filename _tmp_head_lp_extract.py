import subprocess
text = subprocess.check_output(['git','show','HEAD:src/powergama/LpProblemPyomo.py'], text=True, encoding='utf-8')
lines = text.splitlines()
needles = [
    '_rt_target_gen_indices = {',
    'if int(i) in self._rt_target_gen_indices:',
    '_idx_be_peak_gas',
    'self.p_rt_deviation_price_gen_pos[i] = _dev_price_pos',
]
for needle in needles:
    for i,l in enumerate(lines, start=1):
        if needle in l:
            print(f"\\n=== {needle} at line {i} ===")
            for j in range(max(1,i-10), min(len(lines), i+30)+1):
                print(f"{j}: {lines[j-1]}")
            break

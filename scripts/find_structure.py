"""Поиск папки структуры по данным из trace CSV.

Заполни переменные в блоке INPUT ниже и запусти:
    python scripts/find_structure.py
"""
import json
import glob
import re
from pathlib import Path
from ase.data import atomic_numbers

# ============================================================
# INPUT — вставь сюда данные из строки trace CSV
# ============================================================

FORMULA  = "Pd4Sb8"          # формула из колонки formula
MILLER   = None               # "0_1_1" или None — искать по всем граням
OFFSET   = None               # число из колонки offset, или None
TOP      = None               # 0 или 1, или None

STRUCTURES_DIR = "data/structures"   # путь к папке структур
N_OFFSET_BINS  = 8                   # константа из pipeline.py

# ============================================================

def parse_formula(formula):
    """'Pd4Sb8' -> {46: 4, 51: 8}"""
    comp = {}
    for sym, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if sym and sym in atomic_numbers:
            z = atomic_numbers[sym]
            comp[z] = comp.get(z, 0) + int(n or 1)
    return comp

def offset_bin(offset):
    return int(float(offset) * N_OFFSET_BINS)

def dir_pattern(miller, offset, top):
    """Строит glob-паттерн для папки поверхности."""
    m = miller if miller else "*"
    if offset is not None:
        o = f"off{offset_bin(offset)}"
    else:
        o = "off*"
    if top is not None:
        s = "top" if int(top) else "bottom"
    else:
        s = "*"
    return f"{m}_{o}_{s}"

# --- поиск ---
target_comp = parse_formula(FORMULA)
pattern = f"{STRUCTURES_DIR}/*/{dir_pattern(MILLER, OFFSET, TOP)}/manifest.json"
print(f"ищу: {pattern}")
print(f"состав: {target_comp}")
print()

found = []
for mf in glob.glob(pattern):
    try:
        d = json.load(open(mf))
    except Exception:
        continue
    c = d.get("state", {}).get("composition", {})
    # JSON-ключи — строки, сравниваем через int()
    c_int = {int(k): v for k, v in c.items()}
    if c_int == target_comp:
        found.append((Path(mf).parent, d))

if not found:
    print("не найдено")
else:
    for path, d in found:
        state = d.get("state", {})
        desc  = d.get("descriptors", {})
        be    = d.get("BE", {})
        rw    = d.get("reward", {})
        print(f"ПАПКА: {path}")
        print(f"  грань   : {state.get('miller')}  offset={state.get('offset'):.4f}  top={state.get('top')}")
        print(f"  e_hull  : {d.get('e_hull', '?'):.4f} eV/atom")
        print(f"  Eact_CH : {desc.get('e_act_ch', '?'):.3f}  E_sel: {desc.get('e_sel', '?'):.3f}")
        print(f"  BE H={be.get('H','?'):.3f}  CH={be.get('CH','?'):.3f}  C3H7={be.get('C3H7','?'):.3f}")
        print(f"  reward  : stab={rw.get('r_stab','?'):.3f}  act={rw.get('r_act','?'):.3f}  sel={rw.get('r_sel','?'):.3f}  beta={rw.get('reward_beta','?'):.3e}")
        print(f"  файлы   : {[f.name for f in path.iterdir()]}")
        print()

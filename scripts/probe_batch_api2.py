"""Проба 2: паритет энергии/сил predict(батч) vs FAIRChemCalculator.

КРИТИЧНО для BE: последовательный путь берёт энергии через FAIRChemCalculator,
а батчевый будет брать через predictor.predict(батч). Если они в разной шкале
(референсы атомов применяются/не применяются) — BE поедут. Здесь проверяем, что
для тех же систем энергии и силы совпадают, и что батчинг не меняет per-system
результат (системы независимы).

Запуск:  python scripts/probe_batch_api2.py   — пришли вывод.
"""
import torch
from ase.build import fcc111, add_adsorbate

from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.datasets.atomic_data import (
    AtomicData, atomicdata_list_to_batch,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", device=DEVICE)
calc = FAIRChemCalculator(predictor, task_name="oc20")

# три системы разного размера
s1 = fcc111("Pd", size=(2, 2, 3), vacuum=10.0); s1.pbc = True
s2 = fcc111("Pd", size=(2, 2, 3), vacuum=10.0); add_adsorbate(s2, "H", 1.5, "ontop"); s2.pbc = True
s3 = fcc111("Pd", size=(3, 3, 3), vacuum=10.0); s3.pbc = True
systems = [s1, s2, s3]


def calc_energy_forces(atoms):
    a = atoms.copy(); a.calc = calc
    return float(a.get_potential_energy()), a.get_forces()


def predict_batch(atoms_list):
    datas = [AtomicData.from_ase(a, task_name="oc20") for a in atoms_list]
    batch = atomicdata_list_to_batch(datas).to(DEVICE)
    out = predictor.predict(batch)
    e = out["energy"].detach().cpu().numpy()
    f = out["forces"].detach().cpu().numpy()
    nat = batch.natoms.detach().cpu().tolist()
    # режем силы по системам
    forces, i = [], 0
    for n in nat:
        forces.append(f[i:i + n]); i += n
    return e, forces


print("=" * 64)
print("1. ПАРИТЕТ: калькулятор (по одной) vs predict (батч из 3)")
print("=" * 64)
e_calc = []
f_calc = []
for k, s in enumerate(systems):
    e, f = calc_energy_forces(s)
    e_calc.append(e); f_calc.append(f)
    print(f"  calc  sys{k}: E={e:.6f}  natoms={len(s)}")

e_pred, f_pred = predict_batch(systems)
import numpy as np
for k in range(len(systems)):
    de = abs(e_pred[k] - e_calc[k])
    df = float(np.abs(f_pred[k] - f_calc[k]).max())
    print(f"  pred  sys{k}: E={e_pred[k]:.6f}  | dE={de:.2e} eV  max|dF|={df:.2e} eV/A")

print("\n" + "=" * 64)
print("2. НЕЗАВИСИМОСТЬ: батч из 3 vs predict по одной")
print("=" * 64)
for k, s in enumerate(systems):
    e1, _ = predict_batch([s])
    print(f"  sys{k}: dE(батч-3 − одиночный)={abs(e_pred[k]-e1[0]):.2e} eV")

print("\nВЫВОД: если dE и max|dF| ~1e-4 и меньше — батчевый predict шкале")
print("калькулятора эквивалентен, BE считаем напрямую из out['energy'].")

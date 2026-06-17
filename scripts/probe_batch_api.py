"""Диагностика batch-API fairchem v2 на кластере.

ЗАЧЕМ: батч-релаксация адсорбатов (Tier 2) требует знать, КАК в установленной
версии fairchem:
  1) посчитать БАТЧ из N систем за один forward (per-system энергии + per-atom силы),
  2) есть ли готовая батчевая релаксация (ml_relax / OptimizableBatch).

Скрипт ничего не меняет — только пробует разные пути и печатает, что работает.
Запуск:  python scripts/probe_batch_api.py
Пришли весь вывод — по нему реализуем batch_relax под реальный API.
"""
import inspect
import sys
import traceback


def hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def try_(label, fn):
    """Выполнить пробу, напечатать OK/FAIL с краткой инфой."""
    try:
        res = fn()
        print(f"[OK]   {label}: {res}")
        return res
    except Exception as exc:
        print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
        return None


# --- 0. версии -----------------------------------------------------------
hr("0. ВЕРСИИ")
import torch
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
import fairchem.core as fc
print("fairchem.core:", getattr(fc, "__version__", "???"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)


# --- 1. предиктор + калькулятор (как в pdh_gfn/potential/uma.py) ----------
hr("1. ПРЕДИКТОР И КАЛЬКУЛЯТОР")
from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", device=DEVICE)
print("predict_unit type:", type(predictor))
print("predict_unit attrs:", [a for a in dir(predictor) if not a.startswith("_")])
calc = FAIRChemCalculator(predictor, task_name="oc20")
print("calculator type:", type(calc))
print("calculator attrs:", [a for a in dir(calc) if not a.startswith("_")])

# две маленькие тестовые системы (разного размера — проверим гетеро-батч)
from ase.build import fcc111, add_adsorbate
slab1 = fcc111("Pd", size=(2, 2, 3), vacuum=10.0)
slab2 = fcc111("Pd", size=(2, 2, 3), vacuum=10.0)
add_adsorbate(slab2, "H", height=1.5, position="ontop")
for s in (slab1, slab2):
    s.pbc = True
print(f"тест-системы: slab1={len(slab1)} ат., slab2={len(slab2)} ат.")

# single-point sanity через ASE-калькулятор
def _single():
    s = slab1.copy()
    s.calc = calc
    return f"E={s.get_potential_energy():.3f} eV, forces.shape={s.get_forces().shape}"
try_("single-point FAIRChemCalculator", _single)


# --- 2. как построить БАТЧ и сделать один forward ------------------------
hr("2. БАТЧЕВЫЙ FORWARD (per-system E + per-atom F)")

# 2a. ищем конструктор AtomicData
AtomicData = None
for path in [
    "fairchem.core.datasets.atomic_data",
    "fairchem.core.datasets",
    "fairchem.core.data.atomic_data",
]:
    try:
        mod = __import__(path, fromlist=["AtomicData"])
        AtomicData = getattr(mod, "AtomicData", None)
        if AtomicData is not None:
            print(f"[OK]   AtomicData из {path}")
            break
    except Exception as exc:
        print(f"[..]   нет AtomicData в {path}: {exc}")

# 2b. ищем коллатор списка -> батч
collater = None
for path, name in [
    ("fairchem.core.datasets.atomic_data", "atomicdata_list_to_batch"),
    ("fairchem.core.datasets.atomic_data", "data_list_collater"),
    ("fairchem.core.datasets", "data_list_collater"),
]:
    try:
        mod = __import__(path, fromlist=[name])
        collater = getattr(mod, name, None)
        if collater is not None:
            print(f"[OK]   коллатор {name} из {path}: {inspect.signature(collater)}")
            break
    except Exception as exc:
        print(f"[..]   нет {name} в {path}: {exc}")

if AtomicData is not None:
    print("AtomicData.from_ase signature:",
          try_("sig", lambda: str(inspect.signature(AtomicData.from_ase))))

def _batched_forward():
    """Собрать батч из 2 систем и прогнать predictor.predict."""
    if AtomicData is None:
        raise RuntimeError("нет AtomicData — нечем строить батч")
    # пробуем from_ase с task_name (oc20)
    try:
        d1 = AtomicData.from_ase(slab1, task_name="oc20")
        d2 = AtomicData.from_ase(slab2, task_name="oc20")
    except TypeError:
        d1 = AtomicData.from_ase(slab1)
        d2 = AtomicData.from_ase(slab2)
    # коллируем
    if collater is not None:
        try:
            batch = collater([d1, d2])
        except TypeError:
            batch = collater([d1, d2], otf_graph=True)
    else:
        from torch_geometric.data import Batch
        batch = Batch.from_data_list([d1, d2])
    batch = batch.to(DEVICE)
    out = predictor.predict(batch)
    keys = list(out.keys()) if hasattr(out, "keys") else type(out)
    info = [f"out keys/type: {keys}"]
    for k in ("energy", "forces"):
        if hasattr(out, "get") and out.get(k) is not None:
            info.append(f"{k}.shape={tuple(out[k].shape)}")
    # как разделить силы по системам
    for attr in ("natoms", "batch", "ptr"):
        if hasattr(batch, attr):
            info.append(f"batch.{attr}={getattr(batch, attr)}")
    return " | ".join(info)

try_("predictor.predict на батче из 2 систем", _batched_forward)


# --- 3. высокоуровневая батчевая релаксация ------------------------------
hr("3. ГОТОВАЯ БАТЧЕВАЯ РЕЛАКСАЦИЯ")
for path, names in [
    ("fairchem.core.common.relaxation", ["ml_relax"]),
    ("fairchem.core.common.relaxation.optimizable",
     ["OptimizableBatch", "OptimizableUnitCellBatch"]),
    ("fairchem.core.common.relaxation.ase_utils", ["OCPCalculator"]),
    ("fairchem.core.relaxation", ["ml_relax", "OptimizableBatch"]),
]:
    for name in names:
        def _imp(p=path, n=name):
            mod = __import__(p, fromlist=[n])
            obj = getattr(mod, n)
            try:
                return f"есть, signature={inspect.signature(obj)}"
            except (ValueError, TypeError):
                return "есть (класс/без сигнатуры)"
        try_(f"{path}.{name}", _imp)


# --- 4. подсказки для реализации ----------------------------------------
hr("4. ИТОГ")
print("""Что нужно из вывода выше для batch_relax:
  - раздел 2: рабочий способ построить батч (AtomicData.from_ase + коллатор)
    и форма out['energy'] (N систем) / out['forces'] (sum(natoms) атомов),
    плюс чем резать силы по системам (batch.natoms / batch.batch / ptr);
  - раздел 3: если ml_relax/OptimizableBatch доступны — берём их (маскирование
    сходимости из коробки), иначе hand-roll FIRE поверх батч-forward из п.2.
""")
print("Готово. Пришли весь вывод.")

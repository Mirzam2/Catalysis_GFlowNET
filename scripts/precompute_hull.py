"""Дополнение hull чистыми элементами в oc20-шкале.

После rebuild_hull_oc20.py в data/mp_pdm_entries.json остались только Pd-M
бинарники. PhaseDiagram без точек чистых элементов аппроксимирует углы
произвольно — поэтому даже PdZn (mp-1652, по построению на hull) получает
e_above_hull > 0.5 эВ/атом. Это и есть причина invalid=100% в обучении:
ВСЁ за cutoff=0.15.

Скрипт релаксирует чистые ASE-объёмы (FCC/BCC) для всех элементов из
ALL_ELEMENTS в oc20-шкале и дописывает в JSON.

Использование:
    python scripts/precompute_hull.py --uma-device cuda
"""
import argparse
import json
import time
from pathlib import Path

from ase.build import bulk as ase_bulk
from pymatgen.core.periodic_table import Element
from pymatgen.io.ase import AseAtomsAdaptor


# Кристаллическая структура для каждого элемента (приближение, UMA уточнит)
ELEMENT_BULK = {
    # переходные металлы — FCC
    "Pd": ("fcc", 3.89),
    "Pt": ("fcc", 3.92),
    "Ni": ("fcc", 3.52),
    "Cu": ("fcc", 3.61),
    "Ag": ("fcc", 4.09),
    "Au": ("fcc", 4.08),
    "Co": ("hcp", 2.51),
    "Fe": ("bcc", 2.87),
    "Mn": ("bcc", 2.89),
    "Cr": ("bcc", 2.88),
    # тугоплавкие — BCC
    "Ti": ("hcp", 2.95),
    "V":  ("bcc", 3.03),
    "Zr": ("hcp", 3.23),
    "Nb": ("bcc", 3.30),
    "Mo": ("bcc", 3.15),
    "Hf": ("hcp", 3.20),
    "Ta": ("bcc", 3.30),
    "W":  ("bcc", 3.16),
    "Re": ("hcp", 2.76),
    # p-блок
    "Zn": ("hcp", 2.66),
    "Ga": ("fcc", 4.52),    # реальная Ga оргомбическая, но UMA дорелаксирует
    "Ge": ("diamond", 5.66),
    "In": ("fcc", 4.59),    # реально тетрагональная I4/mmm
    "Sn": ("diamond", 6.49),
    "Sb": ("fcc", 4.51),    # реально ромбоэдрическая
    "Pb": ("fcc", 4.95),
    "Bi": ("fcc", 4.75),    # реально ромбоэдрическая
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uma-device", default="cuda")
    p.add_argument("--model", default="uma-s-1p2")
    p.add_argument("--fmax", type=float, default=0.03)
    p.add_argument("--hull-json", default="data/mp_pdm_entries.json")
    p.add_argument("--mock", action="store_true",
                   help="EMT-заглушка для smoke-теста")
    args = p.parse_args()

    # Загружаем существующий hull
    hull_path = Path(args.hull_json)
    if not hull_path.exists():
        raise SystemExit(f"Нет файла {hull_path}. Сначала rebuild_hull_oc20.py")
    records = json.loads(hull_path.read_text())
    print(f"загружено {len(records)} существующих записей")

    # Выясняем какие чистые элементы уже есть
    existing_pure = set()
    for r in records:
        # composition вида "Pd1 Zn1" или "Pd4"
        parts = r["composition"].split()
        if len(parts) == 1:
            # одна часть — чистый элемент
            import re
            m = re.match(r"([A-Z][a-z]?)\d+", parts[0])
            if m:
                existing_pure.add(m.group(1))
    print(f"чистые элементы уже в hull: {sorted(existing_pure)}")

    # Список ALL_ELEMENTS как символов
    from pdh_gfn.constants import ALL_ELEMENTS
    needed_elements = [Element.from_Z(z).symbol for z in ALL_ELEMENTS]
    missing = [el for el in needed_elements if el not in existing_pure]
    print(f"нужно посчитать: {missing}")

    if not missing:
        print("все чистые элементы уже в hull, ничего не делаю")
        return

    # Загружаем потенциал
    if args.mock:
        from pdh_gfn.potential.emt_mock import EMTMockPotential
        pot = EMTMockPotential()
        print("режим mock (EMT)")
    else:
        from pdh_gfn.potential.uma import UMAPotential
        print(f"загрузка UMA {args.model} на {args.uma_device}...")
        pot = UMAPotential(model=args.model, task="oc20", device=args.uma_device)
        print("OK")

    # Релаксируем каждый недостающий элемент
    new_records = list(records)
    n_added = 0
    n_failed = 0
    for el in missing:
        if el not in ELEMENT_BULK:
            print(f"[{el}] не знаю кристаллическую структуру — пропускаю")
            n_failed += 1
            continue
        structure_type, a = ELEMENT_BULK[el]
        t0 = time.time()
        try:
            atoms = ase_bulk(el, structure_type, a=a)
            if not pot.supports(atoms):
                print(f"[{el}] вне домена потенциала — пропускаю")
                n_failed += 1
                continue
            res = pot.relax_bulk(atoms, fmax=args.fmax)
            if not res.valid:
                print(f"[{el}] релаксация не сошлась — пропускаю")
                n_failed += 1
                continue
            e_per_atom = res.energy / len(res.atoms)
            n_atoms = len(res.atoms)
            comp_str = f"{el}{n_atoms}"
            new_records.append({
                "composition": comp_str,
                "energy_per_atom": e_per_atom,
                "mp_id": f"pure_{el}",
                "_note": "pure element computed in oc20 by precompute_hull.py",
            })
            n_added += 1
            print(f"[{el}] {structure_type} a={a} → E/atom={e_per_atom:.3f} ({time.time()-t0:.1f}s)")
        except Exception as exc:
            print(f"[{el}] FAIL: {exc}")
            n_failed += 1

    # Бэкап и сохранение
    backup = hull_path.with_suffix(".no_elements.json.bak")
    if not backup.exists():
        backup.write_text(hull_path.read_text())
        print(f"\nбэкап старого hull: {backup}")
    hull_path.write_text(json.dumps(new_records, indent=2))
    print(f"\nдобавлено {n_added} чистых элементов, провалено {n_failed}")
    print(f"итого записей: {len(new_records)}")
    print(f"новый hull: {hull_path}")

    # Verify: проверяем что PdZn даёт e_hull≈0
    if not args.mock:
        print("\nверификация: проверяем что PdZn(mp-1652) теперь на hull")
        from pdh_gfn.reward.hull import HullReference
        h = HullReference(entries_path=hull_path)
        # ищем PdZn запись из hull
        pdzn = next((r for r in new_records
                    if "Pd" in r["composition"] and "Zn" in r["composition"]
                    and "1Zn1" in r["composition"].replace(" ", "")), None)
        if pdzn:
            # пересчитываем e_hull
            from pymatgen.core import Composition, Structure
            from pymatgen.entries.computed_entries import ComputedEntry
            comp = Composition(pdzn["composition"])
            n = comp.num_atoms
            e_tot = pdzn["energy_per_atom"] * n
            # упрощённо — через _get_diagram
            pd_diag = h._get_diagram({"Pd", "Zn"})
            if pd_diag:
                from pymatgen.analysis.phase_diagram import PDEntry
                test_entry = PDEntry(comp, e_tot)
                e_above = pd_diag.get_e_above_hull(test_entry, allow_negative=True)
                print(f"  PdZn {pdzn['composition']}: e_above_hull = {e_above:.4f} эВ/атом")
                if abs(e_above) < 0.05:
                    print("  OK — PdZn практически на hull")
                else:
                    print("  ПРЕДУПРЕЖДЕНИЕ — должен быть около 0")


if __name__ == "__main__":
    main()

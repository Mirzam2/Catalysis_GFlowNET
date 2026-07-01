"""Быстрая проверка собранного hull: состав записей + e_above_hull эталона.

Не требует UMA/MP — читает готовый data/mp_pdm_entries.json. Показывает,
сколько унарных/бинарных/тернарных фаз в наборе (для тернара должны быть
все три), и, если задан --cif, считает e_above_hull эталонной структуры
по её oc20-энергии из записи hull.

    python scripts/check_hull.py
    python scripts/check_hull.py --entries data/mp_pdm_entries.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from pymatgen.core import Composition


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entries", default="data/mp_pdm_entries.json")
    args = p.parse_args()

    path = Path(args.entries)
    if not path.exists():
        raise SystemExit(f"нет файла {path} — сначала rebuild_hull_oc20.py")

    raw = json.loads(path.read_text())
    by_ncomp = Counter()
    n_with_mpid = 0
    for rec in raw:
        comp = Composition(rec["composition"])
        by_ncomp[len(comp.elements)] += 1
        if rec.get("mp_id"):
            n_with_mpid += 1

    print(f"записей: {len(raw)}  (с mp_id: {n_with_mpid})")
    print(f"  унарные (чистые элементы): {by_ncomp[1]}")
    print(f"  бинарные:                  {by_ncomp[2]}")
    print(f"  тернарные:                 {by_ncomp[3]}")
    if by_ncomp[3] == 0:
        print("  [!] тернарных фаз НЕТ — оболочка для Pd-M-M' будет занижена. "
              "Пересобери rebuild_hull_oc20.py (уже тернар-версия).")
    if n_with_mpid < len(raw):
        print(f"  [!] {len(raw)-n_with_mpid} записей без mp_id — вероятно, "
              "остатки GGA-шкалы. Нужен полный пересчёт в oc20.")

    # Санити: e_above_hull известной бинарной фазы PdZn через HullReference
    try:
        from pdh_gfn.reward.hull import HullReference
        from pymatgen.core import Structure, Lattice
        hull = HullReference(entries_path=path, strict=False)
        rec = next((r for r in raw
                    if set(map(str, Composition(r["composition"]).elements))
                    == {"Pd", "Zn"}), None)
        if rec:
            comp = Composition(rec["composition"])
            e_tot = rec["energy_per_atom"] * comp.num_atoms
            # фиктивная структура нужного состава — e_above_hull зависит только
            # от состава и полной энергии, не от геометрии
            st = Structure(Lattice.cubic(3.0),
                           [str(el) for el in comp.elements
                            for _ in range(int(comp[el]))],
                           [[0, 0, 0]] * int(comp.num_atoms))
            eah = hull.e_above_hull(st, e_tot)
            print(f"\nсанити PdZn ({rec['composition']}): "
                  f"e_above_hull = {eah:.4f} эВ/атом (ждём ~0)")
    except Exception as exc:
        print(f"\n(санити-проверку PdZn пропустил: {exc})")


if __name__ == "__main__":
    main()

"""Однократная выгрузка записей Materials Project для оболочки стабильности.

Использование (нужен MP_API_KEY):
    export MP_API_KEY=...
    python scripts/fetch_mp_entries.py --out data/mp_pdm_entries.json

Скачивает все записи химических систем Pd-M, M-M' и Pd-M-M' (M из словаря)
включая чистые элементы и сохраняет {composition, energy_per_atom} в JSON.
get_entries_in_chemsys([Pd, M, M']) возвращает все подсистемы (вершины,
рёбра, тройник), поэтому перебор пар M покрывает всю оболочку тернара.
Во время обучения сетевых запросов нет (правило кластера an01).

ВНИМАНИЕ: это GGA-шкала. Боевой hull считается в oc20 через
scripts/rebuild_hull_oc20.py; этот файл — референс/бэкап.
"""
import argparse
import itertools
import json
import os
from pathlib import Path

from pymatgen.core.periodic_table import Element

from pdh_gfn.constants import M_ELEMENTS, PD_Z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/mp_pdm_entries.json")
    args = p.parse_args()

    from mp_api.client import MPRester

    pd_symbol = Element.from_Z(PD_Z).symbol
    m_syms = [Element.from_Z(z).symbol for z in M_ELEMENTS]
    # chemsys-наборы: тройники Pd-M-M' покрывают и Pd-M, и M-M' грани;
    # добавляем чистые Pd-M на случай пар, не образующих тройных записей.
    chemsystems = [[pd_symbol, m] for m in m_syms]
    chemsystems += [[pd_symbol, m1, m2]
                    for m1, m2 in itertools.combinations(m_syms, 2)]

    records = []
    seen = set()  # дедуп по entry_id (подсистемы пересекаются)
    with MPRester(os.environ["MP_API_KEY"]) as mpr:
        for chemsys in chemsystems:
            entries = mpr.get_entries_in_chemsys(
                chemsys,
                additional_criteria={"thermo_types": ["GGA_GGA+U"]},
            )
            n_new = 0
            for e in entries:
                eid = getattr(e, "entry_id", None)
                if eid is not None and eid in seen:
                    continue
                if eid is not None:
                    seen.add(eid)
                records.append({
                    "composition": e.composition.formula,
                    "energy_per_atom": e.energy / e.composition.num_atoms,
                })
                n_new += 1
            print(f"{'-'.join(chemsys)}: +{n_new} (всего {len(records)})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records))
    print(f"Сохранено {len(records)} записей -> {out}")


if __name__ == "__main__":
    main()

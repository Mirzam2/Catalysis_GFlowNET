"""Однократная выгрузка записей Materials Project для оболочки стабильности.

Использование (нужен MP_API_KEY):
    export MP_API_KEY=...
    python scripts/fetch_mp_entries.py --out data/mp_pdm_entries.json

Скачивает все записи химических систем Pd-M (M из словаря) включая чистые
элементы и сохраняет {composition, energy_per_atom} в JSON. Во время
обучения сетевых запросов нет (правило кластера an01).
"""
import argparse
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

    records = []
    with MPRester(os.environ["MP_API_KEY"]) as mpr:
        pd_symbol = Element.from_Z(PD_Z).symbol
        for z in M_ELEMENTS:
            m_symbol = Element.from_Z(z).symbol
            entries = mpr.get_entries_in_chemsys(
                [pd_symbol, m_symbol],
                additional_criteria={"thermo_types": ["GGA_GGA+U"]},
            )
            for e in entries:
                records.append({
                    "composition": e.composition.formula,
                    "energy_per_atom": e.energy / e.composition.num_atoms,
                })
            print(f"{pd_symbol}-{m_symbol}: {len(entries)} записей")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records))
    print(f"Сохранено {len(records)} записей -> {out}")


if __name__ == "__main__":
    main()

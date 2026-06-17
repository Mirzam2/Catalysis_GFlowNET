"""Скачивает CIF-файлы референсных структур из Materials Project.

Использование:
    export MP_API_KEY=...
    python scripts/fetch_ref_structures.py

Что скачивает:
    - Конкретные mp-id (PdZn, Pd, GaPd, InPd)
    - Все структуры состава Pd3Sn (или любой другой формулы через --formula)

Файлы сохраняются в data/ref_structures/<name>.cif.
После скачивания печатает таблицу: mp-id, формула, SG, e_hull.
"""
import argparse
import os
from pathlib import Path


EXPLICIT_IDS = {
    "Pd_fcc":   "mp-2",
    "PdZn":     "mp-1652",
    "GaPd":     "mp-1078526",
    "InPd":     "mp-21215",
}

FORMULA_QUERIES = ["Pd3Sn"]   # скачать ВСЕ структуры этого состава


def main():
    p = argparse.ArgumentParser(description="Скачать CIF референсных структур из MP")
    p.add_argument("--out-dir", default="data/ref_structures",
                   help="куда сохранять CIF")
    p.add_argument("--formula", nargs="*", default=None,
                   help="дополнительные формулы для полного скачивания "
                        "(default: Pd3Sn)")
    p.add_argument("--mp-key", default=None,
                   help="MP_API_KEY (или через env-var MP_API_KEY)")
    args = p.parse_args()

    api_key = args.mp_key or os.environ.get("MP_API_KEY")
    if not api_key:
        raise SystemExit("Нужен MP_API_KEY: export MP_API_KEY=... или --mp-key")

    formulas = args.formula if args.formula is not None else FORMULA_QUERIES
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from mp_api.client import MPRester

    rows = []   # для финальной таблицы

    with MPRester(api_key) as mpr:

        # --- 1. Конкретные mp-id ------------------------------------------
        print(f"\n{'='*60}")
        print("Скачиваю конкретные mp-id:")
        print(f"{'='*60}")
        for name, mpid in EXPLICIT_IDS.items():
            try:
                doc = mpr.get_structure_by_material_id(mpid)
                cif_path = out_dir / f"{name}.cif"
                doc.to(filename=str(cif_path))

                # e_hull
                summary = mpr.summary.get_data_by_id(mpid, fields=["energy_above_hull", "symmetry"])
                e_hull = summary.energy_above_hull
                sg = summary.symmetry.number
                formula = doc.composition.reduced_formula

                print(f"  {mpid:12s}  {name:10s}  {formula:10s}  SG={sg:<4}  e_hull={e_hull:.3f}")
                rows.append((name, mpid, formula, sg, e_hull, cif_path))
            except Exception as exc:
                print(f"  {mpid:12s}  {name:10s}  ОШИБКА: {exc}")

        # --- 2. Все структуры по формуле ----------------------------------
        for formula in formulas:
            print(f"\n{'='*60}")
            print(f"Скачиваю все структуры состава {formula}:")
            print(f"{'='*60}")
            try:
                docs = mpr.summary.search(
                    formula=formula,
                    fields=["material_id", "structure", "energy_above_hull",
                            "symmetry", "formula_pretty"],
                )
                docs = sorted(docs, key=lambda d: d.energy_above_hull)
                for doc in docs:
                    mpid = doc.material_id
                    sg = doc.symmetry.number
                    e_hull = doc.energy_above_hull
                    f = doc.formula_pretty
                    # имя файла: формула_mpid (без знаков)
                    safe_id = mpid.replace("-", "_")
                    name = f"{formula}_{safe_id}"
                    cif_path = out_dir / f"{name}.cif"
                    doc.structure.to(filename=str(cif_path))
                    print(f"  {mpid:12s}  {f:10s}  SG={sg:<4}  e_hull={e_hull:.4f}  -> {cif_path.name}")
                    rows.append((name, mpid, f, sg, e_hull, cif_path))
            except Exception as exc:
                print(f"  ОШИБКА для {formula}: {exc}")

    # --- Финальная таблица ------------------------------------------------
    print(f"\n{'='*60}")
    print("ИТОГ — все скачанные структуры:")
    print(f"{'='*60}")
    print(f"  {'файл':<25} {'mp-id':<12} {'формула':<10} {'SG':<5} {'e_hull':>8}")
    print(f"  {'-'*65}")
    for name, mpid, formula, sg, e_hull, cif_path in rows:
        print(f"  {cif_path.name:<25} {mpid:<12} {formula:<10} {sg:<5} {e_hull:>8.4f}")

    # --- Проверка: все ли содержат Pd -------------------------------------
    print(f"\nПроверка содержимого файлов:")
    for _, _, _, _, _, cif_path in rows:
        if not cif_path.exists():
            print(f"  НЕТУ: {cif_path}")
            continue
        content = cif_path.read_text()
        has_pd = "Pd" in content or "pd" in content.lower()
        line = next((l.strip() for l in content.splitlines()
                     if "_chemical_formula_structural" in l), "?")
        marker = "OK" if has_pd else "ВНИМАНИЕ - нет Pd!"
        print(f"  {cif_path.name:<30} {line:<35} {marker}")


if __name__ == "__main__":
    main()

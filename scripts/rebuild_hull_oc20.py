"""Пересчёт ConvexHull в единой oc20-шкале.

Логика: если pipeline считает E_bulk в oc20, то и hull должен быть в oc20.
Тогда сравнение E(кандидат, oc20) с hull(oc20) корректно для всех элементов,
включая тугоплавкие (Nb, Ta, Mo, W и т.п.), где GGA и oc20 расходятся на 4-6 эВ/атом.

Что делает скрипт:
1. Скачивает структуры всех Pd-M систем + чистых элементов из MP.
2. Релаксирует каждую в UMA oc20.
3. Сохраняет новый mp_pdm_entries.json с oc20-энергиями.
4. Старый файл → .gga.json.bak.

Время: ~500 структур × 5-15 сек = 40-120 мин на GPU.

Использование:
    export MP_API_KEY=...
    python scripts/rebuild_hull_oc20.py --uma-device cuda

    # тест на 10 структурах:
    python scripts/rebuild_hull_oc20.py --uma-device cuda --limit 10
"""
import argparse
import json
import os
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uma-device", default="cuda")
    p.add_argument("--model", default="uma-s-1p2")
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--output", default="data/mp_pdm_entries.json")
    p.add_argument("--mp-key", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="ограничить число структур (для отладки)")
    args = p.parse_args()

    api_key = args.mp_key or os.environ.get("MP_API_KEY")
    if not api_key:
        raise SystemExit("Нужен MP_API_KEY: export MP_API_KEY=...")

    out = Path(args.output)

    # Бэкап — только если ещё нет
    bk = out.with_suffix(".gga.json.bak")
    if out.exists() and not bk.exists():
        bk.write_text(out.read_text())
        print(f"бэкап: {bk}")
    elif bk.exists():
        print(f"бэкап уже есть: {bk}")

    # ── Шаг 1: список структур ──────────────────────────────────────────
    print("\nшаг 1: скачивание структур из MP...")
    from pymatgen.core.periodic_table import Element
    from pdh_gfn.constants import ALL_ELEMENTS
    from mp_api.client import MPRester
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.build import bulk as ase_bulk

    structures = []  # (mp_id, formula, atoms)

    with MPRester(api_key) as mpr:
        pd_sym = Element.from_Z(46).symbol

        # Pd-M бинарники
        for z in ALL_ELEMENTS:
            el = Element.from_Z(z).symbol
            if el == pd_sym:
                continue
            chemsys = f"{pd_sym}-{el}"
            try:
                docs = mpr.summary.search(
                    chemsys=chemsys,
                    fields=["material_id", "structure", "formula_pretty"],
                )
                for d in docs:
                    a = AseAtomsAdaptor.get_atoms(d.structure)
                    structures.append((d.material_id, d.formula_pretty, a))
                if docs:
                    print(f"  {chemsys}: {len(docs)}")
            except Exception as exc:
                print(f"  {chemsys}: FAIL {exc}")

        # Чистые элементы — берём из MP по формуле
        print(f"\nчистые элементы ({len(ALL_ELEMENTS)})...")
        for z in ALL_ELEMENTS:
            el = Element.from_Z(z).symbol
            try:
                docs = mpr.summary.search(
                    formula=el,
                    fields=["material_id", "structure", "energy_above_hull",
                            "formula_pretty"],
                )
                if not docs:
                    raise RuntimeError("нет данных")
                # берём самый стабильный
                best = min(docs, key=lambda d: d.energy_above_hull)
                a = AseAtomsAdaptor.get_atoms(best.structure)
                structures.append((best.material_id, el, a))
                print(f"  {el}: {best.material_id}")
            except Exception as exc:
                print(f"  {el}: FAIL {exc}")

    if args.limit:
        structures = structures[:args.limit]
        print(f"\nограничение: {args.limit} структур")
    print(f"\nвсего к релаксации: {len(structures)}")

    # ── Шаг 2: UMA oc20 ─────────────────────────────────────────────────
    print(f"\nшаг 2: загрузка UMA {args.model} oc20 на {args.uma_device}...")
    from pdh_gfn.potential.uma import UMAPotential
    pot = UMAPotential(model=args.model, task="oc20", device=args.uma_device)
    print("OK")

    # ── Шаг 3: релаксация ───────────────────────────────────────────────
    print(f"\nшаг 3: релаксация {len(structures)} структур...")
    new_records = []
    n_done = n_fail = 0
    t0_all = time.time()

    # Подгружаем уже посчитанные (если прогон прерывался)
    already = set()
    if out.exists():
        try:
            prev = json.loads(out.read_text())
            mp_ids = {r.get("mp_id") for r in prev if r.get("mp_id")}
            if mp_ids and all(r.get("mp_id") for r in prev):
                new_records = prev
                already = mp_ids
                print(f"  продолжаем: уже есть {len(new_records)} записей")
        except Exception:
            pass

    for i, (mp_id, formula, atoms) in enumerate(structures):
        if mp_id in already:
            continue  # уже посчитан
        t0 = time.time()
        try:
            res = pot.relax_bulk(atoms, fmax=args.fmax)
            if not res.valid:
                raise RuntimeError("не сошлась")
            e_pa = res.energy / len(res.atoms)
            # строка состава pymatgen-совместимая
            from collections import Counter
            cnt = Counter(res.atoms.get_chemical_symbols())
            comp_str = " ".join(f"{el}{n}" for el, n in sorted(cnt.items()))
            new_records.append({
                "composition": comp_str,
                "energy_per_atom": e_pa,
                "mp_id": mp_id,
            })
            n_done += 1
            already.add(mp_id)
            elapsed = time.time() - t0_all
            n_left = len(structures) - i - 1
            eta = elapsed / max(1, n_done) * n_left
            print(f"[{i+1}/{len(structures)}] {formula:<15s} {mp_id:<12s} "
                  f"E/at={e_pa:.3f} ({time.time()-t0:.1f}s) ETA {eta/60:.0f}min")
        except Exception as exc:
            n_fail += 1
            print(f"[{i+1}/{len(structures)}] {formula:<15s} FAIL: {exc}")

        # Сохранение ПОСЛЕ КАЖДОЙ ЗАПИСИ — если упадёт, не потеряем прогресс
        out.write_text(json.dumps(new_records, indent=2))

    # ── Итог ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"ИТОГ: {n_done} пересчитано, {n_fail} провалено")
    print(f"время: {(time.time()-t0_all)/60:.1f} мин")
    print(f"записей в hull: {len(new_records)}")
    print(f"файл: {out}")
    print(f"\nПроверка PdZn:")
    pdzn = next((r for r in new_records
                 if "Pd" in r["composition"] and "Zn" in r["composition"]
                 and len(r["composition"].split()) == 2), None)
    if pdzn:
        print(f"  {pdzn['composition']}: E/at={pdzn['energy_per_atom']:.3f} oc20")
        print(f"  В бэкапе ZnPd: {next((r['energy_per_atom'] for r in json.loads(bk.read_text()) if 'Zn' in r.get('composition','') and 'Pd' in r.get('composition','') and len(r['composition'].split())==2), 'N/A')} GGA")


if __name__ == "__main__":
    main()

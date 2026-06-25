"""Для кандидатов из CSV (export_candidates) находит их сохранённые структуры.

Структуры лежат на кластере в
    data/structures/<bulk_hash>/<h_k_l>_off<bin>_<side>/{slab_relaxed.xyz, ads_*, manifest.json}
а объёмы — в data/structures/<bulk_hash>/bulk_relaxed.xyz (см. saver.py).
Матчим по (состав, space_group, грань, offset-бин, сторона) из строки кандидата.

Использование:
    python scripts/find_candidate_structures.py \
        --candidates logs/pdh_gfn/candidates_top100.csv \
        --structures-dir data/structures --out candidate_structures --copy
"""
import argparse
import csv
import glob
import json
import re
import shutil
from pathlib import Path

from ase.data import atomic_numbers

N_OFFSET_BINS = 8  # из pipeline.py


def parse_formula(formula):
    """'Pd8Bi8' -> {46: 8, 83: 8}"""
    comp = {}
    for sym, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if sym and sym in atomic_numbers:
            z = atomic_numbers[sym]
            comp[z] = comp.get(z, 0) + int(n or 1)
    return comp


def find_dir(sdir, formula, sg, miller, offset, top):
    """Папка поверхности кандидата или None. Сначала точно (с offset-бином и
    стороной), затем мягко (любой offset/сторона — на случай иной дискретизации)."""
    comp = parse_formula(formula)
    obin = int(float(offset) * N_OFFSET_BINS)
    side = "top" if int(top) else "bottom"
    for pat in (f"{sdir}/*/{miller}_off{obin}_{side}/manifest.json",
                f"{sdir}/*/{miller}_off*/manifest.json"):
        for mf in glob.glob(pat):
            try:
                d = json.load(open(mf))
            except Exception:
                continue
            st = d.get("state", {})
            c = {int(k): v for k, v in st.get("composition", {}).items()}
            if c == comp and str(st.get("space_group")) == str(sg):
                return Path(mf).parent
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--structures-dir", default="data/structures")
    ap.add_argument("--out", default="candidate_structures")
    ap.add_argument("--copy", action="store_true",
                    help="копировать bulk_relaxed/slab_relaxed/manifest в --out")
    ap.add_argument("--with-ads", action="store_true",
                    help="копировать также ads_*.xyz")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.candidates, encoding="utf-8")))
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    map_rows, n_found = [], 0
    for r in rows:
        d = find_dir(args.structures_dir, r["formula"], r["sg"],
                     r["miller"], r["offset"], r["top"])
        files = sorted(p.name for p in d.iterdir()) if d else []
        n_found += d is not None
        map_rows.append({
            "rank": r.get("rank", ""), "formula": r["formula"], "sg": r["sg"],
            "miller": r["miller"], "flag": r.get("flag", ""),
            "structure_dir": str(d) if d else "",
            "files": ";".join(files),
        })
        if d and args.copy:
            dst = outdir / f"{int(r.get('rank', 0)):03d}_{r['formula']}_sg{r['sg']}_{r['miller']}"
            dst.mkdir(parents=True, exist_ok=True)
            for f in (d.parent / "bulk_relaxed.xyz", d / "slab_relaxed.xyz",
                      d / "manifest.json"):
                if f.exists():
                    shutil.copy(f, dst / f.name)
            if args.with_ads:
                for f in d.glob("ads_*.xyz"):
                    shutil.copy(f, dst / f.name)

    mp = outdir / "candidate_structures_map.csv"
    with mp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "rank", "formula", "sg", "miller", "flag", "structure_dir", "files"])
        w.writeheader()
        w.writerows(map_rows)

    print(f"кандидатов: {len(rows)} | найдено структур: {n_found} | "
          f"не найдено: {len(rows) - n_found}")
    print(f"карта: {mp}")
    if args.copy:
        print(f"структуры скопированы: {outdir}/<rank>_<formula>_sg<sg>_<hkl>/")
    if n_found < len(rows):
        print("  [!] часть не найдена — data/structures могла быть перезаписана "
              "более поздним прогоном (структуры от прогона trace, а не текущие).")


if __name__ == "__main__":
    main()

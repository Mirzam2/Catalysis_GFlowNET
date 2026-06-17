"""Сверка батчевой адсорбции с последовательной + замер ускорения.

На наборе слэбов считает BE(H/CH/C3H7) ДВУМЯ путями — последовательным
(base.adsorption_energies) и батчевым (batch.batch_adsorption_energies) — на
ОДНОМ и том же релаксированном слэбе, сравнивает BE и время.

Цель: убедиться, что батч не сдвигает BE (паритет дескрипторов), и измерить
ускорение, прежде чем включать --batch-relax в боевом обучении.

Запуск:  python scripts/bench_batch.py --uma-device cuda
"""
import argparse
import time

import numpy as np
from ase.build import fcc111

from pdh_gfn.potential.uma import UMAPotential
from pdh_gfn.potential.batch import batch_adsorption_energies
from pdh_gfn.reward.pipeline import ADSORBATES

REFS = {"H": 0.0, "CH": 0.0, "C3H7": 0.0}  # обнуляем — в сравнении они сокращаются


def make_test_slabs():
    """Набор слэбов: чистые металлы + псевдо-интерметаллид Pd/Zn."""
    slabs = []
    slabs.append(("Pd(111) 3x3", fcc111("Pd", size=(3, 3, 3), vacuum=12.0)))
    slabs.append(("Pd(111) 2x2x4", fcc111("Pd", size=(2, 2, 4), vacuum=12.0)))
    slabs.append(("Cu(111) 3x3", fcc111("Cu", size=(3, 3, 3), vacuum=12.0)))
    slabs.append(("Ni(111) 3x3", fcc111("Ni", size=(3, 3, 3), vacuum=12.0)))
    # псевдо-интерметаллид: Pd(111) с замещением части атомов на Zn
    pdzn = fcc111("Pd", size=(3, 3, 3), vacuum=12.0)
    sym = list(pdzn.symbols)
    for i in range(0, len(sym), 2):
        sym[i] = "Zn"
    pdzn.symbols = sym
    slabs.append(("Pd/Zn(111) 3x3", pdzn))
    for _, s in slabs:
        s.pbc = True
    return slabs


def seq_adsorption(pot, slab, e_slab, max_sites, max_sites_light, keep, fmax):
    out = {}
    for name in ("CH", "H", "C3H7"):
        ns = max_sites_light if name == "H" else max_sites
        out[name] = pot.adsorption_energies(
            slab, ADSORBATES[name](), e_slab=e_slab, e_adsorbate_ref=0.0,
            max_sites=ns, fmax=fmax, prefilter=True, prefilter_keep=keep)
    return out


def cmp_lists(a, b):
    """Макс. |разница| по выровненным (по длине) спискам BE + флаг длины."""
    n = min(len(a), len(b))
    d = max((abs(a[k] - b[k]) for k in range(n)), default=0.0)
    return d, len(a), len(b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uma-device", default="cuda")
    p.add_argument("--model", default="uma-s-1p2")
    p.add_argument("--fmax", type=float, default=0.10)
    p.add_argument("--max-sites", type=int, default=8)
    p.add_argument("--max-sites-light", type=int, default=4)
    p.add_argument("--prefilter-keep", type=int, default=4)
    p.add_argument("--tol", type=float, default=0.05, help="порог |ΔBE|, эВ")
    args = p.parse_args()

    pot = UMAPotential(model=args.model, task="oc20", device=args.uma_device)
    slabs = make_test_slabs()

    t_seq_tot = t_bat_tot = 0.0
    worst = 0.0
    print(f"{'система':<18} {'адс':<5} {'seq BE':>22} {'batch BE':>22} "
          f"{'maxΔ':>8}")
    for label, slab0 in slabs:
        # один раз релаксируем слэб (последовательно) — общий e_slab для обоих
        res = pot.relax_slab(slab0, fmax=args.fmax)
        if not res.valid:
            print(f"{label:<18} слэб не релаксировался — пропуск")
            continue
        slab, e_slab = res.atoms, res.energy

        t0 = time.perf_counter()
        seq = seq_adsorption(pot, slab, e_slab, args.max_sites,
                             args.max_sites_light, args.prefilter_keep, args.fmax)
        t_seq = time.perf_counter() - t0

        t0 = time.perf_counter()
        bat = batch_adsorption_energies(
            pot, slab, {k: ADSORBATES[k] for k in ("CH", "H", "C3H7")},
            e_slab=e_slab, refs=REFS, fmax=args.fmax,
            max_sites=args.max_sites, max_sites_light=args.max_sites_light,
            prefilter=True, prefilter_keep=args.prefilter_keep)
        t_bat = time.perf_counter() - t0

        t_seq_tot += t_seq
        t_bat_tot += t_bat
        for name in ("CH", "H", "C3H7"):
            d, na, nb = cmp_lists(seq[name], bat[name])
            worst = max(worst, d)
            sa = ",".join(f"{x:.2f}" for x in seq[name][:3])
            sb = ",".join(f"{x:.2f}" for x in bat[name][:3])
            print(f"{label:<18} {name:<5} {sa:>22} {sb:>22} {d:>8.3f}"
                  + ("" if na == nb else f"  [len {na}!={nb}]"))
        print(f"{'':<18} время: seq={t_seq:.1f}s  batch={t_bat:.1f}s  "
              f"speedup×{t_seq / max(t_bat, 1e-9):.1f}")

    print("\n" + "=" * 60)
    print(f"ИТОГО: seq={t_seq_tot:.1f}s  batch={t_bat_tot:.1f}s  "
          f"speedup×{t_seq_tot / max(t_bat_tot, 1e-9):.1f}")
    print(f"макс |ΔBE| по всем = {worst:.3f} эВ  (порог {args.tol})")
    print("ПАРИТЕТ OK" if worst <= args.tol else "ПАРИТЕТ НАРУШЕН — разбираться!")


if __name__ == "__main__":
    main()

"""Калибровка дескрипторов на референсных поверхностях.

Запуск:
    rm -f data/adsorbate_refs.json data/reward_cache.jsonl
    python scripts/calibrate.py --device cuda

Только локальные CIF (никакого MP-фолбэка — он опасный, может скачать не ту фазу).
Печатает таблицу BE/E_act/E_sel и готовые sed-команды для constants.py.
"""
import argparse
import logging
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pdh_gfn.reward.descriptors import compute_descriptors
from pdh_gfn.reward.pipeline import ADSORBATES
from pdh_gfn.reward.refs import compute_adsorbate_refs
from pdh_gfn.structure.slab import cut_slab
from pdh_gfn.potential.base import enumerate_sites

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

CACHE_DIR = Path("data/ref_structures")

# Эталонные поверхности: (метка, имя CIF в data/ref_structures/, грани).
# Несколько граней на структуру — берём первую где CH даёт >=2 минимума.
REFERENCE_SURFACES = [
    ("Pd(111)",        "Pd_fcc",            [(1, 1, 1)]),
    ("PdZn(111)",      "PdZn",              [(1, 1, 1), (1, 0, 1), (1, 1, 0)]),
    ("PdIn(110)",      "PdIn",              [(1, 1, 0), (1, 0, 0), (1, 1, 1)]),
    ("PdGa(111)",      "PdGa",              [(1, 1, 1), (1, 1, 0), (1, 0, 0)]),
    ("Pd3Sn-718(111)", "Pd3Sn_mp_718",      [(1, 1, 1), (1, 0, 0)]),
    ("Pd3Sn-1218937",  "Pd3Sn_mp_1218937",  [(1, 1, 1), (1, 0, 0)]),
]


def load_cif(name):
    """Чтение CIF + извлечение реальной формулы и SG для печати."""
    cif = CACHE_DIR / f"{name}.cif"
    if not cif.exists():
        return None, f"НЕТ файла {cif}"
    struct = Structure.from_file(cif)
    formula = struct.composition.reduced_formula
    sga = SpacegroupAnalyzer(struct, symprec=0.1)
    sg = sga.get_space_group_number()
    has_pd = "Pd" in formula
    info = f"{formula} SG={sg}"
    if not has_pd:
        info += " БЕЗ Pd!"
    return struct, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--model", default="uma-s-1p2")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--fmax", type=float, default=0.03)
    p.add_argument("--max-sites", type=int, default=12)
    args = p.parse_args()

    if args.mock:
        from pdh_gfn.potential.emt_mock import EMTMockPotential
        bulk_pot = ads_pot = EMTMockPotential()
    else:
        from pdh_gfn.potential.uma import UMAPotential
        bulk_pot = UMAPotential(model=args.model, task="oc20", device=args.device)
        ads_pot = bulk_pot  # единый oc20

    print("=" * 88)
    print("КАЛИБРОВКА ДЕСКРИПТОРОВ PDH-GFlowNet")
    print("=" * 88)

    # Проверяем все CIF на старте — сразу видно если что-то не то
    print("\nИсходные CIF в", CACHE_DIR, ":")
    for label, name, _ in REFERENCE_SURFACES:
        struct, info = load_cif(name)
        marker = " " if struct else "!"
        print(f"  [{marker}] {label:<18s} {name}.cif: {info}")

    refs = compute_adsorbate_refs(ads_pot, fmax=0.01)
    print(f"\nadsorbate_refs (Seemakurthi ур. 2):")
    for k, v in refs.items():
        print(f"  {k:>5s}: {v:.3f} эВ")
    print(f"\nfmax = {args.fmax} эВ/Å, max_sites = {args.max_sites}")

    header = (f"{'поверхность':<18} {'формула':<10} {'SG':<5} {'slab':>5} {'сайтов':>6} "
              f"{'BE(H)':>7} {'BE(CH)':>7} {'CHmeta':>7} {'BE(C3H7)':>9} "
              f"{'Eact(CH)':>9} {'E_sel':>7}")
    print(f"\n{header}")
    print("-" * len(header))

    rows = []
    for label, name, hkl_list in REFERENCE_SURFACES:
        struct, info = load_cif(name)
        if struct is None:
            print(f"{label:<18} {info}")
            continue
        if "БЕЗ Pd" in info:
            print(f"{label:<18} {info} — пропускаю")
            continue
        formula = info.split()[0]
        sg = int(info.split("SG=")[1].split()[0])

        atoms = AseAtomsAdaptor.get_atoms(struct)
        if not bulk_pot.supports(atoms):
            print(f"{label:<18} {formula:<10} {sg:<5} вне домена потенциала")
            continue

        # Объём — релаксируем один раз
        bres = bulk_pot.relax_bulk(atoms, fmax=args.fmax)
        if not bres.valid:
            print(f"{label:<18} {formula:<10} {sg:<5} bulk не сошёлся")
            continue
        relaxed = AseAtomsAdaptor.get_structure(bres.atoms)

        # Перебираем грани пока CH не даст >=2 минимумов
        for hkl in hkl_list:
            slab = cut_slab(relaxed, hkl, 0.0, True)
            if slab is None:
                continue
            n_slab = len(slab)
            sres = ads_pot.relax_slab(slab, fmax=args.fmax)
            if not sres.valid:
                continue
            sites = enumerate_sites(sres.atoms, max_sites=args.max_sites)
            n_sites = len(sites)

            def be(adsorbate):
                return ads_pot.adsorption_energies(
                    sres.atoms, ADSORBATES[adsorbate](),
                    e_slab=sres.energy, e_adsorbate_ref=refs[adsorbate],
                    max_sites=args.max_sites, fmax=args.fmax)

            ch = be("CH")
            if len(ch) < 2:
                # пробуем следующую грань
                continue
            h = be("H")
            c3h7 = be("C3H7")
            if not h or not c3h7:
                continue

            d = compute_descriptors(h[0], ch[0], ch[1], c3h7[0])
            display_label = f"{label.split('(')[0]}({hkl[0]}{hkl[1]}{hkl[2]})"
            rows.append((display_label, d, formula, sg))
            print(f"{display_label:<18} {formula:<10} {sg:<5} {n_slab:>5} {n_sites:>6} "
                  f"{h[0]:>7.2f} {ch[0]:>7.2f} {ch[1]:>7.2f} "
                  f"{c3h7[0]:>9.2f} {d.e_act_ch:>9.2f} {d.e_sel:>7.2f}")
            break  # эта структура отработала, идём к следующей
        else:
            print(f"{label:<18} {formula:<10} {sg:<5} "
                  f"ни одна грань не дала ≥2 минимумов CH")

    if not rows:
        print("\nНи одна поверхность не посчиталась.")
        return

    print("\n" + "=" * 88)
    print("РЕКОМЕНДАЦИИ (с фильтрацией физически невозможных)")
    print("=" * 88)

    # Фильтр: Eact > 0 (отрицательная активация — физически невозможно)
    physical = [(l, d) for l, d, _, _ in rows if d.e_act_ch > 0]
    bad = [(l, d) for l, d, _, _ in rows if d.e_act_ch <= 0]
    if bad:
        print("\nОТФИЛЬТРОВАНО (Eact ≤ 0, не физично):")
        for l, d in bad:
            print(f"  {l}: Eact={d.e_act_ch:.2f}, E_sel={d.e_sel:.2f}")

    if not physical:
        print("\nВсе строки нефизичны — порог не построить.")
        return

    e_acts = [d.e_act_ch for _, d in physical]
    e_sels = [d.e_sel for _, d in physical]

    pd_rows = [(l, d) for l, d in physical if l.startswith("Pd(")]
    if pd_rows:
        pd_label, pd = pd_rows[0]
        marker = "✓" if pd.e_act_ch > 0 else "✗"
        print(f"\n[знак] E_act(C-H) {pd_label} = {pd.e_act_ch:.2f} эВ {marker}")

    e_act_max = max(e_acts) + 0.5
    lo, hi = min(e_sels), max(e_sels)
    e_sel_target = (lo + hi) / 2
    e_sel_scale = max(0.3, (hi - lo) / 4)
    e_act_scale = max(0.3, (max(e_acts) - min(e_acts)) / 4) if len(e_acts) > 1 else 0.4

    print(f"\n[порог] E_ACT_CH_MAX = {e_act_max:.2f}  (макс из {len(physical)} + 0.5)")
    print(f"[порог] E_ACT_SCALE  = {e_act_scale:.2f}")
    print(f"[порог] E_SEL_TARGET = {e_sel_target:.2f}  (середина {lo:.2f}..{hi:.2f})")
    print(f"[порог] E_SEL_SCALE  = {e_sel_scale:.2f}")

    print(f"\n{'='*88}\nГотовые команды:\n{'='*88}")
    print(f"sed -i 's/E_ACT_CH_MAX = .*/E_ACT_CH_MAX = {e_act_max:.2f}/' pdh_gfn/constants.py")
    print(f"sed -i 's/E_ACT_SCALE = .*/E_ACT_SCALE = {e_act_scale:.2f}/' pdh_gfn/constants.py")
    print(f"sed -i 's/E_SEL_TARGET = .*/E_SEL_TARGET = {e_sel_target:.2f}/' pdh_gfn/constants.py")
    print(f"sed -i 's/E_SEL_SCALE = .*/E_SEL_SCALE = {e_sel_scale:.2f}/' pdh_gfn/constants.py")
    print(f"\nПосле — удалить старый кэш: rm -f data/reward_cache.jsonl")


if __name__ == "__main__":
    main()

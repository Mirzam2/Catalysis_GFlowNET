"""Тесты механики конвейера на EMT-заглушке (Pd-Cu: оба элемента в EMT).

Проверяется ТОЛЬКО механика (сборка -> релаксация -> поверхность -> сайты
-> BE -> дескрипторы -> награда -> кэш), не химия. Полный цикл с UMA —
на кластере.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdh_gfn import constants as C
from pdh_gfn.potential.emt_mock import EMTMockPotential
from pdh_gfn.reward.cache import RewardCache
from pdh_gfn.reward.descriptors import compute_descriptors
from pdh_gfn.reward.gates import composite_reward, invalid_reward
from pdh_gfn.reward.hull import HullReference
from pdh_gfn.reward.pipeline import CatalystState, RewardPipeline
from pdh_gfn.structure.builder import build_bulk, is_composition_compatible
from pdh_gfn.structure.canonical import structure_hash


def test_builder_pdcu_b2():
    structs = build_bulk(221, {46: 1, 29: 1}, (3.0, 3.0, 3.0, 90, 90, 90),
                         n_samples=3, seed=0)
    assert structs, "pyxtal не собрал B2 PdCu"
    s = structs[0]
    assert s.composition.reduced_formula in ("CuPd", "PdCu")
    assert len(s) == 2
    print("OK builder:", s.composition.reduced_formula, f"{s.volume:.1f} A^3")


def test_wyckoff_compat():
    assert is_composition_compatible(221, [1, 1])
    assert not is_composition_compatible(225, [1])  # 1 атом в Fm-3m: мин. кратность 4
    print("OK wyckoff-совместимость")


def test_structure_hash_invariance():
    """Контракт хэша: структуры, релаксировавшиеся в одну фазу, — один хэш.

    Берём B2 PdCu с разными стартовыми параметрами (и, возможно, разным
    выбором начала координат от pyxtal) — после релаксации EMT обе должны
    попасть в один бин кэша.
    """
    from pymatgen.io.ase import AseAtomsAdaptor
    pot = EMTMockPotential()
    hashes = []
    for lp in [(2.9,) * 3 + (90,) * 3, (3.1,) * 3 + (90,) * 3]:
        s = build_bulk(221, {46: 1, 29: 1}, lp, seed=hash(lp) % 1000)[0]
        res = pot.relax_bulk(AseAtomsAdaptor.get_atoms(s), fmax=0.02)
        assert res.valid
        hashes.append(structure_hash(AseAtomsAdaptor.get_structure(res.atoms)))
    assert hashes[0] == hashes[1], \
        f"релаксированные в одну фазу структуры дали разные хэши: {hashes}"
    print("OK хэш:", hashes[0])


def test_descriptors_and_gates():
    # знаковое соглашение Seemakurthi: BE отрицательны (крепче = отрицательнее)
    be_h, be_ch, be_ch_meta, be_c3h7 = -2.7, -6.0, -5.2, -1.1
    d = compute_descriptors(be_h, be_ch, be_ch_meta, be_c3h7)
    assert math.isfinite(d.e_sel)
    # сверка со свёрнутой прокси-формулой статьи:
    # E_sel = BE(CH) + 0.77*BE(CH_meta) - 0.82*BE(C3H7) - 0.82*BE(H) - 0.64
    e_sel_ref = (be_ch + 0.77 * be_ch_meta
                 - 0.82 * be_c3h7 - 0.82 * be_h - 0.64)
    assert abs(d.e_sel - e_sel_ref) < 1e-9, (d.e_sel, e_sel_ref)
    assert d.be_propyne == be_ch  # Fig S5

    r = composite_reward(d, e_hull=0.02)
    assert 0.0 < r.reward <= 1.0
    assert r.reward_beta == r.reward ** C.REWARD_BETA
    assert invalid_reward().reward == C.R_VALID_EPS
    r_bad = composite_reward(d, e_hull=0.30)
    assert r_bad.r_stab < r.r_stab
    print(f"OK гейты: E_sel={d.e_sel:.2f} эВ (ref {e_sel_ref:.2f}), "
          f"E_act(C-H)={d.e_act_ch:.2f}, R={r.reward:.3g}")


def test_full_pipeline_emt():
    pot = EMTMockPotential()
    hull = HullReference()
    # элементные референсы тем же потенциалом (грубые, для механики)
    from ase.build import bulk as ase_bulk
    refs = {}
    for el, a in [("Pd", 3.89), ("Cu", 3.61)]:
        at = ase_bulk(el, "fcc", a=a)
        res = pot.relax_bulk(at, fmax=0.05)
        refs[el] = res.energy / len(res.atoms)
    hull.add_elemental_references(refs)

    from pdh_gfn.reward.refs import compute_adsorbate_refs
    adsorbate_refs = compute_adsorbate_refs(pot, fmax=0.05)
    cache = RewardCache()
    pipe = RewardPipeline(pot, hull, cache=cache, adsorbate_refs=adsorbate_refs,
                          fmax=0.10, max_sites=4)

    state = CatalystState(
        space_group=221,
        composition={46: 1, 29: 1},
        lattice_params=(3.0, 3.0, 3.0, 90, 90, 90),
        miller=(1, 0, 0),
        offset=0.0,
        top=True,
    )
    r1 = pipe(state)
    print(f"OK конвейер: R={r1.reward:.3g}, E_hull={r1.e_hull}, "
          f"E_sel={r1.e_sel}, кэш={len(cache)}")
    n_before = len(cache)
    r2 = pipe(state)
    assert len(cache) == n_before, "повторный вызов должен бить в кэш"
    assert r2.reward == r1.reward
    print("OK кэш наград")


if __name__ == "__main__":
    test_builder_pdcu_b2()
    test_wyckoff_compat()
    test_structure_hash_invariance()
    test_descriptors_and_gates()
    test_full_pipeline_emt()
    print("\nВСЕ ТЕСТЫ ПРОШЛИ")

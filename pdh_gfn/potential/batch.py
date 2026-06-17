"""УСКОРЕНИЕ #1 (главный рычаг): батч-релаксация адсорбатов на GPU.

~87% времени фаз 2-3 — адсорбционные релаксации, выполняемые ПОСЛЕДОВАТЕЛЬНО
(цикл по сайтам × адсорбатам). UMA на GPU считает БАТЧ систем почти за ту же
латентность, что одну (на одиночных мелких слэбах GPU недозагружен, память почти
пуста). Релаксируем все конфигурации (сайты × 3 адсорбата) одним батчем →
один forward на макрошаг FIRE вместо ~12-24.

Реализация (fairchem 2.21.0): готовой батчевой релаксации в этой версии нет
(`ml_relax`/`OptimizableBatch` отсутствуют), поэтому свой батчевый FIRE поверх
`potential.batch_eval` (батч-forward, см. uma.py). Ячейка фиксирована (адсорбат на
слэбе), оптимизируем только позиции — FIRE без cell-DOF. Параметры FIRE и мягкая
сходимость — как в base.relax, чтобы BE совпадали с последовательным путём.

Интерфейс для RewardPipeline: один вызов batch_adsorption_energies на всю адсорбцию
поверхности вместо 3×N релаксаций.
"""
from typing import Dict, List, Tuple

import numpy as np
from ase import Atoms

from .base import enumerate_sites, _dedup_minima, _geometry_valid
from ..constants import MAX_RELAX_STEPS, RELAX_FMAX_COARSE
from ..profiling import PROFILER

# Параметры FIRE — дефолты ase.optimize.fire.FIRE (для паритета BE)
_FIRE = dict(dt=0.1, maxstep=0.2, dtmax=1.0, Nmin=5,
             finc=1.1, fdec=0.5, astart=0.1, fa=0.99)


def build_adsorbate_configs(
    slab: Atoms,
    adsorbate_factories: Dict[str, callable],
    sites_per_adsorbate: Dict[str, List[np.ndarray]],
) -> List[Tuple[str, Atoms]]:
    """Собирает все конфигурации (адсорбат, сайт) в плоский список для батча.

    Returns список (имя_адсорбата, Atoms) — вход для батч-релаксатора.
    """
    configs = []
    for name, factory in adsorbate_factories.items():
        for site in sites_per_adsorbate.get(name, []):
            combined = slab.copy()
            ads = factory()
            ads.translate(np.asarray(site) - ads.positions[0])
            combined += ads
            configs.append((name, combined))
    return configs


def batch_relax(potential, atoms_list: List[Atoms],
                fmax: float = RELAX_FMAX_COARSE,
                steps: int = MAX_RELAX_STEPS):
    """Батчевый FIRE: релаксирует СПИСОК систем, на каждом макрошаге один
    forward по всем НЕсошедшимся (potential.batch_eval). Позиции-only.

    Возвращает список (energy, atoms, valid) в порядке входа. `valid` —
    сходимость (мягкая, max_force < 3·fmax) И геометрия (_geometry_valid),
    как в base.relax.
    """
    n = len(atoms_list)
    atoms = [a.copy() for a in atoms_list]
    v = [np.zeros((len(a), 3)) for a in atoms]
    dt = [_FIRE["dt"]] * n
    a_mix = [_FIRE["astart"]] * n
    nsteps = [0] * n
    started = [False] * n
    done = [False] * n
    converged = [False] * n
    energy = [float("nan")] * n
    last_fmax = [float("inf")] * n

    for it in range(steps + 1):
        active = [i for i in range(n) if not done[i]]
        if not active:
            break
        es, fs = potential.batch_eval([atoms[i] for i in active])
        for j, i in enumerate(active):
            f = fs[j]
            fmax_i = float(np.sqrt((f ** 2).sum(axis=1)).max())
            last_fmax[i] = fmax_i
            energy[i] = float(es[j])
            if fmax_i <= fmax:           # сошлась (как ASE: проверка ДО шага)
                done[i] = True
                converged[i] = True
                continue
            if it == steps:               # исчерпали бюджет шагов
                done[i] = True
                continue
            # --- один шаг FIRE для системы i (повторяет ASE FIRE.step) ---
            vi = v[i]
            if not started[i]:
                started[i] = True         # первый шаг: ветку vf пропускаем
            else:
                P = float((f * vi).sum())
                if P > 0.0:
                    fnorm = np.linalg.norm(f)
                    vnorm = np.linalg.norm(vi)
                    if fnorm > 0.0:
                        vi = (1.0 - a_mix[i]) * vi + a_mix[i] * (vnorm / fnorm) * f
                    if nsteps[i] > _FIRE["Nmin"]:
                        dt[i] = min(dt[i] * _FIRE["finc"], _FIRE["dtmax"])
                        a_mix[i] *= _FIRE["fa"]
                    nsteps[i] += 1
                else:
                    vi = np.zeros_like(vi)
                    a_mix[i] = _FIRE["astart"]
                    dt[i] *= _FIRE["fdec"]
                    nsteps[i] = 0
            vi = vi + dt[i] * f
            dr = dt[i] * vi
            normdr = float(np.sqrt((dr ** 2).sum()))
            if normdr > _FIRE["maxstep"]:
                dr = _FIRE["maxstep"] * dr / normdr
            atoms[i].set_positions(atoms[i].get_positions() + dr)
            v[i] = vi

    results = []
    for i in range(n):
        ok = converged[i] or (last_fmax[i] < 3.0 * fmax)
        valid = bool(ok and _geometry_valid(atoms[i]))
        results.append((energy[i], atoms[i], valid))
    return results


def batch_adsorption_energies(
    potential,
    slab: Atoms,
    adsorbate_factories: Dict[str, callable],
    e_slab: float,
    refs: Dict[str, float],
    fmax: float = RELAX_FMAX_COARSE,
    max_sites: int = 8,
    max_sites_light: int = 4,
    prefilter: bool = True,
    prefilter_keep: int = 4,
    on_relaxed=None,
) -> Dict[str, List[float]]:
    """BE всех адсорбатов поверхности ОДНИМ батчем. Семантика как у
    последовательного base.adsorption_energies (enumerate_sites → префильтр →
    релаксация → BE → _dedup_minima), но релаксация — один batch_relax по
    конфигурациям ВСЕХ адсорбатов сразу.

    on_relaxed(name, i, be, atoms) — callback сохранения релаксированной
    геометрии (отличается от последовательного именем-аргументом).
    """
    # сайты по адсорбату (H симметричнее → меньше сайтов, как в base)
    sites_per = {}
    for name in adsorbate_factories:
        ns = max_sites_light if name == "H" else max_sites
        sites_per[name] = enumerate_sites(slab, max_sites=ns)

    configs = build_adsorbate_configs(slab, adsorbate_factories, sites_per)
    if not configs:
        return {name: [] for name in adsorbate_factories}
    names = [nm for nm, _ in configs]
    atoms_list = [at for _, at in configs]

    # батчевый single-point префильтр: топ-K по энергии в пределах адсорбата
    if prefilter:
        with PROFILER.timed("ads.batch_prefilter"):
            es, _ = potential.batch_eval(atoms_list)
        keep = []
        for name in adsorbate_factories:
            idxs = [k for k, nm in enumerate(names) if nm == name]
            idxs.sort(key=lambda k: es[k])
            keep.extend(idxs[:prefilter_keep])
        keep.sort()
        names = [names[k] for k in keep]
        atoms_list = [atoms_list[k] for k in keep]

    # один батчевый FIRE по всем оставшимся конфигурациям
    with PROFILER.timed("ads.batch_relax"):
        results = batch_relax(potential, atoms_list, fmax=fmax)
    n_conv = sum(1 for _, _, ok in results if ok)
    PROFILER.add_relax("ads.batch_relax", len(results),
                       n_conv == len(results))

    out: Dict[str, List[float]] = {name: [] for name in adsorbate_factories}
    for k, (energy, atoms, valid) in enumerate(results):
        if not valid:
            continue
        name = names[k]
        be = energy - e_slab - refs[name]
        out[name].append(be)
        if on_relaxed is not None:
            on_relaxed(name, len(out[name]) - 1, be, atoms)
    for name in out:
        out[name] = _dedup_minima(out[name])
    return out

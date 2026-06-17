"""Интерфейс ML-потенциала для конвейера награды.

Один и тот же потенциал используется и как релаксатор (объём + поверхность),
и как прокси адсорбционных энергий — по решению из обсуждения дизайна.
Конкретные реализации: uma.py (боевая), emt_mock.py (тесты).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from ase import Atoms
from ase.optimize import FIRE

from ..constants import MAX_RELAX_STEPS, RELAX_FMAX_COARSE
from ..profiling import PROFILER


@dataclass
class RelaxResult:
    atoms: Optional[Atoms]
    energy: float = np.nan
    converged: bool = False
    valid: bool = False
    info: Dict = field(default_factory=dict)


class Potential(ABC):
    """Базовый интерфейс. Реализации обязаны предоставить ASE-калькулятор."""

    @abstractmethod
    def get_calculator(self):
        ...

    def supports(self, atoms: Atoms) -> bool:
        """Покрывает ли потенциал элементы системы (для заглушек)."""
        return True

    def batch_eval(self, atoms_list):
        """Энергии (N,) + силы [список per-atom] для списка систем.

        Дефолт: по одной через ASE-калькулятор — для заглушек/тестов, чтобы
        оркестрацию батча (build configs → prefilter → split BE) можно было
        прогнать на EMT-mock. UMAPotential переопределяет настоящим батч-forward.
        """
        calc = self.get_calculator()
        energies, forces = [], []
        for a in atoms_list:
            c = a.copy()
            c.calc = calc
            energies.append(float(c.get_potential_energy()))
            forces.append(c.get_forces())
        return np.array(energies), forces

    # -- релаксация ---------------------------------------------------------
    def relax(
        self,
        atoms: Atoms,
        fmax: float = RELAX_FMAX_COARSE,
        steps: int = MAX_RELAX_STEPS,
        relax_cell: bool = False,
        stage: str = "relax",
    ) -> RelaxResult:
        atoms = atoms.copy()
        atoms.calc = self.get_calculator()
        if relax_cell:
            from ase.filters import FrechetCellFilter
            target = FrechetCellFilter(atoms)
        else:
            target = atoms
        try:
            with PROFILER.timed(stage):
                opt = FIRE(target, logfile=None)
                converged = opt.run(fmax=fmax, steps=steps)
                energy = float(atoms.get_potential_energy())
            # Если FIRE не сошёлся formaлно — проверяем макс. силу.
            # На сложных стартах от политики FIRE может не дотянуть за 300 шагов,
            # но геометрия уже почти равновесная (max_force < 3·fmax) — это
            # годится для скрининга. Если силы реально большие — invalid.
            if not converged:
                import numpy as _np
                fmax_actual = _np.linalg.norm(atoms.get_forces(), axis=1).max()
                converged_soft = fmax_actual < 3 * fmax
            else:
                converged_soft = True
            # профиль: сколько FIRE-шагов и сошлась ли (отличает «релаксаций
            # много» от «каждая упирается в steps и не сходится»)
            PROFILER.add_relax(stage, opt.get_number_of_steps(), bool(converged))
        except Exception as exc:
            return RelaxResult(None, info={"error": repr(exc)})
        valid = converged_soft and _geometry_valid(atoms)
        return RelaxResult(atoms, energy, bool(converged), valid)

    def relax_bulk(self, atoms: Atoms, stage: str = "bulk_relax", **kw) -> RelaxResult:
        return self.relax(atoms, relax_cell=True, stage=stage, **kw)

    def relax_slab(self, atoms: Atoms, stage: str = "slab_relax", **kw) -> RelaxResult:
        return self.relax(atoms, relax_cell=False, stage=stage, **kw)

    # -- адсорбция ----------------------------------------------------------
    def adsorption_energies(
        self,
        slab: Atoms,
        adsorbate: Atoms,
        e_slab: float,
        e_adsorbate_ref: float,
        max_sites: int = 12,
        fmax: float = RELAX_FMAX_COARSE,
        prefilter: bool = False,
        prefilter_keep: int = 4,
        on_relaxed=None,
        stage_prefix: str = "ads",
    ) -> List[float]:
        """BE по перечисленным сайтам: E(slab+ads) - E(slab) - E_ref(ads).

        Возвращает отсортированный список (минимум = BE на самом стабильном
        сайте, следующий локальный минимум = BE метастабильного сайта для
        BEP-корреляции C-C по Seemakurthi). Сайты из enumerate_sites уже
        на корректной высоте над поверхностью.

        prefilter (УСКОРЕНИЕ #2): сначала дешёвый single-point по ВСЕМ сайтам,
        затем релаксируем только prefilter_keep лучших. Single-point на UMA
        в разы дешевле полной релаксации, а плохие стартовые позиции почти
        никогда не дают глобальный минимум — отсекаем их заранее.
        """
        with PROFILER.timed(f"{stage_prefix}.sites"):
            sites = enumerate_sites(slab, max_sites=max_sites)
        if not sites:
            return []

        configs = []
        for site in sites:
            combined = slab.copy()
            ads = adsorbate.copy()
            ads.translate(np.asarray(site) - ads.positions[0])  # атом 0 -> сайт
            combined += ads
            configs.append(combined)

        if prefilter and len(configs) > prefilter_keep:
            # single-point энергии (без релаксации), берём топ-K по энергии
            with PROFILER.timed(f"{stage_prefix}.prefilter"):
                scored = []
                calc = self.get_calculator()
                for cfg in configs:
                    c = cfg.copy()
                    c.calc = calc
                    try:
                        scored.append((float(c.get_potential_energy()), cfg))
                    except Exception:
                        continue
                scored.sort(key=lambda t: t[0])
                configs = [cfg for _, cfg in scored[:prefilter_keep]]

        energies = []
        for i, combined in enumerate(configs):
            res = self.relax_slab(combined, fmax=fmax, stage=f"{stage_prefix}.relax")
            if res.valid:
                be = res.energy - e_slab - e_adsorbate_ref
                energies.append(be)
                if on_relaxed is not None:
                    # callback для сохранения релаксированной геометрии
                    on_relaxed(i, be, res.atoms)
        return _dedup_minima(energies)


def enumerate_sites(slab: Atoms, max_sites: int = 12,
                    min_supercell_a: float = 6.0) -> List[np.ndarray]:
    """Перечисление адсорбционных сайтов через pymatgen AdsorbateSiteFinder.

    Возвращает декартовы координаты сайтов (ontop / bridge / hollow). Чтобы
    стабильный и метастабильный сайты реально различались, нужно достаточно
    латеральных неэквивалентных позиций: маленькую ячейку расширяем до
    суперъячейки ~min_supercell_a по a/b. Высоту НЕ обнуляем (finder вернёт
    позицию на поверхности, реальную высоту над ней добавляет вызывающий код).

    Падение на экзотических поверхностях не фатально: запасной вариант —
    on-top над атомами верхнего слоя.
    """
    try:
        from pymatgen.analysis.adsorption import AdsorbateSiteFinder
        from pymatgen.io.ase import AseAtomsAdaptor

        struct = AseAtomsAdaptor.get_structure(slab)
        # расширяем латерально, ТОЛЬКО если ячейка реально мала.
        # cut_slab уже расширяет до min_lateral=9Å, так что обычно na=nb=1.
        # Это защита от случаев, когда slab пришёл откуда-то ещё (тесты).
        a, b = struct.lattice.a, struct.lattice.b
        if a < min_supercell_a - 1.0 or b < min_supercell_a - 1.0:
            na = max(1, int(np.ceil(min_supercell_a / a)))
            nb = max(1, int(np.ceil(min_supercell_a / b)))
            # лимит атомов: 150 — потолок из constants.SLAB_MAX_ATOMS
            if len(struct) * na * nb <= 150:
                struct = struct * (na, nb, 1)

        finder = AdsorbateSiteFinder(struct)
        # symm_reduce>0 убирает симметрийные дубли; distance — высота над пов-стью
        found = finder.find_adsorption_sites(
            distance=2.0, symm_reduce=0.01, near_reduce=0.01,
            positions=("ontop", "bridge", "hollow"),
        )
        sites = [np.array(s) for s in found.get("all", [])]
        if sites:
            return sites[:max_sites]
    except Exception:
        pass
    # запасной вариант: on-top над верхним слоем
    z = slab.positions[:, 2]
    top_idx = np.where(z > z.max() - 0.5)[0]
    return [slab.positions[i].copy() + np.array([0, 0, 2.0]) for i in top_idx[:max_sites]]


def _dedup_minima(energies: List[float], tol: float = 0.02) -> List[float]:
    """Сортирует BE и схлопывает близкие (в пределах tol эВ) в один минимум.

    Несколько стартовых сайтов часто релаксируют в одну и ту же яму —
    без дедупликации стабильный и метастабильный BE оказываются почти
    равны, что ломает BEP-дескриптор C-C. Возвращает РАЗЛИЧНЫЕ минимумы
    по возрастанию (energies отрицательны: первый — самый стабильный).
    """
    if not energies:
        return []
    ordered = sorted(energies)
    distinct = [ordered[0]]
    for e in ordered[1:]:
        if e - distinct[-1] > tol:
            distinct.append(e)
    return distinct


def _geometry_valid(atoms: Atoms, scale: float = 0.7) -> bool:
    """Минимальные межатомные расстояния с учётом периодичности.

    Порог для пары (i, j): scale * (r_cov_i + r_cov_j) — глобальный
    металлический порог (1.9 A) ложно резал бы связи адсорбата
    (C-H 1.09 A). Для Pd-Pd даёт ~1.95 A, для C-H ~0.78 A.
    """
    if len(atoms) < 2:
        return True
    from ase.data import covalent_radii

    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    r = covalent_radii[atoms.numbers]
    dmin = scale * (r[:, None] + r[None, :])
    return bool(np.all(d > dmin))

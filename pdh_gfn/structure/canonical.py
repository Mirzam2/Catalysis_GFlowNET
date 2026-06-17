"""Двухуровневая идентичность (вариант 3 из обсуждения дизайна).

Уровень 1 — канонизация на уровне ДЕЙСТВИЙ (до обучения):
    индексы Миллера сводятся к каноническому представителю своего
    симметрийного семейства для данной структуры. Это уменьшает
    дублирование мод в пространстве x ещё до того, как политика его увидит.

Уровень 2 — канонический ХЭШ релаксированной структуры (после среды):
    используется ТОЛЬКО в кэше наград и в подсчёте уникальных мод.
    В математику GFlowNet (P_F, P_B, Z) не входит — терминальный объект x
    остаётся сырым кортежем (SG, состав, LP, hkl, сдвиг, верх/низ).
"""
import hashlib
from typing import Tuple

from pymatgen.core import Structure
from pymatgen.core.surface import get_symmetrically_distinct_miller_indices
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def canonical_miller(structure: Structure, hkl: Tuple[int, int, int],
                     max_index: int = 1) -> Tuple[int, int, int]:
    """Сводит (hkl) к каноническому представителю симметрийного семейства.

    Например, для кубической структуры (010) -> (100), (-1-1-1) -> (111).
    Представитель — лексикографически максимальный элемент семейства,
    что совпадает с конвенцией pymatgen.
    """
    distinct = get_symmetrically_distinct_miller_indices(structure, max_index)
    sga = SpacegroupAnalyzer(structure, symprec=0.1)
    ops = sga.get_point_group_operations(cartesian=False)
    family = set()
    for op in ops:
        rotated = tuple(int(round(i)) for i in op.operate(hkl))
        family.add(rotated)
        family.add(tuple(-i for i in rotated))
    for representative in distinct:
        if tuple(representative) in family:
            return tuple(representative)
    # Эквивалент не найден (например, hkl вне max_index) — лексикографический максимум
    return max(family)


def structure_hash(structure: Structure, symprec: float = 0.1,
                   cutoff: float = 5.0) -> str:
    """Канонический хэш структуры для кэша наград (уровень 2).

    Инвариантен к выбору начала координат и перестановкам позиций Уайкова
    (один и тот же B2 с Pd в углу или в центре — один хэш): отпечаток
    строится по локальным окружениям — для каждого узла сортированный
    список (вид соседа, квантованное расстояние) в радиусе cutoff, затем
    сортировка отпечатков узлов. Две структуры, релаксировавшиеся в одну
    фазу, получают один хэш и одну оценку UMA.
    """
    try:
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        std = sga.get_primitive_standard_structure()
        sg_number = sga.get_space_group_number()
    except Exception:
        std = structure.get_primitive_structure()
        sg_number = 0

    def _q(x: float, step: float) -> int:
        return int(round(x / step))

    site_fps = []
    neighbors = std.get_all_neighbors(cutoff)
    for site, nbrs in zip(std, neighbors):
        env = sorted((n.species_string, _q(n.nn_distance, 0.1)) for n in nbrs)
        site_fps.append((site.species_string, tuple(env)))
    site_fps.sort()

    parts = [
        f"sg{sg_number}",
        std.composition.reduced_formula,
        f"n{len(std)}",
        f"v{_q(std.volume / len(std), 0.5)}",  # объём/атом, бин 0.5 A^3
        str(site_fps),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:20]


def slab_hash(bulk_hash: str, hkl: Tuple[int, int, int], shift_bin: int,
              top: bool) -> str:
    """Хэш поверхностной конфигурации: хэш объёма + грань + бин сдвига + сторона.

    Сдвиг сечения дискретизируется по бинам (по умолчанию вызывающий код
    передаёт int(offset * n_bins)), потому что близкие сдвиги дают одну
    терминацию после релаксации.
    """
    key = f"{bulk_hash}|{hkl}|{shift_bin}|{int(top)}"
    return hashlib.sha256(key.encode()).hexdigest()[:20]

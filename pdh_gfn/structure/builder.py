"""Сборка объёмной структуры из состояния GFlowNet (SG, состав, параметры решётки).

Паттерн взят из converter.py Catalyst GFlowNet (Podina et al. 2025):
ручная инстанциация pyxtal.Lattice через para2matrix вместо Lattice.from_para
(последний случайно падает на малых ячейках).
"""
from typing import Dict, List, Optional

import numpy as np

from pyxtal import pyxtal
from pyxtal.lattice import Lattice, para2matrix
from pyxtal.msg import Comp_CompatibilityError
from pyxtal.symmetry import Group
from pymatgen.core import Structure


def build_bulk(
    space_group: int,
    composition: Dict[int, int],
    lattice_params: tuple,
    n_samples: int = 1,
    seed: Optional[int] = None,
) -> List[Structure]:
    """Генерирует объёмные структуры pymatgen из кристаллографического описания.

    Parameters
    ----------
    space_group : int
        Международный номер пространственной группы.
    composition : dict
        {атомный номер: число атомов в ячейке}, например {46: 1, 30: 1} для PdZn.
    lattice_params : tuple
        (a, b, c, alpha, beta, gamma) — длины в ангстремах, углы в градусах.
    n_samples : int
        Сколько независимых расстановок по позициям Уайкова сгенерировать
        (pyxtal выбирает позиции стохастически).

    Returns
    -------
    list of pymatgen.Structure (может быть пустым, если состав несовместим с SG).
    """
    if seed is not None:
        np.random.seed(seed)

    group = Group(space_group)
    a, b, c, alpha, beta, gamma = lattice_params

    cell_matrix = para2matrix((a, b, c, alpha, beta, gamma), radians=False)
    volume = float(abs(np.linalg.det(cell_matrix)))
    lattice = Lattice(
        ltype=group.lattice_type,
        volume=volume,
        matrix=cell_matrix,
    )

    species = [int(z) for z in composition.keys()]
    num_ions = [int(n) for n in composition.values()]

    structures = []
    for _ in range(n_samples):
        xtal = pyxtal()
        try:
            xtal.from_random(
                dim=3,
                group=group,
                species=species,
                numIons=num_ions,
                lattice=lattice,
            )
        except (Comp_CompatibilityError, RuntimeError, ValueError):
            continue
        if not xtal.valid:
            continue
        structures.append(xtal.to_pymatgen())
    return structures


def is_composition_compatible(space_group: int, num_ions: List[int]) -> bool:
    """Проверка совместимости состава с позициями Уайкова группы (constraint C2)."""
    group = Group(space_group)
    ok, _ = group.check_compatible(num_ions)
    return bool(ok)

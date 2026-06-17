"""Нарезка поверхностной ячейки из объёмной структуры.

Действия GFlowNet (hkl, сдвиг сечения, верх/низ) -> ASE Atoms с вакуумом
и замороженными нижними слоями. Паттерн SlabGenerator — из converter.py
Catalyst GFlowNet, адаптирован: вместо ocdata.Surface используем pymatgen
напрямую (нам не нужна совместимость с OCP-датасетом).
"""
from typing import Optional, Tuple

import numpy as np

from ase import Atoms
from ase.constraints import FixAtoms
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor

from ..constants import (
    SLAB_FIXED_BOTTOM_LAYERS,
    SLAB_MAX_ATOMS,
    SLAB_MIN_THICKNESS,
    SLAB_VACUUM,
)


def cut_slab(
    bulk: Structure,
    hkl: Tuple[int, int, int],
    offset: float,
    top: bool,
    min_thickness: float = SLAB_MIN_THICKNESS,
    vacuum: float = SLAB_VACUUM,
    min_lateral: float = 9.0,
) -> Optional[Atoms]:
    """Строит поверхностную ячейку.

    Parameters
    ----------
    offset : float in [0, 1]
        Непрерывный сдвиг сечения вдоль нормали к (hkl). Отображается на
        ближайший из терминационных сдвигов, которые pymatgen находит
        автоматически, с дробным дорезанием между ними. Для интерметаллидов
        это и есть выбор терминации (Pd-слой / M-слой / смешанный).
    top : bool
        Использовать верхнюю или нижнюю поверхность сечения (для структур
        без центра инверсии они различны).
    min_lateral : float
        Минимальный латеральный размер slab (ангстремы). Если a или b меньше,
        делается суперъячейка. Критично для C3H7 (~4 Å): на ячейке 2.75 Å
        адсорбат взаимодействует с периодическими образами и BE смещается
        на 5+ эВ. Дефолт 7.0 Å безопасен для всех наших адсорбатов.
    """
    gen = SlabGenerator(
        bulk,
        miller_index=hkl,
        min_slab_size=min_thickness,
        min_vacuum_size=vacuum,
        center_slab=True,
        in_unit_planes=False,
        primitive=True,
    )
    try:
        # Симметрийно-различные терминации; offset in [0,1] индексирует их.
        # Для интерметаллидов это выбор Pd-/M-/смешанной терминации.
        try:
            terminations = gen.get_slabs(filter_out_sym_slabs=True)
        except TypeError:  # pymatgen < 2024.6: аргумента нет, фильтрация по умолчанию
            terminations = gen.get_slabs()
    except Exception:
        return None
    if not terminations:
        return None
    idx = int(round(offset * (len(terminations) - 1))) if len(terminations) > 1 else 0
    slab = terminations[idx]

    # Расширяем латерально, если ячейка мала (C3H7 ~4 Å не помещается в 2.75 Å)
    a, b = slab.lattice.a, slab.lattice.b
    na = max(1, int(np.ceil(min_lateral / a)))
    nb = max(1, int(np.ceil(min_lateral / b)))
    if na > 1 or nb > 1:
        slab = slab * (na, nb, 1)

    if not top:
        # нижняя поверхность: отражаем ячейку по c
        coords = slab.frac_coords.copy()
        coords[:, 2] = 1.0 - coords[:, 2]
        slab = Structure(slab.lattice, slab.species, coords)

    if len(slab) > SLAB_MAX_ATOMS:
        return None

    atoms = AseAtomsAdaptor.get_atoms(slab)
    _freeze_bottom_layers(atoms, n_layers=SLAB_FIXED_BOTTOM_LAYERS)
    return atoms


def _freeze_bottom_layers(atoms: Atoms, n_layers: int, tol: float = 0.5) -> None:
    """Замораживает n нижних атомных слоёв (кластеризация по z с допуском tol, ангстремы)."""
    z = atoms.positions[:, 2]
    order = np.argsort(z)
    layers, current = [], [order[0]]
    for i in order[1:]:
        if z[i] - z[current[-1]] < tol:
            current.append(i)
        else:
            layers.append(current)
            current = [i]
    layers.append(current)
    frozen = [i for layer in layers[:n_layers] for i in layer]
    atoms.set_constraint(FixAtoms(indices=frozen))

"""Конвертер: структура (CIF/xyz/ASE Atoms) -> CatalystState.

Холодный старт: даём 5 известных хороших катализаторов (PdZn, PdIn, PdGa,
Pd3Sn, Pd2Ga), считаем их награду полным конвейером и записываем в
reward_cache.jsonl. Когда политика GFlowNet случайно генерирует структуру
с тем же SG + составом + близкой решёткой, хэш в кэше совпадает и она
сразу получает ненулевую награду — это и есть warm-start без изменения
GFlowNet-математики (только наполнение кэша наград).

Использование:
    from pdh_gfn.structure.from_xyz import structure_to_catalyst_states
    states = structure_to_catalyst_states(
        "PdZn.cif", millers=[(1,1,1), (1,0,0)], offsets=[0.0])
    for s in states:
        pipeline(s)   # это запишет в кэш
"""
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from ase import Atoms
from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ..reward.pipeline import CatalystState


def read_any(path: str) -> Atoms:
    """Читает структуру из любого файла, который понимает ASE (xyz, cif, vasp...)."""
    return ase_read(str(path))


def atoms_to_catalyst_state(
    atoms: Atoms,
    miller: Tuple[int, int, int],
    offset: float = 0.0,
    top: bool = True,
    symprec: float = 0.1,
) -> CatalystState:
    """ASE Atoms (объёмная фаза) -> CatalystState.

    Извлекает SG через SpacegroupAnalyzer, состав как {Z: n}, параметры
    решётки из lattice. Грань/сдвиг/сторону задаёт пользователь.
    """
    struct = AseAtomsAdaptor.get_structure(atoms)
    sga = SpacegroupAnalyzer(struct, symprec=symprec)
    sg = sga.get_space_group_number()

    composition = {}
    for site in struct:
        z = site.specie.Z
        composition[z] = composition.get(z, 0) + 1

    lat = struct.lattice
    lattice_params = (lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma)

    return CatalystState(
        space_group=sg,
        composition=composition,
        lattice_params=lattice_params,
        miller=miller,
        offset=float(offset),
        top=bool(top),
    )


def structure_to_catalyst_states(
    path: str,
    millers: Iterable[Tuple[int, int, int]] = ((1, 1, 1), (1, 0, 0), (1, 1, 0)),
    offsets: Iterable[float] = (0.0,),
    tops: Iterable[bool] = (True,),
    symprec: float = 0.1,
) -> List[CatalystState]:
    """Читает структуру и порождает CatalystState для всех (грань, сдвиг, сторона)."""
    atoms = read_any(path)
    states = []
    for hkl in millers:
        for off in offsets:
            for top in tops:
                states.append(atoms_to_catalyst_state(
                    atoms, hkl, off, top, symprec=symprec))
    return states

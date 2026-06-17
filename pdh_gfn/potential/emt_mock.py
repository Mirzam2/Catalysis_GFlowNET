"""EMT-заглушка для тестов конвейера без GPU и весов UMA.

ASE EMT покрывает Ni, Cu, Pd, Ag, Pt, Au (+ C, H приближённо), то есть
системы Pd-Cu / Pd-Ni / Pd-Ag / Pd-Au и адсорбат CH* считаются "из коробки".
Числа физически грубые — заглушка проверяет ТОЛЬКО механику конвейера
(сборка -> релаксация -> сайты -> BE -> дескрипторы -> награда), не химию.
"""
from ase import Atoms
from ase.calculators.emt import EMT

from .base import Potential

_EMT_ELEMENTS = {"Ni", "Cu", "Pd", "Ag", "Pt", "Au", "Al", "C", "H", "N", "O"}


class EMTMockPotential(Potential):
    def get_calculator(self):
        return EMT()

    def supports(self, atoms: Atoms) -> bool:
        return set(atoms.get_chemical_symbols()) <= _EMT_ELEMENTS

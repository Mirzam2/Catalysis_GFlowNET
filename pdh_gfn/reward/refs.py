"""Газофазные референсы адсорбатов — схема Seemakurthi 2025, уравнение (2).

BE(CxHy*) = E(slab+CxHy*) - E(slab) - E(CxH(2x+2)) + (2x+2-y)/2 * E(H2)

То есть референс, вычитаемый из (E_slab_ads - E_slab):
    E_ref(CxHy) = E(CxH(2x+2)) - (2x+2-y)/2 * E(H2)

Нужны энергии трёх газофазных молекул, посчитанные ТЕМ ЖЕ потенциалом
(UMA) и, для согласованности с поверхностными расчётами Seemakurthi,
в том же режиме (task). Их считаем один раз на старте обучения.

Наши адсорбаты:
    H*     : x=0, y=1  -> источник H2,   баланс 1/2  -> E_ref = 1/2 E(H2)
    CH*    : x=1, y=1  -> источник CH4,   баланс 3/2  -> E_ref = E(CH4) - 3/2 E(H2)
    C3H7*  : x=3, y=7  -> источник C3H8,  баланс 1/2  -> E_ref = E(C3H8) - 1/2 E(H2)
"""
from typing import Dict

from ase import Atoms
from ase.build import molecule

from ..potential.base import Potential

# (x, y) для каждого адсорбата
ADSORBATE_XY = {
    "H": (0, 1),
    "CH": (1, 1),
    "C3H7": (3, 7),
}


def _gas_molecules() -> Dict[str, Atoms]:
    """H2, CH4, C3H8 через ASE g2 (все три присутствуют)."""
    return {
        "H2": molecule("H2"),
        "CH4": molecule("CH4"),
        "C3H8": molecule("C3H8"),
    }


def compute_adsorbate_refs(
    potential: Potential,
    fmax: float = 0.01,
    box: float = 15.0,
    task_molecule: str = None,
) -> Dict[str, float]:
    """Считает E_ref(ads) для H, CH, C3H7 по схеме Seemakurthi.

    Молекулы релаксируются в большой ячейке с PBC=False. Возвращает словарь
    {"H": E_ref, "CH": E_ref, "C3H7": E_ref}, готовый для
    RewardPipeline(adsorbate_refs=...).
    """
    # энергии газофазных молекул
    e_gas: Dict[str, float] = {}
    for name, mol in _gas_molecules().items():
        mol = mol.copy()
        mol.center(vacuum=box / 2)
        mol.pbc = False
        res = potential.relax(mol, fmax=fmax, relax_cell=False)
        if not res.converged:
            raise RuntimeError(f"релаксация {name} не сошлась")
        e_gas[name] = res.energy

    e_h2 = e_gas["H2"]
    source = {"H": "H2", "CH": "CH4", "C3H7": "C3H8"}
    refs: Dict[str, float] = {}
    for ads, (x, y) in ADSORBATE_XY.items():
        balance = (2 * x + 2 - y) / 2.0
        if x == 0:
            # источник — сам H2: E(CxH2x+2)=E(H2), даёт E_ref = (1-balance)?
            # по формуле: E_ref = E(H2) - balance*E(H2) = (1-balance)*E(H2)
            refs[ads] = (1.0 - balance) * e_h2
        else:
            refs[ads] = e_gas[source[ads]] - balance * e_h2
    return refs

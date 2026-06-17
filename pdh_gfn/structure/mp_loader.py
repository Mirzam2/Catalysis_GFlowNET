"""Загрузка эталонных структур из Materials Project по mp-id.

Для калибровки и затравки буфера нужны НАСТОЯЩИЕ релаксированные структуры
(экспериментальные/DFT-параметры решётки и позиции), а не идеальные ячейки
из захардкоженных параметров. MP даёт именно их.

Известные mp-id для эталонных фаз (проверить актуальность на старте —
MP иногда меняет id; есть фолбэк по химической формуле + SG).

Использование (логин-узел, нужен MP_API_KEY):
    from pdh_gfn.structure.mp_loader import load_reference_structures
    structs = load_reference_structures(["mp-2", "mp-1018100"])
"""
import os
from typing import Dict, List, Optional

from pymatgen.core import Structure

# Эталонные фазы: label -> mp-id (проверить на старте; см. поиск-фолбэк ниже)
REFERENCE_MPIDS = {
    "Pd(fcc)": "mp-2",
    "PdZn": "mp-1018100",
    "PdGa": "mp-12768",
    "PdIn": "mp-21503",
    "Pd3Sn": "mp-865925",
}


def load_structure_by_id(mpid: str, api_key: Optional[str] = None) -> Structure:
    from mp_api.client import MPRester
    key = api_key or os.environ["MP_API_KEY"]
    with MPRester(key) as mpr:
        return mpr.get_structure_by_material_id(mpid)


def search_structure(formula: str, spacegroup: Optional[int] = None,
                     api_key: Optional[str] = None) -> Optional[Structure]:
    """Фолбэк: ищем по формуле (и SG), берём самую стабильную (min e_above_hull)."""
    from mp_api.client import MPRester
    key = api_key or os.environ["MP_API_KEY"]
    with MPRester(key) as mpr:
        docs = mpr.materials.summary.search(
            formula=formula,
            fields=["material_id", "structure", "symmetry", "energy_above_hull"],
        )
    if spacegroup is not None:
        docs = [d for d in docs if d.symmetry.number == spacegroup] or docs
    if not docs:
        return None
    best = min(docs, key=lambda d: d.energy_above_hull)
    return best.structure


def load_reference_structures(
    labels: Optional[List[str]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Structure]:
    """Возвращает {label: Structure} для эталонных фаз.

    Сначала пробует mp-id, при неудаче — поиск по формуле. Результат стоит
    закэшировать на диск (CIF/JSON), чтобы не дёргать MP на каждом запуске.
    """
    labels = labels or list(REFERENCE_MPIDS.keys())
    out: Dict[str, Structure] = {}
    for label in labels:
        mpid = REFERENCE_MPIDS.get(label)
        try:
            out[label] = load_structure_by_id(mpid, api_key)
        except Exception:
            # фолбэк по формуле из label (грубо: до скобки)
            formula = label.split("(")[0]
            s = search_structure(formula, api_key=api_key)
            if s is not None:
                out[label] = s
    return out
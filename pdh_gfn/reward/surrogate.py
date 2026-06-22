"""Surrogate-оценщик BE по геометрии сайта релаксированного слэба (вариант B).

Назначение — ПРЕ-ФИЛТР, не замена UMA: после реальной релаксации слэба
предсказывает BE(H/CH/C3H7) по геометрии сайтов, чтобы решить, гнать ли
дорогую релаксацию адсорбатов. Точность (MAE ~0.5-0.75 эВ) хватает на
роутинг, не на финальный reward — пограничные/uncertain случаи уходят на UMA.

Замер предсказуемости: scripts/surrogate_b_probe.py (R²: H 0.46, CH 0.64,
C3H7 0.43 на ручных дескрипторах). Здесь те же дескрипторы, но якорь сайта
передаётся ЯВНО (позиция из enumerate_sites), чтобы train и inference
использовали одинаковую featurization.

Зависимости намеренно лёгкие: numpy + sklearn. ase/pymatgen НЕ импортируются
на уровне модуля — Atoms приходят снаружи как (cell, Z, pos).
"""
from __future__ import annotations

import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

ADSORBATES = ("H", "CH", "C3H7")

# Pauling EN и ковалентный радиус (Å): Pd + M_ELEMENTS (constants.M_ELEMENTS)
EN = {46: 2.20, 22: 1.54, 23: 1.63, 25: 1.55, 26: 1.83, 27: 1.88, 28: 1.91,
      29: 1.90, 30: 1.65, 31: 1.81, 32: 2.01, 49: 1.78, 50: 1.96, 51: 2.05,
      82: 2.33, 83: 2.02, 40: 1.33, 41: 1.60, 42: 2.16, 47: 1.93, 72: 1.30,
      73: 1.50, 74: 2.36, 75: 1.90, 79: 2.54}
RCOV = {46: 1.39, 22: 1.60, 23: 1.53, 25: 1.39, 26: 1.32, 27: 1.26, 28: 1.24,
        29: 1.32, 30: 1.22, 31: 1.22, 32: 1.20, 49: 1.42, 50: 1.39, 51: 1.39,
        82: 1.46, 83: 1.48, 40: 1.75, 41: 1.64, 42: 1.54, 47: 1.45, 72: 1.75,
        73: 1.70, 74: 1.62, 75: 1.51, 79: 1.36}
PD_Z = 46
METALS = np.array(sorted(EN))

R_BIND = 2.9
R_CN = 3.2
CN_MAX = 12.0

FEAT_NAMES = ("gcn", "n_bind", "d_mean", "d_min", "en_mean", "en_min",
              "en_max", "rcov_mean", "frac_pd_bind", "frac_pd_surf",
              "height", "e_hull")


def _lateral_shifts(cell: np.ndarray) -> np.ndarray:
    """Декартовы сдвиги ±1 по a,b (вакуум по c → без сдвигов)."""
    out = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            out.append(i * cell[0] + j * cell[1])
    return np.asarray(out)


def _min_dist(p: np.ndarray, others: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Расстояния от p до others с учётом латеральной PBC (минимальный образ)."""
    if len(others) == 0:
        return np.full(0, np.inf)
    cand = others[:, None, :] + shifts[None, :, :]
    return np.linalg.norm(cand - p[None, None, :], axis=2).min(axis=1)


def site_features(cell: np.ndarray, Z: np.ndarray, pos: np.ndarray,
                  anchor: np.ndarray, e_hull: Optional[float]) -> Optional[List[float]]:
    """Дескрипторы адсорбционного сайта по геометрии МЕТАЛЛА слэба.

    anchor — декартова позиция сайта (из enumerate_sites). Берём только
    металлические атомы; e_hull прокидывается как глобальная фича.
    Возвращает None, если окружение вырождено.
    """
    Z = np.asarray(Z)
    pos = np.asarray(pos, float)
    metal_mask = np.isin(Z, METALS)
    mZ, mpos = Z[metal_mask], pos[metal_mask]
    if len(mpos) < 4:
        return None

    shifts = _lateral_shifts(cell)
    zmax = mpos[:, 2].max()
    surf = mpos[:, 2] > zmax - 2.5
    surf_Z = mZ[surf]

    d_anchor = _min_dist(np.asarray(anchor, float), mpos, shifts)
    bind = d_anchor < R_BIND
    if bind.sum() == 0:
        bind = d_anchor < R_BIND + 0.6
    if bind.sum() == 0:
        return None
    bZ = mZ[bind]
    bd = d_anchor[bind]

    gcn = 0.0
    for k in np.where(bind)[0]:
        dk = _min_dist(mpos[k], mpos, shifts)
        cn = int(((dk > 0.1) & (dk < R_CN)).sum())
        gcn += cn / CN_MAX

    en_b = np.array([EN[int(z)] for z in bZ])
    rc_b = np.array([RCOV[int(z)] for z in bZ])
    return [
        gcn,
        float(bind.sum()),
        float(bd.mean()), float(bd.min()),
        float(en_b.mean()), float(en_b.min()), float(en_b.max()),
        float(rc_b.mean()),
        float(np.mean(bZ == PD_Z)),
        float(np.mean(surf_Z == PD_Z)) if len(surf_Z) else 0.0,
        float(anchor[2] - mpos[surf, 2].mean()) if surf.any() else 0.0,
        float(e_hull) if e_hull is not None else 0.0,
    ]


# --- SOAP пониженной размерности (DScribe, compression mu2) ---------------
# mu2 сворачивает оба видовых канала → размерность НЕ зависит от числа
# элементов (~105 при n_max=6,l_max=4). Геометрия окружения; химию несут
# ручные EN/радиус-фичи. Гибрид руки+SOAP даёт +0.02..0.04 R² (замер
# surrogate_b_probe.py). dscribe/ase импортируются ЛЕНИВО.
_SOAP = None


def _get_soap():
    global _SOAP
    if _SOAP is None:
        from ase.data import chemical_symbols
        from dscribe.descriptors import SOAP
        species = [chemical_symbols[z] for z in sorted(EN)]
        _SOAP = SOAP(species=species, r_cut=5.0, n_max=6, l_max=4,
                     periodic=True, compression={"mode": "mu2"})
    return _SOAP


def soap_vector(cell, Z, pos, anchor) -> Optional[np.ndarray]:
    """SOAP(mu2) в точке anchor по металлическим атомам слэба (без адсорбата)."""
    from ase import Atoms
    Z = np.asarray(Z); pos = np.asarray(pos, float)
    mask = np.isin(Z, METALS)
    if mask.sum() < 4:
        return None
    at = Atoms(numbers=Z[mask], positions=pos[mask],
               cell=np.asarray(cell, float), pbc=[True, True, False])
    try:
        return _get_soap().create(at, centers=[np.asarray(anchor, float)])[0]
    except Exception:
        return None


def full_features(cell, Z, pos, anchor, e_hull, use_soap: bool):
    """Полный вектор фич сайта: ручные (12) [+ SOAP(105)]. None при вырождении."""
    hand = site_features(cell, Z, pos, anchor, e_hull)
    if hand is None:
        return None
    if not use_soap:
        return hand
    sv = soap_vector(cell, Z, pos, anchor)
    if sv is None:
        return None
    return list(hand) + list(sv)


class SurrogateBE:
    """Ансамбль RF по адсорбатам: сайт-фичи → BE. Uncertainty = разброс по
    деревьям агрегированного (min по сайтам) BE. use_soap — обучен ли с SOAP
    (тогда фичи строятся через full_features с тем же флагом)."""

    def __init__(self, models: Optional[Dict[str, object]] = None,
                 use_soap: bool = False):
        self.models: Dict[str, object] = models or {}
        self.use_soap = use_soap

    # -- обучение (офлайн) ------------------------------------------------
    def fit(self, data: Dict[str, Tuple[np.ndarray, np.ndarray]],
            n_estimators: int = 300, random_state: int = 0) -> "SurrogateBE":
        """data[name] = (X[n_sites, n_feat], y[n_sites]) — пер-сайтовые метки."""
        from sklearn.ensemble import RandomForestRegressor
        for name, (X, y) in data.items():
            m = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1,
                                      random_state=random_state)
            m.fit(np.asarray(X, float), np.asarray(y, float))
            self.models[name] = m
        return self

    # -- предсказание по слэбу -------------------------------------------
    def predict_slab(self, site_feats: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """site_feats[n_sites, n_feat] для сайтов этого слэба.
        Возвращает {name: (best_be, uncertainty)}.

        best_be — min(среднее по лесу) по сайтам (самый стабильный сайт).
        uncertainty — std по деревьям величины min-по-сайтам (разброс модели
        на агрегате, который реально используется).
        """
        X = np.asarray(site_feats, float)
        out: Dict[str, Tuple[float, float]] = {}
        for name, m in self.models.items():
            mean_per_site = m.predict(X)                       # (n_sites,)
            best = float(mean_per_site.min())
            # по каждому дереву: min по сайтам → разброс агрегата
            per_tree_best = [float(est.predict(X).min())
                             for est in m.estimators_]
            unc = float(np.std(per_tree_best))
            out[name] = (best, unc)
        return out

    def predict_ch_meta(self, site_feats: np.ndarray, tol: float = 0.02
                        ) -> Optional[float]:
        """2-й различимый минимум CH (нужен для дескриптора C-C/E_sel)."""
        if "CH" not in self.models:
            return None
        vals = np.sort(self.models["CH"].predict(np.asarray(site_feats, float)))
        distinct = [vals[0]]
        for v in vals[1:]:
            if v - distinct[-1] > tol:
                distinct.append(v)
            if len(distinct) >= 2:
                return float(distinct[1])
        return None

    # -- персистентность --------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"models": self.models, "use_soap": self.use_soap}, f)

    @classmethod
    def load(cls, path: str) -> "SurrogateBE":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "models" in obj:
            return cls(obj["models"], use_soap=obj.get("use_soap", False))
        return cls(obj)  # обратная совместимость со старым форматом (только models)

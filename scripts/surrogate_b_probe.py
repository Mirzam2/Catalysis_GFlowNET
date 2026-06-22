"""Вариант B: замер предсказуемости BE по геометрии сайта релаксированного слэба.

В отличие от surrogate_probe.py (досляб-фичи, A), здесь на входе геометрия
сайта адсорбции на релаксированной поверхности — то самое, чего не хватало A.
Метки берём из сохранённых ads_{name}_{i}.xyz (per-site BE в заголовке).

Дескрипторы сайта (numpy, без dscribe/pymatgen): GCN, координация,
расстояния/состав окружения, электроотрицательности/радиусы соседей.

Метрика go/no-go: R² по ЛУЧШЕМУ BE на слэб (min по сайтам) — именно он
входит в дескрипторы Seemakurthi. GroupKFold по слэбу (без утечки сайтов).

Запуск:
    C:/MinGW/miniforge3/envs/hacaton_env/python.exe scripts/surrogate_b_probe.py
"""
import glob
import json
import os
import re

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "structures")
ADSORBATES = ["H", "CH", "C3H7"]
BE_ABS = {"H": 5.0, "CH": 6.0, "C3H7": 5.0}  # фильтр абсурдных BE как в pipeline

# Pauling EN и ковалентный радиус (Å) для Pd + M_ELEMENTS
EN = {46: 2.20, 22: 1.54, 23: 1.63, 25: 1.55, 26: 1.83, 27: 1.88, 28: 1.91,
      29: 1.90, 30: 1.65, 31: 1.81, 32: 2.01, 49: 1.78, 50: 1.96, 51: 2.05,
      82: 2.33, 83: 2.02, 40: 1.33, 41: 1.60, 42: 2.16, 47: 1.93, 72: 1.30,
      73: 1.50, 74: 2.36, 75: 1.90, 79: 2.54}
RCOV = {46: 1.39, 22: 1.60, 23: 1.53, 25: 1.39, 26: 1.32, 27: 1.26, 28: 1.24,
        29: 1.32, 30: 1.22, 31: 1.22, 32: 1.20, 49: 1.42, 50: 1.39, 51: 1.39,
        82: 1.46, 83: 1.48, 40: 1.75, 41: 1.64, 42: 1.54, 47: 1.45, 72: 1.75,
        73: 1.70, 74: 1.62, 75: 1.51, 79: 1.36}
PD_Z = 46
METALS = set(EN)
SYM2Z = {"Pd": 46, "Ti": 22, "V": 23, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28,
         "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "In": 49, "Sn": 50, "Sb": 51,
         "Pb": 82, "Bi": 83, "Zr": 40, "Nb": 41, "Mo": 42, "Ag": 47, "Hf": 72,
         "Ta": 73, "W": 74, "Re": 75, "Au": 79, "H": 1, "C": 6}

R_BIND = 2.9   # связывающие соседи: металл в этом радиусе от якоря
R_CN = 3.2     # для координационного числа металл-металл
CN_MAX = 12.0

# --- SOAP пониженной размерности (DScribe, compression mu2) ---------------
# mu2 сворачивает оба видовых канала → размерность НЕ зависит от числа
# элементов (~105 при n_max=6,l_max=4). Чистая геометрия окружения; химию
# элементов несут ручные EN/радиус-фичи. Центр SOAP = якорь сайта, по МЕТАЛЛУ
# слэба (без адсорбата — как на инференсе по голому слэбу).
from ase import Atoms as _Atoms
from ase.data import chemical_symbols as _CHEM
from dscribe.descriptors import SOAP as _SOAP_CLS

_SPECIES = [_CHEM[z] for z in sorted(METALS)]
_SOAP = _SOAP_CLS(species=_SPECIES, r_cut=5.0, n_max=6, l_max=4,
                  periodic=True, compression={"mode": "mu2"})


def compute_soap(cell, Z, pos, anchor):
    """SOAP(mu2) в точке anchor по металлическим атомам слэба."""
    Z = np.asarray(Z); pos = np.asarray(pos, float)
    mask = np.isin(Z, list(METALS))
    if mask.sum() < 4:
        return None
    at = _Atoms(numbers=Z[mask], positions=pos[mask], cell=cell,
                pbc=[True, True, False])
    try:
        return _SOAP.create(at, centers=[np.asarray(anchor, float)])[0]
    except Exception:
        return None


def parse_xyz(path):
    """Минимальный extxyz: cell(3x3), заголовочные key=val, Z[], pos[]."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    nat = int(lines[0])
    head = lines[1]
    mlat = re.search(r'Lattice="([^"]+)"', head)
    cell = np.array(mlat.group(1).split(), float).reshape(3, 3)
    info = dict(re.findall(r'(\w+)=(-?[\d.eE+]+)', head))
    Z, pos = [], []
    for ln in lines[2:2 + nat]:
        p = ln.split()
        Z.append(SYM2Z.get(p[0], 0))
        pos.append([float(p[1]), float(p[2]), float(p[3])])
    return cell, np.array(Z), np.array(pos, float), info


def lateral_images(pos, cell):
    """Декартовы сдвиги ±1 по a,b (вакуум по c — без сдвигов)."""
    shifts = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            shifts.append(i * cell[0] + j * cell[1])
    return np.array(shifts)  # (9,3)


def min_dist(p, others, shifts):
    """Мин. расстояние от точки p до набора others с учётом лат. PBC."""
    if len(others) == 0:
        return np.full(0, np.inf)
    # (n_others, 9, 3)
    cand = others[:, None, :] + shifts[None, :, :]
    d = np.linalg.norm(cand - p[None, None, :], axis=2)
    return d.min(axis=1)  # (n_others,)


def site_features(cell, Z, pos, adsorbate, e_hull):
    """Дескрипторы сайта: якорь адсорбата → окружение на металле слэба."""
    is_ads = np.isin(Z, [1, 6])
    metal_mask = np.isin(Z, list(METALS))
    mZ, mpos = Z[metal_mask], pos[metal_mask]
    if len(mpos) < 4 or not is_ads.any():
        return None
    # якорь: для H — сам H; иначе — C ближайший к поверхности (min z среди C)
    if adsorbate == "H":
        a_idx = np.where(Z == 1)[0]
    else:
        a_idx = np.where(Z == 6)[0]
    if len(a_idx) == 0:
        return None
    anchor = pos[a_idx][np.argmin(pos[a_idx][:, 2])]

    shifts = lateral_images(mpos, cell)
    zmax = mpos[:, 2].max()
    surf = mpos[:, 2] > zmax - 2.5
    surf_Z = mZ[surf]

    d_anchor = min_dist(anchor, mpos, shifts)
    bind = d_anchor < R_BIND
    if bind.sum() == 0:                      # подняли порог, если ничего не нашли
        bind = d_anchor < R_BIND + 0.6
    if bind.sum() == 0:
        return None
    bZ = mZ[bind]
    bd = d_anchor[bind]

    # CN каждого связывающего металла (металл-металл соседи в R_CN)
    gcn = 0.0
    for k in np.where(bind)[0]:
        dk = min_dist(mpos[k], mpos, shifts)
        cn = int(((dk > 0.1) & (dk < R_CN)).sum())
        gcn += cn / CN_MAX

    en_b = np.array([EN[z] for z in bZ])
    rc_b = np.array([RCOV[z] for z in bZ])
    feat = [
        gcn,
        float(bind.sum()),
        float(bd.mean()), float(bd.min()),
        float(en_b.mean()), float(en_b.min()), float(en_b.max()),
        float(rc_b.mean()),
        float(np.mean(bZ == PD_Z)),                       # доля Pd среди соседей
        float(np.mean(surf_Z == PD_Z)) if len(surf_Z) else 0.0,
        float(anchor[2] - mpos[surf, 2].mean()) if surf.any() else 0.0,  # высота
        float(e_hull) if e_hull is not None else 0.0,
    ]
    vals, counts = np.unique(bZ, return_counts=True)
    dom_z = int(vals[np.argmax(counts)])     # доминирующий элемент-сосед (для cat)
    return feat, dom_z


def load_adsorbate(name):
    cache = os.path.join(ROOT, f"_surr_probe2_{name}.npz")  # v2: +SOAP
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d["X"], d["y"], d["groups"], d["cat"], d["S"]
    X, y, groups, cat, S = [], [], [], [], []
    for path in glob.glob(os.path.join(ROOT, "**", f"ads_{name}_*.xyz"),
                          recursive=True):
        try:
            cell, Z, pos, info = parse_xyz(path)
            be = float(info["BE"])
        except Exception:
            continue
        if abs(be) > BE_ABS[name]:
            continue
        # тот же якорь, что у ручных фич (нижний C / H релакс. адсорбата)
        is_ads = np.where(Z == (1 if name == "H" else 6))[0]
        if len(is_ads) == 0:
            continue
        anchor = pos[is_ads][np.argmin(pos[is_ads][:, 2])]
        soap = compute_soap(cell, Z, pos, anchor)
        if soap is None:
            continue
        surf_dir = os.path.dirname(path)
        e_hull = None
        mpath = os.path.join(surf_dir, "manifest.json")
        if os.path.exists(mpath):
            try:
                e_hull = json.load(open(mpath, encoding="utf-8")).get("e_hull")
            except Exception:
                pass
        res = site_features(cell, Z, pos, name, e_hull)
        if res is None:
            continue
        feat, dom_z = res
        X.append(feat); y.append(be); groups.append(surf_dir)
        cat.append(dom_z); S.append(soap)
    X = np.array(X, float); y = np.array(y, float)
    groups = np.array(groups); cat = np.array(cat, int); S = np.array(S, float)
    np.savez(cache, X=X, y=y, groups=groups, cat=cat, S=S)
    return X, y, groups, cat, S


FEAT_NAMES = ["gcn", "n_bind", "d_mean", "d_min", "en_mean", "en_min",
              "en_max", "rcov_mean", "frac_pd_bind", "frac_pd_surf",
              "height", "e_hull"]


def _scores(y, pred, groups):
    r2s = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    bt, bp = {}, {}
    for g, yt, yp in zip(groups, y, pred):
        bt[g] = min(bt.get(g, yt), yt)
        bp[g] = min(bp.get(g, yp), yp)
    bt = np.array([bt[g] for g in bt])
    bp = np.array([bp[g] for g in bp])
    r2b = 1 - np.sum((bt - bp) ** 2) / np.sum((bt - bt.mean()) ** 2)
    return r2s, r2b, np.mean(np.abs(bt - bp))


def report(name, X, y, groups, cat, S):
    n_slab = len(set(groups))
    print(f"\n=== {name}: {len(y)} сайтов, {n_slab} слэбов, "
          f"SOAP dim={S.shape[1]} ===")
    if len(y) < 50 or n_slab < 5:
        print("мало данных"); return
    cv = GroupKFold(n_splits=min(5, n_slab))
    print(f"{'модель':>20} {'R2(site)':>9} {'R2(best)':>9} {'MAE(best)':>10}")

    def rf_run(label, M):
        rf = RandomForestRegressor(n_estimators=300, n_jobs=4, random_state=0)
        p = cross_val_predict(rf, M, y, groups=groups, cv=cv)
        print("{:>20} {:9.3f} {:9.3f} {:10.3f}".format(label, *_scores(y, p, groups)))

    rf_run("руки(12)", X)                          # текущая (baseline)
    rf_run("SOAP(mu2)", S)                          # только геометрия SOAP
    rf_run("руки+SOAP", np.hstack([X, S]))          # гибрид


def main():
    for name in ADSORBATES:
        X, y, groups, cat, S = load_adsorbate(name)
        report(name, X, y, groups, cat, S)


if __name__ == "__main__":
    main()

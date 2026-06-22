"""Замер предсказуемости BE для surrogate-оценщика (go/no-go).

Читает все manifest.json из data/structures, строит фичи из
[состав, релакс. решётка, e_hull, space_group, грань, offset, top]
и честным GroupKFold (группа = bulk_key) меряет, насколько предсказуемы
BE(H), BE(CH), BE(CH_meta), BE(C3H7) ансамблем деревьев.

Группировка по bulk_key обязательна: разные грани одного объёма
сильно коррелируют, без группировки R² будет оптимистично завышен.

Запуск (env с sklearn):
    C:/MinGW/miniforge3/envs/hacaton_env/python.exe scripts/surrogate_probe.py
"""
import glob
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "structures")
TARGETS = ["H", "CH", "CH_meta", "C3H7"]
BE_ABS_MAX = 8.0  # отсев абсурдных BE (как фильтр в pipeline.py)


def load_rows():
    """Каждый manifest -> (state, e_hull, BE, bulk_key). Рекурсивно по ROOT
    (выкачка с кластера лежит во вложенной structures/structures/...).
    Дедуп по (bulk_key, miller, offset, top)."""
    rows = []
    seen = set()
    for path in glob.glob(os.path.join(ROOT, "**", "manifest.json"),
                          recursive=True):
        try:
            m = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        be = m.get("BE") or {}
        if any(be.get(t) is None for t in TARGETS):
            continue
        if any(abs(be[t]) > BE_ABS_MAX for t in TARGETS):
            continue
        st = m["state"]
        dedup = (m["bulk_key"], tuple(st["miller"]),
                 round(st["offset"], 4), bool(st["top"]))
        if dedup in seen:
            continue
        seen.add(dedup)
        rows.append((st, m.get("e_hull"), be, m["bulk_key"]))
    return rows


def build_matrix(rows):
    # словарь элементов по всему датасету
    elements = sorted({int(z) for st, _, _, _ in rows
                       for z in st["composition"]})
    X, groups = [], []
    Y = {t: [] for t in TARGETS}
    for st, e_hull, be, bkey in rows:
        comp = {int(z): n for z, n in st["composition"].items()}
        tot = sum(comp.values()) or 1
        feat = [comp.get(z, 0) / tot for z in elements]      # доли элементов
        feat.append(tot)                                      # размер ячейки
        feat.append(e_hull if e_hull is not None else 0.0)
        feat.extend(st["lattice_params"])                    # a,b,c,al,be,ga
        feat.append(st["space_group"])
        feat.extend(st["miller"])                            # h,k,l
        feat.append(st["offset"])
        feat.append(1.0 if st["top"] else 0.0)
        X.append(feat)
        groups.append(bkey)
        for t in TARGETS:
            Y[t].append(be[t])
    names = ([f"x_{z}" for z in elements] + ["n_atoms", "e_hull",
             "a", "b", "c", "alpha", "beta", "gamma", "sg",
             "h", "k", "l", "offset", "top"])
    return (np.array(X, float), {t: np.array(Y[t], float) for t in TARGETS},
            np.array(groups), names, elements)


def main():
    rows = load_rows()
    print(f"манифестов с валидными BE: {len(rows)}")
    if len(rows) < 50:
        print("слишком мало данных для замера"); return
    X, Y, groups, names, elements = build_matrix(rows)
    n_bulk = len(set(groups))
    print(f"уникальных bulk_key (групп): {n_bulk}, элементов в составе: "
          f"{len(elements)}, фич: {X.shape[1]}")
    n_splits = min(5, n_bulk)
    schemes = [
        ("GroupKFold (unseen bulk)", GroupKFold(n_splits=n_splits), groups),
        ("KFold random (interpolation)",
         KFold(n_splits=5, shuffle=True, random_state=0), None),
    ]
    for title, cv, grp in schemes:
        print(f"\n=== {title} ===")
        print(f"{'target':>8} {'R2':>7} {'MAE':>7} {'baseMAE':>8} {'std':>7}")
        for t in TARGETS:
            y = Y[t]
            model = RandomForestRegressor(n_estimators=300, n_jobs=-1,
                                          random_state=0)
            pred = cross_val_predict(model, X, y, groups=grp, cv=cv)
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            mae = float(np.mean(np.abs(y - pred)))
            base_mae = float(np.mean(np.abs(y - y.mean())))
            print(f"{t:>8} {r2:7.3f} {mae:7.3f} {base_mae:8.3f} {y.std():7.3f}")

    # важности фич на полной выборке (для интуиции)
    print("\nтоп-12 важностей фич (BE_CH, на всей выборке):")
    m = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0)
    m.fit(X, Y["CH"])
    order = np.argsort(m.feature_importances_)[::-1][:12]
    for i in order:
        print(f"  {names[i]:>10}: {m.feature_importances_[i]:.3f}")


if __name__ == "__main__":
    main()

"""Обучение surrogate-оценщика BE (вариант B) из готового npz-датасета.

Лёгкий шаг (только sklearn + matplotlib) — можно крутить локально/итеративно.
Датасет строит build_surrogate_dataset.py (тяжёлый, на кластере).

Делает: честный GroupKFold по слэбу → метрики (per-site + best/slab),
обучает финальный SurrogateBE на всех данных, сохраняет модель и метрики,
рисует стандартные графики (parity, остатки, learning/validation curves,
важности фич).

RF не итеративная модель — «loss от шага» в эпохах нет; стандартные аналоги:
learning curve (ошибка vs объём данных) и validation curve (vs число деревьев).

    python scripts/train_surrogate.py --dataset data/surrogate_dataset.npz \
        --out data/surrogate_be.pkl --plots
"""
import argparse
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (GroupKFold, cross_val_predict,
                                     learning_curve, validation_curve)

from pdh_gfn.reward.surrogate import ADSORBATES, FEAT_NAMES, SurrogateBE

N_TREES = 300       # финальная модель
N_TREES_CV = 150    # для CV/важностей: метрики стабильны, но ~2x быстрее


def load_dataset(path):
    d = np.load(path, allow_pickle=True)
    use_soap = bool(d["use_soap"])
    data = {}
    for name in ADSORBATES:
        if f"{name}_X" in d and len(d[f"{name}_y"]) >= 50:
            data[name] = (d[f"{name}_X"], d[f"{name}_y"], d[f"{name}_g"])
    return data, use_soap


def feat_names(n_feat):
    names = list(FEAT_NAMES)
    names += [f"soap_{i}" for i in range(n_feat - len(names))]
    return names


def best_per_slab(y, pred, groups):
    """min BE на слэб (то, что входит в дескрипторы) — true и pred."""
    bt, bp = {}, {}
    for g, yt, yp in zip(groups, y, pred):
        bt[g] = min(bt.get(g, yt), yt)
        bp[g] = min(bp.get(g, yp), yp)
    return (np.array([bt[g] for g in bt]), np.array([bp[g] for g in bt]))


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


# ----------------------------------------------------------------------
def cv_predict(name, X, y, groups):
    cv = GroupKFold(n_splits=min(5, len(set(groups))))
    rf = RandomForestRegressor(n_estimators=N_TREES_CV, n_jobs=4, random_state=0)
    return cross_val_predict(rf, X, y, groups=groups, cv=cv)


def metrics_for(y, pred, groups):
    bt, bp = best_per_slab(y, pred, groups)
    return {
        "n_sites": int(len(y)), "n_slabs": int(len(set(groups))),
        "r2_site": float(r2(y, pred)),
        "mae_site": float(np.mean(np.abs(y - pred))),
        "r2_best": float(r2(bt, bp)),
        "mae_best": float(np.mean(np.abs(bt - bp))),
    }


# ----------------------------- графики ---------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_basic(name, X, y, groups, pred, plot_dir):
    """Дешёвые графики (без переобучения по сетке): parity, остатки, важности."""
    plt = _plt()
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(f"Surrogate BE — {name}  ({len(y)} сайтов, "
                 f"{len(set(groups))} слэбов)", fontsize=13)

    a = ax[0, 0]                                   # parity per-site (out-of-fold)
    a.scatter(y, pred, s=4, alpha=0.25)
    lim = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    a.plot(lim, lim, "r--", lw=1)
    a.set(xlabel="true BE", ylabel="pred BE",
          title=f"parity (site)  R²={r2(y, pred):.3f}  "
                f"MAE={np.mean(np.abs(y-pred)):.3f}")

    bt, bp = best_per_slab(y, pred, groups)        # parity best/slab
    a = ax[0, 1]
    a.scatter(bt, bp, s=8, alpha=0.4, color="C2")
    lim = [min(bt.min(), bp.min()), max(bt.max(), bp.max())]
    a.plot(lim, lim, "r--", lw=1)
    a.set(xlabel="true best BE", ylabel="pred best BE",
          title=f"parity (best/slab)  R²={r2(bt, bp):.3f}  "
                f"MAE={np.mean(np.abs(bt-bp)):.3f}")

    a = ax[1, 0]                                   # остатки
    a.hist(pred - y, bins=60, color="C1")
    a.axvline(0, color="r", lw=1)
    a.set(xlabel="pred − true (эВ)", ylabel="частота", title="остатки")

    a = ax[1, 1]                                   # важности фич (топ-15)
    m = RandomForestRegressor(n_estimators=N_TREES_CV, n_jobs=4,
                              random_state=0).fit(X, y)
    names = feat_names(X.shape[1])
    order = np.argsort(m.feature_importances_)[::-1][:15]
    a.barh(range(len(order))[::-1], m.feature_importances_[order], color="C4")
    a.set_yticks(range(len(order))[::-1])
    a.set_yticklabels([names[i] for i in order], fontsize=8)
    a.set(title="важности фич (топ-15)")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(plot_dir, f"surrogate_{name}.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  график: {path}")


def plot_curves(name, X, y, groups, plot_dir):
    """Дорогие кривые (много переобучений RF): learning + validation."""
    plt = _plt()
    cv = GroupKFold(n_splits=min(5, len(set(groups))))
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Surrogate BE — {name}: кривые обучения", fontsize=13)

    # learning curve (ошибка vs объём данных) — аналог loss-от-шага для RF
    sizes, tr, va = learning_curve(
        RandomForestRegressor(n_estimators=100, n_jobs=4, random_state=0),
        X, y, groups=groups, cv=cv, train_sizes=np.linspace(0.15, 1.0, 5),
        scoring="neg_mean_absolute_error", n_jobs=1)
    ax[0].plot(sizes, -tr.mean(1), "o-", label="train")
    ax[0].plot(sizes, -va.mean(1), "o-", label="CV")
    ax[0].set(xlabel="объём обучающих данных", ylabel="MAE (эВ)",
              title="learning curve"); ax[0].legend()

    # validation curve (ошибка vs число деревьев)
    n_range = [50, 100, 200, 300]
    tr, va = validation_curve(
        RandomForestRegressor(n_jobs=4, random_state=0), X, y,
        param_name="n_estimators", param_range=n_range,
        groups=groups, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1)
    ax[1].plot(n_range, -tr.mean(1), "o-", label="train")
    ax[1].plot(n_range, -va.mean(1), "o-", label="CV")
    ax[1].set(xlabel="n_estimators", ylabel="MAE (эВ)",
              title="validation curve"); ax[1].legend()

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(plot_dir, f"surrogate_{name}_curves.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  кривые: {path}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/surrogate_dataset.npz")
    ap.add_argument("--out", default="data/surrogate_be.pkl")
    ap.add_argument("--metrics", default="data/surrogate_metrics.json")
    ap.add_argument("--plots", action="store_true",
                    help="parity / остатки / важности (дёшево)")
    ap.add_argument("--curves", action="store_true",
                    help="+ learning/validation curves (дорого: много RF-фитов)")
    ap.add_argument("--plot-dir", default="data/surrogate_plots")
    args = ap.parse_args()

    data, use_soap = load_dataset(args.dataset)
    print(f"датасет: {args.dataset} | use_soap={use_soap} | "
          f"адсорбаты: {list(data)}")
    if args.plots or args.curves:
        os.makedirs(args.plot_dir, exist_ok=True)

    metrics, fit_data = {}, {}
    print(f"\n{'адсорбат':>8} {'site R²':>8} {'site MAE':>9} "
          f"{'best R²':>8} {'best MAE':>9}")
    for name, (X, y, g) in data.items():
        pred = cv_predict(name, X, y, g)
        m = metrics_for(y, pred, g)
        metrics[name] = m
        print(f"{name:>8} {m['r2_site']:8.3f} {m['mae_site']:9.3f} "
              f"{m['r2_best']:8.3f} {m['mae_best']:9.3f}")
        if args.plots:
            plot_basic(name, X, y, g, pred, args.plot_dir)
        if args.curves:
            plot_curves(name, X, y, g, args.plot_dir)
        fit_data[name] = (X, y)

    SurrogateBE(use_soap=use_soap).fit(fit_data, n_estimators=N_TREES).save(args.out)
    json.dump(metrics, open(args.metrics, "w"), indent=2)
    print(f"\nмодель:  {args.out} (use_soap={use_soap})")
    print(f"метрики: {args.metrics}")


if __name__ == "__main__":
    main()

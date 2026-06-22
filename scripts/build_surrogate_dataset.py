"""Построение датасета для surrogate-оценщика BE (вариант B) → npz.

Тяжёлый шаг (нужны ase + pymatgen + при --soap dscribe): для каждой
поверхностной конфигурации читает голый релаксированный слэб, перечисляет
сайты (enumerate_sites, как на инференсе), а метки BE берёт из сохранённых
ads_{name}_{i}.xyz; якорь релаксированного адсорбата матчит к БЛИЖАЙШЕМУ
перечисленному сайту (при prefilter=True callback пишет индекс топ-K, а не
enum-индекс — base.py:155), фичи считает full_features той же функцией, что
инференс. Результат — npz с пер-адсорбатными (X, y, groups).

Гоняется раз на кластере; дальше train_surrogate.py работает с npz быстро.

    python scripts/build_surrogate_dataset.py --soap --out data/surrogate_dataset.npz
"""
import argparse
import glob
import json
import os

import numpy as np
from ase.io import read as ase_read

from pdh_gfn.potential.base import enumerate_sites
from pdh_gfn.reward.surrogate import ADSORBATES, full_features

BE_ABS = {"H": 5.0, "CH": 6.0, "C3H7": 5.0}
MATCH_MAX = 1.5  # Å: релакс. адсорбат уехал дальше любого сайта — пропуск


def adsorbate_anchor(atoms, name):
    """Якорь релаксированного адсорбата: H — сам H; иначе нижний C."""
    Z, pos = atoms.numbers, atoms.positions
    idx = np.where(Z == (1 if name == "H" else 6))[0]
    if len(idx) == 0:
        return None
    return pos[idx][np.argmin(pos[idx][:, 2])]


def build(root, use_soap):
    data = {n: ([], [], []) for n in ADSORBATES}
    surf_dirs = {os.path.dirname(p)
                 for p in glob.glob(os.path.join(root, "**", "slab_relaxed.xyz"),
                                    recursive=True)}
    n_dirs = 0
    for sd in sorted(surf_dirs):
        try:
            slab = ase_read(os.path.join(sd, "slab_relaxed.xyz"))
            sites = enumerate_sites(slab, max_sites=16)
        except Exception:
            continue
        if not sites:
            continue
        site_pos = np.asarray(sites, float)
        cell = np.asarray(slab.cell[:], float)
        Z, pos = slab.numbers, slab.positions
        e_hull = None
        mpath = os.path.join(sd, "manifest.json")
        if os.path.exists(mpath):
            try:
                e_hull = json.load(open(mpath, encoding="utf-8")).get("e_hull")
            except Exception:
                pass
        n_dirs += 1
        for name in ADSORBATES:
            for ap in glob.glob(os.path.join(sd, f"ads_{name}_*.xyz")):
                try:
                    ads = ase_read(ap)
                    be = float(ads.info["BE"])
                except Exception:
                    continue
                if abs(be) > BE_ABS[name]:
                    continue
                anc = adsorbate_anchor(ads, name)
                if anc is None:
                    continue
                dxy = np.linalg.norm(site_pos[:, :2] - anc[:2], axis=1)
                j = int(np.argmin(dxy))
                if dxy[j] > MATCH_MAX:
                    continue
                feat = full_features(cell, Z, pos, site_pos[j], e_hull, use_soap)
                if feat is None:
                    continue
                X, y, g = data[name]
                X.append(feat); y.append(be); g.append(sd)
        if n_dirs % 200 == 0:
            print(f"  обработано surf_dirs: {n_dirs}", flush=True)
    print(f"обработано surf_dirs: {n_dirs}")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structures", default="data/structures")
    ap.add_argument("--out", default="data/surrogate_dataset.npz")
    ap.add_argument("--soap", action="store_true",
                    help="добавить SOAP(mu2) к ручным фичам (нужен dscribe)")
    args = ap.parse_args()

    data = build(args.structures, args.soap)
    out = {"use_soap": np.array(args.soap)}
    for name in ADSORBATES:
        X, y, g = data[name]
        out[f"{name}_X"] = np.array(X, float)
        out[f"{name}_y"] = np.array(y, float)
        out[f"{name}_g"] = np.array(g)
        print(f"{name}: {len(y)} пер-сайтовых примеров, "
              f"{X[0].__len__() if X else 0} фич")
    np.savez(args.out, **out)
    print(f"\nдатасет сохранён: {args.out} (use_soap={args.soap})")


if __name__ == "__main__":
    main()

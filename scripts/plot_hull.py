"""Визуализация выпуклой оболочки Pd-M систем.

Рисует тернарные/бинарные диаграммы для каждой системы Pd-M:
- точки из MP (старые DFT-данные) — синие
- наши oc20-пересчитанные — красные
- чистые элементы — отмечены отдельно

Использование:
    python scripts/plot_hull.py
    python scripts/plot_hull.py --system Pd-Zn   # только одна система
    python scripts/plot_hull.py --out hull_plots/
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_composition(comp_str):
    """'Pd2 Zn1' -> {'Pd': 2, 'Zn': 1}"""
    out = {}
    for part in comp_str.split():
        m = re.match(r"([A-Z][a-z]?)(\d+)", part)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hull-json", default="data/mp_pdm_entries.json")
    p.add_argument("--out", default="data/hull_plots",
                   help="директория для PNG-файлов")
    p.add_argument("--system", default=None,
                   help='только одна система, например "Pd-Zn"')
    args = p.parse_args()

    records = json.loads(Path(args.hull_json).read_text())
    print(f"загружено {len(records)} записей из {args.hull_json}")

    # Группируем по бинарной системе
    systems = {}   # {frozenset({'Pd','Zn'}): [(records...)]}
    for r in records:
        comp = parse_composition(r["composition"])
        elements = frozenset(comp.keys())
        if "Pd" not in elements and len(elements) == 1:
            # чистый не-Pd элемент: добавляем ко всем Pd-M системам где он есть
            for el in elements:
                key = frozenset({"Pd", el})
                systems.setdefault(key, []).append((r, comp))
        elif elements <= {"Pd"} and len(elements) == 1:
            # чистый Pd: во все системы
            for key in list(systems.keys()):
                systems[key].append((r, comp))
        else:
            systems.setdefault(elements, []).append((r, comp))

    # фильтр Pd-X систем
    pd_systems = {k: v for k, v in systems.items()
                  if "Pd" in k and len(k) == 2}

    # Добавляем чистый Pd ко всем системам, если ещё нет
    for key in pd_systems:
        has_pure_pd = any(len(parse_composition(r["composition"])) == 1
                          and "Pd" in parse_composition(r["composition"])
                          for r, _ in pd_systems[key])
        if not has_pure_pd:
            # ищем чистый Pd в общем списке
            pure_pd = [r for r in records
                       if parse_composition(r["composition"]) == {"Pd": _}
                       for _ in [1, 2, 4]]
            # найдём любой чистый Pd
            for r in records:
                comp = parse_composition(r["composition"])
                if list(comp.keys()) == ["Pd"]:
                    pd_systems[key].append((r, comp))
                    break

    print(f"найдено {len(pd_systems)} Pd-M систем")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.system:
        # фильтр
        wanted = frozenset(args.system.split("-"))
        pd_systems = {k: v for k, v in pd_systems.items() if k == wanted}
        print(f"фильтр {args.system}: осталось {len(pd_systems)} систем")

    for sys_key in sorted(pd_systems, key=lambda k: sorted(k)):
        other = next(e for e in sys_key if e != "Pd")
        entries = pd_systems[sys_key]
        if len(entries) < 3:
            print(f"  Pd-{other}: {len(entries)} точек, пропускаю")
            continue

        # Считаем x_Pd и E/atom
        xs_mp, es_mp = [], []           # без mp_id или с MP-DFT энергией
        xs_oc20, es_oc20 = [], []       # с mp_id (oc20-пересчёт)
        labels_mp, labels_oc20 = [], []

        for r, comp in entries:
            n_pd = comp.get("Pd", 0)
            n_other = comp.get(other, 0)
            n_total = n_pd + n_other
            if n_total == 0:
                continue
            x_pd = n_pd / n_total
            e = r["energy_per_atom"]
            label = r["composition"]

            if "mp_id" in r and r["mp_id"].startswith("mp-"):
                xs_oc20.append(x_pd)
                es_oc20.append(e)
                labels_oc20.append(label)
            else:
                xs_mp.append(x_pd)
                es_mp.append(e)
                labels_mp.append(label)

        # Построение
        fig, ax = plt.subplots(figsize=(10, 6))
        if xs_mp:
            ax.scatter(xs_mp, es_mp, c='steelblue', s=60, alpha=0.7,
                       label=f"MP-DFT (старые, {len(xs_mp)})",
                       edgecolors='black', linewidth=0.5)
        if xs_oc20:
            ax.scatter(xs_oc20, es_oc20, c='crimson', s=60, alpha=0.7,
                       label=f"oc20 (пересчёт, {len(xs_oc20)})",
                       edgecolors='black', linewidth=0.5, marker='s')

        # Чистые: выделяем
        for x, e, lab in zip(xs_mp + xs_oc20, es_mp + es_oc20,
                              labels_mp + labels_oc20):
            comp = parse_composition(lab)
            if len(comp) == 1:
                ax.annotate(f"чист.{list(comp.keys())[0]}", (x, e),
                            fontsize=8, ha='center', va='bottom',
                            xytext=(0, 5), textcoords='offset points')

        ax.set_xlabel(f"x(Pd)  →  Pd:0 = чистый {other},  1 = чистый Pd")
        ax.set_ylabel("E (эВ/атом)")
        ax.set_title(f"Hull: Pd-{other} (всего {len(entries)} записей)")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)

        # Сохранение
        out_file = out_dir / f"hull_Pd-{other}.png"
        fig.tight_layout()
        fig.savefig(out_file, dpi=120)
        plt.close(fig)
        print(f"  Pd-{other}: MP={len(xs_mp)}, oc20={len(xs_oc20)} -> {out_file}")

    print(f"\nготово, графики в: {out_dir}")
    print("посмотри Pd-Zn, Pd-Ga, Pd-In, Pd-Sn — где наши эталоны")


if __name__ == "__main__":
    main()

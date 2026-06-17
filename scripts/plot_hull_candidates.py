"""Convex hull с наложением кандидатов из GFlowNet.

Для каждой Pd-M системы:
  - рисует нижнюю выпуклую оболочку (lower convex hull) по MP oc20 данным
  - MP oc20 точки: яркие маркеры (crimson), размер 90
  - кандидаты GFlowNet: полупрозрачные, окрашены по reward (colormap)

Источники кандидатов (в порядке приоритета):
  1. logs/pdh_gfn/trace_*.csv (авто-поиск по последней дате)
  2. data/structures/*/manifest.json (если trace не найден)

Позиция кандидата по Y = hull_line(x_Pd) + e_hull, то есть
кандидаты прямо на hull — когда e_hull ≈ 0.

Использование:
    python scripts/plot_hull_candidates.py
    python scripts/plot_hull_candidates.py --system Pd-Zn
    python scripts/plot_hull_candidates.py --trace logs/pdh_gfn/trace_X.csv
    python scripts/plot_hull_candidates.py --min-step 500
"""
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from ase.data import chemical_symbols, atomic_numbers


# ---- helpers ----------------------------------------------------------------

def parse_formula(formula: str) -> Dict[str, int]:
    """'Pd8Zn8' -> {'Pd': 8, 'Zn': 8}"""
    out = {}
    for sym, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if sym:
            out[sym] = out.get(sym, 0) + int(n or 1)
    return out


def parse_composition_str(comp_str: str) -> Dict[str, int]:
    """'Pd2 Zn1' -> {'Pd': 2, 'Zn': 1}"""
    out = {}
    for part in comp_str.split():
        m = re.match(r"([A-Z][a-z]?)(\d+)", part)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def x_pd(comp: Dict[str, int]) -> float:
    """Мольная доля Pd."""
    n_pd = comp.get("Pd", 0)
    n_total = sum(comp.values())
    return n_pd / n_total if n_total else 0.5


def m_element(comp: Dict[str, int]) -> Optional[str]:
    """Второй (не-Pd) элемент в бинарной Pd-M системе. None если не бинарная."""
    others = [el for el in comp if el != "Pd"]
    return others[0] if len(others) == 1 else None


def lower_convex_hull(xs: List[float], ys: List[float]):
    """Andrew's monotone chain — нижняя оболочка точек (x, y).

    Возвращает (hull_xs, hull_ys) — отсортированные по x вершины нижней оболочки.
    """
    points = sorted(zip(xs, ys))
    if len(points) < 2:
        return [p[0] for p in points], [p[1] for p in points]

    def cross(O, A, B):
        return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    hx, hy = zip(*lower)
    return list(hx), list(hy)


def hull_interpolate(hull_xs: List[float], hull_ys: List[float], x: float) -> float:
    """Интерполяция значения hull в точке x (по-нижнему выпуклому кусочно-линейному)."""
    return float(np.interp(x, hull_xs, hull_ys))


# ---- data loading -----------------------------------------------------------

def load_mp_entries(path: str) -> List[dict]:
    records = json.loads(Path(path).read_text())
    out = []
    for r in records:
        comp = parse_composition_str(r["composition"])
        if not comp:
            continue
        out.append({
            "composition": comp,
            "energy_per_atom": r["energy_per_atom"],
            "is_oc20": "mp_id" in r and str(r["mp_id"]).startswith("mp-"),
            "label": r.get("composition", ""),
        })
    return out


def load_trace_candidates(trace_path: str) -> List[dict]:
    """Читает trace CSV -> список {formula, x_pd, m_element, e_hull, reward, step, status}."""
    candidates = []
    with open(trace_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] == "invalid":
                continue
            formula = row.get("formula", "")
            if not formula:
                continue
            comp = parse_formula(formula)
            m = m_element(comp)
            if m is None or "Pd" not in comp:
                continue
            try:
                e_h = float(row["e_hull"]) if row.get("e_hull") else None
            except ValueError:
                e_h = None
            if e_h is None:
                continue
            try:
                step = int(row.get("step", -1))
            except ValueError:
                step = -1
            # reward: предпочитаем reward_used, fallback r_stab
            rw = None
            for col in ("reward_used", "reward_gated", "r_stab"):
                try:
                    v = float(row.get(col) or "")
                    if v > 0:
                        rw = v
                        break
                except ValueError:
                    pass
            candidates.append({
                "formula": formula,
                "composition": comp,
                "x_pd": x_pd(comp),
                "m_element": m,
                "e_hull": e_h,
                "reward": rw,
                "step": step,
                "status": row["status"],
                "e_act_ch": _float_or_none(row.get("e_act_ch")),
                "e_sel": _float_or_none(row.get("e_sel")),
            })
    return candidates


def load_manifest_candidates(structures_dir: str) -> List[dict]:
    """Fallback: читает manifest.json из data/structures/*/..."""
    root = Path(structures_dir)
    candidates = []
    for mf in root.rglob("manifest.json"):
        try:
            data = json.loads(mf.read_text())
        except Exception:
            continue
        state = data.get("state", {})
        raw_comp = state.get("composition", {})
        # JSON-ключи строковые (int -> str при сериализации)
        comp_z = {int(k): v for k, v in raw_comp.items()}
        comp = {chemical_symbols[z]: n for z, n in comp_z.items()
                if 0 < z < len(chemical_symbols)}
        m = m_element(comp)
        if m is None or "Pd" not in comp:
            continue
        e_h = data.get("e_hull")
        if e_h is None:
            continue
        desc = data.get("descriptors", {})
        rw_data = data.get("reward", {})
        candidates.append({
            "formula": "".join(f"{el}{n}" for el, n in comp.items()),
            "composition": comp,
            "x_pd": x_pd(comp),
            "m_element": m,
            "e_hull": float(e_h),
            "reward": rw_data.get("reward_beta"),
            "step": -1,
            "status": "full",
            "e_act_ch": desc.get("e_act_ch"),
            "e_sel": desc.get("e_sel"),
        })
    return candidates


def _float_or_none(s) -> Optional[float]:
    try:
        return float(s) if s else None
    except (ValueError, TypeError):
        return None


# ---- find latest trace -------------------------------------------------------

def find_latest_trace(log_dir: str = "logs/pdh_gfn") -> Optional[str]:
    """Находит самый свежий trace_*.csv в log_dir."""
    traces = sorted(Path(log_dir).glob("trace_*.csv"))
    return str(traces[-1]) if traces else None


# ---- plotting ---------------------------------------------------------------

def plot_system(
    system_name: str,
    mp_oc20_pts: List[Tuple[float, float]],    # (x_pd, e_per_atom)
    mp_gga_pts: List[Tuple[float, float]],
    candidates: List[dict],
    out_dir: Path,
):
    """Строит и сохраняет PNG для одной Pd-M системы."""
    if not mp_oc20_pts:
        return

    xs_oc20, es_oc20 = zip(*mp_oc20_pts) if mp_oc20_pts else ([], [])
    xs_gga, es_gga = zip(*mp_gga_pts) if mp_gga_pts else ([], [])

    # Нижняя выпуклая оболочка (hull line) только по oc20 точкам
    all_hull_xs = list(xs_oc20) + list(xs_gga)
    all_hull_es = list(es_oc20) + list(es_gga)
    hull_xs, hull_ys = lower_convex_hull(all_hull_xs, all_hull_es)

    # Позиции кандидатов GFlowNet на осях энергии
    cand_x, cand_y, cand_r, cand_labels = [], [], [], []
    for c in candidates:
        hv = hull_interpolate(hull_xs, hull_ys, c["x_pd"])
        cand_x.append(c["x_pd"])
        cand_y.append(hv + c["e_hull"])
        cand_r.append(c["reward"] if c["reward"] is not None else 0.0)
        cand_labels.append(c["formula"])

    # --- figure ---
    fig, ax = plt.subplots(figsize=(11, 6))

    # 1. Hull line
    ax.plot(hull_xs, hull_ys, color="gray", lw=1.5, ls="--",
            zorder=2, label="lower convex hull (MP oc20)")

    # 2. MP GGA (старые, если есть)
    if xs_gga:
        ax.scatter(xs_gga, es_gga, c="steelblue", s=55, alpha=0.6,
                   edgecolors="navy", lw=0.5, zorder=4,
                   label=f"MP GGA ({len(xs_gga)})")

    # 3. MP oc20
    ax.scatter(xs_oc20, es_oc20, c="crimson", s=90, alpha=0.85,
               edgecolors="darkred", lw=0.7, marker="s", zorder=5,
               label=f"MP oc20 ({len(xs_oc20)})")

    # 4. Кандидаты GFlowNet
    if cand_x:
        # log-масштаб reward -> цвет (reward_beta в диапазоне ~1e-8 ... 1e-1)
        rw_arr = np.array(cand_r, dtype=float)
        # log-нормировка для более выразительного цвета
        with np.errstate(divide="ignore", invalid="ignore"):
            log_r = np.log10(np.where(rw_arr > 1e-10, rw_arr, 1e-10))
        vmin, vmax = log_r.min(), max(log_r.max(), log_r.min() + 1e-3)
        norm = plt.Normalize(vmin, vmax)
        sc = ax.scatter(cand_x, cand_y, c=log_r, cmap="plasma",
                        norm=norm, s=35, alpha=0.45, edgecolors="none",
                        zorder=3, label=f"GFlowNet ({len(cand_x)})")
        cbar = fig.colorbar(sc, ax=ax, pad=0.01)
        cbar.set_label("log10(reward_used)", fontsize=9)

    # 5. Подписи у известных составов (только крупные, не дублируем)
    _label_pure_elements(ax, xs_oc20, es_oc20, mp_oc20_pts, system_name)

    # Formatting
    m_el = system_name.replace("Pd-", "")
    ax.set_xlabel(f"x(Pd)   [0 = чистый {m_el}, 1 = чистый Pd]", fontsize=11)
    ax.set_ylabel("E (эВ/атом)", fontsize=11)
    n_full = sum(1 for c in candidates if c["status"] == "full")
    n_stab = sum(1 for c in candidates if c["status"] == "stab_only")
    ax.set_title(
        f"Hull: {system_name}   |   MP oc20: {len(xs_oc20)}   "
        f"GFlowNet full: {n_full}, stab_only: {n_stab}",
        fontsize=12,
    )
    ax.legend(loc="upper center", fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-0.05, 1.05)

    out_file = out_dir / f"hull_cand_{system_name.replace('-', '_')}.png"
    fig.tight_layout()
    fig.savefig(out_file, dpi=130)
    plt.close(fig)
    return out_file


def _label_pure_elements(ax, xs_oc20, es_oc20, mp_oc20_pts, system_name):
    """Аннотируем чистые элементы (x=0 и x=1) маленьким текстом."""
    m = system_name.replace("Pd-", "")
    for x, e in zip(xs_oc20, es_oc20):
        if abs(x) < 0.02:
            ax.annotate(f"чист.{m}", (x, e), fontsize=8, ha="left", va="bottom",
                        xytext=(4, 4), textcoords="offset points", color="darkred")
        elif abs(x - 1.0) < 0.02:
            ax.annotate("чист.Pd", (x, e), fontsize=8, ha="right", va="bottom",
                        xytext=(-4, 4), textcoords="offset points", color="darkred")


# ---- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hull-json", default="data/mp_pdm_entries.json",
                    help="MP entries JSON (из rebuild_hull_oc20.py)")
    ap.add_argument("--trace", default=None,
                    help="trace CSV (авто-поиск последнего если не указан)")
    ap.add_argument("--manifests", default="data/structures",
                    help="директория с manifest.json (fallback если нет trace)")
    ap.add_argument("--out", default="data/hull_plots",
                    help="директория для PNG")
    ap.add_argument("--system", default=None,
                    help='только одна система, например "Pd-Zn"')
    ap.add_argument("--min-step", type=int, default=0,
                    help="фильтр: только шаги >= min_step (0 = все)")
    ap.add_argument("--max-e-hull", type=float, default=3.0,
                    help="отсечение кандидатов с e_hull > порога (шум/невалид)")
    args = ap.parse_args()

    # --- загрузка MP ---
    mp_records = load_mp_entries(args.hull_json)
    print(f"MP записей: {len(mp_records)}")

    # --- загрузка кандидатов ---
    trace_file = args.trace or find_latest_trace()
    candidates_all: List[dict] = []
    if trace_file and Path(trace_file).exists():
        candidates_all = load_trace_candidates(trace_file)
        print(f"trace: {trace_file}  -> {len(candidates_all)} кандидатов")
    else:
        manifests_dir = Path(args.manifests)
        if manifests_dir.exists():
            candidates_all = load_manifest_candidates(str(manifests_dir))
            print(f"manifests: {manifests_dir}  -> {len(candidates_all)} кандидатов")
        else:
            print("!  ни trace CSV, ни manifests не найдены -- только MP hull")

    # фильтр по шагу
    if args.min_step > 0:
        before = len(candidates_all)
        candidates_all = [c for c in candidates_all if c["step"] >= args.min_step]
        print(f"фильтр min_step={args.min_step}: {before} -> {len(candidates_all)}")

    # фильтр по e_hull
    candidates_all = [c for c in candidates_all if c["e_hull"] <= args.max_e_hull]

    # --- группировка по системам ---
    systems_mp: Dict[str, dict] = {}    # "Pd-Zn" -> {oc20: [(x,e)], gga: [(x,e)]}
    for r in mp_records:
        comp = r["composition"]
        m = m_element(comp)
        if m is None:
            continue
        key = f"Pd-{m}"
        if key not in systems_mp:
            systems_mp[key] = {"oc20": [], "gga": []}
        pt = (x_pd(comp), r["energy_per_atom"])
        if r["is_oc20"]:
            systems_mp[key]["oc20"].append(pt)
        else:
            systems_mp[key]["gga"].append(pt)

    # Добавляем чистый Pd в каждую систему
    pure_pd_pts = [(1.0, r["energy_per_atom"]) for r in mp_records
                   if r["composition"].get("Pd", 0) > 0
                   and len(r["composition"]) == 1 and r["is_oc20"]]
    for key in systems_mp:
        if pure_pd_pts:
            systems_mp[key]["oc20"].extend(pure_pd_pts[:1])

    systems_cand: Dict[str, List[dict]] = {}
    for c in candidates_all:
        key = f"Pd-{c['m_element']}"
        systems_cand.setdefault(key, []).append(c)

    # фильтр системы
    if args.system:
        systems_mp = {k: v for k, v in systems_mp.items() if k == args.system}
        systems_cand = {k: v for k, v in systems_cand.items() if k == args.system}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Строим для каждой Pd-M системы (где есть >= 3 MP точки)
    generated = []
    for sys_key in sorted(systems_mp):
        pts_oc20 = systems_mp[sys_key]["oc20"]
        pts_gga = systems_mp[sys_key]["gga"]
        if len(pts_oc20) + len(pts_gga) < 3:
            continue
        cands = systems_cand.get(sys_key, [])
        out_file = plot_system(sys_key, pts_oc20, pts_gga, cands, out_dir)
        if out_file:
            generated.append(out_file)
            n_full = sum(1 for c in cands if c["status"] == "full")
            n_stab = sum(1 for c in cands if c["status"] == "stab_only")
            print(f"  {sys_key}: MP oc20={len(pts_oc20)}, GFN full={n_full} "
                  f"stab_only={n_stab}  -> {out_file.name}")

    # Сводный по ВСЕМ системам (если ≥ 2 систем)
    if len(generated) >= 2 and not args.system:
        _plot_summary(systems_mp, systems_cand, out_dir)

    print(f"\nготово: {len(generated)} систем в {out_dir}")
    print("ключевые системы: Pd-Zn, Pd-Ga, Pd-In, Pd-Sn")


def _plot_summary(systems_mp, systems_cand, out_dir: Path):
    """Сводная панель: все Pd-M системы на одном figure (grid)."""
    keys = sorted(k for k, v in systems_mp.items()
                  if len(v["oc20"]) + len(v["gga"]) >= 3)
    if not keys:
        return
    n = len(keys)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                              squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for i, key in enumerate(keys):
        ax = axes_flat[i]
        pts_oc20 = systems_mp[key]["oc20"]
        pts_gga = systems_mp[key]["gga"]
        cands = systems_cand.get(key, [])

        xs_oc20, es_oc20 = zip(*pts_oc20) if pts_oc20 else ([], [])
        xs_gga, es_gga = zip(*pts_gga) if pts_gga else ([], [])

        all_xs = list(xs_oc20) + list(xs_gga)
        all_es = list(es_oc20) + list(es_gga)
        if len(all_xs) < 2:
            ax.set_visible(False)
            continue

        hull_xs, hull_ys = lower_convex_hull(all_xs, all_es)
        ax.plot(hull_xs, hull_ys, "k--", lw=1.0, alpha=0.5)

        if xs_gga:
            ax.scatter(xs_gga, es_gga, c="steelblue", s=25, alpha=0.5)
        ax.scatter(xs_oc20, es_oc20, c="crimson", s=45, alpha=0.85,
                   marker="s", zorder=4)

        if cands:
            cand_x = [c["x_pd"] for c in cands]
            cand_y = [hull_interpolate(hull_xs, hull_ys, c["x_pd"]) + c["e_hull"]
                      for c in cands]
            rw_arr = np.array([c["reward"] or 0 for c in cands])
            with np.errstate(divide="ignore", invalid="ignore"):
                log_r = np.log10(np.where(rw_arr > 1e-10, rw_arr, 1e-10))
            norm = plt.Normalize(log_r.min(), max(log_r.max(), log_r.min() + 0.1))
            ax.scatter(cand_x, cand_y, c=log_r, cmap="plasma",
                       norm=norm, s=15, alpha=0.35, zorder=3)

        ax.set_title(f"{key} ({len(cands)} cand.)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(-0.05, 1.05)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Hull + GFlowNet candidates — все Pd-M системы", fontsize=13)
    fig.tight_layout()
    out_file = out_dir / "hull_cand_ALL.png"
    fig.savefig(out_file, dpi=110)
    plt.close(fig)
    print(f"  сводный: {out_file.name}")


if __name__ == "__main__":
    main()

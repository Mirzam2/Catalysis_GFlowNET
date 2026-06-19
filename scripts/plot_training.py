"""Графики динамики обучения из trace CSV (+ опц. loss из stdout-лога).

Trace пишет PDHProxy по каждой оценке (step, phase, status, дескрипторы,
награды). Здесь агрегируем по шагам и рисуем читаемые кривые вместо tqdm-строк.

Использование:
    python scripts/plot_training.py --trace logs/pdh_gfn/trace_XXXX.csv
    python scripts/plot_training.py --trace ...csv --log train_1500.log
    python scripts/plot_training.py --trace ...csv --out logs/pdh_gfn/plots
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless кластер
import matplotlib.pyplot as plt
import numpy as np

import pdh_gfn.constants as C


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_trace(path):
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        r["step"] = _num(r["step"])
        for k in ("e_hull", "e_act_ch", "e_sel", "reward_gated", "reward_used"):
            r[k] = _num(r.get(k))
        # отбрасываем warmup/сид-оценки (step=1e9 — _current_step до обучения):
        # это не шаги обучения, и они растягивают ось X.
        if r["step"] is not None and r["step"] < 1e8:
            rows.append(r)
    return rows


def phase_boundaries(rows):
    """Шаги, где меняется фаза curriculum (для вертикалей)."""
    bounds = []
    last = None
    for r in sorted(rows, key=lambda r: r["step"]):
        ph = r["phase"]
        if ph != last and last is not None:
            bounds.append((r["step"], ph))
        last = ph
    return bounds


def per_step(rows):
    """Агрегаты по шагу: доли статусов, средние награды."""
    by = defaultdict(list)
    for r in rows:
        by[int(r["step"])].append(r)
    steps = sorted(by)
    out = {"step": steps, "inv_frac": [], "full_frac": [],
           "mean_used": [], "mean_gated_full": [], "max_gated_full": []}
    for s in steps:
        rs = by[s]
        n = len(rs)
        full = [r for r in rs if r["status"] == "full" and r["reward_gated"] is not None]
        out["inv_frac"].append(sum(r["status"] == "invalid" for r in rs) / n)
        out["full_frac"].append(len(full) / n)
        used = [r["reward_used"] for r in rs if r["reward_used"] is not None]
        out["mean_used"].append(np.mean(used) if used else np.nan)
        g = [r["reward_gated"] for r in full]
        out["mean_gated_full"].append(np.mean(g) if g else np.nan)
        out["max_gated_full"].append(np.max(g) if g else np.nan)
    return out


def parse_loss(log_path):
    """Достаёт (step, loss, mean_reward) из tqdm-строк stdout."""
    pat = re.compile(r"Loss:\s*([\d.eE+-]+)\s*\|\s*Mean rewards:\s*([\d.eE+-]+)"
                     r".*?(\d+)/\d+\s*\[")
    by = {}
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    for m in pat.finditer(text):
        loss, mr, step = float(m.group(1)), float(m.group(2)), int(m.group(3))
        by[step] = (loss, mr)  # последнее значение на шаг
    steps = sorted(by)
    return steps, [by[s][0] for s in steps], [by[s][1] for s in steps]


def _vlines(ax, bounds):
    for s, ph in bounds:
        ax.axvline(s, color="grey", ls="--", lw=0.8, alpha=0.6)
        ax.text(s, ax.get_ylim()[1], ph, fontsize=7, color="grey",
                rotation=90, va="top", ha="right")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True)
    p.add_argument("--log", default=None, help="stdout-лог для кривой loss")
    p.add_argument("--out", default="logs/pdh_gfn/plots")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_trace(args.trace)
    if not rows:
        print("trace пуст"); return
    agg = per_step(rows)
    bounds = phase_boundaries(rows)
    full = [r for r in rows if r["status"] == "full"
            and r["e_act_ch"] is not None and r["e_sel"] is not None]

    # --- дашборд 2x2 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    ax.plot(agg["step"], agg["mean_used"], label="mean reward_used (сигнал)", lw=1)
    ax.plot(agg["step"], agg["mean_gated_full"], label="mean R_gated (full)", lw=1)
    ax.plot(agg["step"], agg["max_gated_full"], label="max R_gated (full)",
            lw=1, alpha=0.6)
    ax.set_title("Награды по шагам"); ax.set_xlabel("step"); ax.set_ylabel("reward")
    ax.legend(fontsize=8); _vlines(ax, bounds)

    ax = axes[0, 1]
    ax.plot(agg["step"], agg["inv_frac"], label="invalid", color="tab:red", lw=1)
    ax.plot(agg["step"], agg["full_frac"], label="full", color="tab:green", lw=1)
    ax.set_title("Доли статусов"); ax.set_xlabel("step"); ax.set_ylabel("доля")
    ax.set_ylim(0, 1); ax.legend(fontsize=8); _vlines(ax, bounds)

    ax = axes[1, 0]
    gated = [(int(r["step"]), r["reward_gated"]) for r in rows
             if r["reward_gated"] is not None and r["status"] == "full"]
    if gated:
        gated.sort()
        gs = [s for s, _ in gated]
        cummax = np.maximum.accumulate([v for _, v in gated])
        ax.plot(gs, cummax, color="tab:purple", lw=1.5)
    ax.set_title("Лучший R_gated на текущий момент")
    ax.set_xlabel("step"); ax.set_ylabel("cummax R_gated"); _vlines(ax, bounds)

    ax = axes[1, 1]
    if args.log and Path(args.log).exists():
        ls, loss, _ = parse_loss(args.log)
        if ls:
            ax.plot(ls, loss, color="tab:orange", lw=1)
            ax.set_yscale("log")
        ax.set_title("Loss (из stdout-лога)")
    else:
        ax.set_title("Loss — нет --log")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); _vlines(ax, bounds)

    fig.tight_layout()
    f1 = out / "training_dashboard.png"
    fig.savefig(f1, dpi=120); plt.close(fig)

    # --- облако дескрипторов E_act vs E_sel ---
    fig, ax = plt.subplots(figsize=(8, 6))
    if full:
        x = [r["e_act_ch"] for r in full]
        y = [r["e_sel"] for r in full]
        c = [r["reward_gated"] if r["reward_gated"] is not None else 0 for r in full]
        sc = ax.scatter(x, y, c=c, cmap="viridis", s=18, alpha=0.8)
        fig.colorbar(sc, label="R_gated")
    ax.axvline(C.E_ACT_CH_MIN, color="red", ls="--", lw=1,
               label=f"E_ACT_CH_MIN={C.E_ACT_CH_MIN} (клэмп)")
    ax.axvline(C.E_ACT_CH_MAX, color="grey", ls=":", lw=1,
               label=f"E_ACT_CH_MAX={C.E_ACT_CH_MAX}")
    ax.axhline(C.E_SEL_TARGET, color="blue", ls=":", lw=1,
               label=f"E_SEL_TARGET={C.E_SEL_TARGET}")
    ax.set_xlabel("E_act(C-H), эВ (ниже=активнее)")
    ax.set_ylabel("E_sel, эВ (выше=селективнее)")
    ax.set_title(f"Дескрипторы full-оценок ({len(full)} шт.)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    f2 = out / "descriptors_scatter.png"
    fig.savefig(f2, dpi=120); plt.close(fig)

    print(f"сохранено: {f1}")
    print(f"           {f2}")
    print(f"full-оценок: {len(full)} | шагов: {len(agg['step'])} "
          f"| фаз-переходов: {len(bounds)}")


if __name__ == "__main__":
    main()

"""Выгрузка top-N кандидатов из trace CSV с пометкой битых/подозрительных.

Дедуп по грани (formula, sg, miller) — лучший reward_gated на поверхность.
Каждой строке ставится флаг качества дескрипторов: OK либо предупреждения о
нефизичных/экстраполированных числах (см. CLAUDE.md, reward-hack гейта).

Использование:
    python scripts/export_candidates.py --trace logs/pdh_gfn/trace_XXXX.csv
    python scripts/export_candidates.py --trace ...csv --top 100 --out cand.csv
"""
import argparse
import csv
from pathlib import Path

import pdh_gfn.constants as C

# Домен доверия дескрипторов (калибровка на PdZn/PdIn/PdGa/Pd3Sn):
EACT_LO, EACT_HI = C.E_ACT_CH_MIN, 2.2     # физичный барьер C-H
ESEL_LO, ESEL_HI = 3.3, 4.5                # домен BEP по эталонам (~3.3-3.9 + запас)
ESEL_WILD = 6.0                            # дальше — дичь
EHULL_MAX = 0.1                            # эВ/атом, метастабильность


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def flags(eact, esel, ehull):
    """Список предупреждений о качестве дескрипторов ('' = OK)."""
    f = []
    if eact is None or esel is None:
        return ["no-desc"]
    if eact < EACT_LO:
        f.append("over-bind")          # Eact ниже физичного → BEP в минус
    if esel > ESEL_WILD:
        f.append("Esel-wild")          # дикая экстраполяция
    elif esel > ESEL_HI:
        f.append("Esel-extrap")        # за доменом BEP
    if ehull is not None and ehull > EHULL_MAX:
        f.append("unstable")
    return f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True)
    p.add_argument("--top", type=int, default=100)
    p.add_argument("--out", default="logs/pdh_gfn/candidates_top100.csv")
    args = p.parse_args()

    # дедуп по грани (formula, sg, miller): лучший reward_gated
    best = {}
    for r in csv.DictReader(open(args.trace, encoding="utf-8")):
        if r["status"] != "full":
            continue
        g = _num(r["reward_gated"])
        s = _num(r["step"])
        if g is None or s is None or s >= 1e8:   # пропускаем warmup/сиды (step=1e9)
            continue
        key = (r["formula"], r["sg"], r["miller"])
        if key not in best or g > best[key]["_g"]:
            r["_g"] = g
            best[key] = r

    rows = sorted(best.values(), key=lambda r: r["_g"], reverse=True)[:args.top]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["rank", "formula", "sg", "miller", "offset", "top", "e_hull",
            "e_act_ch", "e_sel", "r_stab", "r_act", "r_sel", "reward_gated",
            "phase", "step", "flag"]
    n_ok = 0
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i, r in enumerate(rows, 1):
            fl = flags(_num(r["e_act_ch"]), _num(r["e_sel"]), _num(r["e_hull"]))
            if not fl:
                n_ok += 1
            w.writerow([
                i, r["formula"], r["sg"], r["miller"], r["offset"], r["top"],
                r["e_hull"], r["e_act_ch"], r["e_sel"], r["r_stab"], r["r_act"],
                r["r_sel"], r["reward_gated"], r["phase"], r["step"],
                ";".join(fl) if fl else "OK",
            ])

    print(f"выгружено: {out}")
    print(f"всего уникальных граней: {len(best)} | в таблице top-{len(rows)}")
    print(f"  OK (надёжные числа):   {n_ok}")
    print(f"  с флагами (битые/?):   {len(rows) - n_ok}")
    # топ-10 в stdout для быстрого взгляда
    print(f"\n{'#':>3} {'состав':<10} {'sg':>4} {'hkl':<7} {'Eact':>6} "
          f"{'Esel':>6} {'ehull':>7} {'R':>6}  флаг")
    for i, r in enumerate(rows[:10], 1):
        fl = flags(_num(r["e_act_ch"]), _num(r["e_sel"]), _num(r["e_hull"]))
        print(f"{i:>3} {r['formula']:<10} {r['sg']:>4} {r['miller']:<7} "
              f"{_num(r['e_act_ch']):>6.2f} {_num(r['e_sel']):>6.2f} "
              f"{_num(r['e_hull']):>7.3f} {r['_g']:>6.3f}  "
              f"{';'.join(fl) if fl else 'OK'}")


if __name__ == "__main__":
    main()

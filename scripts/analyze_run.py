"""Один отчёт по прогону обучения — консолидация ручных анализов.

Печатает: статусы/награды по фазам, топ-15 кандидатов с флагами качества,
концентрацию M-элементов, и (если дан --log) loss-траекторию + статистику кэша.

Использование:
    python scripts/analyze_run.py --trace logs/pdh_gfn/trace_X.csv
    python scripts/analyze_run.py --trace ...csv --log train_X.log
"""
import argparse
import csv
import re
import statistics as S
from collections import Counter

# Домен доверия дескрипторов (см. export_candidates.py / CLAUDE.md)
EACT_LO, ESEL_HI, ESEL_WILD, EHULL_MAX = 1.0, 4.5, 6.0, 0.1
PH = {"1/0/0": "ф1 stab", "1/1/0": "ф2 +act",
      "1/1/1": "ф3 +sel", "1/1/1g": "ф4 gated"}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def flags(ea, es, eh):
    f = []
    if ea is None or es is None:
        return ["no-desc"]
    if ea < EACT_LO:
        f.append("over-bind")
    if es > ESEL_WILD:
        f.append("Esel-wild")
    elif es > ESEL_HI:
        f.append("Esel-extrap")
    if eh is not None and eh > EHULL_MAX:
        f.append("unstable")
    return f


def melems(formula):
    return [s for s in re.findall(r"[A-Z][a-z]?", formula) if s != "Pd"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--log", default=None, help="stdout-лог для loss/кэша")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    rows = []
    for r in csv.DictReader(open(args.trace, encoding="utf-8")):
        r["_s"] = num(r["step"])
        if r["_s"] is None or r["_s"] >= 1e8:  # warmup/сиды
            continue
        r["_g"] = num(r["reward_gated"])
        r["_ea"] = num(r["e_act_ch"])
        r["_es"] = num(r["e_sel"])
        rows.append(r)

    # --- статусы/награды по фазам ---
    print("=" * 64)
    print("СТАТУСЫ И НАГРАДЫ ПО ФАЗАМ")
    print("=" * 64)
    for ph, nm in PH.items():
        rs = [r for r in rows if r["phase"] == ph]
        if not rs:
            continue
        n = len(rs)
        st = Counter(r["status"] for r in rs)
        g = [r["_g"] for r in rs if r["status"] == "full" and r["_g"] is not None]
        print(f"  {nm:<9} {n:>5} оц. | full={st.get('full',0):>4}"
              f"({100*st.get('full',0)//n:>2}%) inv={st.get('invalid',0):>4}"
              f"({100*st.get('invalid',0)//n:>2}%) | "
              f"meanR={S.mean(g) if g else 0:.3f} maxR={max(g) if g else 0:.3f}")

    # --- топ кандидатов (дедуп по грани) с флагами ---
    best = {}
    for r in rows:
        if r["status"] != "full" or r["_g"] is None:
            continue
        key = (r["formula"], r["sg"], r["miller"])
        if key not in best or r["_g"] > best[key]["_g"]:
            best[key] = r
    top = sorted(best.values(), key=lambda r: r["_g"], reverse=True)
    n_ok = sum(1 for r in top if not flags(r["_ea"], r["_es"], num(r["e_hull"])))
    print("\n" + "=" * 64)
    print(f"ТОП-{args.top} (дедуп по грани) | всего граней={len(best)} "
          f"| OK={n_ok}/{len(top)}")
    print("=" * 64)
    print(f"{'#':>3} {'состав':<10} {'sg':>4} {'hkl':<7} {'Eact':>6} "
          f"{'Esel':>6} {'ehull':>7} {'R':>6}  флаг")
    for i, r in enumerate(top[:args.top], 1):
        fl = flags(r["_ea"], r["_es"], num(r["e_hull"]))
        print(f"{i:>3} {r['formula']:<10} {r['sg']:>4} {r['miller']:<7} "
              f"{r['_ea']:>6.2f} {r['_es']:>6.2f} {num(r['e_hull']):>7.3f} "
              f"{r['_g']:>6.3f}  {';'.join(fl) if fl else 'OK'}")

    # --- концентрация M-элементов по фазам ---
    print("\n" + "=" * 64)
    print("КОНЦЕНТРАЦИЯ M-ЭЛЕМЕНТА (full, по фазам)")
    print("=" * 64)
    for ph, nm in PH.items():
        c = Counter()
        for r in rows:
            if r["phase"] == ph and r["status"] == "full":
                for e in melems(r["formula"]):
                    c[e] += 1
        if c:
            print(f"  {nm}: {dict(c.most_common(6))}")

    # --- loss и кэш из stdout-лога ---
    if args.log:
        try:
            text = open(args.log, encoding="utf-8", errors="ignore").read()
        except OSError:
            text = ""
        pat = re.compile(r"Loss:\s*([\d.eE+-]+)\s*\|\s*Mean rewards:\s*"
                         r"([\d.eE+-]+).*?(\d+)/\d+\s*\[")
        by = {int(m.group(3)): float(m.group(1)) for m in pat.finditer(text)}
        if by:
            print("\n" + "=" * 64)
            print("LOSS (из stdout-лога)")
            print("=" * 64)
            ks = sorted(by)
            mn = min(by.items(), key=lambda kv: kv[1])
            print(f"  старт(шаг {ks[0]})={by[ks[0]]:.1f} | "
                  f"финал(шаг {ks[-1]})={by[ks[-1]]:.1f} | "
                  f"мин={mn[1]:.1f}@шаг{mn[0]}")
        cache = re.findall(r"proxy stats: calls=(\d+).*?cache=(\d+)", text)
        if cache:
            c0, cN = int(cache[0][1]), int(cache[-1][1])
            calls = int(cache[-1][0])
            print(f"\nКЭШ: старт={c0} → финал={cN} (+{cN-c0}) за {calls} вызовов "
                  f"| hit-rate≈{100*(1-(cN-c0)/max(1,calls)):.0f}%")


if __name__ == "__main__":
    main()

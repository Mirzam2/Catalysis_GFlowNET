"""Лёгкий стадийный профайлер reward-конвейера.

Зачем: оценка награды дорогая (~50 c) и мы не знаем, где именно горит время —
в релаксации объёма, поверхности или адсорбатов, и почему: «релаксаций много»
или «каждая упирается в MAX_RELAX_STEPS=300 и не сходится». cProfile тут врёт:
UMA считает на GPU асинхронно через C-расширения torch. Простые таймеры на
time.perf_counter точнее и почти бесплатны.

Использование:
    from pdh_gfn.profiling import PROFILER
    PROFILER.enabled = True            # включается флагом --profile в train.py
    with PROFILER.timed("slab_relax"):
        ...
    PROFILER.add_relax("slab_relax", n_steps, converged)
    print(PROFILER.report())
"""
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict


class Profiler:
    """Аккумулятор времени и статистики FIRE по именованным стадиям.

    Все операции — no-op при enabled=False, поэтому врезки можно оставлять
    в горячем пути без накладных расходов в обычном прогоне.
    """

    def __init__(self):
        self.enabled = False
        # стадия -> {total_time, count, fire_steps, fire_unconverged, n_relax}
        self._stages: Dict[str, dict] = {}

    def _slot(self, stage: str) -> dict:
        s = self._stages.get(stage)
        if s is None:
            s = {"total_time": 0.0, "count": 0,
                 "fire_steps": 0, "fire_unconverged": 0, "n_relax": 0}
            self._stages[stage] = s
        return s

    @contextmanager
    def timed(self, stage: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            slot = self._slot(stage)
            slot["total_time"] += time.perf_counter() - t0
            slot["count"] += 1

    def add_relax(self, stage: str, n_steps: int, converged: bool) -> None:
        """Статистика по одной FIRE-релаксации: сколько шагов и сошлась ли."""
        if not self.enabled:
            return
        slot = self._slot(stage)
        slot["fire_steps"] += int(n_steps)
        slot["n_relax"] += 1
        if not converged:
            slot["fire_unconverged"] += 1

    def reset(self) -> None:
        self._stages.clear()

    # -- отчёты ------------------------------------------------------------
    def to_dict(self) -> dict:
        total = sum(s["total_time"] for s in self._stages.values()) or 1.0
        out = {}
        for stage, s in self._stages.items():
            d = dict(s)
            d["pct"] = 100.0 * s["total_time"] / total
            d["mean_time"] = s["total_time"] / max(1, s["count"])
            if s["n_relax"]:
                d["mean_fire_steps"] = s["fire_steps"] / s["n_relax"]
                d["unconverged_frac"] = s["fire_unconverged"] / s["n_relax"]
            out[stage] = d
        return out

    def report(self) -> str:
        if not self._stages:
            return "(профиль пуст)"
        data = self.to_dict()
        # сортировка по суммарному времени убыв.
        rows = sorted(data.items(), key=lambda kv: kv[1]["total_time"], reverse=True)
        lines = [
            f"{'стадия':<18} {'total,s':>9} {'%':>6} {'count':>6} "
            f"{'mean,s':>8} {'FIRE_avg':>9} {'unconv%':>8}",
        ]
        for stage, d in rows:
            fire = f"{d.get('mean_fire_steps', 0):>9.1f}" if d["n_relax"] else f"{'-':>9}"
            unc = f"{100*d['unconverged_frac']:>7.0f}%" if d["n_relax"] else f"{'-':>8}"
            lines.append(
                f"{stage:<18} {d['total_time']:>9.1f} {d['pct']:>5.1f}% "
                f"{d['count']:>6} {d['mean_time']:>8.2f} {fire} {unc}"
            )
        return "\n".join(lines)

    def dump_json(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))


# Модульный синглтон — один на процесс.
PROFILER = Profiler()

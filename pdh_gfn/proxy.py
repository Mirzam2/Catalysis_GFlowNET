"""Прокси для тренера gflownet: обёртка над RewardPipeline.

Конвенция репозитория Mila: proxy(states) -> тензор значений, из которых
reward_function делает награду. У нас конвейер сам возвращает готовую
R^beta, поэтому reward_function="identity".

Стоимость: одна оценка = релаксация объёма + поверхности + ~max_sites
релаксаций адсорбата. Бюджет обучения ~1e4-1e5 оценок (НЕ 5e5 как у
Crystal-GFN с дешёвым MLP-прокси) — кэш и затравка буфера обязательны.

Curriculum: если schedule передан, pipeline считает дескрипторы (BE,
Eact, E_sel) как обычно, но reward_beta пересчитывается под текущую
фазу обучения (см. reward/schedule.py).
"""
import csv
import logging
from typing import Callable, List, Optional

import torch
from torchtyping import TensorType
from ase.data import chemical_symbols

from gflownet.proxy.base import Proxy

from .reward.gates import invalid_reward
from .reward.pipeline import CatalystState, RewardPipeline
from .reward.schedule import RewardSchedule
from .profiling import PROFILER

logger = logging.getLogger(__name__)


def _formula(composition: dict) -> str:
    """{Z: n} -> 'Pd8Zn8' (Pd первым, остальное по Z)."""
    items = sorted(composition.items(), key=lambda kv: (kv[0] != 46, kv[0]))
    return "".join(f"{chemical_symbols[z]}{n}" for z, n in items)


class PDHProxy(Proxy):
    def __init__(self, pipeline: RewardPipeline,
                 schedule: Optional[RewardSchedule] = None,
                 step_callback: Optional[Callable[[], int]] = None,
                 debug_errors: bool = True,
                 trace_path: Optional[str] = None, **kwargs):
        kwargs.setdefault("reward_function", "identity")
        super().__init__(**kwargs)
        self.pipeline = pipeline
        self.schedule = schedule
        # step_callback: lambda которая возвращает текущий шаг GFN-агента
        # (например, lambda: gfn.it). Если None — фаза всегда последняя.
        self.step_callback = step_callback
        # Считаем долю невалидных/упавших оценок — диагностика пустого replay.
        self.debug_errors = debug_errors
        self.n_calls = 0
        self.n_invalid = 0
        self.n_errors = 0
        # Логируем переход фаз
        self._last_phase_label = None
        # CSV-трейс: по строке на оценку (шаг, состав, дескрипторы, награды) —
        # чтобы смотреть динамику обучения, а не парсить текстовый лог.
        self._trace_writer = None
        self._trace_file = None
        if trace_path is not None:
            self._trace_file = open(trace_path, "w", newline="", encoding="utf-8")
            self._trace_writer = csv.writer(self._trace_file)
            self._trace_writer.writerow([
                "step", "phase", "status", "sg", "formula", "miller",
                "offset", "top", "e_hull", "e_act_ch", "e_sel",
                "be_h", "be_ch", "be_ch_meta", "be_c3h7",
                "r_stab", "r_act", "r_sel", "reward_gated", "reward_used",
            ])
            self._trace_file.flush()

    def setup(self, env=None):
        pass

    def _current_step(self) -> int:
        if self.step_callback is None:
            return 10**9   # "после всех фаз" = последняя фаза
        try:
            return int(self.step_callback())
        except Exception:
            return 10**9

    @torch.no_grad()
    def __call__(self, states: List[CatalystState]) -> TensorType["batch"]:
        rewards = []
        step = self._current_step()

        # Текущая фаза + признак «только стабильность» (Tier 0)
        phase = None
        stab_only = False
        if self.schedule is not None:
            phase = self.schedule.current_phase(step)
            label = f"step={step} α={phase.alpha} β={phase.beta} γ={phase.gamma} mode={phase.mode}"
            if label != self._last_phase_label:
                logger.info("curriculum фаза: %s", label)
                self._last_phase_label = label
            # фаза 1: β=γ=0, linear → адсорбцию не считаем (см. pipeline)
            stab_only = (phase.mode == "linear"
                         and phase.beta == 0 and phase.gamma == 0)

        for s in states:
            self.n_calls += 1
            try:
                breakdown = self.pipeline(s, stability_only=stab_only)
                if self.schedule is not None:
                    reward = self.schedule.compute(breakdown, step)
                else:
                    reward = breakdown.reward_beta
                rewards.append(reward)
                # invalid = не смогли оценить даже стабильность (bulk упал).
                # stab_only-пропуски НЕ считаем невалидными (e_hull известен).
                if breakdown.e_hull is None:
                    self.n_invalid += 1
                self._trace(step, phase, s, breakdown, reward)
            except Exception as exc:
                self.n_errors += 1
                if self.debug_errors and self.n_errors <= 20:
                    logger.warning("ошибка награды для %s: %r", s, exc)
                rewards.append(invalid_reward().reward_beta)
            # периодическая статистика каждые 50 вызовов
            if self.n_calls % 50 == 0:
                logger.info("proxy stats: calls=%d invalid=%.0f%% errors=%.0f%% "
                            "cache=%d",
                            self.n_calls,
                            100 * self.n_invalid / max(1, self.n_calls),
                            100 * self.n_errors / max(1, self.n_calls),
                            len(self.pipeline.cache))
                if PROFILER.enabled:
                    logger.info("profile:\n%s", PROFILER.report())
        return torch.tensor(rewards, dtype=self.float, device=self.device)

    def _trace(self, step, phase, state, b, reward_used) -> None:
        """Одна строка CSV-трейса на оценку награды."""
        if self._trace_writer is None:
            return
        if b.e_hull is None:
            status = "invalid"       # bulk/pyxtal упали
        elif b.e_sel is None:
            status = "stab_only"     # адсорбция пропущена (фаза 1 или cutoff)
        else:
            status = "full"
        phase_lbl = "none"
        if phase is not None:
            phase_lbl = (f"{phase.alpha:.0f}/{phase.beta:.0f}/{phase.gamma:.0f}"
                         + ("g" if phase.mode == "gated" else ""))

        def f(x):
            return "" if x is None else f"{x:.4g}"

        self._trace_writer.writerow([
            step, phase_lbl, status, state.space_group,
            _formula(state.composition),
            "_".join(map(str, state.miller)), state.offset, int(state.top),
            f(b.e_hull), f(b.e_act_ch), f(b.e_sel),
            f(b.be_h), f(b.be_ch), f(b.be_ch_meta), f(b.be_c3h7),
            f(b.r_stab), f(b.r_act), f(b.r_sel),
            f(b.reward_beta), f(reward_used),
        ])
        self._trace_file.flush()

    def error_stats(self) -> dict:
        return {
            "calls": self.n_calls,
            "invalid": self.n_invalid,
            "errors": self.n_errors,
            "invalid_frac": self.n_invalid / max(1, self.n_calls),
            "error_frac": self.n_errors / max(1, self.n_calls),
        }

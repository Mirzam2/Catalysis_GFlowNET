"""Curriculum learning для награды: динамическое переключение фаз.

Идея: GFlowNet с холодного старта в плоском ландшафте награды (когда
произведение трёх гейтов почти всегда у пола) ничему не учится. Решение —
поэтапное обучение:

  фаза 1 (0..500 шагов):    только стабильность → найди интерметаллиды
  фаза 2 (500..1000):       + активность → научись адсорбировать C3H7
  фаза 3 (1000..1500):      + селективность → балансируй с CH/H
  фаза 4 (1500..):          gated с сигмоидами → строгие критерии

Reward внутри каждой фазы — truncated linear с насыщением:

  score_stab = clip(1 - E_hull / 0.15,           0, 1)
  score_act  = clip(1 - (Eact - 0.5) / 2.5,      0, 1)
  score_sel  = clip((E_sel - 1.5) / 2.5,         0, 1)
  R = exp(α·s_stab + β·s_act + γ·s_sel - (α+β+γ))

Нормировка `-(α+β+γ)`: идеальная структура (все score=1) → R=exp(0)=1.0,
плохая (все score=0) → R=exp(-3)≈0.05. Диапазон ~20x — TB-loss это любит.

Усиление одного score сверх 1.0 не помогает (clip сверху) — нет reward
hacking через выкручивание одного параметра.

После фазы 4 переходим на исходный gated режим — для финальной выборки
кандидатов с физическими порогами E_ACT_CH_MAX, E_SEL_TARGET.
"""
from dataclasses import dataclass
from typing import List, Optional

import math

from .gates import RewardBreakdown, composite_reward, GATE_FLOOR
from ..reward.descriptors import compute_descriptors
from .. import constants as C


# ---- Truncated linear scores -------------------------------------------

# Границы для нормализации (откалиброваны на PdZn/PdGa/PdIn/Pd3Sn):
# хорошая структура попадает в score≈1, мусор в score=0.
# Границы для нормализации (откалиброваны на PdZn/PdGa/PdIn/Pd3Sn):
# хорошая структура попадает в score≈1, мусор в score=0.
# STAB_E_HULL_MAX = 0.5 — даёт мягкий градиент по стабильности. pyxtal часто
# генерирует МЕТАСТАБИЛЬНЫЕ полиморфы стабильных составов (например Pd3Nb в
# SG=140 вместо L1₂ SG=221) с E_hull~1+ эВ/атом — это физически правильно,
# но жёсткий cutoff 0.15 отсекал 100% политики. С 0.5 score плавно убывает,
# даёт градиент к стабильности.
STAB_E_HULL_MAX = 0.5    # эВ/атом; за этой границей score=0
ACT_E_MIN, ACT_E_MAX = 0.5, 3.0   # Eact_CH: ниже 0.5 → score=1, выше 3.0 → 0
SEL_E_MIN, SEL_E_MAX = 1.5, 4.0   # E_sel: ниже 1.5 → score=0, выше 4.0 → 1


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def truncated_linear_scores(e_hull: float, e_act_ch: float, e_sel: float):
    """Три score'а в [0,1] для curriculum.

    Все clip сверху единицей → нельзя выкрутить один за счёт остальных.
    """
    s_stab = _clip01(1.0 - max(0.0, e_hull) / STAB_E_HULL_MAX)
    # клэмп Eact снизу на E_ACT_CH_MIN: ниже валидированного диапазона НЕ активнее
    # (анти-hack; BEP экстраполируется в отрицательный барьер на переусиленных)
    e_act_eff = max(e_act_ch, C.E_ACT_CH_MIN)
    s_act = _clip01(1.0 - (e_act_eff - ACT_E_MIN) / (ACT_E_MAX - ACT_E_MIN))
    s_sel = _clip01((e_sel - SEL_E_MIN) / (SEL_E_MAX - SEL_E_MIN))
    return s_stab, s_act, s_sel


# ---- Schedule -----------------------------------------------------------

@dataclass
class Phase:
    """Одна фаза curriculum: до какого шага и с какими весами."""
    until_step: Optional[int]   # None = "до конца"
    alpha: float = 1.0          # вес стабильности
    beta: float = 0.0           # вес активности
    gamma: float = 0.0          # вес селективности
    mode: str = "linear"        # "linear" или "gated"


# Дефолтный 4-фазный curriculum (рекомендация: см. docstring файла)
DEFAULT_PHASES: List[Phase] = [
    Phase(until_step=500,  alpha=1.0, beta=0.0, gamma=0.0, mode="linear"),
    Phase(until_step=1000, alpha=1.0, beta=1.0, gamma=0.0, mode="linear"),
    Phase(until_step=1500, alpha=1.0, beta=1.0, gamma=1.0, mode="linear"),
    Phase(until_step=None, alpha=1.0, beta=1.0, gamma=1.0, mode="gated"),
]


class RewardSchedule:
    """Награда с расписанием фаз. По дескрипторам + текущему шагу
    выдаёт reward_beta.

    Использование в proxy:
        schedule = RewardSchedule(phases=DEFAULT_PHASES)
        proxy = PDHProxy(..., schedule=schedule, step_callback=lambda: gfn.it)
    """

    def __init__(self, phases: Optional[List[Phase]] = None):
        self.phases = phases or DEFAULT_PHASES

    def current_phase(self, step: int) -> Phase:
        for phase in self.phases:
            if phase.until_step is None or step < phase.until_step:
                return phase
        return self.phases[-1]   # после всех фаз — последняя

    def compute(self, breakdown: RewardBreakdown, step: int) -> float:
        """Берёт сохранённые в breakdown дескрипторы (e_hull, e_act_ch, e_sel)
        и пересчитывает reward_beta под текущую фазу curriculum.

        Возвращает reward_beta — то, что подаётся в TB-loss.
        """
        phase = self.current_phase(step)

        if phase.mode == "gated":
            # Финальный режим: сигмоиды с порогами из constants.py.
            # Пересчитываем из дескрипторов СВЕЖИМ гейтом (с клэмпом Eact) —
            # иначе кэш-хит отдаёт reward_beta, посчитанный СТАРЫМ гейтом до фикса.
            if breakdown.e_sel is None or breakdown.be_h is None:
                return breakdown.reward_beta
            desc = compute_descriptors(breakdown.be_h, breakdown.be_ch,
                                       breakdown.be_ch_meta, breakdown.be_c3h7)
            return composite_reward(desc, breakdown.e_hull or 0.0).reward_beta

        # --- phase.mode == "linear" ---
        # invalid: не смогли оценить даже стабильность (bulk/pyxtal упали) → пол
        if breakdown.e_hull is None:
            return breakdown.reward_beta

        # Дескриптор НУЖЕН фазе (вес>0), но отсутствует (адсорбция не дала
        # результата) — возвращаем пол, чтобы не наградить «дыру».
        if (phase.beta > 0 and breakdown.e_act_ch is None) or \
           (phase.gamma > 0 and breakdown.e_sel is None):
            return breakdown.reward_beta

        # Tier 0: в фазе 1 (β=γ=0) дескрипторы не нужны — адсорбцию пропускаем,
        # e_act_ch/e_sel приходят None; их вес 0, подставляем 0.0 безопасно.
        s_stab, s_act, s_sel = truncated_linear_scores(
            breakdown.e_hull,
            breakdown.e_act_ch if breakdown.e_act_ch is not None else 0.0,
            breakdown.e_sel if breakdown.e_sel is not None else 0.0,
        )
        alpha, beta, gamma = phase.alpha, phase.beta, phase.gamma
        weighted = alpha * s_stab + beta * s_act + gamma * s_sel
        norm = alpha + beta + gamma   # нормировка: идеал → exp(0) = 1
        if norm <= 0:
            return 1.0
        # exp(weighted - norm): диапазон [exp(-norm), 1]
        # пол на R_VALID_EPS чтобы log(R) был конечным для TB-loss
        r = math.exp(weighted - norm)
        return max(r, 1e-6)

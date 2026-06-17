"""Композитная награда: R = R_valid * R_stab * R_act * R_sel, затем R^beta.

Дизайн-решения (из обсуждения):
- R_valid = eps, не 0: Trajectory Balance работает с log R.
- R_stab — мягкая экспонента по E_hull: метастабильные интерметаллиды
  до ~100 мэВ/атом синтезируемы.
- R_sel — СИГМОИДА, не экспонента: насыщение защищает от эксплуатации
  ошибок прокси в области экстраполяции BEP (E_sel вне домена).
- R_act — отдельный гейт активности: E_sel измеряет относительную
  селективность и может быть накручен неактивным катализатором.

ВАЖНО (почему пол на КАЖДЫЙ гейт, а не только на произведение):
  Если любой гейт даёт строгий 0 (например, E_act далеко за порогом при
  резкой сигмоиде), произведение зануляется и ВСЕ кандидаты получают
  одинаковый eps — GFlowNet теряет градиент и не учится (симптом: пустой
  replay-буфер).  Поэтому каждый гейт ограничен снизу GATE_FLOOR: даже
  «проваленный» кандидат сохраняет относительный сигнал по другим осям,
  и модель может двигаться к лучшим областям.
"""
import math
from dataclasses import dataclass
from typing import Optional

from .. import constants as C

from .descriptors import Descriptors

# Пол на каждый гейт: маленький, но НЕнулевой, чтобы сохранить ранжирование
# по остальным осям, когда один гейт провален.
GATE_FLOOR = 1e-3


def _sigmoid(x: float) -> float:
    if x < -40:
        return 0.0
    if x > 40:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _floored_sigmoid(x: float) -> float:
    """Сигмоида с полом GATE_FLOOR и потолком 1: сохраняет градиент."""
    return GATE_FLOOR + (1.0 - GATE_FLOOR) * _sigmoid(x)


@dataclass
class RewardBreakdown:
    r_valid: float
    r_stab: float
    r_act: float
    r_sel: float
    reward: float          # произведение гейтов, ДО beta
    reward_beta: float     # R^beta — то, что видит GFlowNet
    e_hull: Optional[float] = None
    e_sel: Optional[float] = None
    e_act_ch: Optional[float] = None  # для пересчёта reward по schedule
    # raw BE — для отладки/анализа, не для гейтов
    be_h: Optional[float] = None
    be_ch: Optional[float] = None
    be_ch_meta: Optional[float] = None
    be_c3h7: Optional[float] = None


def stability_only_reward(e_hull: float) -> RewardBreakdown:
    """Награда для раннего выхода (УСКОРЕНИЕ #4): структура слишком далеко
    от выпуклой оболочки, адсорбцию не считаем. Активность/селективность
    ставим на пол — кандидат всё равно отбракован стабильностью, но награда
    остаётся различимой по E_hull (градиент к более стабильным областям).
    """
    r_stab = max(GATE_FLOOR, math.exp(-max(0.0, e_hull) / C.SIGMA_STAB))
    r = r_stab * GATE_FLOOR * GATE_FLOOR  # r_act, r_sel на полу
    r = max(r, C.R_VALID_EPS)
    return RewardBreakdown(
        1.0, r_stab, GATE_FLOOR, GATE_FLOOR, r, r ** C.REWARD_BETA,
        e_hull=e_hull, e_sel=None,
    )


def invalid_reward() -> RewardBreakdown:
    r = C.R_VALID_EPS
    return RewardBreakdown(C.R_VALID_EPS, 1.0, 1.0, 1.0, r, r ** C.REWARD_BETA)


def composite_reward(desc: Descriptors, e_hull: float) -> RewardBreakdown:
    # Стабильность: мягкая экспонента, тоже с полом (далёкие от hull не нулевые)
    r_stab = max(GATE_FLOOR, math.exp(-max(0.0, e_hull) / C.SIGMA_STAB))
    # Активность: floored-сигмоида (ниже порога E_act = активнее)
    r_act = _floored_sigmoid((C.E_ACT_CH_MAX - desc.e_act_ch) / C.E_ACT_SCALE)
    # Селективность: floored-сигмоида
    r_sel = _floored_sigmoid((desc.e_sel - C.E_SEL_TARGET) / C.E_SEL_SCALE)
    r = r_stab * r_act * r_sel
    r = max(r, C.R_VALID_EPS)
    return RewardBreakdown(
        1.0, r_stab, r_act, r_sel, r, r ** C.REWARD_BETA,
        e_hull=e_hull, e_sel=desc.e_sel, e_act_ch=desc.e_act_ch,
    )

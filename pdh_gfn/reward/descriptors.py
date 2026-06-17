"""Дескрипторы селективности PDH — Seemakurthi et al. ChemCatChem 2025.

Знаковое соглашение Seemakurthi: BE отрицательны (крепче связь = более
отрицательно).

Четыре исходные величины на поверхность (все эВ):
    BE(H*)         — атомарный водород
    BE(CH*)        — метилидин на стабильном сайте (= BE(propyne*), Fig S5)
    BE(CH*_meta)   — CH* на метастабильном сайте (вход BEP для C-C)
    BE(C3H7*)      — пропил (вход BEP для C-H вместе с BE(H*))

Уравнение (1):
    E_sel = BE(propyne*) + E_act(C-C) - BE(propane*) - E_act(C-H)

Прокси-подстановки:
    E_act(C-H)   = 0.82 * [BE(C3H7*) + BE(H*)] + 0.83     (Fig 2a)
    E_act(C-C)   = 0.77 *  BE(CH*_meta)        + 0.19     (Fig 2d)
    BE(propyne*) = BE(CH*)                                (Fig S5)
    BE(propane*) = 0                                       (физисорбция)

Свёрнутая форма:
    E_sel = BE(CH*) + 0.77*BE(CH*_meta) - 0.82*BE(C3H7*) - 0.82*BE(H*) - 0.64

Три целевых числа на поверхность:
    Активность   : E_act(C-H)   (ниже = активнее)
    Стабильность : BE(propyne*) = BE(CH*) (предшественник кокса)
    Селективность: E_sel        (выше = селективнее)
"""
from dataclasses import dataclass

from .. import constants as C


@dataclass
class Descriptors:
    be_h: float            # BE(H*), эВ
    be_ch: float           # BE(CH*), стабильный сайт, эВ
    be_ch_meta: float      # BE(CH*) на метастабильном сайте, эВ
    be_c3h7: float         # BE(C3H7*), пропил, эВ
    e_act_ch: float        # барьер C-H (активность; ниже = активнее), эВ
    e_act_cc: float        # барьер разрыва C-C, эВ
    be_propyne: float      # BE(propyne*) = BE(CH*) (стабильность/кокс), эВ
    e_sel: float           # композит селективности (выше = селективнее), эВ


def compute_descriptors(be_h: float, be_ch: float,
                        be_ch_meta: float, be_c3h7: float) -> Descriptors:
    """Дескрипторы из четырёх BE по Seemakurthi 2025, уравнение (1).

    Все BE в знаковом соглашении Seemakurthi (отрицательные = крепкая связь).
    """
    e_act_ch = C.BEP_CH_A * (be_c3h7 + be_h) + C.BEP_CH_B
    e_act_cc = C.BEP_CC_A * be_ch_meta + C.BEP_CC_B
    be_propyne = be_ch                      # Fig S5
    be_propane = 0.0                        # физисорбция
    e_sel = be_propyne + e_act_cc - be_propane - e_act_ch
    return Descriptors(
        be_h=be_h, be_ch=be_ch, be_ch_meta=be_ch_meta, be_c3h7=be_c3h7,
        e_act_ch=e_act_ch, e_act_cc=e_act_cc,
        be_propyne=be_propyne, e_sel=e_sel,
    )

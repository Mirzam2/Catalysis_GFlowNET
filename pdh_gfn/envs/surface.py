"""Поверхностный блок: три подокружения, складываемых в Stack.

1. MillerChoice — категориальный выбор грани из канонического набора
   (|h|,|k|,|l| <= 1 без (000), 13 представителей; пары (hkl)/(-h-k-l)
   сведены, потому что сторона выбирается отдельным действием).
   Симметрийная канонизация по конкретной структуре (вариант 3, уровень 1)
   выполняется в конвейере награды (structure.canonical.canonical_miller).
2. OffsetCube — непрерывный сдвиг сечения, ContinuousCube размерности 1
   (та же смесь Beta-распределений, что у параметров решётки Crystal-GFN).
   Для интерметаллидов это выбор терминации: Pd-слой / M-слой / смешанный
   — ключевой механизм геометрической изоляции активных центров.
3. TopBottomChoice — верхняя или нижняя поверхность сечения (различны
   для структур без центра инверсии).
"""
from gflownet.envs.choice import Choice
from gflownet.envs.cube import ContinuousCube

from .. import constants as C


class MillerChoice(Choice):
    def __init__(self, miller_set=None, **kwargs):
        miller_set = miller_set or C.MILLER_SET
        self.miller_set = [tuple(m) for m in miller_set]
        options = [str(m) for m in self.miller_set]
        super().__init__(options=options, **kwargs)

    def state2miller(self, state):
        """[idx] -> (h, k, l); индексация опций с 1, 0 = источник."""
        if state[0] == 0:
            return None
        return self.miller_set[state[0] - 1]


class OffsetCube(ContinuousCube):
    def __init__(self, **kwargs):
        defaults = dict(n_dim=1, n_comp=3, min_incr=0.05)
        defaults.update(kwargs)
        super().__init__(**defaults)

    @staticmethod
    def state2offset(state):
        return float(state[0])


class TopBottomChoice(Choice):
    def __init__(self, **kwargs):
        super().__init__(options=["top", "bottom"], **kwargs)

    @staticmethod
    def state2top(state):
        if state[0] == 0:
            return None
        return state[0] == 1

"""Композиция Pd-M: подкласс Composition из Mila gflownet.

Отличия от базового окружения:
1. Pd обязателен (required_elements), проверка заряда выключена —
   электронейтральность не имеет смысла для интерметаллидов (решение
   из обсуждения дизайна: выбрасываем constraint C3 Crystal-GFN).
2. Жёсткий гейт x_Pd = n_Pd / n_total в [X_PD_MIN, X_PD_MAX]:
   - EOS маскируется, пока доля Pd вне диапазона;
   - действия (element, n) маскируются, если после них диапазон
     аналитически недостижим ни при каком достраивании (консервативная
     отсечка без учёта Уайкова — Уайков проверяет super-маска).
3. Страховка от тупиков: если совместное наложение масок (Уайков +
   x_Pd) не оставило ни одного действия, разрешается EOS — невалидная
   доля тогда ловится наградой как invalid (R = eps). На практике
   с нашими диапазонами это редкий случай, но GFlowNet не должен
   попадать в состояния без действий.
"""
from typing import List, Optional

from gflownet.envs.crystals.composition import Composition

from .. import constants as C


class PdMComposition(Composition):
    def __init__(
        self,
        elements: Optional[List[int]] = None,
        x_pd_min: Optional[float] = None,
        x_pd_max: Optional[float] = None,
        **kwargs,
    ):
        # Всё читаем из C в момент вызова (не как дефолт-аргумент): так
        # search_space-переопределения из run.yaml, применённые до сборки env,
        # доходят сюда. См. apply_search_space_overrides в scripts/train.py.
        elements = elements or C.ALL_ELEMENTS
        defaults = dict(
            max_diff_elem=C.MAX_DIFF_ELEM,
            min_diff_elem=C.MIN_DIFF_ELEM,
            min_atoms=C.MIN_ATOMS_PER_CELL,
            max_atoms=C.MAX_ATOMS_PER_CELL,
            required_elements=[C.PD_Z],
            do_charge_check=False,
        )
        defaults.update(kwargs)
        self.x_pd_min = C.X_PD_MIN if x_pd_min is None else x_pd_min
        self.x_pd_max = C.X_PD_MAX if x_pd_max is None else x_pd_max
        super().__init__(elements=elements, **defaults)

    # ------------------------------------------------------------------
    def _x_pd_ok(self, n_pd: int, n_total: int) -> bool:
        if n_total == 0 or n_pd == 0:
            return False
        x = n_pd / n_total
        return self.x_pd_min <= x <= self.x_pd_max

    def _completable(self, n_pd: int, n_other: int, n_elems_used: int,
                     pd_used: bool) -> bool:
        """Существует ли достраивание состава с валидной долей Pd.

        Аналитический перебор по агрегатам (доля зависит только от сумм):
        можно добавить до (max_diff_elem - n_elems_used) новых элементов,
        каждый с числом атомов в [min_atom_i, max_atom_i], общий максимум
        атомов max_atoms. Уайков не учитывается (консервативно вширь):
        если недостижимо даже без Уайкова — недостижимо вообще.
        """
        slots = self.max_diff_elem - n_elems_used
        budget = self.max_atoms - n_pd - n_other
        if budget < 0:
            return False
        # Вариант: завершить прямо сейчас
        if pd_used and n_pd + n_other >= self.min_atoms \
                and self._x_pd_ok(n_pd, n_pd + n_other):
            return True
        if slots == 0 or budget < self.min_atom_i:
            return False
        # Pd ещё не добавлен: перебираем его будущее количество
        if not pd_used:
            for add_pd in range(self.min_atom_i,
                                min(self.max_atom_i, budget) + 1):
                if self._completable(add_pd, n_other, n_elems_used + 1, True):
                    return True
            return False
        # Pd добавлен: добавляем ещё один M и рекурсия
        for add_m in range(self.min_atom_i, min(self.max_atom_i, budget) + 1):
            if self._completable(n_pd, n_other + add_m, n_elems_used + 1, True):
                return True
        return False

    # ------------------------------------------------------------------
    def get_mask_invalid_actions_forward(self, state=None, done=None):
        mask = super().get_mask_invalid_actions_forward(state, done)
        state = self._get_state(state)
        done = self._get_done(done)
        if done:
            return mask

        n_pd = state.get(C.PD_Z, 0)
        n_other = sum(v for k, v in state.items() if k != C.PD_Z)
        n_elems = len(state)

        for i, action in enumerate(self.action_space):
            if mask[i]:
                continue
            if action == self.eos:
                if not self._x_pd_ok(n_pd, n_pd + n_other):
                    mask[i] = True
                continue
            element, n = action
            new_pd = n_pd + (n if element == C.PD_Z else 0)
            new_other = n_other + (n if element != C.PD_Z else 0)
            pd_used = (element == C.PD_Z) or n_pd > 0
            if not self._completable(new_pd, new_other, n_elems + 1, pd_used):
                mask[i] = True

        # Страховка от тупика: пересечение масок опустошило действия
        if all(mask):
            eos_idx = self.action_space.index(self.eos)
            mask[eos_idx] = False
        return mask

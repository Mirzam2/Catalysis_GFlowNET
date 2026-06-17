"""Затравка обучения известными фазами (warm-start вместо холодного старта).

GFlowNet — off-policy метод (Bengio et al. 2021, A.6: обучение на фиксированных
траекториях работает). Берём известные хорошие катализаторы (PdZn, PdIn, PdGa,
Pd3Sn, Pd2Ga) и строим для каждого терминальное состояние окружения, проходя
окружение FORWARD форсированными действиями. Forward-построение надёжнее ручной
сборки словаря состояния: формат вложенных Stack (Crystal -> LatticeParameters
сам Stack) меняется между версиями Mila, а прохождение через нативные step/
set_state даёт корректный формат любой версии.

Затравки опциональны: если построить какую-то фазу не удалось (несовместимость
состава с группой по Уайкову, отсутствие cube-действия под нужную решётку),
она пропускается с предупреждением — обучение продолжается без неё.
"""
import itertools
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import constants as C

logger = logging.getLogger(__name__)


@dataclass
class SeedPhase:
    formula_dict: Dict[int, int]      # {Z: n} в ячейке
    space_group: int
    lattice_params: Tuple[float, float, float, float, float, float]
    label: str = ""


KNOWN_GOOD_PHASES = [
    SeedPhase({46: 1, 30: 1}, 123, (2.90, 2.90, 3.34, 90, 90, 90), "PdZn L1_0"),
    SeedPhase({46: 1, 49: 1}, 221, (3.25, 3.25, 3.25, 90, 90, 90), "PdIn B2"),
    SeedPhase({46: 4, 31: 4}, 198, (4.90, 4.90, 4.90, 90, 90, 90), "PdGa FeSi"),
    SeedPhase({46: 3, 50: 1}, 221, (3.97, 3.97, 3.97, 90, 90, 90), "Pd3Sn L1_2"),
    SeedPhase({46: 2, 31: 2}, 62, (5.48, 4.05, 7.81, 90, 90, 90), "Pd2Ga"),
]

SEED_MILLERS = [(1, 1, 1), (1, 0, 0), (1, 1, 0)]
SEED_OFFSETS = [0.0, 0.5]
SEED_TOPS = [True]


def _set_spacegroup(sg_env, target: int) -> bool:
    """Устанавливает SpaceGroup в target. True при успехе.

    SpaceGroup может требовать действие выбора группы + EOS для завершения.
    Перебираем действия выбора, затем добиваем EOS.
    """
    for a in sg_env.action_space:
        sg_env.set_state(sg_env.source, done=False)
        try:
            _, _, valid = sg_env.step(a)
        except Exception:
            continue
        if not valid:
            continue
        try:
            reached = sg_env.get_space_group() == target
        except Exception:
            reached = False
        if not reached:
            continue
        # группа выбрана; завершаем EOS, если ещё не done
        if not sg_env.done:
            try:
                sg_env.step(sg_env.eos)
            except Exception:
                pass
        if sg_env.done:
            return True
    return False


def _build_terminal_state(env, phase: SeedPhase,
                          hkl: Tuple[int, int, int], offset: float,
                          top: bool) -> Optional[Dict]:
    """Строит одно терминальное состояние окружения для фазы+поверхности.

    Возвращает state-dict или None, если построить не удалось.
    Подход: свежая копия окружения, прохождение forward по стадиям.
    """
    e = env.copy()
    e.reset()

    crystal = e.subenvs[0]
    sg_env = crystal.subenvs[crystal.idx_spacegroup]
    comp_env = crystal.subenvs[crystal.idx_composition]
    lp_env = crystal.subenvs[crystal.idx_latticeparameters]

    # 1) SpaceGroup
    if not _set_spacegroup(sg_env, phase.space_group):
        logger.warning("seed %s: не удалось установить SG=%d",
                       phase.label, phase.space_group)
        return None

    # 2) Composition: устанавливаем dict напрямую и помечаем терминальным
    comp_env.set_state(dict(phase.formula_dict), done=True)

    # 3) LatticeParameters (вложенный Stack: lattice_system + cube).
    #    Строим cube-состояние напрямую из физических параметров через
    #    нативные преобразователи _length2statevalue / _angle2statevalue.
    lp_ok = False
    try:
        a, b, c, al, be, ga = phase.lattice_params
        cube_vals = [
            lp_env._length2statevalue(a), lp_env._length2statevalue(b),
            lp_env._length2statevalue(c), lp_env._angle2statevalue(al),
            lp_env._angle2statevalue(be), lp_env._angle2statevalue(ga),
        ]
        cube_env = lp_env.subenvs[lp_env.idx_cube]
        cube_env.set_state(list(cube_vals), done=True)
        # система решётки (Dummy-подокружение) — завершаем как есть
        lat_sys_env = lp_env.subenvs[0]
        lat_sys_env.set_state(lat_sys_env.source, done=True)
        # собираем состояние LP-Stack
        lp_state = lp_env._get_state(None)
        lp_env._set_substate(0, lat_sys_env.state, lp_state)
        lp_env._set_substate(lp_env.idx_cube, cube_env.state, lp_state)
        lp_env.set_state(lp_state, done=True)
        lp_ok = True
    except Exception as exc:
        logger.warning("seed %s: не удалось построить решётку (%r)",
                       phase.label, exc)
    if not lp_ok:
        return None

    # Собираем состояние Crystal вручную из готовых под-состояний
    crystal_state = crystal._get_state(None)
    crystal._set_substate(crystal.idx_spacegroup, sg_env.state, crystal_state)
    crystal._set_substate(crystal.idx_composition, comp_env.state, crystal_state)
    crystal._set_substate(crystal.idx_latticeparameters, lp_env.state, crystal_state)
    crystal.set_state(crystal_state, done=True)

    # Поверхностные подокружения
    miller_env = e.subenvs[1]
    offset_env = e.subenvs[2]
    tb_env = e.subenvs[3]

    miller_idx = miller_env.miller_set.index(hkl) + 1 if hkl in miller_env.miller_set else None
    if miller_idx is None:
        return None
    miller_env.set_state([miller_idx], done=True)
    offset_env.set_state([offset], done=True)
    tb_env.set_state([1 if top else 2], done=True)

    # Собираем полное состояние Stack
    full = e._get_state(None)
    e._set_substate(0, crystal.state, full)
    e._set_substate(1, miller_env.state, full)
    e._set_substate(2, offset_env.state, full)
    e._set_substate(3, tb_env.state, full)
    e.set_state(full, done=True)

    # Валидация: распаковывается ли в CatalystState
    try:
        e.unpack_state(full)
    except Exception as exc:
        logger.warning("seed %s: состояние не распаковалось (%r)", phase.label, exc)
        return None
    return full


def seed_terminal_states(env, phases: List[SeedPhase] = None) -> List[Dict]:
    """Список валидных терминальных состояний для буфера (пропускает неудачные)."""
    phases = phases or KNOWN_GOOD_PHASES
    states = []
    for phase, hkl, off, top in itertools.product(
            phases, SEED_MILLERS, SEED_OFFSETS, SEED_TOPS):
        if phase.space_group not in C.SPACE_GROUP_SUBSET:
            continue
        s = _build_terminal_state(env, phase, hkl, off, top)
        if s is not None:
            states.append(s)
    logger.info("seed: построено %d терминальных состояний из %d фаз",
                len(states), len(phases))
    return states

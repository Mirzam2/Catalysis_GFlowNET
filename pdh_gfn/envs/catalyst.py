"""PdMCatalyst — главное окружение: Stack(Crystal, Miller, Offset, TopBottom).

Траектория (см. диаграмму из обсуждения):
    SG (12 групп) -> состав Pd-M -> параметры решётки
    -> грань (hkl) -> сдвиг сечения -> верх/низ -> EOS

Crystal внутри уже Stack(SpaceGroup, Composition, LatticeParameters)
с межблочными ограничениями C2/C4; Stack поддерживает вложенность.

Терминальный объект x — сырой кортеж (вариант 3): unpack_state()
переводит его в CatalystState для конвейера награды; канонизация
живёт в кэше, не здесь.
"""
from typing import Dict, List, Optional

from gflownet.envs.composite.stack import Stack
from gflownet.envs.crystals.crystal import Crystal

from .. import constants as C
from ..reward.pipeline import CatalystState
from .composition_pdm import PdMComposition
from .surface import MillerChoice, OffsetCube, TopBottomChoice

IDX_CRYSTAL, IDX_MILLER, IDX_OFFSET, IDX_TOPBOTTOM = 0, 1, 2, 3


class PdMCrystal(Crystal):
    """Crystal с нашей конфигурацией: SG-подмножество + PdMComposition.

    ВАЖНО: PdMComposition подставляется ДО вызова Stack.__init__(), а не
    после.  Оригинальный Crystal жёстко создаёт базовый Composition внутри
    своего __init__; подмена после super().__init__() приводила к
    рассинхронизации envs_unique (Composition с action_space_dim=1505 вместо
    PdMComposition с asd=401) и, как следствие, к невалидным действиям при
    сэмплировании из вложенного Stack.  Поэтому мы собираем subenvs вручную и
    вызываем Stack.__init__() напрямую, минуя Crystal.__init__().
    """

    def __init__(self, space_group_kwargs: Optional[Dict] = None,
                 composition_kwargs: Optional[Dict] = None,
                 lattice_parameters_kwargs: Optional[Dict] = None, **kwargs):
        # --- Настройки ограничений (как в Crystal.__init__) ---
        self.do_spacegroup = True
        self.do_lattice_parameters = True
        self.do_projected_lattice_parameters = False
        self.do_sg_before_composition = True
        self.do_sg_to_composition_constraints = True
        self.do_composition_to_sg_constraints = False
        self.do_sg_to_lp_constraints = True

        sg_kwargs = dict(space_groups_subset=C.SPACE_GROUP_SUBSET)
        sg_kwargs.update(space_group_kwargs or {})
        lp_kwargs = dict(
            min_length=C.LENGTH_MIN, max_length=C.LENGTH_MAX,
            min_angle=C.ANGLE_MIN, max_angle=C.ANGLE_MAX,
        )
        lp_kwargs.update(lattice_parameters_kwargs or {})

        # --- Собираем subenvs с PdMComposition вместо Composition ---
        from gflownet.envs.crystals.lattice_parameters import LatticeParameters
        from gflownet.envs.crystals.spacegroup import SpaceGroup
        from gflownet.utils.crystals.constants import TRICLINIC

        space_group = SpaceGroup(**sg_kwargs)
        composition = PdMComposition(
            **dict(do_spacegroup_check=self.do_sg_to_composition_constraints,
                   **(composition_kwargs or {})))
        lattice_parameters = LatticeParameters(
            lattice_system=TRICLINIC, **lp_kwargs)

        # Порядок: SpaceGroup -> Composition -> LatticeParameters
        self.idx_spacegroup = 0
        self.idx_composition = 1
        self.idx_latticeparameters = 2
        subenvs = (space_group, composition, lattice_parameters)

        # --- Сохраняем kwargs для совместимости с Crystal API ---
        self.composition_kwargs = dict(
            composition_kwargs or {},
            do_spacegroup_check=self.do_sg_to_composition_constraints,
        )
        self.space_group_kwargs = sg_kwargs
        self.lattice_parameters_kwargs = lp_kwargs

        # --- Инициализация через Stack (а не Crystal) ---
        Stack.__init__(self, subenvs=subenvs, **kwargs)


class PdMCatalyst(Stack):
    def __init__(self, crystal_kwargs: Optional[Dict] = None,
                 miller_kwargs: Optional[Dict] = None,
                 offset_kwargs: Optional[Dict] = None, **kwargs):
        crystal = PdMCrystal(**(crystal_kwargs or {}))
        miller = MillerChoice(**(miller_kwargs or {}))
        offset = OffsetCube(**(offset_kwargs or {}))
        topbottom = TopBottomChoice()
        super().__init__(
            subenvs=(crystal, miller, offset, topbottom), **kwargs
        )
        # Примечание: policy/env работают на device='cpu' (дефолт).
        # UMA-потенциал запускается на GPU независимо (device передаётся
        # в UMAPotential напрямую, не через env). Это правильная архитектура:
        # GFlowNet MLP маленький (минуты на CPU), bottleneck — UMA-релаксации.

    # ------------------------------------------------------------------
    def unpack_state(self, state: Optional[Dict] = None) -> CatalystState:
        """Терминальное состояние Stack -> CatalystState для награды."""
        state = self._get_state(state)
        crystal_env: Crystal = self.subenvs[IDX_CRYSTAL]
        crystal_state = self._get_substate(state, IDX_CRYSTAL)

        comp_state = crystal_env._get_substate(
            crystal_state, crystal_env.idx_composition)
        sg_state = crystal_env._get_substate(
            crystal_state, crystal_env.idx_spacegroup)
        lp_state = crystal_env._get_substate(
            crystal_state, crystal_env.idx_latticeparameters)

        sg_env = crystal_env.subenvs[crystal_env.idx_spacegroup]
        lp_env = crystal_env.subenvs[crystal_env.idx_latticeparameters]

        space_group = sg_env.get_state_space_group_int(sg_state) \
            if hasattr(sg_env, "get_state_space_group_int") else sg_state[-1]

        # LatticeParameters — вложенный Stack (lattice_system + cube).
        # Физические (a,b,c,alpha,beta,gamma) получаем через apply_lattice_constraints
        # на cube-подсостоянии с учётом симметрии решёточной системы.
        cube_state = lp_env._get_substate(lp_state, lp_env.idx_cube)
        lattice_system = lp_env.get_lattice_system(lp_state)
        lengths, angles = lp_env.apply_lattice_constraints(cube_state, lattice_system)
        lattice_params = tuple(lengths) + tuple(angles)

        miller_env: MillerChoice = self.subenvs[IDX_MILLER]
        offset_env: OffsetCube = self.subenvs[IDX_OFFSET]
        tb_env: TopBottomChoice = self.subenvs[IDX_TOPBOTTOM]

        return CatalystState(
            space_group=int(space_group),
            composition={int(k): int(v) for k, v in comp_state.items()},
            lattice_params=lattice_params,
            miller=miller_env.state2miller(self._get_substate(state, IDX_MILLER)),
            offset=offset_env.state2offset(self._get_substate(state, IDX_OFFSET)),
            top=bool(tb_env.state2top(self._get_substate(state, IDX_TOPBOTTOM))),
        )

    def states2proxy(self, states: List[Dict]) -> List[CatalystState]:
        """Прокси получает список CatalystState (не тензоры): награда
        вычисляется физическим конвейером, а не нейросетью по эмбеддингу."""
        return [self.unpack_state(s) for s in states]

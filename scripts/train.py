"""
Сборка и запуск обучения PDH-GFlowNet.

Использование:
    conda activate <env>
    pip install -e /path/to/mila-gflownet -e /path/to/pdh-gfn fairchem-core
    python scripts/train.py                        # полный запуск (UMA)
    python scripts/train.py --mock                 # EMT вместо UMA (тесты)
    python scripts/train.py --n-steps 10000 --lr 5e-4  # кастомные параметры
    python scripts/train.py --resume               # продолжить с последнего чекпоинта
"""
import argparse
import gc
import json
import logging
import os
import pickle
import random
import sys
import time
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd


def setup_logging(log_dir: str = "logs/pdh_gfn", level: str = "INFO"):
    """Настройка логирования: консоль + файл.

    Все WARNING/ERROR из конвейера награды, прокси, окружения пишутся в файл,
    чтобы диагностировать проблемы обучения (пустой replay, device-ошибки,
    падения UMA) постфактум.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"

    fmt = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
    datefmt = "%H:%M:%S"

    # корневой логгер
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # файловый хэндлер — всё от DEBUG
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(fh)

    # консольный хэндлер — WARNING+ (чтобы не забивать tqdm)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(ch)

    # наши модули — INFO в файл
    for name in ["pdh_gfn", "pdh_gfn.reward", "pdh_gfn.potential", "pdh_gfn.envs"]:
        logging.getLogger(name).setLevel(logging.DEBUG)

    # gflownet (Mila) — WARNING только
    logging.getLogger("gflownet").setLevel(logging.WARNING)

    logging.info("логирование: %s (файл: %s)", level, log_file)
    return log_file
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pdh_gfn.envs.catalyst import PdMCatalyst
from pdh_gfn.proxy import PDHProxy
from pdh_gfn.reward.cache import RewardCache
from pdh_gfn.reward.hull import HullReference
from pdh_gfn.reward.pipeline import RewardPipeline

from gflownet.utils.common import set_device, set_float_precision
from gflownet.utils.policy import parse_policy_config


# ---------------------------------------------------------------------------
#  Reward pipeline
# ---------------------------------------------------------------------------

def build_pipeline(args) -> RewardPipeline:
    """Собирает RewardPipeline: потенциал + hull + cache + газофазные референсы."""
    if args.mock:
        from pdh_gfn.potential.emt_mock import EMTMockPotential
        bulk_pot = ads_pot = EMTMockPotential()
    else:
        try:
            from pdh_gfn.potential.uma import UMAPotential
        except ImportError as e:
            print(f"ERROR: fairchem-core не установлен или недоступен: {e}")
            print("Установите: pip install fairchem-core")
            print("Или запустите с --mock для тестов: python scripts/train.py --mock")
            sys.exit(1)

        uma_device = args.uma_device or args.device  # uma_device приоритетнее
        print(f"Загрузка UMA модели '{args.model}' (device={uma_device})...")
        t0 = time.time()
        bulk_pot = UMAPotential(model=args.model, task="oc20", device=uma_device)
        ads_pot = bulk_pot  # единый oc20
        print(f"UMA загружена за {time.time() - t0:.1f}s")

    # Газофазные референсы адсорбатов (Seemakurthi ур. 2)
    refs_path = Path(args.cache).with_name("adsorbate_refs.json")
    if refs_path.exists():
        adsorbate_refs = json.loads(refs_path.read_text())
        print(f"adsorbate_refs загружены из {refs_path}: {adsorbate_refs}")
    else:
        print("Вычисление газофазных референсов адсорбатов...")
        from pdh_gfn.reward.refs import compute_adsorbate_refs
        t0 = time.time()
        adsorbate_refs = compute_adsorbate_refs(ads_pot)
        refs_path.parent.mkdir(parents=True, exist_ok=True)
        refs_path.write_text(json.dumps(adsorbate_refs))
        print(f"adsorbate_refs посчитаны за {time.time() - t0:.1f}s: {adsorbate_refs}")

    # Выпуклая оболочка Materials Project
    hull_entries_path = Path(args.hull_entries) if args.hull_entries else None
    if hull_entries_path and not hull_entries_path.exists():
        print(f"WARNING: {hull_entries_path} не найден — HullReference будет пустой.")
        print("  Скачайте: python scripts/fetch_mp_entries.py --api-key YOUR_KEY")
        print("  Без MP-данных e_above_hull=0 для всех структур (нет штрафа за нестабильность).")
        hull_entries_path = None
    # Проверка hull: ОБЯЗАТЕЛЬНО наличие всех чистых элементов (Pd + 25 M),
    # иначе PhaseDiagram аппроксимирует углы произвольно и даёт мусорные
    # e_above_hull > 0.5 для известных стабильных фаз (диагностика 15.06.2026).
    from pymatgen.core.periodic_table import Element as _Element
    from pdh_gfn import constants as _C
    _required_pure = [_Element.from_Z(z).symbol for z in _C.ALL_ELEMENTS]
    hull = HullReference(entries_path=hull_entries_path,
                         require_pure_elements=_required_pure, strict=False)
    if hull._entries:
        print(f"HullReference: загружено {len(hull._entries)} MP-записей")
    else:
        print("HullReference: пустая (нет MP-данных)")

    cache = RewardCache(path=Path(args.cache))
    print(f"RewardCache: {len(cache)} записей в {args.cache}")

    # surrogate-оценщик BE (вариант B): пре-фильтр перед адсорбцией
    surrogate = None
    surrogate_mode = getattr(args, "surrogate", "off")
    if surrogate_mode != "off":
        from pdh_gfn.reward.surrogate import SurrogateBE
        surrogate = SurrogateBE.load(args.surrogate_model)
        print(f"SurrogateBE: загружен {args.surrogate_model}, "
              f"режим={surrogate_mode} (адсорбаты: {list(surrogate.models)})")

    return RewardPipeline(
        potential=bulk_pot,
        adsorption_potential=ads_pot,
        hull=hull,
        cache=cache,
        adsorbate_refs=adsorbate_refs,
        fmax=args.fmax,
        max_sites=args.max_sites,
        e_hull_cutoff=args.e_hull_cutoff,        # #4 ранний выход
        max_sites_light=args.max_sites_light,    # #7 меньше сайтов для H
        prefilter_sites=not args.no_prefilter,   # #2 предфильтр сайтов
        prefilter_keep=args.prefilter_keep,
        save_dir=None if args.save_structures == "off" else args.save_structures,
        use_batch_relaxation=getattr(args, "batch_relax", False),
        cache_only=getattr(args, "cache_only", False),
        surrogate=surrogate,
        surrogate_mode=surrogate_mode,
        surrogate_log=getattr(args, "surrogate_log", None),
        surrogate_gate_thresh=getattr(args, "surrogate_gate_thresh", 0.05),
        surrogate_unc_max=getattr(args, "surrogate_unc_max", 0.5),
        surrogate_cache=getattr(args, "surrogate_cache", None),
    )


# ---------------------------------------------------------------------------
#  Seed buffer
# ---------------------------------------------------------------------------

def build_seed_buffer(env, args) -> Path:
    """Создаёт pkl-файл с затравочными терминальными состояниями из seeding.py."""
    seed_path = Path(args.seed_path)
    seed_path.parent.mkdir(parents=True, exist_ok=True)

    if seed_path.exists():
        print(f"Seed buffer уже существует: {seed_path}")
        return seed_path

    print("Создание seed buffer из известных Pd-M фаз...")
    from pdh_gfn.seeding import seed_terminal_states
    t0 = time.time()
    states = seed_terminal_states(env)
    print(f"  Сгенерировано {len(states)} затравочных состояний за {time.time() - t0:.1f}s")

    # Сохраняем в формате DataFrame, как ожидает BaseBuffer
    rows = []
    for s in states:
        try:
            readable = env.state2readable(s)
        except Exception:
            readable = str(s)
        rows.append({
            "samples": s,
            "samples_readable": readable,
        })
    df = pd.DataFrame(rows)
    df.to_pickle(seed_path)
    print(f"  Seed buffer сохранён: {seed_path}")
    return seed_path


# ---------------------------------------------------------------------------
#  OmegaConf config — программный аналог config/train.yaml из mila-gflownet
# ---------------------------------------------------------------------------

def build_config(args, seed_path=None) -> DictConfig:
    """
    Строит OmegaConf-конфиг, совместимый с gflownet_from_config().

    Структура полностью повторяет config/train.yaml + субконфиги из
    mila-gflownet, но env и proxy будут подставлены напрямую
    (не через instantiate), поэтому их конфиги здесь — заглушки.
    """
    # Настройки buffer
    buffer_train = {"type": None, "path": None, "n": None, "seed": None}
    if seed_path and seed_path.exists():
        buffer_train = {"type": "pkl", "path": str(seed_path), "n": None, "seed": args.seed}

    # Backward batch sizes
    # backward_dataset отключён: наши затравочные состояния (seeding.py)
    # построены через forward-установку subenvs, но backward-валидация
    # Stack требует точного формата «done»-флагов на каждом уровне
    # вложенности, который вручную воспроизвести хрупко. Forward-обучение
    # (основной режим GFlowNet) + replay buffer работает без этого.
    bw_dataset = 0
    bw_replay = args.backward_replay if args.replay_capacity > 0 else 0

    cfg = OmegaConf.create({
        # ---- top-level ----
        "device": args.gfn_device,
        "float_precision": 32,
        "seed": args.seed,
        "n_samples": args.n_samples,

        # ---- logger (gflownet/utils/logger.py:Logger) ----
        "logger": {
            "_target_": "gflownet.utils.logger.Logger",
            "do": {"online": args.wandb, "times": False},
            "project_name": "PDH-GFN",
            "logdir": {
                "root": args.logdir,
                "ckpts": "ckpts",
                "overwrite": not args.resume,
            },
            "lightweight": False,
            "debug": False,
            "progressbar": {"skip": False, "n_iters_mean": 100},
            "context": "0",
            "notes": None,
            "entity": None,
            "run_id": None,
            "is_resumed": bool(args.resume),
            "run_name": None,
            "run_name_date": True,
            "run_name_job": True,
        },

        # ---- proxy — заглушка; подставляется напрямую ----
        "proxy": {
            "_target_": "gflownet.proxy.base.Proxy",
        },

        # ---- env — заглушка; подставляется напрямую ----
        "env": {
            "_target_": "gflownet.envs.base.GFlowNetEnv",
        },

        # ---- buffer ----
        "buffer": {
            "_target_": "gflownet.buffer.base.BaseBuffer",
            "replay_buffer": None,
            "replay_capacity": args.replay_capacity,
            "train": buffer_train,
            "test": {"type": None, "path": None, "n": None, "seed": None},
            "use_main_buffer": False,
            "check_diversity": False,
            "diversity_check_reward_similarity": 0.1,
            "progress_process_dataset": False,
        },

        # ---- evaluator ----
        "evaluator": {
            "_target_": "gflownet.evaluator.base.BaseEvaluator",
            "first_it": True,
            "period": args.eval_period,
            "n": 100,
            "kde": {"bandwidth": 0.1, "kernel": "gaussian"},
            "n_top_k": 5000,
            "top_k": 100,
            "top_k_period": -1,
            "n_trajs_logprobs": 10,
            "logprobs_batch_size": 100,
            "logprobs_bootstrap_size": 10000,
            "max_data_logprobs": 1e5,
            "n_grid": 40000,
            "train_log_period": 1,
            "checkpoints_period": args.ckpt_period,
            "metrics": "all",
        },

        # ---- policy (forward/backward) ----
        "policy": {
            "_target_": "gflownet.policy.base.Policy",
            "shared": None,
            "forward": {
                "type": "mlp",
                "n_hid": args.n_hid,
                "n_layers": args.n_layers,
                "checkpoint": None,
                "reload_ckpt": False,
            },
            "backward": {
                "shared_weights": True,
                "checkpoint": None,
                "reload_ckpt": False,
            },
        },

        # ---- gflownet agent ----
        "gflownet": {
            "_target_": "gflownet.gflownet.GFlowNetAgent",
            "seed": args.seed,
            "optimizer": {
                "lr": args.lr,
                "lr_z_mult": args.lr_z_mult,
                "z_dim": args.z_dim,
                "lr_decay_period": 1000000,
                "lr_decay_gamma": 0.5,
                "method": "adam",
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "sgd_momentum": 0.9,
                "batch_size": {
                    "forward": args.batch_size,
                    "backward_dataset": bw_dataset,
                    "backward_replay": bw_replay,
                },
                "train_to_sample_ratio": 1,
                "n_train_steps": args.n_steps,
                "bootstrap_tau": 0.0,
                "clip_grad_norm": args.clip_grad_norm,
            },
            "state_flow": None,
            "mask_invalid_actions": True,
            "temperature_logits": 1.0,
            "random_action_prob": args.random_action_prob,
            "replay_sampling": "permutation",
            "train_sampling": "permutation",
            "use_context": False,
            "garbage_collection_period": args.gc_period,
            "collect_reversed_logprobs": False,
        },

        # ---- loss (Trajectory Balance) ----
        "loss": {
            "_target_": "gflownet.losses.trajectorybalance.TrajectoryBalance",
            "early_stopping_th": 0.0,
            "ema_alpha": 0.5,
        },
    })
    return cfg


# ---------------------------------------------------------------------------
#  Custom factory — аналог gflownet_from_config, но с явными env и proxy
# ---------------------------------------------------------------------------

def _patch_cube_device_bug():
    """
    Monkey-patch ContinuousCube для исправления бага device в mila-gflownet.

    В gflownet/envs/cube.py строки ~1131, ~1222 и ~1230 создают тензоры через
    torch.zeros() / torch.ones() без указания device=self.device.
    На CUDA это приводит к RuntimeError: tensors on different devices
    при torch.cat().

    Патч оборачивает _sample_actions_batch_forward и
    _sample_actions_batch_backward, чтобы гарантировать что все
    создаваемые внутри torch.zeros/ones используют self.device.
    """
    from gflownet.envs.cube import ContinuousCube

    _orig_forward = ContinuousCube._sample_actions_batch_forward
    _orig_backward = ContinuousCube._sample_actions_batch_backward

    def _patched_forward(self, *args, **kwargs):
        _orig_zeros = torch.zeros
        _orig_ones = torch.ones
        _device = self.device

        def _device_aware_zeros(*size, **kw):
            if 'device' not in kw:
                kw['device'] = _device
            return _orig_zeros(*size, **kw)

        def _device_aware_ones(*size, **kw):
            if 'device' not in kw:
                kw['device'] = _device
            return _orig_ones(*size, **kw)

        torch.zeros = _device_aware_zeros
        torch.ones = _device_aware_ones
        try:
            return _orig_forward(self, *args, **kwargs)
        finally:
            torch.zeros = _orig_zeros
            torch.ones = _orig_ones

    def _patched_backward(self, *args, **kwargs):
        _orig_zeros = torch.zeros
        _orig_ones = torch.ones
        _device = self.device

        def _device_aware_zeros(*size, **kw):
            if 'device' not in kw:
                kw['device'] = _device
            return _orig_zeros(*size, **kw)

        def _device_aware_ones(*size, **kw):
            if 'device' not in kw:
                kw['device'] = _device
            return _orig_ones(*size, **kw)

        torch.zeros = _device_aware_zeros
        torch.ones = _device_aware_ones
        try:
            return _orig_backward(self, *args, **kwargs)
        finally:
            torch.zeros = _orig_zeros
            torch.ones = _orig_ones

    ContinuousCube._sample_actions_batch_forward = _patched_forward
    ContinuousCube._sample_actions_batch_backward = _patched_backward
    print("ContinuousCube patched for CUDA device compatibility.")


def _patch_log_reward_bug():
    """Monkey-patch бага np.log в mila-gflownet (gflownet.py log_train_iteration).

    `np.log(rewards_replay)` / `np.log(rewards)` падают на pandas object-Series
    (`'float' object has no attribute 'log'`) в свежих numpy/pandas. Апстрим-фикс —
    обернуть в `np.asarray(..., dtype=float)`. Чтобы НЕ патчить клон gflownet вручную
    (важно для uv-сборки с чистым git-gflownet), на время вызова log_train_iteration
    подменяем np.log на коэрцящую версию — она меняет поведение ТОЛЬКО при сбое,
    обычные массивы не трогает.
    """
    import numpy as _np
    from gflownet.gflownet import GFlowNetAgent

    _orig = GFlowNetAgent.log_train_iteration
    _real_log = _np.log

    def _safe_log(x, *a, **k):
        try:
            return _real_log(x, *a, **k)
        except (TypeError, AttributeError):
            return _real_log(_np.asarray(x, dtype=float), *a, **k)

    def _patched(self, *args, **kwargs):
        _np.log = _safe_log
        try:
            return _orig(self, *args, **kwargs)
        finally:
            _np.log = _real_log

    GFlowNetAgent.log_train_iteration = _patched
    print("gflownet.log_train_iteration patched (np.log coercion).")


def _move_tensor(x, device):
    """Переносит тензор на device, если он ещё не там."""
    if torch.is_tensor(x) and x.device != device:
        return x.to(device)
    return x


def _set_env_device(env, device):
    """
    Рекурсивно переводит окружение (и все sub-env) на нужный device.

    GFlowNetEnv.__init__ хранит тензоры (action_space_torch, маски и т.д.)
    на self.device.  Когда env создаётся как PdMCatalyst() без device,
    всё остаётся на CPU.  Эта функция переносит все тензорные атрибуты
    и сам env.device на CUDA, чтобы не было RuntimeError: devices differ.

    Подходит для CompositeBase / Stack: обходит subenvs и envs_unique.
    Обходит вложенные Stack-окружения рекурсивно.
    """
    device = set_device(device)
    env.device = device
    env.float = set_float_precision(getattr(env, 'float_precision', 32))

    # Переносим ВСЕ тензорные атрибуты на device
    for attr_name in list(vars(env)):
        attr = getattr(env, attr_name)
        if torch.is_tensor(attr):
            try:
                setattr(env, attr_name, attr.to(device))
            except Exception:
                pass
        elif isinstance(attr, list):
            new_list = [_move_tensor(t, device) if torch.is_tensor(t) else t
                        for t in attr]
            try:
                setattr(env, attr_name, new_list)
            except Exception:
                pass
        elif isinstance(attr, dict):
            new_dict = {}
            for k, v in attr.items():
                if torch.is_tensor(v):
                    new_dict[k] = v.to(device)
                else:
                    new_dict[k] = v
            try:
                setattr(env, attr_name, new_dict)
            except Exception:
                pass

    # Sub-environments в CompositeBase / Stack
    visited = set()
    for subenv_list_attr in ('subenvs', 'envs_unique'):
        if hasattr(env, subenv_list_attr):
            subenv_list = getattr(env, subenv_list_attr)
            if subenv_list is not None:
                for subenv in subenv_list:
                    if id(subenv) not in visited:
                        visited.add(id(subenv))
                        _set_env_device(subenv, device)

    # Дополнительно: вложенные атрибуты, которые сами являются Stack
    for attr_name in list(vars(env)):
        attr = getattr(env, attr_name)
        if hasattr(attr, 'device') and hasattr(attr, 'action_space') and id(attr) not in visited:
            visited.add(id(attr))
            try:
                _set_env_device(attr, device)
            except Exception:
                pass


def _validate_env(env):
    """
    Проверяет целостность структуры Stack-окружения.

    Проверяет:
    1. Соответствие envs_unique[i] типа и action_space с subenvs[j]
    2. Соответствие policy_output_dim = sum(unique_subenv.policy_output_dim)
    3. Согласованность envs_unique и subenvs
    """
    if not hasattr(env, 'subenvs'):
        return

    print("\n=== ENV VALIDATION ===")
    print(f"  env type:          {type(env).__name__}")
    print(f"  env.device:        {env.device}")
    print(f"  action_space_dim:  {env.action_space_dim}")
    print(f"  policy_output_dim: {env.policy_output_dim}")
    print(f"  mask_dim:          {env.mask_dim}")
    print(f"  n_subenvs:         {env.n_subenvs}")
    print(f"  n_unique_envs:     {env.n_unique_envs}")
    print(f"  action_dim:        {env.action_dim}")
    print(f"  eos:               {env.eos}")
    print(f"  continuous:        {env.continuous}")
    print(f"  unique_indices:    {env.unique_indices}")

    for idx_unique in range(env.n_unique_envs):
        uenv = env._get_env_unique(idx_unique)
        subenvs_with_this_idx = [
            i for i, ui in enumerate(env.unique_indices) if ui == idx_unique
        ]
        print(f"\n  Unique env [{idx_unique}]:")
        print(f"    type:              {type(uenv).__name__}")
        print(f"    device:            {uenv.device}")
        print(f"    action_space_dim:  {uenv.action_space_dim}")
        print(f"    policy_output_dim: {uenv.policy_output_dim}")
        print(f"    mask_dim:          {uenv.mask_dim}")
        print(f"    eos:               {uenv.eos}")
        print(f"    continuous:        {uenv.continuous}")
        print(f"    maps to subenvs:   {subenvs_with_this_idx}")

        if hasattr(uenv, 'action_space_torch') and uenv.action_space_torch is not None:
            print(f"    action_space_torch device: {uenv.action_space_torch.device}")

    sum_unique_asd = sum(env._get_env_unique(i).action_space_dim
                        for i in range(env.n_unique_envs))
    sum_unique_pod = sum(env._get_env_unique(i).policy_output_dim
                        for i in range(env.n_unique_envs))

    if env.action_space_dim != sum_unique_asd:
        print(f"\n  WARNING: action_space_dim mismatch!")
        print(f"    env.action_space_dim = {env.action_space_dim}")
        print(f"    sum(unique.action_space_dim) = {sum_unique_asd}")

    if env.policy_output_dim != sum_unique_pod:
        print(f"\n  WARNING: policy_output_dim mismatch!")
        print(f"    env.policy_output_dim = {env.policy_output_dim}")
        print(f"    sum(unique.policy_output_dim) = {sum_unique_pod}")

    type_mismatch_found = False
    for idx_subenv, subenv in enumerate(env.subenvs):
        idx_unique = env._get_unique_idx_of_subenv(idx_subenv)
        uenv = env._get_env_unique(idx_unique)
        if type(subenv) != type(uenv):
            type_mismatch_found = True
            print(f"\n  ERROR: subenv[{idx_subenv}] type {type(subenv).__name__} "
                  f"!= unique[{idx_unique}] type {type(uenv).__name__}")
            print(f"    This is a CRITICAL bug! envs_unique was computed before "
                  f"subenv was replaced with a subclass.")
            print(f"    subenv asd={subenv.action_space_dim}, unique asd={uenv.action_space_dim}")
        if tuple(subenv.action_space) != tuple(uenv.action_space):
            print(f"\n  WARNING: subenv[{idx_subenv}] action_space differs from "
                  f"unique[{idx_unique}] action_space!")
            print(f"    subenv asd={subenv.action_space_dim}, unique asd={uenv.action_space_dim}")

    if not type_mismatch_found:
        print(f"\n  OK: All subenvs and envs_unique are type-consistent.")

    print("\n=== END VALIDATION ===\n")


def pdh_gflownet_from_config(config: DictConfig, env, proxy):
    """
    Собирает GFlowNetAgent из OmegaConf-конфига, используя
    переданные напрямую env и proxy (вместо instantiate из YAML).

    Логика идентична gflownet.utils.common.gflownet_from_config,
    но proxy и env не создаются через Hydra — они уже готовы.
    """
    # Logger
    logger = instantiate(config.logger, config, _recursive_=False)

    # ALWAYS apply _set_env_device
    _set_env_device(env, config.device)

    # Validate environment structure
    _validate_env(env)

    env_maker = partial(env.copy)

    # Buffer
    buffer = instantiate(
        config.buffer,
        env=env,
        proxy=proxy,
        datadir=logger.datadir,
    )

    # Evaluator
    evaluator = instantiate(config.evaluator)

    # Policy (forward / backward)
    forward_config = parse_policy_config(config, kind="forward")
    backward_config = parse_policy_config(config, kind="backward")

    forward_policy = instantiate(
        forward_config,
        env=env,
        device=config.device,
        float_precision=config.float_precision,
    )
    backward_policy = instantiate(
        backward_config,
        env=env,
        device=config.device,
        float_precision=config.float_precision,
        base=forward_policy,
    )

    # State flow (не нужен для Trajectory Balance)
    if config.gflownet.state_flow is not None:
        state_flow = instantiate(
            config.gflownet.state_flow,
            env=env,
            device=config.device,
            float_precision=config.float_precision,
            base=forward_policy,
        )
    else:
        state_flow = None

    # Loss
    loss = instantiate(
        config.loss,
        forward_policy=forward_policy,
        backward_policy=backward_policy,
        state_flow=state_flow,
        device=config.device,
        float_precision=config.float_precision,
    )

    # GFlowNet Agent
    gflownet = instantiate(
        config.gflownet,
        device=config.device,
        float_precision=config.float_precision,
        env_maker=env_maker,
        proxy=proxy,
        loss=loss,
        forward_policy=forward_policy,
        backward_policy=backward_policy,
        state_flow=state_flow,
        buffer=buffer,
        logger=logger,
        evaluator=evaluator,
    )

    return gflownet


# ---------------------------------------------------------------------------
#  Resume from checkpoint
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(logdir: str) -> Path:
    """Ищет последний .ckpt файл в logdir/ckpts/."""
    ckpt_dir = Path(logdir) / "ckpts"
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        return None
    return ckpts[-1]


def _resume_from_checkpoint(gfn, ckpt_path: Path):
    """Загружает чекпоинт в GFlowNetAgent."""
    print(f"Загрузка чекпоинта: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=gfn.device)
    gfn.load_checkpoint(checkpoint)
    print(f"  Восстановлено с итерации {gfn.it}")
    del checkpoint
    gc.collect()


# ---------------------------------------------------------------------------
#  Pre-flight test
# ---------------------------------------------------------------------------

def _preflight_test(env, n_tests=5):
    """
    Выполняет тестовую траекторию в окружении, чтобы убедиться,
    что step() работает корректно с маскированием действий.
    """
    print(f"\n=== PREFLIGHT TEST ({n_tests} trajectories) ===")

    for trial in range(n_tests):
        test_env = env.copy()
        test_env.reset()
        step_count = 0
        max_steps = 50

        while not test_env.done and step_count < max_steps:
            try:
                valid_actions = test_env.get_valid_actions()
            except Exception as e:
                print(f"  Trial {trial}: get_valid_actions() failed at step "
                      f"{step_count}: {e}")
                break

            if not valid_actions:
                active = test_env.state.get('_active', 'N/A')
                print(f"  Trial {trial}: no valid actions at step {step_count}, "
                      f"state['_active']={active}")
                break

            import random as _rng
            action = _rng.choice(valid_actions)

            try:
                state, action, valid = test_env.step(action)
                if not valid:
                    active = test_env.state.get('_active', 'N/A')
                    print(f"  Trial {trial}: step {step_count} returned invalid "
                          f"for action {action}, active_subenv={active}")
                    break
                step_count += 1
            except (ValueError, RuntimeError) as e:
                active = test_env.state.get('_active', 'N/A')
                print(f"  Trial {trial}: error at step {step_count}: {type(e).__name__}: {e}")
                print(f"    action={action}, active_subenv={active}")
                if hasattr(test_env, 'subenvs') and isinstance(active, int):
                    subenv = test_env.subenvs[active]
                    print(f"    subenv type={type(subenv).__name__}, "
                          f"subenv asd={subenv.action_space_dim}")
                break
        else:
            if test_env.done:
                print(f"  Trial {trial}: completed in {step_count} steps")
            else:
                print(f"  Trial {trial}: hit max_steps limit ({max_steps})")

    print("=== END PREFLIGHT TEST ===\n")


# ---------------------------------------------------------------------------
#  Utils
# ---------------------------------------------------------------------------

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
#  Единый YAML-конфиг (config/run.yaml). CLI-флаги переопределяют значения.
# ---------------------------------------------------------------------------

def load_yaml_config(parser):
    """Загружает --config YAML и задаёт ИМ дефолты argparse (CLI override сверху).
    Ключи секций gflownet/pipeline/uma/run/data = argparse-dest имена.
    Возвращает полный cfg-dict (нужны секции reward/curriculum)."""
    pre, _ = parser.parse_known_args()
    path = Path(getattr(pre, "config", None) or "config/run.yaml")
    if not path.exists():
        return {}
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    flat = {}
    for sec in ("gflownet", "pipeline", "uma", "run", "data"):
        flat.update(cfg.get(sec) or {})
    parser.set_defaults(**flat)
    return cfg


def apply_reward_overrides(rcfg):
    """Переопределяет пороги награды в pdh_gfn.constants из cfg['reward']."""
    if not rcfg:
        return
    from pdh_gfn import constants as C
    mapping = {"e_act_ch_max": "E_ACT_CH_MAX", "e_act_ch_min": "E_ACT_CH_MIN",
               "e_act_scale": "E_ACT_SCALE", "e_sel_target": "E_SEL_TARGET",
               "e_sel_scale": "E_SEL_SCALE", "e_sel_max": "E_SEL_MAX",
               "e_sel_fall_scale": "E_SEL_FALL_SCALE", "sigma_stab": "SIGMA_STAB",
               "reward_beta": "REWARD_BETA", "r_valid_eps": "R_VALID_EPS"}
    for yk, ck in mapping.items():
        if rcfg.get(yk) is not None:
            setattr(C, ck, float(rcfg[yk]))


def apply_search_space_overrides(scfg):
    """Переопределяет search-space в pdh_gfn.constants из cfg['search_space'].

    Вызывать ДО построения env (PdMCatalyst): sub-env читают константы в момент
    сборки. Скалярные ключи маппятся в имена констант; списки (m_elements,
    space_groups, miller_set) заменяют соответствующие наборы. m_elements
    пересобирает ALL_ELEMENTS = [Pd] + M.

    ВНИМАНИЕ: space_groups / miller_set / m_elements меняют размер action space
    → чекпоинт с другим набором несовместим (учить с нуля)."""
    if not scfg:
        return
    from pdh_gfn import constants as C
    scalar = {"min_diff_elem": "MIN_DIFF_ELEM", "max_diff_elem": "MAX_DIFF_ELEM",
              "x_pd_min": "X_PD_MIN", "x_pd_max": "X_PD_MAX",
              "min_atoms": "MIN_ATOMS_PER_CELL", "max_atoms": "MAX_ATOMS_PER_CELL",
              "length_min": "LENGTH_MIN", "length_max": "LENGTH_MAX",
              "angle_min": "ANGLE_MIN", "angle_max": "ANGLE_MAX"}
    ints = {"MIN_DIFF_ELEM", "MAX_DIFF_ELEM", "MIN_ATOMS_PER_CELL",
            "MAX_ATOMS_PER_CELL"}
    for yk, ck in scalar.items():
        if scfg.get(yk) is not None:
            setattr(C, ck, int(scfg[yk]) if ck in ints else float(scfg[yk]))
    if scfg.get("m_elements"):
        C.M_ELEMENTS = [int(z) for z in scfg["m_elements"]]
        C.ALL_ELEMENTS = [C.PD_Z] + C.M_ELEMENTS
    if scfg.get("space_groups"):
        C.SPACE_GROUP_SUBSET = [int(s) for s in scfg["space_groups"]]
    if scfg.get("miller_set"):
        C.MILLER_SET = [tuple(int(i) for i in m) for m in scfg["miller_set"]]


def build_schedule_from_cfg(ccfg):
    """RewardSchedule из cfg['curriculum'] (пороги score + фазы). None → дефолт."""
    from pdh_gfn.reward.schedule import RewardSchedule, Phase
    from pdh_gfn.reward import schedule as sched_mod
    if not ccfg:
        return RewardSchedule()
    mapping = {"stab_e_hull_max": "STAB_E_HULL_MAX", "act_e_min": "ACT_E_MIN",
               "act_e_max": "ACT_E_MAX", "sel_e_min": "SEL_E_MIN",
               "sel_e_max": "SEL_E_MAX"}
    for yk, ck in mapping.items():
        if ccfg.get(yk) is not None:
            setattr(sched_mod, ck, float(ccfg[yk]))
    phases = None
    if ccfg.get("phases"):
        phases = [Phase(until_step=ph.get("until"),
                        alpha=float(ph.get("alpha", 1.0)),
                        beta=float(ph.get("beta", 0.0)),
                        gamma=float(ph.get("gamma", 0.0)),
                        mode=ph.get("mode", "linear"))
                  for ph in ccfg["phases"]]
    return RewardSchedule(phases=phases)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="PDH-GFlowNet training — поиск селективных Pd-M катализаторов PDH",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Быстрый тест (EMT, 10 шагов):
  python scripts/train.py --mock --n-steps 10

  # Полный запуск (UMA, 5000 шагов, production-параметры):
  python scripts/train.py

  # Продолжить с чекпоинта:
  python scripts/train.py --resume

  # Кастомный запуск:
  python scripts/train.py --n-steps 10000 --batch-size 4 --lr 5e-4
        """,
    )

    p.add_argument("--config", default="config/run.yaml",
                   help="YAML с параметрами запуска (дефолт config/run.yaml). "
                        "CLI-флаги переопределяют значения из него.")

    # --- Потенциал / pipeline ---
    pot = p.add_argument_group("Потенциал / Pipeline")
    pot.add_argument("--mock", action="store_true",
                     help="EMT вместо UMA (для тестов)")
    pot.add_argument("--model", default="uma-s-1p2",
                     help="UMA model name (default: uma-s-1p2)")
    pot.add_argument("--device", default="cuda",
                     help="(устарел, используй --uma-device) device для UMA")
    pot.add_argument("--uma-device", default=None,
                     help="Device для UMA-потенциала (default: cuda). "
                          "GFlowNet policy ВСЕГДА на cpu — UMA не зависит от этого.")
    pot.add_argument("--gfn-device", default="cpu",
                     help="Device для GFlowNet policy/env (default: cpu). "
                          "MLP маленький, cpu достаточно. Меняй только если знаешь зачем.")
    pot.add_argument("--hull-entries", default="data/mp_pdm_entries.json",
                     help="Путь к JSON с MP-записями для HullReference")
    pot.add_argument("--cache", default="data/reward_cache.jsonl",
                     help="Путь к кэшу наград (JSONL)")
    pot.add_argument("--fmax", type=float, default=0.10,
                     help="Порог сил при релаксации (эВ/A), default: 0.10 (coarse)")
    pot.add_argument("--max-sites", type=int, default=8,
                     help="Макс. число адсорбционных сайтов на поверхность")
    # --- Ускорение пайплайна ---
    pot.add_argument("--e-hull-cutoff", type=float, default=2.0,
                     help="Ранний выход если E_hull > этого (эВ/атом). 2.0 — "
                          "физический потолок (отсекает явный мусор), не "
                          "пытается отсечь метастабильные полиморфы. "
                          "Curriculum-награда сама штрафует за E_hull через "
                          "STAB_E_HULL_MAX (см. reward/schedule.py).")
    pot.add_argument("--max-sites-light", type=int, default=4,
                     help="#7 число сайтов для лёгких симметричных адсорбатов (H*)")
    pot.add_argument("--no-prefilter", action="store_true",
                     help="отключить #2 предфильтр сайтов (single-point перед "
                          "релаксацией)")
    pot.add_argument("--save-structures", default="data/structures",
                     help="Директория для сохранения структур (extxyz) на всех "
                          "этапах: bulk/slab/adsorbates. 'off' отключает.")
    pot.add_argument("--prefilter-keep", type=int, default=4,
                     help="#2 сколько лучших сайтов релаксировать после префильтра")
    pot.add_argument("--surrogate", choices=["off", "shadow", "gate"],
                     default="off",
                     help="surrogate-оценщик BE (вариант B): off | shadow "
                          "(лог pred-vs-real, reward не трогает) | gate (роутинг)")
    pot.add_argument("--surrogate-model", default="data/surrogate_be.pkl",
                     help="путь к обученной модели (scripts/train_surrogate.py)")
    pot.add_argument("--surrogate-log", default="data/surrogate_shadow.jsonl",
                     help="JSONL для shadow-режима: pred-vs-real BE")
    pot.add_argument("--surrogate-gate-thresh", type=float, default=0.05,
                     help="gate: порог r_act*r_sel — выше уходит на полный UMA "
                          "(ниже = агрессивнее скип). Консервативно: 0.05")
    pot.add_argument("--surrogate-unc-max", type=float, default=0.5,
                     help="gate: макс. неопределённость BE (эВ); выше → UMA")
    pot.add_argument("--surrogate-cache", default="data/surrogate_cache.jsonl",
                     help="gate: отдельный кэш предсказаний (НЕ reward_cache)")

    # --- Обучение ---
    train = p.add_argument_group("Обучение")
    train.add_argument("--log-dir", default="logs/pdh_gfn",
                       help="Директория для лог-файлов (default: logs/pdh_gfn)")
    train.add_argument("--n-steps", type=int, default=5000,
                       help="Число итераций обучения (default: 5000)")
    train.add_argument("--batch-size", type=int, default=8,
                       help="Forward траекторий на итерацию (default: 8)")
    train.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate (default: 1e-4)")
    train.add_argument("--lr-z-mult", type=float, default=100,
                       help="Множитель lr для logZ при TB (default: 100)")
    train.add_argument("--z-dim", type=int, default=16,
                       help="Размерность logZ при TB (default: 16)")
    train.add_argument("--n-hid", type=int, default=512,
                       help="Hidden dim в MLP политике (default: 512)")
    train.add_argument("--n-layers", type=int, default=4,
                       help="Число скрытых слоёв в MLP (default: 4)")
    train.add_argument("--clip-grad-norm", type=float, default=0.0,
                       help="Grad clipping (0 = disabled, default: 0)")
    train.add_argument("--random-action-prob", type=float, default=0.0,
                       help="Вероятность случайного действия (exploration)")
    train.add_argument("--seed", type=int, default=0)

    # --- Buffer / Replay ---
    buf = p.add_argument_group("Buffer / Replay")
    buf.add_argument("--replay-capacity", type=int, default=1000,
                     help="Ёмкость replay buffer (0 = disabled)")
    buf.add_argument("--backward-dataset", type=int, default=4,
                     help="Число backward-траекторий из seed buffer (default: 4)")
    buf.add_argument("--backward-replay", type=int, default=4,
                     help="Число backward-траекторий из replay buffer (default: 4)")
    buf.add_argument("--seed-path", default="data/seed_states.pkl",
                     help="Путь к seed buffer (pkl)")

    # --- Curriculum ---
    cur = p.add_argument_group("Curriculum learning")
    cur.add_argument("--curriculum", action="store_true",
                     help="Включить поэтапную награду (stab → +act → +sel → gated). "
                          "Помогает с холодным стартом — в первой фазе политика "
                          "учится только стабильности, потом постепенно "
                          "добавляются активность и селективность.")

    # --- Профайлинг ---
    prof = p.add_argument_group("Профайлинг")
    prof.add_argument("--profile", action="store_true",
                      help="Включить стадийный профайлер reward-конвейера: "
                           "периодический вывод в лог + финальная сводка + "
                           "JSON-дамп в <log-dir>/profile_<timestamp>.json. "
                           "Показывает, где горит время (bulk/slab/H/CH/C3H7/hull) "
                           "и сколько FIRE-шагов уходит на релаксации.")
    prof.add_argument("--batch-relax", action="store_true",
                      help="Батч-релаксация адсорбатов на GPU: все конфигурации "
                           "(сайты × H/CH/C3H7) поверхности релаксируются одним "
                           "батчевым forward UMA вместо последовательных. "
                           "ВНИМАНИЕ: на реальных слэбах оказалась ×3 МЕДЛЕННЕЕ "
                           "(связка по медленной конфигурации) — не использовать.")
    prof.add_argument("--cache-only", action="store_true",
                      help="Офлайн-обучение на УЖЕ посчитанных данных: на промахе "
                           "кэша адсорбция НЕ считается (награда по стабильности). "
                           "Шаг ~3 c вместо ~50 c — быстрая итерация политики на "
                           "закэшированной хорошей области без дорогого UMA.")

    # --- Семплирование ---
    samp = p.add_argument_group("Семплирование после обучения")
    samp.add_argument("--n-samples", type=int, default=1000,
                      help="Число структур после обучения (0 = skip)")
    samp.add_argument("--sample-only", action="store_true",
                      help="Только семплирование (нужен --resume)")

    # --- Логирование ---
    log = p.add_argument_group("Логирование")
    log.add_argument("--logdir", default="./logs/pdh_gfn",
                     help="Корневая директория логов")
    log.add_argument("--wandb", action="store_true",
                     help="Включить wandb логирование")
    log.add_argument("--eval-period", type=int, default=100,
                     help="Evaluировать каждые N шагов")
    log.add_argument("--ckpt-period", type=int, default=50,
                     help="Сохранять чекпоинт каждые N шагов (дёшево; чаще = "
                          "меньше потерь при падении/перезапуске)")
    log.add_argument("--gc-period", type=int, default=0,
                     help="Сборка мусора каждые N шагов (0 = disabled)")

    # --- Resume ---
    resume = p.add_argument_group("Resume")
    resume.add_argument("--resume", action="store_true",
                        help="Продолжить с последнего чекпоинта")
    resume.add_argument("--resume-from", default=None,
                        help="Путь к конкретному .ckpt файлу")

    # --- Debug ---
    debug = p.add_argument_group("Debug")
    debug.add_argument("--preflight", action="store_true",
                       help="Запустить pre-flight тест перед обучением")
    debug.add_argument("--no-train", action="store_true",
                       help="Пропустить обучение (--preflight diagnostics)")

    # Единый YAML-конфиг задаёт дефолты; CLI-флаги ниже переопределяют.
    cfg = load_yaml_config(p)
    args = p.parse_args()

    # ------------------------------------------------------------------
    #  Logging
    # ------------------------------------------------------------------
    log_file = setup_logging(
        log_dir=args.log_dir if hasattr(args, 'log_dir') else "logs/pdh_gfn",
        level="DEBUG",
    )
    logger = logging.getLogger("pdh_gfn.train")
    if cfg:
        logger.info("конфиг: %s", args.config)
    logger.info("аргументы: %s", vars(args))

    # Пороги награды из cfg['reward'] → pdh_gfn.constants (до построения proxy).
    apply_reward_overrides(cfg.get("reward") if cfg else None)

    # Search-space из cfg['search_space'] → pdh_gfn.constants (ДО сборки env).
    apply_search_space_overrides(cfg.get("search_space") if cfg else None)
    if cfg and cfg.get("search_space"):
        from pdh_gfn import constants as _C
        logger.info("search_space: diff_elem=[%d,%d] x_pd=[%.2f,%.2f] "
                    "atoms=[%d,%d] |M|=%d |SG|=%d |miller|=%d",
                    _C.MIN_DIFF_ELEM, _C.MAX_DIFF_ELEM, _C.X_PD_MIN, _C.X_PD_MAX,
                    _C.MIN_ATOMS_PER_CELL, _C.MAX_ATOMS_PER_CELL,
                    len(_C.M_ELEMENTS), len(_C.SPACE_GROUP_SUBSET),
                    len(_C.MILLER_SET))

    # Профайлинг reward-конвейера (флаг --profile)
    if getattr(args, "profile", False):
        from pdh_gfn.profiling import PROFILER
        PROFILER.enabled = True
        logger.info("профайлинг включён (--profile)")

    # ------------------------------------------------------------------
    #  Seeds
    # ------------------------------------------------------------------
    set_seeds(args.seed)

    # ------------------------------------------------------------------
    #  Patches for mila-gflownet bugs
    # ------------------------------------------------------------------
    _patch_cube_device_bug()
    _patch_log_reward_bug()

    # ------------------------------------------------------------------
    #  Environment + Proxy
    # ------------------------------------------------------------------
    # GFlowNet policy + env — на cpu. UMA — на cuda независимо.
    # Это правильная архитектура: MLP маленький (cpu достаточно),
    # bottleneck — UMA-релаксации (GPU внутри RewardPipeline).
    gfn_device = args.gfn_device  # default: "cpu"
    env = PdMCatalyst(device=gfn_device)

    print(f"env type:         {type(env).__name__}")
    print(f"env action_space_dim: {env.action_space_dim}")
    print(f"env policy_output_dim: {env.policy_output_dim}")
    print(f"env mask_dim:     {env.mask_dim}")
    print(f"env device:       {env.device}")

    # Seed buffer (для backward_dataset — warm-start из известных фаз)
    seed_path = build_seed_buffer(env, args)

    pipeline = build_pipeline(args)

    # Curriculum: если флаг --curriculum, проксируем награду через schedule.
    schedule = None
    if args.curriculum:
        # фазы и пороги score'ов — из cfg['curriculum'] (или дефолтные)
        schedule = build_schedule_from_cfg(cfg.get("curriculum") if cfg else None)
        print(f"curriculum: включён, фаз: {len(schedule.phases)}")
        for i, ph in enumerate(schedule.phases):
            until = "до конца" if ph.until_step is None else f"до шага {ph.until_step}"
            print(f"  фаза {i+1}: {until}, "
                  f"α(stab)={ph.alpha} β(act)={ph.beta} γ(sel)={ph.gamma} "
                  f"mode={ph.mode}")

    # CSV-трейс динамики обучения рядом с лог-файлом (trace_<ts>.csv)
    trace_path = log_file.with_name(
        log_file.stem.replace("train_", "trace_") + ".csv")

    # step_callback заполним после создания агента
    proxy = PDHProxy(pipeline=pipeline, schedule=schedule,
                     device=gfn_device, float_precision=32,
                     trace_path=str(trace_path))
    print(f"trace: {trace_path}")

    print(f"proxy: {proxy.__class__.__name__}")

    # ------------------------------------------------------------------
    #  Config + Agent
    # ------------------------------------------------------------------
    config = build_config(args, seed_path=seed_path)
    gfn = pdh_gflownet_from_config(config, env, proxy)

    # Теперь когда агент создан — подключаем callback для curriculum
    if schedule is not None:
        proxy.step_callback = lambda: getattr(gfn, "it", 0)
        print("curriculum step_callback подключён к gfn.it")

    print(f"\nGFlowNet agent assembled.")
    print(f"  device:       {gfn.device}")
    print(f"  n_train_steps: {gfn.n_train_steps}")
    print(f"  batch_size:    {gfn.batch_size}")
    print(f"  forward_policy: {gfn.forward_policy.model}")
    print(f"  loss:          {gfn.loss.name}")
    print(f"  mask_invalid_actions: {gfn.mask_invalid_actions}")
    print(f"  template env type: {type(gfn.env).__name__}")
    print(f"  template env policy_output_dim: {gfn.env.policy_output_dim}")
    print(f"  template env action_space_dim: {gfn.env.action_space_dim}")
    if hasattr(gfn, 'logZ') and gfn.logZ is not None:
        print(f"  logZ shape:    {gfn.logZ.shape}")
    print()

    # ------------------------------------------------------------------
    #  Resume from checkpoint
    # ------------------------------------------------------------------
    if args.resume or args.resume_from:
        if args.resume_from:
            ckpt_path = Path(args.resume_from)
        else:
            ckpt_path = _find_latest_checkpoint(args.logdir)
        if ckpt_path is None or not ckpt_path.exists():
            print(f"ERROR: чекпоинт не найден!")
            if args.resume:
                print(f"  Искал в {Path(args.logdir) / 'ckpts'}")
            sys.exit(1)
        _resume_from_checkpoint(gfn, ckpt_path)

    # ------------------------------------------------------------------
    #  Pre-flight test
    # ------------------------------------------------------------------
    if args.preflight:
        _preflight_test(gfn.env, n_tests=5)
        if args.no_train:
            print("Preflight complete, --no-train specified, exiting.")
            return

    # ------------------------------------------------------------------
    #  Train
    # ------------------------------------------------------------------
    if args.sample_only:
        print("--sample-only: пропускаем обучение")
    else:
        print(f"Starting training: {args.n_steps} steps, "
              f"batch_size={args.batch_size}, lr={args.lr}")
        print(f"  policy: MLP(n_hid={args.n_hid}, n_layers={args.n_layers})")
        print(f"  loss: Trajectory Balance (lr_z_mult={args.lr_z_mult}, z_dim={args.z_dim})")
        print(f"  fmax: {args.fmax} eV/A")
        print(f"  replay_capacity: {args.replay_capacity}")
        if seed_path and seed_path.exists():
            print(f"  seed buffer: {seed_path}")
        print()

        try:
            gfn.train()
        except (ValueError, RuntimeError) as e:
            print(f"\nTraining failed with {type(e).__name__}: {e}")
            if "not present in action space" in str(e):
                print("\nThe policy sampled an action not in the sub-environment's "
                      "action space.")
                print("This is typically caused by a mismatch between envs_unique "
                      "and subenvs in a nested Stack environment.")
                print("Make sure PdMCrystal creates PdMComposition BEFORE calling "
                      "Stack.__init__(), not after.")
                print("See the fixed catalyst.py for the correct pattern.")
            print("\nTry running with --preflight --no-train for diagnostics.")
            raise

        # --- Диагностика награды (чтобы пустой replay не был «немым») ---
        if hasattr(proxy, "error_stats"):
            st = proxy.error_stats()
            print("\n=== reward diagnostics ===")
            print(f"  proxy вызовов:        {st['calls']}")
            print(f"  invalid (R=eps):      {st['invalid']} ({st['invalid_frac']:.1%})")
            print(f"  ошибок конвейера:     {st['errors']} ({st['error_frac']:.1%})")
            if st["calls"]:
                cache = pipeline.cache
                print(f"  записей в кэше наград: {len(cache)}")
                # распределение reward_beta по кэшу
                vals = [b.reward_beta for b in cache._mem.values()]
                if vals:
                    import numpy as _np
                    from pdh_gfn import constants as _C
                    C_REWARD_EPS, _beta = _C.R_VALID_EPS, _C.REWARD_BETA
                    arr = _np.array(vals)
                    print(f"  reward_beta: min={arr.min():.2e} "
                          f"med={_np.median(arr):.2e} max={arr.max():.2e}")
                    n_floor = int((arr <= (C_REWARD_EPS ** _beta) * 1.01).sum())
                    print(f"  на полу (eps^beta):   {n_floor}/{len(arr)}")
            if st["invalid_frac"] > 0.9:
                print("  [!] >90% структур невалидны — проверь fmax, "
                      "max_sites, домен потенциала, пороги гейтов.")
            if st["error_frac"] > 0.1:
                print("  [!] >10% оценок упали с исключением — смотри "
                      "WARNING-логи выше (первые 20 ошибок).")
            print("==========================\n")

        # --- Профиль reward-конвейера (флаг --profile) ---
        if getattr(args, "profile", False):
            from pdh_gfn.profiling import PROFILER
            print("\n=== reward pipeline profile ===")
            print(PROFILER.report())
            print("===============================\n")
            ts = time.strftime("%Y%m%d_%H%M%S")
            prof_path = Path(args.log_dir) / f"profile_{ts}.json"
            PROFILER.dump_json(prof_path)
            print(f"  JSON-профиль: {prof_path}")

    # ------------------------------------------------------------------
    #  Sample from trained GFlowNet
    # ------------------------------------------------------------------
    if args.n_samples > 0 and args.n_samples <= 1e5:
        print(f"\nSampling {args.n_samples} structures...")
        t0 = time.time()
        batch, times = gfn.sample_batch(n_forward=args.n_samples, train=False)
        x_sampled = batch.get_terminating_states(proxy=True)
        energies = gfn.proxy(x_sampled)
        x_sampled = batch.get_terminating_states()
        print(f"  Sampling completed in {time.time() - t0:.1f}s")

        # Сохраняем результаты
        samples_dir = Path("./samples/")
        samples_dir.mkdir(parents=True, exist_ok=True)

        # CSV с читаемыми состояниями и энергиями
        readable_list = []
        for x in x_sampled:
            try:
                readable_list.append(gfn.env.state2readable(x))
            except Exception:
                readable_list.append(str(x))

        df = pd.DataFrame({
            "readable": readable_list,
            "energies": energies.tolist(),
        })

        # Сортируем по энергии (лучшие первые)
        df = df.sort_values("energies", ascending=False).reset_index(drop=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = samples_dir / f"gfn_samples_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  CSV: {csv_path}")

        # Полный pickle
        pkl_path = samples_dir / f"gfn_samples_{timestamp}.pkl"
        dct = {"x": x_sampled, "energy": energies, "readable": readable_list}
        pickle.dump(dct, open(pkl_path, "wb"))
        print(f"  PKL: {pkl_path}")

        # Статистика
        print(f"\n  Energy statistics:")
        print(f"    max  = {energies.max().item():.6f}")
        print(f"    mean = {energies.mean().item():.6f}")
        print(f"    min  = {energies.min().item():.6f}")
        n_positive = (energies > 1e-3).sum().item()
        print(f"    structures with R > 0: {n_positive}/{len(energies)}")

        # Топ-10
        if len(df) >= 10:
            print(f"\n  Top-10 structures:")
            for i, row in df.head(10).iterrows():
                print(f"    #{i}: reward={row['energies']:.6f}  {row['readable'][:100]}")

    # ------------------------------------------------------------------
    #  Close
    # ------------------------------------------------------------------
    if not gfn.use_context:
        gfn.logger.end()


if __name__ == "__main__":
    main()

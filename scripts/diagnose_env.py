"""
Диагностический скрипт для проверки PdMCatalyst окружения.

Запуск:
    python scripts/diagnose_env.py
    python scripts/diagnose_env.py --device cuda
    python scripts/diagnose_env.py --device cpu

Проверяет:
1. Структуру Stack-окружения (subenvs, unique_indices, action spaces)
2. Корректность масок (get_mask_invalid_actions_forward)
3. Корректность sample_actions_batch (действия в action_space)
4. Целостность step() при случайных валидных действиях
5. Совместимость policy output dim с mask dim
6. Синхронность envs_unique и subenvs после deepcopy
"""
import argparse
import random
from copy import deepcopy
from functools import partial

import numpy as np
import torch

from pdh_gfn.envs.catalyst import PdMCatalyst

from gflownet.envs.composite.stack import Stack
from gflownet.envs.base import GFlowNetEnv


def diagnose(env, device="cpu"):
    print("=" * 70)
    print("PDH-GFlowNet Environment Diagnostics")
    print("=" * 70)

    # 1. Basic info
    print(f"\n--- Basic Info ---")
    print(f"  env type:          {type(env).__name__}")
    print(f"  MRO:               {[c.__name__ for c in type(env).__mro__]}")
    print(f"  env.device:        {env.device}")
    print(f"  action_space_dim:  {env.action_space_dim}")
    print(f"  policy_output_dim: {env.policy_output_dim}")
    print(f"  mask_dim:          {env.mask_dim}")
    print(f"  action_dim:        {env.action_dim}")
    print(f"  eos:               {env.eos}")
    print(f"  continuous:        {env.continuous}")
    print(f"  source:            {env.source}")

    if not isinstance(env, Stack):
        print("\n  env is NOT a Stack — skipping composite diagnostics")
        return

    # 2. Sub-environment structure
    print(f"\n--- Sub-environments ({env.n_subenvs}) ---")
    for i, subenv in enumerate(env.subenvs):
        idx_unique = env._get_unique_idx_of_subenv(i)
        print(f"  subenv[{i}]:")
        print(f"    type:              {type(subenv).__name__}")
        print(f"    device:            {subenv.device}")
        print(f"    action_space_dim:  {subenv.action_space_dim}")
        print(f"    policy_output_dim: {subenv.policy_output_dim}")
        print(f"    mask_dim:          {subenv.mask_dim}")
        print(f"    eos:               {subenv.eos}")
        print(f"    continuous:        {subenv.continuous}")
        print(f"    idx_unique:        {idx_unique}")
        if hasattr(subenv, 'action_space_torch') and subenv.action_space_torch is not None:
            print(f"    action_space_torch device: {subenv.action_space_torch.device}")

    # 3. Unique environments
    print(f"\n--- Unique Environments ({env.n_unique_envs}) ---")
    for idx in range(env.n_unique_envs):
        uenv = env._get_env_unique(idx)
        subenvs_with_idx = [i for i, ui in enumerate(env.unique_indices) if ui == idx]
        is_same_as_subenv = any(uenv is env.subenvs[i] for i in subenvs_with_idx)
        print(f"  unique[{idx}]:")
        print(f"    type:              {type(uenv).__name__}")
        print(f"    device:            {uenv.device}")
        print(f"    action_space_dim:  {uenv.action_space_dim}")
        print(f"    policy_output_dim: {uenv.policy_output_dim}")
        print(f"    mask_dim:          {uenv.mask_dim}")
        print(f"    eos:               {uenv.eos}")
        print(f"    maps to subenvs:   {subenvs_with_idx}")
        print(f"    same object as subenv: {is_same_as_subenv}")

    # 4. Dimension consistency checks
    print(f"\n--- Dimension Consistency ---")
    sum_unique_asd = sum(env._get_env_unique(i).action_space_dim
                        for i in range(env.n_unique_envs))
    sum_unique_pod = sum(env._get_env_unique(i).policy_output_dim
                        for i in range(env.n_unique_envs))

    print(f"  env.action_space_dim:  {env.action_space_dim}")
    print(f"  sum(unique.asd):       {sum_unique_asd}")
    print(f"  match:                 {env.action_space_dim == sum_unique_asd}")

    print(f"  env.policy_output_dim: {env.policy_output_dim}")
    print(f"  sum(unique.pod):       {sum_unique_pod}")
    print(f"  match:                 {env.policy_output_dim == sum_unique_pod}")

    # 5. Action space consistency between unique env and subenvs
    print(f"\n--- Action Space Consistency ---")
    for idx_subenv, subenv in enumerate(env.subenvs):
        idx_unique = env._get_unique_idx_of_subenv(idx_subenv)
        uenv = env._get_env_unique(idx_unique)
        subenv_as = tuple(subenv.action_space)
        uenv_as = tuple(uenv.action_space)
        if subenv_as != uenv_as:
            print(f"  MISMATCH: subenv[{idx_subenv}] != unique[{idx_unique}]")
            print(f"    subenv asd: {subenv.action_space_dim}, unique asd: {uenv.action_space_dim}")
            print(f"    subenv first 3: {subenv.action_space[:3]}")
            print(f"    unique first 3: {uenv.action_space[:3]}")
            print(f"    subenv last 3:  {subenv.action_space[-3:]}")
            print(f"    unique last 3:  {uenv.action_space[-3:]}")
        else:
            print(f"  OK: subenv[{idx_subenv}] == unique[{idx_unique}] "
                  f"(type={type(subenv).__name__}, asd={subenv.action_space_dim})")

    # 6. Mask test
    print(f"\n--- Mask Test ---")
    test_env = env.copy()
    test_env.reset()
    for step in range(3):
        if test_env.done:
            break
        active = test_env._get_active_subenv(test_env.state)
        idx_unique = test_env._get_unique_idx_of_subenv(active)
        subenv = test_env.subenvs[active]

        # Get mask from Stack
        mask = test_env.get_mask_invalid_actions_forward()
        print(f"  Step {step}: active_subenv={active}, idx_unique={idx_unique}, "
              f"subenv_type={type(subenv).__name__}")
        print(f"    mask length: {len(mask)} (env.mask_dim={env.mask_dim})")
        print(f"    one-hot prefix: {mask[:env.n_subenvs]}")
        print(f"    n_invalid: {sum(mask)}, n_valid: {sum(not m for m in mask)}")

        # Шаг через step_random: внешний Stack сам уважает активный subenv
        # (наивный random.choice(valid) из всего action_space ломается на
        # вложенных Stack — действие может попасть в не-активный subenv).
        try:
            state, action, valid = test_env.step_random()
            print(f"    step_random: action prefix={action[0]}, valid={valid}")
        except Exception as exc:
            print(f"    step_random упал: {exc!r}")
            break

    # 7. sample_actions_batch test with random policy
    print(f"\n--- sample_actions_batch Test ---")
    test_env2 = env.copy()
    test_env2.reset()

    # Create a random policy output
    policy_output = env.fixed_policy_output.clone().unsqueeze(0).to(env.device)
    mask = test_env2.get_mask_invalid_actions_forward()
    mask_tensor = torch.tensor([mask], dtype=torch.bool, device=env.device)

    try:
        actions = env.sample_actions_batch(
            policy_outputs=policy_output,
            mask=mask_tensor,
            states_from=[test_env2.state],
            is_backward=False,
        )
        action = actions[0]
        print(f"  sampled action: {action}")
        print(f"  action[0] (idx_unique from action): {action[0]}")

        # Verify action is in Stack's action space
        if action in env.action_space:
            print(f"  action in Stack.action_space: YES")
        else:
            print(f"  action in Stack.action_space: NO — BUG!")

        # Depad the action
        active = test_env2._get_active_subenv(test_env2.state)
        idx_unique = test_env2._get_unique_idx_of_subenv(active)
        action_subenv = env._depad_action(action, idx_unique)
        subenv = test_env2.subenvs[active]

        print(f"  depadded action: {action_subenv}")
        print(f"  depadded using idx_unique={idx_unique} (from active_subenv={active})")
        print(f"  subenv type: {type(subenv).__name__}")

        if action_subenv in subenv.action_space:
            print(f"  depadded action in subenv.action_space: YES")
        else:
            print(f"  depadded action in subenv.action_space: NO — THIS IS THE BUG!")
            # Try with the action's own idx_unique
            action_idx_unique = action[0]
            action_subenv_alt = env._depad_action(action, action_idx_unique)
            uenv_alt = env._get_env_unique(action_idx_unique)
            print(f"  trying with action[0]={action_idx_unique}:")
            print(f"    alt depadded: {action_subenv_alt}")
            print(f"    alt unique env type: {type(uenv_alt).__name__}")
            if action_subenv_alt in uenv_alt.action_space:
                print(f"    alt action in alt unique.action_space: YES")
                print(f"    >>> MISMATCH between state's idx_unique and action's idx_unique!")
            else:
                print(f"    alt action in alt unique.action_space: NO")

    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    # 8. deepcopy consistency test
    print(f"\n--- deepcopy Consistency Test ---")
    env_copy = env.copy()
    print(f"  Original env.action_space_dim: {env.action_space_dim}")
    print(f"  Copy env.action_space_dim:      {env_copy.action_space_dim}")
    print(f"  Match: {env.action_space_dim == env_copy.action_space_dim}")

    if hasattr(env, 'subenvs'):
        for i, (orig, copy) in enumerate(zip(env.subenvs, env_copy.subenvs)):
            if type(orig) != type(copy):
                print(f"  subenv[{i}] type MISMATCH: {type(orig).__name__} vs {type(copy).__name__}")
            elif orig.action_space_dim != copy.action_space_dim:
                print(f"  subenv[{i}] asd MISMATCH: {orig.action_space_dim} vs {copy.action_space_dim}")
            else:
                print(f"  subenv[{i}] OK: type={type(orig).__name__}, asd={orig.action_space_dim}")

        # Check envs_unique consistency
        for i, (orig, copy) in enumerate(zip(env.envs_unique, env_copy.envs_unique)):
            same_as_subenv_orig = any(orig is env.subenvs[j] for j in range(env.n_subenvs))
            same_as_subenv_copy = any(copy is env_copy.subenvs[j] for j in range(env_copy.n_subenvs))
            print(f"  unique[{i}]: orig_same_as_subenv={same_as_subenv_orig}, "
                  f"copy_same_as_subenv={same_as_subenv_copy}")

    # 9. Full trajectory test
    print(f"\n--- Full Trajectory Test (3 random trajectories) ---")
    for trial in range(3):
        test_env3 = env.copy()
        test_env3.reset()
        steps = 0
        while not test_env3.done and steps < 100:
            try:
                state, action, valid = test_env3.step_random()
                if not valid:
                    print(f"  Trial {trial}: invalid at step {steps}, action={action}")
                    break
                steps += 1
            except (ValueError, RuntimeError) as e:
                print(f"  Trial {trial}: {type(e).__name__} at step {steps}: {e}")
                active = test_env3.state.get('_active', 'N/A')
                print(f"    active_subenv={active}")
                break
        else:
            if test_env3.done:
                print(f"  Trial {trial}: completed in {steps} steps")
            else:
                print(f"  Trial {trial}: hit max steps")

    print("\n" + "=" * 70)
    print("Diagnostics complete.")
    print("=" * 70)


def main():
    p = argparse.ArgumentParser(description="Diagnose PdMCatalyst environment")
    p.add_argument("--device", default="cpu",
                     help="Device для env (default: cpu — UMA на cuda отдельно)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Creating PdMCatalyst(device='{args.device}')...")
    env = PdMCatalyst(device=args.device)

    diagnose(env, device=args.device)


if __name__ == "__main__":
    main()

"""Масштабирование batch-forward: где горит время и помогает ли батч.

Отвечает на два вопроса, от которых зависит архитектура batch_relax:
  1) predict(N систем) растёт ЛИНЕЙНО по N (GPU compute-bound → батч не спасёт)
     или сублинейно (GPU недозагружен → батч даёт выигрыш, стоит бить крупнее)?
  2) Сколько стоит AtomicData.from_ase + коллация + перенос на КАЖДОМ шаге, и
     сколько даст подход "собрать батч один раз → обновлять позиции in-place"?

Запуск:  python scripts/probe_batch_scaling.py --uma-device cuda
"""
import argparse
import time

import numpy as np
import torch
from ase.build import fcc111, add_adsorbate

from fairchem.core import pretrained_mlip
from fairchem.core.datasets.atomic_data import (
    AtomicData, atomicdata_list_to_batch)


def make_config():
    """Реалистичная конфигурация slab+адсорбат (~46 атомов)."""
    s = fcc111("Pd", size=(3, 3, 4), vacuum=12.0)
    add_adsorbate(s, "C", height=2.0, position="fcc")  # тяжёлый адсорбат-прокси
    s.pbc = True
    return s


def timed(fn, n=10):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return (time.perf_counter() - t0) / n * 1e3  # ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uma-device", default="cuda")
    p.add_argument("--model", default="uma-s-1p2")
    args = p.parse_args()
    dev = args.uma_device

    predictor = pretrained_mlip.get_predict_unit(args.model, device=dev)
    base = make_config()
    print(f"система: {len(base)} атомов\n")

    # --- 1. стоимость from_ase + коллации (CPU) ---
    print("=== 1. CPU-стоимость подготовки батча ===")
    t_fa = timed(lambda: AtomicData.from_ase(base, task_name="oc20", r_edges=False), 20)
    print(f"  from_ase 1 система:        {t_fa:.1f} ms")
    for N in (12, 48):
        datas = [AtomicData.from_ase(base, task_name="oc20", r_edges=False)
                 for _ in range(N)]
        t_col = timed(lambda: atomicdata_list_to_batch(datas), 20)
        print(f"  from_ase×{N} + collate:     {N*t_fa + t_col:.1f} ms "
              f"(collate {t_col:.1f})")

    # --- 2. масштабирование predict по размеру батча ---
    print("\n=== 2. predict(N): compute-bound или батчуется? ===")
    print(f"  {'N':>4} {'total ms':>10} {'ms/система':>12}")
    t1 = None
    for N in (1, 4, 12, 24, 48, 96):
        datas = [AtomicData.from_ase(base, task_name="oc20", r_edges=False)
                 for _ in range(N)]
        batch = atomicdata_list_to_batch(datas).to(dev)
        predictor.predict(batch)  # warmup
        t = timed(lambda: predictor.predict(batch), 10)
        if t1 is None:
            t1 = t
        print(f"  {N:>4} {t:>10.1f} {t/N:>12.2f}"
              + (f"   (×{t/t1:.1f} за ×{N} работы)" if N > 1 else ""))

    # --- 3. rebuild-each-step vs in-place позиции ---
    print("\n=== 3. шаг релаксации: rebuild from_ase vs in-place pos ===")
    N = 12
    atoms_list = [base.copy() for _ in range(N)]

    def step_rebuild():
        datas = [AtomicData.from_ase(a, task_name="oc20", r_edges=False)
                 for a in atoms_list]
        batch = atomicdata_list_to_batch(datas).to(dev)
        out = predictor.predict(batch)
        _ = out["forces"].detach().cpu().numpy()

    datas = [AtomicData.from_ase(a, task_name="oc20", r_edges=False) for a in atoms_list]
    batch0 = atomicdata_list_to_batch(datas).to(dev)
    predictor.predict(batch0)

    def step_inplace():
        # имитируем обновление позиций на GPU без пересборки
        batch0.pos = batch0.pos + 1e-4
        out = predictor.predict(batch0)
        _ = out["forces"]  # держим на GPU

    t_rb = timed(step_rebuild, 10)
    t_ip = timed(step_inplace, 10)
    print(f"  rebuild from_ase каждый шаг (N={N}): {t_rb:.1f} ms/шаг")
    print(f"  in-place pos на GPU        (N={N}): {t_ip:.1f} ms/шаг")
    print(f"  потенциальный выигрыш in-place: ×{t_rb/max(t_ip,1e-9):.1f}")

    print("\nВЫВОД:")
    print("  - если predict(N) почти не растёт с N → бить батч КРУПНЕЕ (поперёк 8 состояний);")
    print("  - если in-place ≪ rebuild → переписать batch_relax на in-place GPU-позиции;")
    print("  - если predict линеен по N и in-place≈rebuild → GPU compute-bound, батч на пределе.")


if __name__ == "__main__":
    main()

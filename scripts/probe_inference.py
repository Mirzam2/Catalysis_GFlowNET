"""Ускорение forward UMA через настройки инференса — с проверкой паритета.

forward ~30 мс на 37 атомов бьёт по КАЖДОЙ релаксации. Пробуем ускорить
(tf32 / bf16-autocast / inference_settings="turbo") и для каждого варианта
меряем скорость И отклонение энергии/сил от базового fp32 — менять точность
можно только если паритет держится (вся калибровка завязана на энергиях).

Запуск:  python scripts/probe_inference.py --uma-device cuda
"""
import argparse
import inspect
import time

import numpy as np
import torch
from ase.build import fcc111, add_adsorbate

from fairchem.core import pretrained_mlip
from fairchem.core.datasets.atomic_data import (
    AtomicData, atomicdata_list_to_batch)


def make_batch(dev, n=12):
    s = fcc111("Pd", size=(3, 3, 4), vacuum=12.0)
    add_adsorbate(s, "C", height=2.0, position="fcc")
    s.pbc = True
    datas = [AtomicData.from_ase(s, task_name="oc20", r_edges=False) for _ in range(n)]
    return atomicdata_list_to_batch(datas).to(dev)


def timed(fn, n=10):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def stats(out):
    return (out["energy"].detach().float().cpu().numpy().copy(),
            out["forces"].detach().float().cpu().numpy().copy())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uma-device", default="cuda")
    p.add_argument("--model", default="uma-s-1p2")
    args = p.parse_args()
    dev = args.uma_device

    print("get_predict_unit signature:")
    print(" ", inspect.signature(pretrained_mlip.get_predict_unit), "\n")

    # --- базовый fp32 ---
    pred = pretrained_mlip.get_predict_unit(args.model, device=dev)
    batch = make_batch(dev)
    pred.predict(batch)  # warmup
    e0, f0 = stats(pred.predict(batch))
    t0 = timed(lambda: pred.predict(batch))
    print(f"{'вариант':<28} {'мс (N=12)':>10} {'dE макс':>12} {'dF макс':>12}")
    print(f"{'baseline fp32':<28} {t0:>10.1f} {'—':>12} {'—':>12}")

    def report(name, fn):
        try:
            fn()  # warmup
            e, f = stats(fn())
            t = timed(fn)
            de = float(np.abs(e - e0).max())
            df = float(np.abs(f - f0).max())
            print(f"{name:<28} {t:>10.1f} {de:>12.2e} {df:>12.2e}"
                  f"   ×{t0/max(t,1e-9):.2f}")
        except Exception as exc:
            print(f"{name:<28} FAIL: {type(exc).__name__}: {exc}")

    # --- tf32 (глобальные флаги Ampere) ---
    def with_tf32():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return pred.predict(batch)
    report("tf32 on", with_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def with_high():
        torch.set_float32_matmul_precision("high")
        return pred.predict(batch)
    report("matmul_precision=high", with_high)
    torch.set_float32_matmul_precision("highest")

    # --- bf16 autocast ---
    def with_bf16():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return pred.predict(batch)
    report("autocast bf16", with_bf16)

    def with_fp16():
        with torch.autocast("cuda", dtype=torch.float16):
            return pred.predict(batch)
    report("autocast fp16", with_fp16)

    # --- inference_settings пресеты, если API принимает ---
    for preset in ("turbo", "default"):
        try:
            pr = pretrained_mlip.get_predict_unit(
                args.model, device=dev, inference_settings=preset)
            b = make_batch(dev)
            pr.predict(b)
            e, f = stats(pr.predict(b))
            t = timed(lambda: pr.predict(b))
            de = float(np.abs(e - e0).max())
            df = float(np.abs(f - f0).max())
            print(f"{'inference='+preset:<28} {t:>10.1f} {de:>12.2e} {df:>12.2e}"
                  f"   ×{t0/max(t,1e-9):.2f}")
        except Exception as exc:
            print(f"{'inference='+preset:<28} FAIL: {type(exc).__name__}: {exc}")

    print("\nБрать вариант с лучшим ×ускорения при dE ≲ 1e-3 эВ и dF ≲ 1e-2.")


if __name__ == "__main__":
    main()

"""Боевой потенциал: UMA (fairchem-core v2).

Использование на кластере an01:
    pip install fairchem-core
    huggingface-cli login   # доступ к весам facebook/UMA

    from pdh_gfn.potential.uma import UMAPotential
    pot = UMAPotential(model="uma-s-1p2", task="oc20", device="cuda")

Задача (task_name):
    oc20 — единый режим для всего пайплайна (объём + поверхность + адсорбция).
            Раньше bulk считали в omat, но переход omat→oc20 между объёмом и
            поверхностью давал разные стартовые геометрии и нефизические BE.
            Сейчас всё на oc20: согласованная энергетическая шкала,
            референсы адсорбатов из ASE g2 тоже в oc20.
"""
from .base import Potential


class UMAPotential(Potential):
    def __init__(self, model: str = "uma-s-1p2", task: str = "oc20",
                 device: str = "cuda"):
        import torch
        from fairchem.core import pretrained_mlip, FAIRChemCalculator

        # tf32 для matmul на Ampere+ (A5000): ~×1.15 к forward при сдвиге
        # энергии ~1 мэВ / сил ~5e-4 (проверено probe_inference). Для BE
        # (разности) и дескрипторов незаметно. Ставится здесь, чтобы
        # calibrate / warmup / train считали в ОДНОЙ точности.
        torch.set_float32_matmul_precision("high")

        self.task = task
        self._predictor = pretrained_mlip.get_predict_unit(model, device=device)
        self._calc = FAIRChemCalculator(self._predictor, task_name=task)

    def get_calculator(self):
        return self._calc

    def batch_eval(self, atoms_list):
        """Энергии (N,) + силы [список per-atom] для СПИСКА систем за ОДИН
        forward UMA. Эквивалент FAIRChemCalculator по системе (проверено:
        dE~1e-6 эВ, max|dF|~1e-7). Основа батчевой релаксации адсорбатов.
        """
        from fairchem.core.datasets.atomic_data import (
            AtomicData, atomicdata_list_to_batch)

        datas = [AtomicData.from_ase(a, task_name=self.task, r_edges=False)
                 for a in atoms_list]
        batch = atomicdata_list_to_batch(datas).to(self._predictor.device)
        out = self._predictor.predict(batch)
        energies = out["energy"].detach().cpu().numpy()
        forces = out["forces"].detach().cpu().numpy()
        nat = batch.natoms.detach().cpu().tolist()
        split, i = [], 0
        for n in nat:
            split.append(forces[i:i + n])
            i += n
        return energies, split

    def with_task(self, task: str) -> "UMAPotential":
        """Лёгкая копия с другим task_name (один predictor, два режима)."""
        clone = object.__new__(UMAPotential)
        clone.task = task
        clone._predictor = self._predictor
        from fairchem.core import FAIRChemCalculator
        clone._calc = FAIRChemCalculator(self._predictor, task_name=task)
        return clone

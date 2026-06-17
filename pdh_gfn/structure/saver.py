"""Сохранение структур конвейера в extxyz на каждом этапе.

Организация (root = data/structures по умолчанию):

    structures/
      <bulk_hash>/                      # канонический хэш релаксированного объёма
        bulk_initial.xyz                #  pyxtal-сборка (до релаксации)
        bulk_relaxed.xyz                #  после UMA (energy в заголовке)
        <h_k_l>_off<bin>_<top|bottom>/  # поверхностная конфигурация
          slab_initial.xyz
          slab_relaxed.xyz              #  energy в заголовке
          ads_H_0.xyz, ads_H_1.xyz ...  #  релаксированные slab+адсорбат,
          ads_CH_0.xyz ...              #  BE и site_idx в заголовке
          ads_C3H7_0.xyz ...
          manifest.json                 #  state, дескрипторы, гейты, награда

Формат extxyz: энергия/метаданные в комментарии второй строки —
читается ase.io.read, совместимо с рабочим скриптом run_uma.py.
"""
import json
from pathlib import Path
from typing import Optional

from ase import Atoms
from ase.io import write as ase_write


class StructureSaver:
    def __init__(self, root: Optional[str] = None):
        """root=None отключает сохранение (все методы — no-op)."""
        self.root = Path(root) if root else None

    @property
    def enabled(self) -> bool:
        return self.root is not None

    # ------------------------------------------------------------------
    def save_atoms(self, atoms: Atoms, relpath: str, **info) -> None:
        """Сохраняет ASE Atoms в extxyz; info уходит в заголовок.

        Constraints снимаются: смесь атомов с/без FixAtoms даёт extxyz
        с разнородными колонками (move_mask), который ase.io.read не
        может прочитать обратно.
        """
        if not self.enabled or atoms is None:
            return
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        a = atoms.copy()
        a.set_constraint()  # снять FixAtoms
        # выкинуть нестандартные per-atom массивы (bulk_wyckoff от pymatgen
        # Slab и т.п.): у адсорбата их нет -> разнородные колонки extxyz,
        # которые ase.io.read не может прочитать
        for key in list(a.arrays.keys()):
            if key not in ("numbers", "positions"):
                del a.arrays[key]
        a.info.update({k: v for k, v in info.items() if v is not None})
        try:
            ase_write(path, a, format="extxyz")
        except Exception:
            pass  # сохранение не должно ронять конвейер

    def save_structure(self, structure, relpath: str, **info) -> None:
        """pymatgen Structure -> extxyz."""
        if not self.enabled or structure is None:
            return
        from pymatgen.io.ase import AseAtomsAdaptor
        self.save_atoms(AseAtomsAdaptor.get_atoms(structure), relpath, **info)

    def save_manifest(self, reldir: str, data: dict) -> None:
        if not self.enabled:
            return
        path = self.root / reldir / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(data, indent=2, default=str,
                                       ensure_ascii=False))
        except Exception:
            pass

    # ------------------------------------------------------------------
    @staticmethod
    def surface_dir(bulk_key: str, miller, offset_bin: int, top: bool) -> str:
        h, k, l = miller
        side = "top" if top else "bottom"
        return f"{bulk_key}/{h}_{k}_{l}_off{offset_bin}_{side}"

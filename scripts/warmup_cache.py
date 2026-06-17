"""Прогрев кэша наград известными хорошими структурами (warm-start).

Решает проблему холодного старта: GFlowNet с нулевой начальной политикой
сэмплирует случайные структуры, почти все получают R≈eps^beta, ландшафт
плоский — обучение не сдвигается. Если в кэше уже есть ненулевые награды
для PdZn/PdIn/PdGa и т.п., политика при первом же случайном попадании
в похожее состояние (тот же SG+состав+близкая решётка) получит сигнал.

Использование:
    # из CIF файлов в data/ref_structures + дефолтные грани
    python scripts/warmup_cache.py --uma-device cuda

    # явный список структур
    python scripts/warmup_cache.py --uma-device cuda \\
        --structures PdZn.cif PdGa.cif Pd3Sn.cif \\
        --millers "1,1,1" "1,0,0"

Что делает:
1. Читает каждую структуру через ASE (xyz/cif/vasp/...).
2. Извлекает SG+composition+lattice через SpacegroupAnalyzer.
3. Для каждой (грань × сдвиг × сторона) собирает CatalystState.
4. Прогоняет pipeline(state) — это релаксирует объём+поверхность,
   считает все 4 BE, дескрипторы, награду И ЗАПИСЫВАЕТ В КЭШ.
5. Параллельно сохраняет xyz по этапам в data/structures/ (через
   StructureSaver, уже встроен в pipeline).
"""
import argparse
import logging
import sys
import time
from pathlib import Path


# Дефолтный набор известных хороших катализаторов PDH из data/ref_structures.
# Если файла нет — структура пропускается с предупреждением. PdGa/Pd3Sn/Pd2Ga
# можно докинуть, если есть CIF (например, из MP по MP_API_KEY).
DEFAULT_STRUCTURES = [
    ("data/ref_structures/PdZn.cif", "PdZn"),
    ("data/ref_structures/Pd_fcc.cif", "Pd"),
    ("data/ref_structures/PdGa.cif", "PdGa"),
    ("data/ref_structures/PdIn.cif", "PdIn"),
    ("data/ref_structures/Pd3Sn.cif", "Pd3Sn"),
]

DEFAULT_MILLERS = [(1, 1, 1), (1, 0, 0), (1, 1, 0)]
DEFAULT_OFFSETS = [0.0, 0.5]


def parse_miller(s: str):
    parts = [int(x) for x in s.replace("(", "").replace(")", "").split(",")]
    if len(parts) != 3:
        raise ValueError(f"некорректный hkl: {s}")
    return tuple(parts)


def main():
    p = argparse.ArgumentParser(description="Прогрев кэша наград known phases")
    p.add_argument("--mock", action="store_true",
                   help="EMT вместо UMA (тесты, не реальный прогрев)")
    p.add_argument("--model", default="uma-s-1p2")
    p.add_argument("--uma-device", default="cuda")
    p.add_argument("--structures", nargs="*", default=None,
                   help="пути к xyz/cif файлам (default: data/ref_structures/*.cif)")
    p.add_argument("--millers", nargs="*", default=None,
                   help='индексы Миллера через запятую, напр. "1,1,1" "1,0,0"')
    p.add_argument("--offsets", nargs="*", type=float, default=None,
                   help="дискретные сдвиги сечения (default: 0.0 0.5)")
    p.add_argument("--cache", default="data/reward_cache.jsonl")
    p.add_argument("--hull-entries", default="data/mp_pdm_entries.json")
    p.add_argument("--save-structures", default="data/structures",
                   help="директория для xyz (off — отключить)")
    p.add_argument("--fmax", type=float, default=0.05,
                   help="строгий fmax: warm-start пишет на много обучения вперёд")
    p.add_argument("--max-sites", type=int, default=8)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("warmup")

    # Структуры
    if args.structures:
        paths = [(s, Path(s).stem) for s in args.structures]
    else:
        paths = [(p_, n) for p_, n in DEFAULT_STRUCTURES if Path(p_).exists()]
        if not paths:
            log.error("ни одной дефолтной структуры не найдено в data/ref_structures/")
            sys.exit(1)

    millers = [parse_miller(m) for m in args.millers] if args.millers else DEFAULT_MILLERS
    offsets = args.offsets if args.offsets is not None else DEFAULT_OFFSETS

    log.info("структур: %d, граней: %d, сдвигов: %d -> %d состояний всего",
             len(paths), len(millers), len(offsets),
             len(paths) * len(millers) * len(offsets) * 2)  # *2 = top/bottom

    # Потенциал
    if args.mock:
        from pdh_gfn.potential.emt_mock import EMTMockPotential
        bulk_pot = ads_pot = EMTMockPotential()
        log.info("режим mock (EMT)")
    else:
        from pdh_gfn.potential.uma import UMAPotential
        log.info("загрузка UMA %s на %s...", args.model, args.uma_device)
        t0 = time.time()
        bulk_pot = UMAPotential(model=args.model, task="oc20",
                                device=args.uma_device)
        ads_pot = bulk_pot  # единый oc20
        log.info("UMA загружена за %.1fs", time.time() - t0)

    # Hull
    from pdh_gfn.reward.hull import HullReference
    hull_path = args.hull_entries if Path(args.hull_entries).exists() else None
    from pymatgen.core.periodic_table import Element as _Element
    from pdh_gfn import constants as _C
    _required_pure = [_Element.from_Z(z).symbol for z in _C.ALL_ELEMENTS]
    hull = HullReference(entries_path=Path(hull_path) if hull_path else None,
                         require_pure_elements=_required_pure, strict=False)
    if hull_path:
        log.info("HullReference: загружено из %s", hull_path)
    else:
        log.warning("hull entries %s не найден — стабильность не оценится",
                    args.hull_entries)

    # Кэш + adsorbate_refs
    from pdh_gfn.reward.cache import RewardCache
    from pdh_gfn.reward.refs import compute_adsorbate_refs
    from pdh_gfn.reward.pipeline import RewardPipeline

    cache = RewardCache(path=args.cache)
    log.info("кэш: %d существующих записей в %s", len(cache), args.cache)

    refs_path = Path(args.cache).with_name("adsorbate_refs.json")
    if refs_path.exists():
        import json
        adsorbate_refs = json.loads(refs_path.read_text())
        log.info("adsorbate_refs загружены: %s",
                 {k: round(v, 3) for k, v in adsorbate_refs.items()})
    else:
        log.info("считаем adsorbate_refs (Seemakurthi ур. 2)...")
        adsorbate_refs = compute_adsorbate_refs(ads_pot, fmax=0.01)
        refs_path.write_text(__import__("json").dumps(adsorbate_refs, indent=2))

    # Pipeline (e_hull_cutoff отключён — известные фазы пропускаем без отсечки)
    save_dir = None if args.save_structures == "off" else args.save_structures
    pipeline = RewardPipeline(
        potential=bulk_pot, adsorption_potential=ads_pot, hull=hull,
        cache=cache, adsorbate_refs=adsorbate_refs,
        fmax=args.fmax, max_sites=args.max_sites,
        e_hull_cutoff=None,    # warm-start: считаем всё, без отсечек
        save_dir=save_dir,
    )

    # Конвертер
    from pdh_gfn.structure.from_xyz import read_any

    # Гонка по структурам
    n_done, n_good, n_skip = 0, 0, 0
    t_start = time.time()
    for path, name in paths:
        log.info("=== %s (%s) ===", name, path)
        try:
            atoms = read_any(path)
        except Exception as exc:
            log.warning("не удалось прочитать %s: %r", path, exc)
            n_skip += len(millers) * len(offsets) * 2
            continue

        # Печатаем РЕАЛЬНУЮ формулу из CIF — поможет сразу заметить,
        # что в файле PdZn.cif на самом деле что-то другое.
        formula = atoms.get_chemical_formula(empirical=True)
        log.info("  загружено: %s, %d атомов в ячейке", formula, len(atoms))
        if "Pd" not in formula:
            log.warning("  ВНИМАНИЕ: в %s нет Pd! пропускаю.", path)
            n_skip += len(millers) * len(offsets) * 2
            continue

        for hkl in millers:
            for off in offsets:
                for top in (True, False):
                    n_done += 1
                    t0 = time.time()
                    try:
                        r = pipeline.evaluate_from_atoms(
                            atoms, miller=hkl, offset=off, top=top)
                        dt = time.time() - t0
                        ok = r.e_sel is not None  # успешно посчитан
                        marker = "OK" if ok else "skip"
                        if ok:
                            n_good += 1
                        log.info("  %s %s hkl=%s off=%.1f top=%s | "
                                 "R=%.2e r=(%.2f,%.2f,%.2f) | %.1fs",
                                 marker, name, hkl, off, top,
                                 r.reward_beta, r.r_stab, r.r_act, r.r_sel, dt)
                    except Exception as exc:
                        log.warning("  FAIL %s hkl=%s off=%.1f top=%s: %r",
                                    name, hkl, off, top, exc)

    log.info("=" * 60)
    log.info("ИТОГО: %d состояний, %d с ненулевой наградой, %d пропущено, %.1fs",
             n_done, n_good, n_skip, time.time() - t_start)
    log.info("кэш теперь: %d записей в %s", len(cache), args.cache)
    log.info("структуры (xyz): %s",
             save_dir if save_dir else "не сохранялись")


if __name__ == "__main__":
    main()

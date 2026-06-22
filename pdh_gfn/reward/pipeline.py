"""Полный конвейер награды — детерминированная часть среды.

Состояние GFlowNet x = (SG, состав, LP, hkl, сдвиг, верх/низ)
    -> pyxtal: объёмная структура
    -> UMA: релаксация объёма (ячейка + атомы)
    -> валидность, E_form, E_hull (оболочка MP)
    -> нарезка поверхностной ячейки -> UMA: релаксация
    -> сайты адсорбции: H*, CH* (стаб+мета), C3H7*
    -> BE по Seemakurthi ур.(2) -> BEP -> E_sel
    -> R = R_valid * R_stab * R_act * R_sel

Кэширование на двух уровнях:
- объём: structure_hash релаксированной структуры (E_hull, E_form);
- поверхность+награда: slab_hash (хэш объёма, грань, бин сдвига, сторона).
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from ase import Atoms

from .. import constants as C
from ..potential.base import Potential, enumerate_sites
from ..structure.builder import build_bulk
from ..structure.canonical import slab_hash, structure_hash
from ..structure.slab import cut_slab
from .cache import RewardCache
from .descriptors import compute_descriptors
from .gates import (RewardBreakdown, composite_reward, invalid_reward,
                    stability_only_reward)
from .hull import HullReference
from ..structure.saver import StructureSaver
from ..profiling import PROFILER

logger = logging.getLogger(__name__)

N_OFFSET_BINS = 8  # дискретизация сдвига сечения для кэша


def make_h_adsorbate() -> Atoms:
    """H* — одиночный атом водорода."""
    return Atoms("H", positions=[[0.0, 0.0, 0.0]])


def make_ch_adsorbate() -> Atoms:
    """CH* — углерод к поверхности, водород сверху (1.09 ангстрема)."""
    return Atoms("CH", positions=[[0, 0, 0], [0, 0, 1.09]])


def make_c3h7_adsorbate() -> Atoms:
    """C3H7* (пропил) — связь с поверхностью через C1.

    Геометрия: C1 в (0,0,0), C2 выше и сбоку, C3 ещё выше. Все 7 H выше z=0.
    sp3-углы приближены (UMA досчитает релаксацией). При translate(site - C1)
    точка крепления попадает на сайт, остальное уходит ВВЕРХ от поверхности.
    """
    # zigzag C-C-C идёт в +z с боковым смещением в y; H1a/H1b у C1 — латерально
    # с z=+0.5 (чуть выше плоскости C1, чтобы не уходили внутрь slab)
    return Atoms(
        "C3H7",
        positions=[
            [ 0.00,  0.00, 0.00],  # C1 — точка крепления
            [ 0.00,  1.27, 0.85],  # C2
            [ 0.00,  0.00, 1.70],  # C3
            [-0.88, -0.51, 0.30],  # H1a
            [ 0.88, -0.51, 0.30],  # H1b
            [-0.88,  1.78, 1.15],  # H2a
            [ 0.88,  1.78, 1.15],  # H2b
            [-0.88, -0.51, 2.00],  # H3a (CH3)
            [ 0.88, -0.51, 2.00],  # H3b
            [ 0.00,  0.00, 2.79],  # H3c (вершина)
        ],
    )


# Реестр адсорбатов: имя -> (фабрика, нужен ли метастабильный сайт)
ADSORBATES = {
    "H": make_h_adsorbate,
    "CH": make_ch_adsorbate,
    "C3H7": make_c3h7_adsorbate,
}


@dataclass
class CatalystState:
    """Распакованное терминальное состояние GFlowNet (сырой кортеж, вариант 3)."""
    space_group: int
    composition: Dict[int, int]            # {Z: n_atoms}
    lattice_params: Tuple[float, ...]      # (a, b, c, alpha, beta, gamma)
    miller: Tuple[int, int, int]
    offset: float                          # [0, 1]
    top: bool


class RewardPipeline:
    def __init__(
        self,
        potential: Potential,
        hull: HullReference,
        cache: Optional[RewardCache] = None,
        adsorbate_refs: Optional[Dict[str, float]] = None,
        fmax: float = C.RELAX_FMAX_COARSE,
        max_sites: int = 8,
        adsorption_potential: Optional[Potential] = None,
        e_hull_cutoff: Optional[float] = None,
        max_sites_light: int = 4,
        prefilter_sites: bool = True,
        prefilter_keep: int = 4,
        save_dir: Optional[str] = None,
        use_batch_relaxation: bool = False,
        cache_only: bool = False,
        surrogate=None,
        surrogate_mode: str = "off",
        surrogate_log: Optional[str] = None,
        surrogate_gate_thresh: float = 0.05,
        surrogate_unc_max: float = 0.5,
        surrogate_cache: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        potential : Potential
            Релаксации объёма и поверхности (UMA task=omat на бою).
        adsorption_potential : Potential, optional
            Отдельный режим для адсорбции (UMA task=oc20). Если None —
            используется ``potential``.
        adsorbate_refs : dict, optional
            Референсные энергии адсорбатов {"H": E, "CH": E, "C3H7": E} в
            выбранной схеме отсчёта, согласованной с Seemakurthi/Purdy.
            BE(ads) = E(slab+ads) - E(slab) - adsorbate_refs[ads].
            По умолчанию нули (сырые энергии связывания относительно
            изолированных фрагментов того же потенциала).
        e_hull_cutoff : float, optional
            УСКОРЕНИЕ #4 (ранний выход): если E_hull выше этого порога
            (эВ/атом), структура заведомо нестабильна — пропускаем дорогую
            адсорбцию (84% времени) и возвращаем награду по одной стабильности.
            None отключает. Рекомендация: ~3*SIGMA_STAB (где R_stab уже мал).
        max_sites_light : int
            УСКОРЕНИЕ #7: число сайтов для симметричных лёгких адсорбатов (H*).
            H* почти не зависит от латеральной позиции — хватает 2-4 сайтов
            против max_sites для CH*/C3H7*.
        prefilter_sites : bool
            УСКОРЕНИЕ #2: single-point по всем сайтам, релаксировать только
            prefilter_keep лучших (по энергии). Отсекает заведомо плохие
            стартовые позиции до дорогой релаксации.
        prefilter_keep : int
            Сколько лучших сайтов релаксировать после предфильтра.
        """
        self.potential = potential
        self.ads_potential = adsorption_potential or potential
        self.hull = hull
        self.cache = cache if cache is not None else RewardCache()
        self.adsorbate_refs = adsorbate_refs or {"H": 0.0, "CH": 0.0, "C3H7": 0.0}
        self.fmax = fmax
        self.max_sites = max_sites
        self.e_hull_cutoff = e_hull_cutoff
        self.max_sites_light = max_sites_light
        self.prefilter_sites = prefilter_sites
        self.prefilter_keep = prefilter_keep
        self.use_batch_relaxation = use_batch_relaxation
        self.cache_only = cache_only
        # surrogate-оценщик BE (вариант B). mode: off | shadow | gate.
        # shadow: считаем pred, логируем pred-vs-real, на reward НЕ влияет.
        # gate: уверенно-низкий хвост скипаем (отдаём pred reward), остальное
        #       и неуверенное — на полный UMA.
        self.surrogate = surrogate
        self.surrogate_mode = surrogate_mode if surrogate is not None else "off"
        self.surrogate_log = surrogate_log
        self.surrogate_gate_thresh = surrogate_gate_thresh  # порог r_act*r_sel
        self.surrogate_unc_max = surrogate_unc_max          # макс. unc (эВ)
        # отдельный кэш предсказаний (НЕ смешиваем с ground-truth reward_cache)
        self.surrogate_cache = (RewardCache(path=Path(surrogate_cache))
                                if surrogate_cache and self.surrogate_mode == "gate"
                                else None)
        self._gate_stats = {"skip": 0, "run": 0}
        self.saver = StructureSaver(save_dir)
        self._bulk_cache: Dict[str, Tuple] = {}

    # ------------------------------------------------------------------
    def __call__(self, state: CatalystState,
                 stability_only: bool = False) -> RewardBreakdown:
        # --- объём -----------------------------------------------------
        with PROFILER.timed("bulk"):  # pyxtal + relax_bulk + hull
            bulk_result = self._relaxed_bulk(state)
        if bulk_result is None:
            return invalid_reward()
        bulk_struct, e_bulk, bulk_key, e_hull = bulk_result

        # --- TIER 0: фаза «только стабильность» (curriculum фаза 1) -------
        # Награда там зависит ТОЛЬКО от e_hull (β=γ=0), а адсорбция — 78%
        # времени. Пропускаем slab+адсорбцию, отдаём награду по стабильности.
        # Кэш поверхности НЕ трогаем: в фазах 2-3 дескрипторы досчитаются.
        if stability_only:
            return stability_only_reward(e_hull)

        # --- УСКОРЕНИЕ #4: ранний выход для нестабильных структур --------
        # Если E_hull выше порога, R_stab уже мал и доминирует в награде —
        # дорогая адсорбция (84% времени) не изменит вывод «плохой кандидат».
        if self.e_hull_cutoff is not None and e_hull > self.e_hull_cutoff:
            logger.debug("ранний выход: E_hull=%.3f > cutoff=%.3f, SG=%d %s",
                         e_hull, self.e_hull_cutoff,
                         state.space_group, state.composition)
            return stability_only_reward(e_hull)

        # --- кэш поверхности+награды ------------------------------------
        sl_key = slab_hash(
            bulk_key, state.miller,
            int(state.offset * N_OFFSET_BINS), state.top,
        )
        cached = self.cache.get(sl_key)
        if cached is not None:
            logger.debug("кэш-хит: %s, reward=%.2e", sl_key[:8], cached.reward_beta)
            return cached
        # gate: повторный скип берём из surrogate-кэша ДО slab-релаксации
        if self.surrogate_cache is not None:
            scached = self.surrogate_cache.get(sl_key)
            if scached is not None:
                logger.debug("surrogate-кэш-хит: %s", sl_key[:8])
                return scached

        # ОФЛАЙН-режим: учим политику на УЖЕ посчитанных данных. На промахе
        # кэша не считаем дорогую адсорбцию — отдаём награду по стабильности
        # (bulk нужен лишь для ключа/hull). Так шаг идёт за ~3 c вместо ~50 c,
        # политика концентрируется на закэшированной хорошей области.
        if self.cache_only:
            return stability_only_reward(e_hull)

        # --- поверхность -------------------------------------------------
        surf_dir = StructureSaver.surface_dir(
            bulk_key, state.miller, int(state.offset * N_OFFSET_BINS), state.top)
        with PROFILER.timed("slab_cut"):
            slab = cut_slab(bulk_struct, state.miller, state.offset, state.top)
        if slab is None or not self.potential.supports(slab):
            result = invalid_reward()
            self.cache.put(sl_key, result)
            return result
        self.saver.save_atoms(slab, f"{surf_dir}/slab_initial.xyz")
        slab_res = self.potential.relax_slab(slab, fmax=self.fmax)
        if not slab_res.valid:
            result = invalid_reward()
            self.cache.put(sl_key, result)
            return result
        self.saver.save_atoms(slab_res.atoms, f"{surf_dir}/slab_relaxed.xyz",
                              energy=slab_res.energy)

        # surrogate (вариант B): предсказание BE по геометрии сайтов слэба.
        # В shadow только логируем pred-vs-real (см. ниже) — reward не трогаем.
        surr_pred = None
        if self.surrogate_mode != "off":
            surr_pred = self._surrogate_predict(slab_res.atoms, e_hull)
        # gate: уверенно-низкий хвост — скип адсорбции, отдаём pred reward
        if self.surrogate_mode == "gate" and surr_pred is not None:
            skipped = self._gate_skip(surr_pred, e_hull)
            self._gate_stats["skip" if skipped is not None else "run"] += 1
            tot = self._gate_stats["skip"] + self._gate_stats["run"]
            if tot % 100 == 0:
                logger.info("surrogate gate: skip=%d run=%d (%.0f%% скип)",
                            self._gate_stats["skip"], self._gate_stats["run"],
                            100 * self._gate_stats["skip"] / tot)
            if skipped is not None:
                if self.surrogate_cache is not None:
                    self.surrogate_cache.put(sl_key, skipped)
                return skipped

        # --- адсорбция: H*, CH* (стаб+мета), C3H7* ------------------------
        if self.use_batch_relaxation:
            # БАТЧ: все конфигурации всех адсорбатов одним GPU-вызовом.
            from ..potential.batch import batch_adsorption_energies

            def saver_cb(name, i, be, atoms):
                if self.saver.enabled:
                    self.saver.save_atoms(
                        atoms, f"{surf_dir}/ads_{name}_{i}.xyz",
                        BE=be, adsorbate=name, site_idx=i)

            with PROFILER.timed("ads.batch"):
                be_map = batch_adsorption_energies(
                    self.ads_potential, slab_res.atoms,
                    {k: ADSORBATES[k] for k in ("CH", "H", "C3H7")},
                    e_slab=slab_res.energy,
                    refs=self.adsorbate_refs,
                    fmax=self.fmax,
                    max_sites=self.max_sites,
                    max_sites_light=self.max_sites_light,
                    prefilter=self.prefilter_sites,
                    prefilter_keep=self.prefilter_keep,
                    on_relaxed=saver_cb if self.saver.enabled else None,
                )
            ch_energies = be_map["CH"]
            h_energies = be_map["H"]
            c3h7_energies = be_map["C3H7"]
        else:
            def be_list(name):
                # УСКОРЕНИЕ #7: H* симметричен -> меньше сайтов
                n_sites = self.max_sites_light if name == "H" else self.max_sites
                # сохранение релаксированных slab+адсорбат через callback
                saver_cb = None
                if self.saver.enabled:
                    def saver_cb(i, be, atoms, _name=name):
                        self.saver.save_atoms(
                            atoms, f"{surf_dir}/ads_{_name}_{i}.xyz",
                            BE=be, adsorbate=_name, site_idx=i)
                return self.ads_potential.adsorption_energies(
                    slab_res.atoms, ADSORBATES[name](),
                    e_slab=slab_res.energy,
                    e_adsorbate_ref=self.adsorbate_refs[name],  # KeyError если нет
                    max_sites=n_sites,
                    fmax=self.fmax,
                    prefilter=self.prefilter_sites,       # УСКОРЕНИЕ #2
                    prefilter_keep=self.prefilter_keep,
                    on_relaxed=saver_cb,
                    stage_prefix=name,                    # профиль: H/CH/C3H7 раздельно
                )

            ch_energies = be_list("CH")      # нужны два минимума (стаб + мета)
            h_energies = be_list("H")
            c3h7_energies = be_list("C3H7")

        if len(ch_energies) < 2:
            result = invalid_reward()
            self.cache.put(sl_key, result)
            return result
        if not h_energies or not c3h7_energies:
            result = invalid_reward()
            self.cache.put(sl_key, result)
            return result

        be_ch, be_ch_meta = ch_energies[0], ch_energies[1]
        be_h = h_energies[0]
        be_c3h7 = c3h7_energies[0]

        # ФИЛЬТР АБСУРДНЫХ BE: физически адсорбация даёт BE в окне
        # примерно ±3 эВ для H/C3H7, ±5 для CH. Если UMA выдаёт −18 эВ —
        # это патология геометрии (атом провалился внутрь slab, ВдВ-конфликт
        # после релаксации, развалившийся slab). Принимать такие BE опасно:
        # они дают огромные E_sel/Eact и reward hacking. Лучше выкинуть как
        # invalid и потом разобраться по сохранённой xyz.
        if (abs(be_h) > 5.0 or abs(be_ch) > 6.0 or abs(be_ch_meta) > 6.0
                or abs(be_c3h7) > 5.0):
            logger.warning("отказ по абсурдным BE: SG=%d %s | "
                           "H=%.2f CH=%.2f meta=%.2f C3H7=%.2f",
                           state.space_group, state.composition,
                           be_h, be_ch, be_ch_meta, be_c3h7)
            result = invalid_reward()
            self.cache.put(sl_key, result)
            return result

        # surrogate shadow: лог pred-vs-real на живых данных (валидация B
        # перед включением гейта). На reward не влияет.
        if self.surrogate_mode == "shadow" and surr_pred is not None:
            self._log_surrogate(sl_key, e_hull, surr_pred,
                                be_h, be_ch, be_ch_meta, be_c3h7)

        # --- дескрипторы и награда ----------------------------------------
        desc = compute_descriptors(be_h, be_ch, be_ch_meta, be_c3h7)
        result = composite_reward(desc, e_hull)
        # Сохраняем raw BE в breakdown для пересчёта reward через schedule
        result.be_h = be_h
        result.be_ch = be_ch
        result.be_ch_meta = be_ch_meta
        result.be_c3h7 = be_c3h7
        logger.info("награда: SG=%d %s hkl=%s | BE: H=%.2f CH=%.2f meta=%.2f "
                    "C3H7=%.2f | Eact_CH=%.2f E_sel=%.2f E_hull=%.3f | "
                    "r=(%.3f,%.3f,%.3f) R=%.2e",
                    state.space_group, state.composition, state.miller,
                    be_h, be_ch, be_ch_meta, be_c3h7,
                    desc.e_act_ch, desc.e_sel, e_hull,
                    result.r_stab, result.r_act, result.r_sel,
                    result.reward_beta)
        self.cache.put(sl_key, result)
        # манифест: всё для анализа этой поверхности (state, BE, дескрипторы, награда)
        self.saver.save_manifest(surf_dir, {
            "state": {"space_group": state.space_group,
                      "composition": state.composition,
                      "lattice_params": state.lattice_params,
                      "miller": state.miller, "offset": state.offset,
                      "top": state.top},
            "bulk_key": bulk_key,
            "e_hull": e_hull,
            "BE": {"H": be_h, "CH": be_ch, "CH_meta": be_ch_meta,
                   "C3H7": be_c3h7},
            "descriptors": {"e_act_ch": desc.e_act_ch,
                            "e_act_cc": desc.e_act_cc,
                            "e_sel": desc.e_sel},
            "reward": {"r_stab": result.r_stab, "r_act": result.r_act,
                       "r_sel": result.r_sel,
                       "reward_beta": result.reward_beta},
        })
        return result

    # ------------------------------------------------------------------
    def _surrogate_predict(self, slab_atoms, e_hull):
        """Сайт-фичи релаксированного слэба → {name:(best_be, unc)} + CH_meta.
        Не должен ронять конвейер: на любом сбое возвращает None."""
        try:
            from .surrogate import full_features
            sites = enumerate_sites(slab_atoms, max_sites=self.max_sites)
            if not sites:
                return None
            cell = np.asarray(slab_atoms.cell[:], float)
            Z, pos = slab_atoms.numbers, slab_atoms.positions
            use_soap = getattr(self.surrogate, "use_soap", False)
            feats = [f for s in sites
                     if (f := full_features(cell, Z, pos, np.asarray(s, float),
                                            e_hull, use_soap)) is not None]
            if not feats:
                return None
            feats = np.asarray(feats, float)
            pred = self.surrogate.predict_slab(feats)      # {name:(best,unc)}
            pred["CH_meta"] = self.surrogate.predict_ch_meta(feats)
            return pred
        except Exception:
            logger.debug("surrogate predict failed", exc_info=True)
            return None

    def _log_surrogate(self, sl_key, e_hull, pred,
                       be_h, be_ch, be_ch_meta, be_c3h7):
        """JSONL pred-vs-real для офлайн-оценки точности surrogate в shadow."""
        if not self.surrogate_log:
            return
        import json
        g = lambda n, i: pred.get(n, (None, None))[i]
        rec = {
            "key": sl_key, "e_hull": e_hull,
            "pred": {"H": g("H", 0), "CH": g("CH", 0), "C3H7": g("C3H7", 0),
                     "CH_meta": pred.get("CH_meta")},
            "unc": {"H": g("H", 1), "CH": g("CH", 1), "C3H7": g("C3H7", 1)},
            "real": {"H": be_h, "CH": be_ch, "CH_meta": be_ch_meta,
                     "C3H7": be_c3h7},
        }
        try:
            with open(self.surrogate_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    def _gate_skip(self, pred, e_hull):
        """Решение gate: вернуть предсказанный RewardBreakdown, если уверенно
        низко (скип адсорбции), иначе None (гнать полный UMA).

        На UMA уходит всё, где предсказанная активность×селективность не
        пренебрежимо мала ИЛИ велика неопределённость — ловит и потенциально
        интересные структуры, и переоценку оценщика (защита от reward hacking).
        Стабильность (e_hull) известна точно, гейтим только act/sel."""
        try:
            be_h, be_ch, be_c3h7 = pred["H"][0], pred["CH"][0], pred["C3H7"][0]
            be_ch_meta = pred.get("CH_meta") or be_ch
            unc = max(pred["H"][1], pred["CH"][1], pred["C3H7"][1])
        except Exception:
            return None
        if unc > self.surrogate_unc_max:        # не доверяем → UMA
            return None
        desc = compute_descriptors(be_h, be_ch, be_ch_meta, be_c3h7)
        result = composite_reward(desc, e_hull)
        if result.r_act * result.r_sel > self.surrogate_gate_thresh:
            return None                          # потенциально хорошо/переоценка → UMA
        # уверенно-низко: raw BE кладём для пересчёта schedule под фазу curriculum
        result.be_h, result.be_ch = be_h, be_ch
        result.be_ch_meta, result.be_c3h7 = be_ch_meta, be_c3h7
        return result

    # ------------------------------------------------------------------
    def _relaxed_bulk(self, state: CatalystState):
        """Сборка + релаксация объёма с кэшем по хэшу релаксированной структуры."""
        candidates = build_bulk(
            state.space_group, state.composition, state.lattice_params,
            n_samples=1,
        )
        if not candidates:
            return None
        return self._relax_and_hull(candidates[0], state)

    def evaluate_from_atoms(self, atoms, miller, offset, top):
        """Прогон конвейера, начиная с ГОТОВОЙ структуры (CIF/xyz).

        Используется для warm-start: даём известные хорошие катализаторы
        напрямую, без pyxtal-сборки. Возвращает RewardBreakdown и кэширует
        результат в reward_cache.jsonl под обычным slab_hash.
        """
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        struct = AseAtomsAdaptor.get_structure(atoms)
        sga = SpacegroupAnalyzer(struct, symprec=0.1)
        composition = {}
        for site in struct:
            z = site.specie.Z
            composition[z] = composition.get(z, 0) + 1
        # Защита от подмены: в структуре ДОЛЖЕН быть Pd (Z=46).
        # Без этого warmup может тихо посчитать награду для AlSb из файла
        # PdZn.cif (реальный кейс из практики: CIF с правильным именем,
        # но неправильным содержимым).
        if 46 not in composition:
            logger.warning("evaluate_from_atoms: в композиции нет Pd! "
                           "композиция=%s — пропускаем (вероятно, в CIF "
                           "не та фаза, что предполагалось)", composition)
            return invalid_reward()
        # Защита от моноатомных structures: нужно >= 4 атомов для slab
        n_atoms = sum(composition.values())
        if n_atoms < 4:
            logger.warning("evaluate_from_atoms: %d атомов в ячейке (<4) "
                           "— расширяем в (2,2,2) суперъячейку", n_atoms)
            atoms = atoms.repeat((2, 2, 2))
            struct = AseAtomsAdaptor.get_structure(atoms)
            composition = {}
            for site in struct:
                z = site.specie.Z
                composition[z] = composition.get(z, 0) + 1
        lat = struct.lattice
        state = CatalystState(
            space_group=sga.get_space_group_number(),
            composition=composition,
            lattice_params=(lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma),
            miller=miller, offset=float(offset), top=bool(top),
        )
        # Сначала пробуем нормальный путь через pyxtal
        bulk_result = self._relaxed_bulk(state)
        if bulk_result is None:
            # pyxtal не справился (моноатом/сложная Wyckoff/...) —
            # используем структуру из CIF напрямую.
            bulk_result = self._relax_and_hull(struct, state)
            if bulk_result is None:
                return invalid_reward()
            # Помещаем в _bulk_cache по правильному ключу, чтобы повторный
            # _relaxed_bulk в __call__ сразу вернул его.
            self._bulk_cache[bulk_result[2]] = bulk_result
        # Теперь обычный __call__ найдёт bulk в кэше и не пойдёт через pyxtal.
        # Но __call__ всё равно зовёт _relaxed_bulk заново — поэтому
        # подменяем его на одноразовый, который вернёт готовый результат.
        original = self._relaxed_bulk
        self._relaxed_bulk = lambda _s: bulk_result
        try:
            return self.__call__(state)
        finally:
            self._relaxed_bulk = original

    def _relax_and_hull(self, struct, state):
        """Релаксация объёма + e_hull. Используется и из _relaxed_bulk, и из
        evaluate_from_atoms. Принимает pymatgen Structure напрямую."""
        from pymatgen.io.ase import AseAtomsAdaptor
        atoms = AseAtomsAdaptor.get_atoms(struct)
        if not self.potential.supports(atoms):
            return None
        # Объём релаксируем ЖЁСТЧЕ поверхности (fmax<=0.05 всегда):
        # недорелаксированная ячейка (стартовое a от политики может быть
        # далёким) даёт растянутую решётку -> слабые BE -> e_act завышен
        # на 3+ эВ -> r_act на полу у ВСЕХ кандидатов (диагноз из лога
        # обучения 2026-06-12). Объёмная релаксация дешёвая (<=16 атомов),
        # экономить на ней нельзя.
        bulk_fmax = min(self.fmax, 0.05)
        res = self.potential.relax_bulk(atoms, fmax=bulk_fmax)
        if not res.valid:
            return None
        # sanity: объём на атом металлического интерметаллида 8-35 A^3;
        # вне диапазона = недорелаксированная/развалившаяся ячейка
        v_per_atom = res.atoms.get_volume() / len(res.atoms)
        if not (8.0 < v_per_atom < 35.0):
            logger.debug("отказ по V/atom=%.1f A^3: SG=%d %s",
                         v_per_atom, state.space_group, state.composition)
            return None
        relaxed = AseAtomsAdaptor.get_structure(res.atoms)
        key = structure_hash(relaxed)
        if key in self._bulk_cache:
            return self._bulk_cache[key]
        with PROFILER.timed("hull"):
            e_hull = self.hull.e_above_hull(relaxed, res.energy)
        # сохранение: исходная и релаксированный объём
        self.saver.save_structure(struct, f"{key}/bulk_initial.xyz",
                                  space_group=state.space_group)
        self.saver.save_atoms(res.atoms, f"{key}/bulk_relaxed.xyz",
                              energy=res.energy, e_hull=e_hull,
                              space_group=state.space_group)
        out = (relaxed, res.energy, key, e_hull)
        self._bulk_cache[key] = out
        return out

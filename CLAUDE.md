# CLAUDE.md — операционная память проекта pdh-gfn

Это инструкция для Claude Code. Прочти **полностью** перед первым ответом в новой сессии — без этого ты будешь предлагать вещи, которые уже пробовались и не работают.

---

## 1. Что это за проект (5 секунд)

ML-ускоренный поиск селективных Pd-M интерметаллидов для дегидрирования пропана (PDH). GFlowNet (от Mila) сэмплирует структуры (space_group + состав + параметры решётки + грань + сдвиг + сторона), а универсальный нейронный потенциал UMA считает энергии связывания H/CH/C3H7 на поверхности. Из них по формулам Seemakurthi 2025 вычисляются дескрипторы активности и селективности; reward — сигмоидное произведение трёх гейтов (стабильность, активность, селективность).

Цель: найти PdZn-подобные катализаторы (высокая селективность к пропилену) без полного DFT-скрининга. Заказчик — Норникель, научный руководитель — А. Квашнин (Сколтех), скептичный экспериментатор — Алексей.

---

## 2. Принципы работы Григоши

- **Терсный, минимум формализма.** Не делай длинных вступлений. Сразу к делу.
- **Сначала обсуждаем в чате**, потом пишем код. Не лезь сразу писать всё — задай уточнения если есть сомнения.
- **Русский в чате**, английский в коде/комментариях за редкими исключениями (доменные термины часто на русском).
- **Терминология**: "поверхностные конфигурации" / "поверхностные структуры" вместо "сляб" (транслит slab).
- **Короткие фиксы > обширные объяснения.** Не переписывай весь модуль если можно поменять 3 строки.
- **Скриптами автоматизирует охотно** — Claude Code специально для этого ставился.
- Формат вывода: минималистичный для слайдов/спецификаций, детальный/структурированный — для лит-обзоров и формальных документов.

---

## 3. КРИТИЧЕСКИ ВАЖНОЕ — что НЕ трогать

### 3.1 Архитектура потенциалов: ВСЁ на oc20

Долгое время был ад с двумя task: `bulk=omat`, `surface=oc20`. Это давало:
- разные стартовые геометрии после переключения task
- BE на больших ячейках уходили в +30...+50 эВ (артефакты UMA на патологических геометриях)
- E_hull казался корректным для p-блока, но был сломан для d-металлов

Решение: **ВСЁ считаем на oc20**. И bulk, и slab, и адсорбаты. Hull тоже пересчитываем в oc20-шкале через `scripts/rebuild_hull_oc20.py`.

```python
# scripts/train.py, warmup_cache.py, calibrate.py:
bulk_pot = UMAPotential(model=args.model, task="oc20", device=...)
ads_pot = bulk_pot  # единый oc20
```

**Не предлагай вернуть omat для bulk.** Мы это пробовали, оно сломанное. Если кажется что hull нелогичный — нужно дополнить его недостающими точками или перекалибровать, **НЕ менять task на omat**.

### 3.2 fairchem-core API

UMA подгружается через **fairchem-core v2 API**:
```python
from fairchem.core import pretrained_mlip, FAIRChemCalculator
predictor = pretrained_mlip.get_predict_unit(model, device=device)
calc = FAIRChemCalculator(predictor, task_name=task)
```

**НЕ** через `OCPCalculator` / `model_name_to_local_file` — это старый API, его в текущем fairchem нет. Уже сломал так v2.5, откатывали. См. `pdh_gfn/potential/uma.py`.

### 3.3 PdZn.cif и другие референсные CIF

**НЕ перезаписывай** `data/ref_structures/*.cif`. Был долгий эпизод где `PdZn.cif` оказался AlSb (mp-1652 вместо правильного), и calibrate с MP-фолбэком молча скачивал не ту фазу. Если нужна структура — скачивать вручную, проверять `head -3` после.

Правильные mp-id (проверены):
- PdZn: `mp-1652` (тетрагональная B2-like)
- Pd: `mp-2` (FCC)
- GaPd: `mp-1078526`
- InPd: `mp-21215`
- Pd3Sn — два полиморфа, оба используем: `mp_718` (стабильный L1₂) и `mp_1218937` (триклинный, аномалия — отфильтровать в калибровке)

### 3.4 Архив (zip) НЕ должен содержать пользовательские данные

Когда упаковываешь архив для Григоши на кластер — **исключай**:
- `data/ref_structures/*` — у него правильные локальные, наши могут быть мусором
- `data/mp_pdm_entries.json` — у него пересчитанный hull, перезапишем чужими данными
- `data/reward_cache.jsonl`, `data/structures/*`, `data/adsorbate_refs.json` — все вычисленные кэши
- `data/seed_states.pkl` — затравка

Шаблон команды:
```bash
zip -qr archive.zip . \
  -x "*/__pycache__/*" -x "*.egg-info/*" -x "logs/*" \
  -x "data/structures/*" -x "data/reward_cache.jsonl" \
  -x "data/adsorbate_refs.json" -x "data/ref_structures/*" \
  -x "data/hull_plots/*" -x "data/mp_pdm_entries.json" \
  -x "data/mp_pdm_entries.gga.json.bak" -x "data/seed_states.pkl"
```

---

## 4. Текущее состояние

### v2.6 (актуальное) — изменения этой линии

- **Единый `config/run.yaml`** — все параметры запуска (обучение/пайплайн/UMA/пороги
  награды/curriculum) в одном YAML; CLI-флаги переопределяют. `train.py` грузит его
  (`--config`, дефолт `config/run.yaml`), reward/curriculum применяются к
  `constants`/`schedule` на старте. Старый `config/pdh.yaml` УДАЛЁН (hydra-stale,
  врал `bulk_task: omat`). Search-space по-прежнему в `constants.py`.
- **Tier 0** (×8.5): в curriculum-фазе 1 пропуск адсорбции (награда там только от
  e_hull). `proxy.py` (stab_only) + `pipeline.py` (ранний выход до slab).
- **tf32** в `UMAPotential` (×1.15, ~1 мэВ). Всё ещё всё на oc20.
- **Батч-релаксация — ТУПИК.** За `--batch-relax` есть, но НЕ использовать: на боевых
  слэбах ×3 МЕДЛЕННЕЕ (GPU compute-bound на A5000, связка по медленной конфигурации,
  89% invalid). Доказано `scripts/probe_batch_scaling.py`. Память: `batching-dead-end`.
- **Reward-hacking гейтов** (BEP экстраполируется вне домена калибровки):
  активность — мягкий клэмп `E_ACT_CH_MIN=1.0`; селективность — `r_sel` стало ОКНОМ,
  падает за `E_SEL_MAX=5.5`. gated-режим `schedule.compute` пересчитывает награду из
  дескрипторов. Сырой топ фильтровать по `flag=OK`. Память: `activity-gate-reward-hack`.
- **Анализ-тулкит** (`scripts/`): `analyze_run`, `export_candidates`,
  `find_candidate_structures`, `plot_training`. Профайлер стадий + CSV-трейс (`--profile`).
- **Офлайн-режим** `--cache-only` (политика на кэше без UMA-адсорбции). Чекпоинты — каждые 50.
- **uv-сборка** (`pyproject.toml [tool.uv]` + `uv.lock`, проверена на zeus): `uv sync` +
  `uv run` воспроизводят валидированный стек (torch 2.8.0 cu128 / fairchem 2.21.0 /
  gflownet@git) без conda. torch-индекс cu128, gflownet git@21ebf039,
  `override-dependencies = [torch==2.8.0, numpy==2.4.6]` (поверх устаревших пинов
  gflownet: torch 2.5.1 / numpy<2.0; fairchem требует numpy 2.0..2.5). conda-путь жив,
  но теперь `pip install -e . --no-deps`. Детали — INSTALL_an01.md §3b.
- **Баги Mila gflownet — монкипатчи в train.py** (`_patch_cube_device_bug`,
  `_patch_log_reward_bug`): ContinuousCube device + `np.log(rewards_*)` на pandas
  object-Series. Раньше патчили клон gflownet руками — теперь в нашем коде, работает на
  любой установке (важно для uv с чистым git-gflownet).

### Что РАБОТАЕТ
- Калибровка дескрипторов на 5 эталонах. PdZn(111), PdGa(111), PdIn(110), Pd(111), Pd3Sn(111) дают физичные числа (BE(H) ∈ [-0.5, +0.2], Eact ∈ [1.2, 2.2], E_sel ∈ [1.4, 3.9]).
- Warmup кэша через `evaluate_from_atoms` (обходит pyxtal-сборку, читает CIF напрямую).
- Curriculum learning (`scripts/train.py --curriculum`) с 4 фазами: только R_stab → +R_act → +R_sel → gated.
- Сохранение структур по этапам в `data/structures/<hash>/{bulk_initial, bulk_relaxed, slab_relaxed, ads_X_Y}.xyz` для офлайн-анализа.
- Визуализация hull через `scripts/plot_hull.py` — рисует PNG на каждую Pd-M систему.
- `scripts/rebuild_hull_oc20.py` (исправленный) — пересчитывает весь hull в oc20-шкале с продолжением после прерывания.

### Что СЛОМАНО или НЕДОДЕЛАНО

**КРИТИЧНОЕ**: на момент последнего обучения 16.06 hull частично GGA, частично oc20 — смешанная шкала. Нужно полностью пересчитать в oc20:
```bash
export MP_API_KEY=...
python scripts/rebuild_hull_oc20.py --uma-device cuda
```
У Григоши это занимает несколько минут (не 40-120 как я когда-то писал). После пересчёта проверить: для PdZn в новом hull должен быть e_above_hull ≈ 0.

**Известные баги Mila GFlowNet** (требуют workaround):
1. `np.log(rewards_replay/dataset)` падает на тензорах float32 — обернуть в `np.asarray(..., dtype=float)` в `gflownet.py`. Уже исправлено в нашем коде.
2. `backward_replay` ломает TB-loss с вложенным Stack-окружением → workaround: `--backward-replay 0`.
3. `ContinuousCube` device-баг при работе на CPU — патч в `train.py` через `_patch_cube_device_bug()`.
4. `backward_dataset` отключён (затравки несовместимы с backward-валидацией Stack).

**TODO (отложено)**:
- Подключить `canonical_miller` в конвейер — оставлено как TODO в обзоре архитектуры (требует пересчёта кэша при изменении набора граней, риск > польза сейчас).
- ~~Перенос порогов из `constants.py` в конфиг/CLI~~ — СДЕЛАНО (v2.6, `config/run.yaml`).
- Tuning порогов `STAB_E_HULL_MAX`, `E_ACT_CH_MAX`, `E_SEL_TARGET` после первого полного прогона curriculum.

### Откалиброванные пороги (актуальные)

В `pdh_gfn/constants.py`:
```python
E_ACT_CH_MAX = 2.70    # эВ, из калибровки (макс + 0.5)
E_ACT_SCALE = 0.30     # эВ, ширина сигмоиды активности
E_SEL_TARGET = 3.56    # эВ, середина PdZn(3.80)...PdIn(3.87)
E_SEL_SCALE = 0.30     # эВ
REWARD_BETA = 2.0      # log-диапазон сжат (было 4 → давало -37 на полу)
R_VALID_EPS = 1e-4     # пол invalid_reward
```

В `pdh_gfn/reward/schedule.py`:
```python
STAB_E_HULL_MAX = 0.5  # эВ/атом (было 0.15 — отсекало 100%)
ACT_E_MIN, ACT_E_MAX = 0.5, 3.0
SEL_E_MIN, SEL_E_MAX = 1.5, 4.0
```

В `pdh_gfn/reward/pipeline.py` и `scripts/train.py`:
```python
e_hull_cutoff = 2.0    # эВ/атом (было 0.15 — отсекало pyxtal-полиморфы)
fmax = 0.10            # для обучения; 0.03 для калибровки/warmup
```

---

## 5. Структура репозитория

```
pdh-gfn/
├── pdh_gfn/                    # пакет
│   ├── constants.py            # M_ELEMENTS, ALL_ELEMENTS, пороги, X_PD_MIN/MAX, MAX_ATOMS_PER_CELL
│   ├── proxy.py                # PDHProxy: оборачивает RewardPipeline + curriculum
│   ├── seeding.py              # затравочный буфер
│   ├── envs/                   # GFlowNet окружения
│   │   ├── catalyst.py         # PdMCatalyst (Stack)
│   │   ├── composition_pdm.py  # PdMComposition (биномиальная)
│   │   └── surface.py          # MillerChoice, OffsetCube, TopBottomChoice
│   ├── structure/              # построение и манипуляции
│   │   ├── builder.py          # pyxtal: SG+composition → Structure
│   │   ├── canonical.py        # каноническая форма грани
│   │   ├── from_xyz.py         # читалка CIF/xyz → CatalystState
│   │   ├── mp_loader.py        # выкачка из Materials Project (НЕ для CIF, только для hull)
│   │   ├── saver.py            # StructureSaver — pyxtal-этапы в extxyz
│   │   └── slab.py             # cut_slab (pymatgen SlabGenerator)
│   ├── potential/              # UMA, EMT-mock
│   │   ├── base.py             # Potential ABC, _relax, BasePotential, enumerate_sites
│   │   ├── batch.py            # батчевая обёртка
│   │   ├── emt_mock.py         # ASE EMT для тестов (без UMA)
│   │   └── uma.py              # UMAPotential — fairchem-core v2 API
│   └── reward/                 # дескрипторы, гейты, hull, pipeline
│       ├── cache.py            # RewardCache: JSONL append-only по slab_hash
│       ├── descriptors.py      # Seemakurthi: Eact_CH, Eact_CC, E_sel
│       ├── gates.py            # composite_reward, sigmoid с порогами, RewardBreakdown
│       ├── hull.py             # HullReference: pymatgen PhaseDiagram
│       ├── pipeline.py         # RewardPipeline: главная функция (bulk → slab → ads → reward)
│       ├── refs.py             # compute_adsorbate_refs (ур.2 Seemakurthi)
│       └── schedule.py         # RewardSchedule — curriculum learning
├── scripts/                    # CLI
│   ├── calibrate.py            # калибровка на 5+ эталонах, печатает sed-команды
│   ├── diagnose_env.py         # smoke-тест GFlowNet окружения
│   ├── fetch_mp_entries.py     # выкачка Pd-M записей для hull
│   ├── fetch_ref_structures.py # выкачка референсных CIF
│   ├── plot_hull.py            # визуализация Pd-M hull (PNG)
│   ├── precompute_hull.py      # добавление чистых элементов в hull
│   ├── rebuild_hull_oc20.py    # пересчёт всего hull в oc20-шкале
│   ├── train.py                # основной — pdh_gflownet_from_config + цикл
│   └── warmup_cache.py         # прогрев кэша на ref_structures
├── tests/
│   └── test_pipeline.py        # smoke на EMT-mock
└── data/                       # НЕ в git
    ├── ref_structures/         # CIF — НЕ трогать, у Григоши свои
    ├── mp_pdm_entries.json     # hull (oc20-пересчёт)
    ├── reward_cache.jsonl      # кэш наград по slab_hash
    ├── adsorbate_refs.json     # E_ref(H), E_ref(CH), E_ref(C3H7)
    ├── seed_states.pkl         # затравочный буфер
    └── structures/             # extxyz сохранения по этапам
```

---

## 6. Главный поток данных

GFlowNet политика семплирует `CatalystState`:
```python
@dataclass
class CatalystState:
    space_group: int           # 12 групп: 225, 221, 123, 139, 194, 227, 216, 198, 62, 63, 140, 59
    composition: Dict[int, int] # {Z: n_atoms}; ALL_ELEMENTS = [46] + 25 M-элементов
    lattice_params: Tuple      # (a, b, c, alpha, beta, gamma)
    miller: Tuple[int, int, int]
    offset: float              # ∈ {0.0, 0.5}
    top: bool                  # верхняя или нижняя грань
```

`RewardPipeline.__call__(state)`:
1. **`_relaxed_bulk(state)`** — pyxtal сборка → UMA oc20 релаксация ячейки → e_hull через `HullReference.e_above_hull`. Если `e_hull > e_hull_cutoff` — ранний выход.
2. **`cut_slab(bulk, hkl, offset, top)`** — pymatgen SlabGenerator: вакуум 15 Å, латеральное расширение до min_lateral=9 Å. UMA oc20 релаксация атомов.
3. **`enumerate_sites(slab, max_sites=8)`** — AdsorbateSiteFinder: ontop/bridge/hollow.
4. **`adsorption_energies` × 3** — для H, CH, C3H7: на каждом сайте релакс адсорбата → BE = E(slab+ads) − E(slab) − E_ref.
5. **`compute_descriptors`** — формулы Seemakurthi 2025:
   - `Eact(C-H) = 0.82·[BE(C3H7*) + BE(H*)] + 0.83`
   - `Eact(C-C) = 0.77·BE(CH*_meta) + 0.19`
   - `E_sel = BE(CH*) + 0.77·BE(CH*_meta) − 0.82·[BE(C3H7*) + BE(H*)] − 0.64`
6. **`composite_reward(desc, e_hull)`** — три гейта × сигмоиды → R^beta.
7. **`schedule.compute(breakdown, step)`** — если включён curriculum, пересчитывает R под текущую фазу. Cache хранит **дескрипторы**, не reward — переключение фазы не требует релаксаций.

---

## 7. Запуск с нуля (на кластере)

```bash
# Окружение: conda env /home/heimao/pdhgfn на zeus
cd ~/work/egor/Nornikel/pdh-gfn

# 1. Установка/обновление (один раз)
pip install -e . --no-deps
pip install fairchem-core mp-api pymatgen pyxtal ase

# 2. Hull в oc20-шкале (несколько минут на GPU)
export MP_API_KEY=...
python scripts/rebuild_hull_oc20.py --uma-device cuda
# проверка: PdZn должен дать E_hull ≈ 0

# 3. Калибровка (опционально, если нужно перекалибровать)
rm -f data/adsorbate_refs.json
python scripts/calibrate.py --device cuda
# применить sed-команды из вывода для constants.py

# 4. Чистка кэша
rm -f data/reward_cache.jsonl
rm -rf data/structures

# 5. Warmup — затравка кэша эталонами
python scripts/warmup_cache.py --uma-device cuda --fmax 0.03 \
  --structures data/ref_structures/PdZn.cif \
               data/ref_structures/PdGa.cif \
               data/ref_structures/PdIn.cif \
               data/ref_structures/Pd3Sn_mp_718.cif

# 6. Обучение
python scripts/train.py --uma-device cuda --n-steps 1500 \
    --backward-replay 0 --curriculum

# Опционально: визуализация hull
python scripts/plot_hull.py
ls data/hull_plots/
```

---

## 8. Что делать когда что-то идёт не так

### invalid=100% / mean_reward = 1e-8 на старте
- **Проверь hull.** `python3 -c "import json; r=json.load(open('data/mp_pdm_entries.json')); print(len(r), 'mp_id:', sum(1 for x in r if 'mp_id' in x))"`. Если `mp_id: 0` — hull всё ещё GGA, нужен `rebuild_hull_oc20.py`.
- **Проверь PdZn.cif.** `head -3 data/ref_structures/PdZn.cif` — должно быть `data_PdZn`, не `data_AlSb`.
- **Проверь cutoff.** `--e-hull-cutoff 2.0` (не 0.15). Метастабильные полиморфы из pyxtal имеют E_hull в районе 1-2 эВ/атом, это нормально.

### BE гигантские (>10 эВ)
- Фильтр абсурдных BE уже стоит в `pipeline.py` (отказ если `|BE(H)|>5` или `|BE(CH)|>6` или `|BE(C3H7)|>5`).
- Если всё равно: смотри сохранённые структуры в `data/structures/<hash>/ads_C3H7_0.xyz` — скорее всего там адсорбат проваливается в slab из-за патологической геометрии. C3H7 геометрия должна быть с `все z ≥ 0 относительно C1`.

### Reward тонет в плато
- Curriculum фаза 1 (только R_stab) — политика быстро находит стабильные. Если нет — hull сломан (см. пункт первый).
- `STAB_E_HULL_MAX = 0.5` (был 0.15 — отсекало всё).
- `REWARD_BETA = 2.0` (был 4.0 — диапазон log был -37, TB-loss задыхался).

### "ModuleNotFoundError: fairchem.core.X"
- Скорее всего код пытается импортировать старый API. Проверь `pdh_gfn/potential/uma.py` — должно быть `from fairchem.core import pretrained_mlip, FAIRChemCalculator`. Если там `OCPCalculator` или `model_name_to_local_file` — это сломанная версия из v2.5, откати к снапшоту кластера.

---

## 9. Конвенции стиля

- **Docstrings — короткие, по-делу**, без шаблонного "Args/Returns/Raises". Сразу пиши зачем функция, какой инвариант она держит.
- **Комментарии в коде — на русском**, если поясняют доменную логику. На английском — если про чисто технический code path.
- **Не плодить файлы.** Если функция помещается в существующий модуль — не делай новый.
- **Тесты — smoke на EMT-mock** (`tests/test_pipeline.py`). Это быстрая проверка что pipeline собирается и считает; не unit-тесты в стиле «каждая функция изолированно».

---

## 10. История версий (для контекста)

- **v1.5 и раньше**: всё ломалось, BE были по +30 эВ, reward на полу. Эпоха «omat для bulk + oc20 для surface», смешанные шкалы.
- **v1.7-1.9**: фильтр абсурдных BE, фикс C3H7 геометрии (все z ≥ 0), warmup через `evaluate_from_atoms`, защита от AlSb-CIF.
- **v2.0**: переход на единый oc20. BE стали корректными для калибровочных структур. Но hull остался GGA → e_above_hull стал бессмысленным для тугоплавких.
- **v2.1**: `rebuild_hull_oc20.py` (с багом дублированного main).
- **v2.2**: curriculum learning (4 фазы), фильтр |BE|, RewardBreakdown с raw дескрипторами. **Эта версия рабочая, на ней Григоша запускал ночное обучение.**
- **v2.3-2.4**: проверка чистых элементов в hull, plot_hull.py.
- **v2.5 (НЕ РАБОЧАЯ)**: я попытался вернуть omat для bulk, при этом сломал uma.py (старый API fairchem). Откатили.
- **v2.5 (рабочая, эта)**: ВСЁ на oc20, omat нигде нет, uma.py из снапшота кластера, rebuild_hull_oc20.py с исправленным дублем main(), STAB_E_HULL_MAX=0.5, e_hull_cutoff=2.0.

---

## 11. Связанная литература (что цитируем в работе)

- **Seemakurthi 2025** (ChemCatChem): PDH-дескрипторы, BEP-корреляции, формулы для Eact и E_sel.
- **Purdy et al. 2020**: бенчмарк Pd-M интерметаллидов для PDH (6 поверхностей, против которых валидируемся).
- **Park et al. 2025**: ML-скрининг HER, похожая схема для другой реакции.
- **Malkin et al. NeurIPS 2022**: TB-GFlowNet (на ней наш агент).
- **Podina et al. NeurIPS WS 2025**: Catalyst GFlowNet (близкая работа, чтобы цитировать в related work).

---

Конец CLAUDE.md.

# PDH-GFlowNet

Генеративная модель на основе GFlowNet для поиска Pd-M интерметаллидов — селективных катализаторов дегидрирования пропана (PDH). Проект Норникель × Сколтех.

## Идея

Дегидрирование пропана (`C3H8 → C3H6 + H2`) — промышленная реакция получения пропилена. Чистый Pd активен, но плохо селективен — теряется в крекинг C-C. Сплавы Pd с p-блоком (Zn, Ga, In, Sn) показывают высокую селективность, но пространство Pd-M интерметаллидов огромное (десятки тысяч кандидатов с разной стехиометрией, кристаллической структурой, гранями), и DFT-скрининг непрактичен.

Мы:
1. Семплируем структуры через GFlowNet (Trajectory Balance loss)
2. Считаем энергии связывания H/CH/C3H7 универсальным потенциалом UMA (fairchem-core)
3. Из BE по формулам Seemakurthi 2025 вычисляем дескрипторы активности и селективности
4. Используем три гейта (стабильность по E_hull, активность Eact_CH, селективность E_sel) как награду
5. Curriculum learning: сначала учим стабильности, потом активности, потом селективности

Архитектура GFlowNet адаптирована из Crystal-GFN (Mila) и Catalyst GFlowNet (Podina et al.).

## Стек методов

- **GFlowNet / Trajectory Balance** — Bengio et al. NeurIPS 2021; Malkin et al. NeurIPS 2022
- **Crystal-GFN** — Mila gflownet, пространство `SG → состав → решётка`
- **Catalyst GFlowNet** — Podina et al. NeurIPS WS 2025: поверхностный блок (грани, сдвиг, сторона)
- **PDH-дескрипторы** — Seemakurthi et al. ChemCatChem 2025: BEP, E_sel
- **Универсальный потенциал** — UMA (fairchem-core v2), task=oc20

## Архитектура

### Траектория GFlowNet

```
SG (12 групп) → состав Pd-M → параметры решётки → (hkl) → сдвиг → верх/низ
```

- 12 пространственных групп: 225, 221, 123, 139, 194, 227, 216, 198, 62, 63, 140, 59 (покрывают L1₂, B2, L1₀, FCC и Лавес-фазы)
- 25 M-элементов: 3d (Ti, V, Mn, Fe, Co, Ni, Cu), p-блок (Zn, Ga, Ge, In, Sn, Sb, Pb, Bi), 4d/5d (Zr, Nb, Mo, Ag, Hf, Ta, W, Re, Au)
- x(Pd) ∈ [0.25, 0.80] — захватывает PdM₃...Pd₃M
- Максимум 16 атомов в ячейке (вычислительный бюджет)

### Награда

Все вычисления в единой шкале UMA-oc20:

```
pyxtal sampling → UMA oc20 relax(bulk) → E_hull через MP-PhaseDiagram(oc20)
              → cut_slab(hkl, offset, top) → UMA oc20 relax(slab)
              → adsorption sites (ontop/bridge/hollow)
              → UMA oc20 relax(H, CH, C3H7) на каждом сайте
              → BE = E(slab+ads) − E(slab) − E_ref(oc20)
              → дескрипторы Seemakurthi
              → R = R_stab · R_act · R_sel (curriculum по фазам)
```

Формулы Seemakurthi:
```
Eact(C-H) = 0.82·[BE(C3H7*) + BE(H*)] + 0.83
Eact(C-C) = 0.77·BE(CH*_meta) + 0.19
E_sel = BE(CH*) + 0.77·BE(CH*_meta) − 0.82·[BE(C3H7*) + BE(H*)] − 0.64
```

### Curriculum learning

Четыре фазы:
1. **0–500 шагов**: R = exp(score_stab − 1) — только стабильность. Политика учится генерировать интерметаллиды близко к выпуклой оболочке.
2. **500–1000**: + score_act. Добавляется активность дегидрирования.
3. **1000–1500**: + score_sel. Учим балансировать с селективностью.
4. **1500+**: gated mode с сигмоидами и порогами из калибровки.

Score-функции — truncated linear с насыщением сверху (clip 0..1), нельзя выкрутить один параметр за счёт других.

### Ключевые дизайн-решения

- Терминальный объект — сырой кортеж; канонизация только в кэше наград
- R_sel через сигмоиду (не экспоненту) для защиты от эксплуатации ошибок прокси
- R_act как отдельный гейт активности — E_sel может быть высоким на мёртвом катализаторе
- Жёсткие гейты x_Pd ∈ [0.25, 0.80] и MAX_ATOMS=16 в маске действий
- Hull в той же UMA-oc20 шкале что и pipeline (через `rebuild_hull_oc20.py`)

### Reward-hacking и фиксы гейтов

Дескрипторы — линейные BEP-корреляции (Seemakurthi), откалиброванные на 5 эталонах.
**Вне домена калибровки они экстраполируются**, что даёт reward-hacking:

- **Активность**: BEP экстраполирует Eact(C-H) в отрицательный (нефизичный) барьер
  для переусиленных связывателей → мягкий клэмп `E_ACT_CH_MIN=1.0` (ниже не активнее).
- **Селективность**: для тех же сильных связывателей формула даёт огромный E_sel
  (физически часто наоборот — пере-дегидрирование/кокс). `r_sel` сделан **окном**:
  растёт к `E_SEL_TARGET`, но **падает за `E_SEL_MAX=5.5`** (экстраполяция не хакает награду).

Пороги — в `pdh_gfn/constants.py` / `config/run.yaml` (секция `reward`). Сырой топ
кандидатов всё равно фильтровать по `flag=OK` (`export_candidates.py`) и валидировать DFT.

### Инженерные рычаги (что реально ускоряет)

- **Tier 0** — в curriculum-фазе 1 (только стабильность) пропускаем адсорбцию (78% времени): **×8.5**.
- **tf32** — matmul на Ampere: ×1.15 ко всем forward, сдвиг энергии ~1 мэВ.
- Кэш наград (по дескрипторам), ранний выход по E_hull, предфильтр сайтов.
- **Батч-релаксация — работает с ограничениями**: GPU compute-bound на A5000, ×3 *медленнее* на боевых
  слэбах (связка по самой медленной конфигурации). Перед использованием проверить ускорение на видеокарте.

## Быстрый старт

### Установка

```bash
git clone <repo>
cd pdh-gfn
pip install -e . --no-deps
pip install fairchem-core mp-api pymatgen pyxtal ase
huggingface-cli login   # для доступа к facebook/UMA
export MP_API_KEY=...   # ключ Materials Project
```

### Подготовка

```bash
# 1. Скачать референсные структуры
python scripts/fetch_ref_structures.py

# 2. Скачать MP-записи для hull
python scripts/fetch_mp_entries.py

# 3. Пересчитать hull в oc20-шкале (несколько минут на GPU)
python scripts/rebuild_hull_oc20.py --uma-device cuda

# 4. Откалибровать дескрипторы
python scripts/calibrate.py --device cuda
# применить sed-команды из вывода
```

### Обучение

Все параметры — в едином `config/run.yaml` (обучение, пайплайн, UMA, пороги награды,
фазы curriculum). Любой CLI-флаг переопределяет значение оттуда.

```bash
# Затравка кэша эталонами
python scripts/warmup_cache.py --uma-device cuda --fmax 0.03 \
  --structures data/ref_structures/PdZn.cif data/ref_structures/PdGa.cif \
               data/ref_structures/PdIn.cif data/ref_structures/Pd3Sn_mp_718.cif

# Обучение — всё из config/run.yaml (curriculum, tf32, ckpt-50, n_samples=0…)
python scripts/train.py

# переопределить отдельное:
python scripts/train.py --n-steps 50            # быстрый тест
python scripts/train.py --resume                # продолжить с чекпоинта
```

> ⚠️ **Не** использовать `--batch-relax`: на боевых слэбах оказался ×3 медленнее
> (GPU compute-bound на A5000) — тупик, оставлен под флагом только для истории и других вычислительных архитектур.

### Анализ результатов

```bash
TR=$(ls -t logs/pdh_gfn/trace_*.csv | head -1)
python scripts/analyze_run.py   --trace "$TR" --log train_run.log   # отчёт по прогону
python scripts/plot_training.py --trace "$TR" --log train_run.log   # графики (PNG)
python scripts/export_candidates.py --trace "$TR" --top 100         # таблица + флаги битых
python scripts/find_candidate_structures.py \
    --candidates logs/pdh_gfn/candidates_top100.csv --copy           # кандидат → .xyz
```

### Диагностика

```bash
# Smoke-тест pipeline на EMT (без UMA)
python tests/test_pipeline.py

# Тест GFlowNet окружения
python scripts/diagnose_env.py

# Визуализация hull (PNG на каждую Pd-M систему)
python scripts/plot_hull.py
ls data/hull_plots/
```

## Структура репозитория

```
pdh-gfn/
├── pdh_gfn/                # пакет
│   ├── constants.py        # пороги, M_ELEMENTS, ограничения состава
│   ├── proxy.py            # PDHProxy — обёртка над RewardPipeline для GFlowNet
│   ├── envs/               # пространство действий (Stack из 4 окружений)
│   ├── structure/          # pyxtal сборка, нарезка slab, чтение CIF
│   ├── potential/          # UMA, EMT-mock
│   └── reward/             # дескрипторы, гейты, hull, curriculum
├── scripts/                # CLI-скрипты
├── tests/
└── data/                   # генерируется при запуске
```

### CLI-скрипты

| Скрипт | Что делает |
|---|---|
| `fetch_ref_structures.py` | Скачать референсные CIF (PdZn, Pd, PdGa, PdIn, Pd3Sn) из MP |
| `fetch_mp_entries.py` | Скачать Pd-M записи для hull |
| `rebuild_hull_oc20.py` | Пересчитать hull в UMA-oc20 шкале |
| `precompute_hull.py` | Дополнить hull чистыми элементами |
| `plot_hull.py` | Визуализация Pd-M фазовых диаграмм |
| `calibrate.py` | Калибровка дескрипторов на эталонах |
| `warmup_cache.py` | Затравка reward_cache известными катализаторами |
| `train.py` | Главный — обучение GFlowNet (читает `config/run.yaml`) |
| `analyze_run.py` | Отчёт по прогону: статусы/награды по фазам, топ с флагами, loss, кэш |
| `plot_training.py` | Графики обучения (loss/награды/статусы + облако E_act–E_sel) |
| `export_candidates.py` | Таблица top-N кандидатов с пометкой битых (over-bind / Esel-extrap) |
| `find_candidate_structures.py` | По таблице кандидатов находит/копирует сохранённые .xyz |
| `diagnose_env.py` | Smoke-тест GFlowNet окружения |

## Калибровочные числа

После полного пайплайна на правильно настроенном UMA-oc20 (fmax=0.03, max_sites=12):

| Поверхность | SG | BE(H) | BE(CH) | BE(C3H7) | Eact_CH | E_sel |
|---|---|---|---|---|---|---|
| Pd(111) | 225 | −0.46 | 1.37 | 0.88 | 1.18 | 1.48 |
| PdZn(111) | 123 | −0.14 | 3.15 | 1.50 | 1.94 | 3.84 |
| PdIn(110) | 221 | +0.19 | 3.29 | 1.42 | 2.15 | 3.87 |
| PdGa(111) | 198 | −0.44 | 2.37 | 1.13 | 1.40 | 3.45 |
| Pd3Sn(111) | 221 | −0.53 | 2.44 | 1.14 | 1.33 | 3.25 |

Согласуется с Seemakurthi 2025: PdZn — золотой стандарт, Pd — неселективный baseline, монотонный тренд по E_sel.

## Литература

- **Bengio et al.** Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation. NeurIPS 2021.
- **Malkin et al.** Trajectory Balance: Improved Credit Assignment in GFlowNets. NeurIPS 2022.
- **Podina et al.** Catalyst GFlowNet for Electrocatalyst Design. NeurIPS Workshop 2025.
- **Seemakurthi et al.** Descriptor-Based Selectivity Predictions for PDH on Intermetallic Catalysts. ChemCatChem 2025.
- **Purdy et al.** Selective Propane Dehydrogenation on Pd Intermetallic Compounds: A Computational Study. ACS Catal. 2020.

## Лицензия

Исследовательский код, не для продакшена. Используется под лицензиями оригинальных проектов (Mila gflownet — MIT, fairchem — MIT, pymatgen — MIT).

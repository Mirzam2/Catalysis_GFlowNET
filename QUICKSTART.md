# QUICKSTART

Актуальная последовательность для запуска на zeus / an01. Подробности — `INSTALL_an01.md`.

## 0. Окружение

Если `pdhgfn` уже собран (torch + gflownet + pdh-gfn + fairchem) — переходи к шагу 1.

Иначе — два пути:
- **uv (быстро, воспроизводимо)** на CUDA-узле: `uv sync` (ставит стек по `uv.lock`), запуск через `uv run python …`. Детали — `INSTALL_an01.md §3b`.
- **conda** (исходный): клон UMA-окружения + `pip install -e . --no-deps` — `INSTALL_an01.md §0–§3`.

Баги Mila gflownet (cube device + np.log) обходятся монкипатчами в `scripts/train.py` — на обоих путях ручной патч клона не нужен.

## 1. Дымовой тест без GPU (~2 мин)

Проверяет: pyxtal-сборка, канонизация структур, дескрипторы Seemakurthi, конвейер награды на EMT, кэш.

```bash
cd pdh-gfn
python tests/test_pipeline.py
# ожидаем: ВСЕ ТЕСТЫ ПРОШЛИ
```

## 2. Диагностика окружения GFlowNet (без GPU, ~30 сек)

Проверяет: `PdMCatalyst` собирается, маски x_Pd работают, `step_random` доходит до терминала, `unpack_state` корректно извлекает SG/состав/решётку, seeding строит все 30 затравок.

```bash
python scripts/diagnose_env.py
# ожидаем: 5 случайных траекторий завершаются, 30 затравок построено
```

Если этот шаг падает — НЕ запускай обучение, пиши мне с выводом.

## 3. Калибровка дескрипторов (GPU, ~5–15 мин)

Прогоняет Pd(111), PdZn(111), PdGa(111) через UMA, печатает E_act(C-H) и E_sel,
рекомендует значения порогов `E_ACT_CH_MAX` и `E_SEL_TARGET` для `constants.py`.

```bash
python scripts/calibrate.py --device cuda
```

**Сверь с ожиданиями статьи:**
- `E_act(C-H)` на Pd(111) должен быть положительным (~0.7–0.9 эВ);
- BE(CH) и CHmeta должны различаться (не равны) — это проверяет, что фикс перечисления сайтов сработал;
- E_sel на PdZn должен быть заметно выше Pd.

Если знак E_act правильный и значения разумные — впиши рекомендованные пороги в `pdh_gfn/constants.py` (`E_SEL_TARGET`, `E_ACT_CH_MAX`).

## 4. Smoke-run обучения (~5 мин)

Маленький прогон с EMT — проверка сборки тренера, device-логики и **что грузится
`config/run.yaml`** (в логе: `конфиг: config/run.yaml`):

```bash
python scripts/train.py --mock --n-steps 2 --n-samples 0
```

После прогона смотри блок `=== reward diagnostics ===`:
- `invalid_frac` (на EMT-mock 100% — норма, EMT даёт мусорную адсорбцию);
- `error_frac > 10%` — ошибки в конвейере, ищи WARNING выше;
- `reward_beta min/med/max` должны различаться.

## 5. Полное обучение

Все параметры — в `config/run.yaml` (n_steps, curriculum, tf32, ckpt-50, n_samples=0…).
Обычно достаточно:

```bash
nohup python scripts/train.py > train_run.log 2>&1 &
# продолжить с чекпоинта:
nohup python scripts/train.py --resume > train_resume.log 2>&1 &
```

> ⚠️ Не использовать `--batch-relax` — ×3 медленнее на боевых слэбах (см. ниже).

`reward_cache.jsonl` персистентен — перезапуски переиспользуют посчитанные
UMA-оценки. **Не удалять между запусками** (фикс гейтов пересчитывает награду из
дескрипторов — кэш остаётся валидным).

## 5b. Анализ прогона

```bash
TR=$(ls -t logs/pdh_gfn/trace_*.csv | head -1)
python scripts/analyze_run.py   --trace "$TR" --log train_run.log   # отчёт
python scripts/plot_training.py --trace "$TR" --log train_run.log   # графики PNG
python scripts/export_candidates.py --trace "$TR" --top 100         # таблица + флаги
```

## Что менять под себя

**`config/run.yaml`** — ЕДИНЫЙ конфиг запуска: параметры тренера (lr, batch, n_steps),
пайплайн (fmax, max_sites…), пороги награды (`E_SEL_TARGET`, `E_ACT_CH_MAX/MIN`…),
фазы curriculum. Любой CLI-флаг переопределяет значение отсюда.

**`pdh_gfn/constants.py`** — search-space: словарь M-элементов, подмножество SG,
диапазоны решётки/состава, miller (физика домена, меняется редко).

**CLI-флаги ускорения** (см. `python scripts/train.py --help`):
- `--e-hull-cutoff 0.15` — ранний выход для нестабильных структур (×1.5–3)
- `--prefilter-keep 4` — предфильтр сайтов (×2–4 на больших slab'ах)
- `--max-sites-light 3` — меньше сайтов для H* (×1.3)
- `--no-prefilter` отключить предфильтр

## Что НЕ работает / ограничения

- **Батч-релаксация UMA — ТУПИК** (реализована за `--batch-relax`, но НЕ использовать):
  на боевых слэбах ×3 *медленнее* (GPU compute-bound на A5000, связка по самой
  медленной конфигурации). Реальные рычаги — Tier 0 (×8.5) + tf32 (×1.15), включены сами.
- **Reward-hacking дескрипторов**: BEP экстраполируется вне домена → нефизичный
  Eact<0 и завышенный E_sel. Закрыто частично (клэмп `E_ACT_CH_MIN`, окно `r_sel` за
  `E_SEL_MAX`), но сырой топ всё равно фильтровать по `flag=OK` (`export_candidates.py`).
- **backward-replay / backward-dataset отключены** (`--backward-replay 0` в конфиге) —
  баг Stack-среды Mila. Обучение идёт только forward-траекториями.
- **Затравки**: терминальные состояния от 5 эталонных фаз через seed buffer.

## Файлы данных (что уже есть в архиве)

- `data/mp_pdm_entries.json` — твоя выкачка Materials Project (28 КБ).
- `data/ref_structures/{Pd_fcc,PdZn}.cif` — закэшированы для калибратора.

Что **нет** и пересоберётся при первом запуске:
- `data/adsorbate_refs.json` — пересчитается на старте `train.py`/`calibrate.py`;
- `data/seed_states.pkl` — пересоберётся из `pdh_gfn/seeding.py`;
- `data/reward_cache.jsonl` — пустой, наполняется во время обучения.

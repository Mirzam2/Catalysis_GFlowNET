# QUICKSTART

Актуальная последовательность для запуска на zeus / an01. Подробности — `INSTALL_an01.md`.

## 0. Окружение

Если `pdhgfn` уже собран (torch + gflownet + pdh-gfn + fairchem) — переходи к шагу 1.
Иначе — раздел 0–3 в `INSTALL_an01.md` (клон UMA-окружения, ослабление пинов gflownet, `pip install -e . --no-deps` + ручная доустановка).

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

## 4. Smoke-run обучения (GPU, ~10 мин)

Маленький прогон с EMT для проверки сборки тренера и device-логики (не обучение по существу):

```bash
python scripts/train.py --mock --n-steps 20 --batch-size 4 --max-sites 4 --preflight
```

Затем то же на UMA, тоже короткое:
```bash
python scripts/train.py --device cuda --n-steps 50 --batch-size 4 --max-sites 6 --preflight
```

После прогона смотри блок `=== reward diagnostics ===`:
- если `invalid_frac > 90%` — структуры не считаются (плохие пороги/cutoff);
- если `error_frac > 10%` — ошибки в конвейере, ищи WARNING выше;
- `reward_beta min/med/max` должны различаться (не все одинаковые).

## 5. Полное обучение

```bash
sbatch sbatch/train_pdhgfn.sh
# или интерактивно:
python scripts/train.py --device cuda \
    --hull-entries data/mp_pdm_entries.json \
    --cache data/reward_cache.jsonl \
    --fmax 0.10 --n-steps 5000
```

`reward_cache.jsonl` персистентен — перезапуски джоба переиспользуют все
посчитанные UMA-оценки. **Не удалять между запусками.**

## Что менять под себя

**`pdh_gfn/constants.py`** — пороги награды после калибровки (`E_SEL_TARGET`, `E_ACT_CH_MAX`), при желании словарь M-элементов и подмножество SG.

**`config/pdh.yaml`** — параметры тренера (lr, batch, steps).

**CLI-флаги ускорения** (см. `python scripts/train.py --help`):
- `--e-hull-cutoff 0.15` — ранний выход для нестабильных структур (×1.5–3)
- `--prefilter-keep 4` — предфильтр сайтов (×2–4 на больших slab'ах)
- `--max-sites-light 3` — меньше сайтов для H* (×1.3)
- `--no-prefilter` отключить предфильтр

## Что НЕ работает / ограничения

- **Батч-релаксация UMA (#1, ×5–20)** — каркас в `pdh_gfn/potential/batch.py`,
  доделать на кластере по fairchem v2 API.
- **Коэффициенты BEP** в `constants.py` — точно по Seemakurthi 2025 (твоя
  поправка). `adsorbate_refs` — Seemakurthi ур. (2) через H₂/CH₄/C₃H₈.
- **Затравки**: 30 терминальных состояний от 5 фаз (PdZn, PdIn, PdGa,
  Pd₃Sn, Pd₂Ga) × 3 грани × 2 сдвига. Подмешиваются через replay buffer.

## Файлы данных (что уже есть в архиве)

- `data/mp_pdm_entries.json` — твоя выкачка Materials Project (28 КБ).
- `data/ref_structures/{Pd_fcc,PdZn}.cif` — закэшированы для калибратора.

Что **нет** и пересоберётся при первом запуске:
- `data/adsorbate_refs.json` — пересчитается на старте `train.py`/`calibrate.py`;
- `data/seed_states.pkl` — пересоберётся из `pdh_gfn/seeding.py`;
- `data/reward_cache.jsonl` — пустой, наполняется во время обучения.

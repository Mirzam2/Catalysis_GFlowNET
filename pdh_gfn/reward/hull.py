"""Энергия над выпуклой оболочкой Materials Project.

Решение из обсуждения: генерируем ЗА пределами MP, а MP используем как
референсную оболочку стабильности и детектор новизны.

Рабочий режим — офлайн: записи Pd-M химических систем скачиваются один раз
скриптом scripts/fetch_mp_entries.py (mp-api, нужен ключ) и кэшируются
в JSON; во время обучения сетевых запросов нет.

ВАЖНО про совместимость энергий: E_form кандидата считается UMA-потенциалом
в режиме omat (MP-уровень теории, PBE+U-поправки MP2020), поэтому смешение
с записями MP корректно с точностью до ошибки UMA. Это нужно проговорить
в методике, Алексей спросит.
"""
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Structure
from pymatgen.entries.computed_entries import ComputedEntry


class HullReference:
    def __init__(self, entries_path: Optional[Path] = None,
                 require_pure_elements: Optional[Iterable[str]] = None,
                 strict: bool = False):
        """entries_path: JSON со списком {composition: str, energy_per_atom: float}.

        require_pure_elements: проверить что эти элементы есть как ЧИСТЫЕ
            фазы в hull (углы выпуклой оболочки). Без них PhaseDiagram
            аппроксимирует границы произвольно — даже идеальный PdZn даёт
            e_above_hull > 0.5 эВ/атом. Если strict=True — падать с ошибкой
            при отсутствии, иначе только warning.

        strict: True — падать при проблемах с hull (рекомендуется для
            продакшена). False — только warning (тесты, отладка).
        """
        import logging
        log = logging.getLogger(__name__)

        self._diagrams: Dict[frozenset, PhaseDiagram] = {}
        self._entries = []
        self.pure_elements_present = set()

        if entries_path is not None and Path(entries_path).exists():
            raw = json.loads(Path(entries_path).read_text())
            for rec in raw:
                comp = Composition(rec["composition"])
                self._entries.append(
                    PDEntry(comp, rec["energy_per_atom"] * comp.num_atoms)
                )
                if len(comp.elements) == 1:
                    self.pure_elements_present.add(str(comp.elements[0]))

        # Проверка наличия чистых элементов
        if require_pure_elements:
            required = set(require_pure_elements)
            missing = required - self.pure_elements_present
            if missing:
                msg = (f"HullReference: в {entries_path} отсутствуют чистые "
                       f"элементы {sorted(missing)}. Без них PhaseDiagram "
                       f"даёт некорректные e_above_hull. Запусти "
                       f"scripts/precompute_hull.py чтобы дополнить hull.")
                if strict:
                    raise RuntimeError(msg)
                else:
                    log.warning(msg)
            else:
                log.info("HullReference: все требуемые чистые элементы "
                         "присутствуют (%d): %s",
                         len(required), sorted(required))

    def add_elemental_references(self, refs: Dict[str, float]) -> None:
        """Энергии чистых элементов (эВ/атом) тем же потенциалом, что и кандидаты.

        Обязательны для согласованного E_form, даже когда записи MP не загружены
        (режим тестов с заглушкой).
        """
        for el, e in refs.items():
            comp = Composition(el)
            self._entries.append(PDEntry(comp, e))
        self._diagrams.clear()

    def _get_diagram(self, elements: Iterable[str]) -> Optional[PhaseDiagram]:
        key = frozenset(elements)
        if key not in self._diagrams:
            relevant = [
                e for e in self._entries
                if set(map(str, e.composition.elements)) <= key
            ]
            try:
                self._diagrams[key] = PhaseDiagram(relevant)
            except Exception:
                self._diagrams[key] = None
        return self._diagrams[key]

    def e_above_hull(self, structure: Structure, energy_total: float) -> float:
        """E_hull (эВ/атом) кандидата с полной энергией energy_total (эВ).

        Если оболочка для системы не построена (нет записей) — возвращает 0.0
        и помечать это должен вызывающий код; в боевом конфиге записи обязаны
        покрывать все Pd-M системы словаря.
        """
        elements = {str(el) for el in structure.composition.elements}
        pd = self._get_diagram(elements)
        if pd is None:
            return 0.0
        entry = ComputedEntry(structure.composition, energy_total)
        try:
            return float(pd.get_e_above_hull(entry, allow_negative=True))
        except Exception:
            return 0.0

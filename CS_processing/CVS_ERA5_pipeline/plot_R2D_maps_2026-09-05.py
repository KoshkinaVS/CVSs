"""
Отрисовка карт R2D (Stage A: DBSCAN "сырые" вихри, ДО TempestExtremes-трекинга)
в стиле article_2026_FAO.ipynb — единый ГЛОБАЛЬНЫЙ экстремум на кластер (звезда)
+ граница оболочки кластера (ConvexHull), БЕЗ окружностей радиусов, для ТРЁХ
типов исходных данных: LoRes, HiRes (WRF NAAD) и ERA5.

История файла
-------------
Предыдущая версия скрипта (см. git-историю) была портирована из plot_R2D() в
ERA5_plot_test.ipynb только для ERA5 и по умолчанию рисовала ВСЕ локальные
экстремумы кластера (`local_extr_cluster`, маленькие звёздочки) плюс окружности
радиусов вихрей (полная версия + версия "_no_rads"). Эта версия — адаптация по
прямой просьбе пользователя:

1. Оформление карты приведено к стилю `plot_DBSCAN_centers_map()` из
   article_2026_FAO.ipynb: градусные gridlines (LongitudeFormatter/
   LatitudeFormatter, шаг 15°/10°), заливка суши `cfeature.LAND`
   бледно-коричневым (`#f0e0c0`, alpha=0.3), coastlines. Сама проекция осей —
   `ccrs.LambertConformal(central_latitude=45.0, central_longitude=-45)`
   (вернули по повторной просьбе пользователя, 2026-09-05 — см. "Раунд 3"
   ниже; ноутбук использовал PlateCarree, но это была временная правка).
2. Добавлена поддержка трёх типов исходных данных — LoRes, HiRes, ERA5
   (раньше скрипт понимал только ERA5).
3. Вместо локальных экстремумов теперь рисуется ОДИН глобальный экстремум на
   кластер (`center_cluster`, крупная звезда) + граница оболочки кластера
   (ConvexHull) — это ровно то, что раньше называлось `draw_contours` в
   `plot_R2D()`; теперь это параметр `draw_envelope` (по умолчанию `True`,
   "флаг=True" из просьбы пользователя), выведенный в CLI как
   `--no-envelope`, если понадобится отключить.
4. Радиусы вихрей убраны полностью: нет `show_radius`, нет `RENDER_VARIANTS`,
   нет второго прохода "_no_rads" — на каждую комбинацию ровно один PNG.

Правки по итогам первого реального запуска (2026-09-05)
---------------------------------------------------------
- Легенда (Cyclones/Anticyclones) сначала была убрана по просьбе пользователя,
  затем возвращена обратно (тоже по просьбе) — на карте она ЕСТЬ.
- На самой картинке убраны технические параметры eps/size_filter/hPa из
  заголовка и подписи в углу (для ERA5 подпись теперь просто "ERA5", как у
  LoRes/HiRes) — в имени файла они по-прежнему сохраняются для идентификации.
- Граница оболочки (ConvexHull) на LoRes/HiRes рисовалась неверно: один и тот
  же `cluster_id` мог занимать НЕСКОЛЬКО физически несвязанных областей на
  сетке, и общая выпуклая оболочка через все точки сразу "перетягивала"
  контур через пустое пространство между разными вихрями. Теперь кластер
  сначала разбивается на связные компоненты в ИНДЕКСНОМ пространстве сетки
  (`scipy.ndimage.label`, 4-связность — годится и для регулярной сетки ERA5,
  и для криволинейной сетки WRF), оболочка строится отдельно для каждой
  компоненты. НЕ проверено на реальных данных (нет доступа к файлам) — если
  контур всё ещё выглядит не так, это, вероятно, другая причина (см. issue-лог
  переписки/проекта) — пришлите PNG для точной диагностики.
- Раскладка каталогов LoRes и HiRes на диске оказалась РАЗНОЙ (не опечатка в
  одной ячейке ноутбука, а системное отличие): у LoRes имя типа данных
  повторяется дважды (`{WRF_BASE}/LoRes/LoRes/...`), у HiRes — один раз
  (`{WRF_BASE}/HiRes/...`). См. `WRF_DOUBLE_TYPE_DIR`.
- Координата `Time` в реальных LoRes/HiRes R2D .nc (ежемесячные файлы)
  оказалась НЕ календарным временем, а сырыми номерами шагов —
  `pd.to_datetime()` без ошибки трактовала их как наносекунды с эпохи 1970
  года и тихо подставляла физически бессмысленный "ближайший срез" около
  1970 года. Первая версия фикса (порог year>=1900) была НЕДОСТАТОЧНОЙ: 1970
  проходит проверку `>= 1900`. Порог поднят до 1990 + добавлена проверка, что
  диапазон значений не подозрительно узкий (см. `find_time_index()`).
  Откатывается на: 1) переменную `Times`/`times` (WRF-строки), 2) позиционный
  расчёт индекса по числу срезов в файле и числу дней в месяце (см. докстринг
  `find_time_index` и `--wrf-samples-per-day`, если авто-расчёт числа срезов
  в сутки неверен).

Раунд 3 (тоже 2026-09-05)
--------------------------
- Подпись "{data_type} (XX км)" в правом верхнем углу убрана полностью —
  тип данных виден только в заголовке (`ax.set_title`).
- Проекция осей — снова `ccrs.LambertConformal(central_latitude=45.0,
  central_longitude=-45)` (как в самой первой версии скрипта, до подгонки
  под article_2026_FAO.ipynb) для ВСЕХ трёх типов данных, а не PlateCarree.
  Экстент/gridlines/LAND/coastlines по-прежнему задаются через
  `ax.set_extent(extent, ccrs.PlateCarree())` — cartopy сам трансформирует
  географические границы в display-проекцию, менять их не пришлось.
- LoRes/HiRes теперь тоже поддерживают `--size-filter` (по умолчанию —
  тот же список `SIZE_FILTER_VALUES`, что и для ERA5, обычно 10/25/49):
  базовый пул на диске у LoRes/HiRes — "02-04-10" (eps=2, min_samples=4,
  SOURCE_SIZE_FILTER=10, тот же SOURCE_SIZE_FILTER, что и у ERA5 — конвенция
  именования и переменные идентичны), для size_filter=25/49 кластеры
  пост-фильтруются той же `filter_clusters_by_size()`, что и для ERA5 —
  отдельных .nc для других size_filter на диске у LoRes/HiRes НЕТ, это тот
  же файл. Имя выходного файла теперь всегда содержит `_sf{size_filter:02d}`
  (раньше — только у ERA5).

Раунд 4 (тоже 2026-09-05)
--------------------------
- **Экстент ERA5 заменён** на тот же, что у LoRes: `(-79, -11, 5, 81)` вместо
  рабочего домена ERA5-пайплайна NA_for_TC `(-110, 17, 0, 75)`.
- **Число Ц/АЦ в легенде/заголовке пересчитано под этот экстент**: раньше
  `unique_clusters` (и, соответственно, счётчики Cyclones/Anticyclones в
  легенде, и число CVSs в заголовке) считались по ВСЕЙ сетке датасета —
  после сужения ERA5 до extent LoRes часть кластеров оказалась ЗА пределами
  видимой карты, и легенда считала бы "лишние" кластеры, которых не видно.
  Теперь кластер попадает в счёт (и рисуется оболочкой) только если у него
  есть хотя бы одна точка ВНУТРИ отображаемого extent (см. `in_extent` в
  `plot_R2D()`) — сама оболочка по-прежнему строится по всем точкам кластера
  (края естественно обрезаются осями), меняется только критерий отбора
  "штриховать/считать или нет". Проверка применена одинаково для всех трёх
  типов данных.

Раунд 5 (тоже 2026-09-05)
--------------------------
- **`--vmax-frac` по умолчанию уменьшен с 0.5 до 0.2** (значение из самого
  article_2026_FAO.ipynb) — по просьбе пользователя сделать картинки ярче:
  чем меньше эта доля от max|R2D|, тем быстрее значения упираются в края
  палитры PiYG, то есть меньше блёклых/полупрозрачных промежуточных тонов.
- **Отрисовка заголовка (`ax.set_title`) ВРЕМЕННО отключена** флагом
  `DRAW_TITLE = False` (текст заголовка по-прежнему считается, просто не
  рисуется) — верните `DRAW_TITLE = True`, когда понадобится обратно.

Раунд 6 (тоже 2026-09-05)
--------------------------
- **Добавлен второй, "точный" способ построения границы кластера** —
  `envelope_method="exact"`, портирован по прямой просьбе пользователя из
  присланного им файла `save_eddy_boundary_combined_exact.py` (отдельный
  пайплайн для океанских вихрей). В отличие от `"hull"` (ConvexHull по
  связным компонентам, Раунд 2), метод `"exact"` строит РЕАЛЬНУЮ растровую
  границу маски кластера: клетки сетки представляются рёбрами ("crack
  code"), общие рёбра соседних заполненных клеток взаимно уничтожаются,
  оставшиеся рёбра собираются в замкнутые контуры (`_raster_mask_to_loops`),
  внутренние дыры маски заполняются (`ndimage.binary_fill_holes`) — то есть
  контур облегает кластер вплотную по границам ячеек сетки, а не "натягивает
  плёнку" через выпуклую оболочку. Контур затем упрощается алгоритмом
  Дугласа-Пекера с проверкой на самопересечения (`_simplify_closed_polygon`,
  `_has_self_intersection`) до не более `max_contour_points` точек (CLI:
  `--max-contour-points`, по умолчанию 150). Портированные функции:
  `_pixel_edges`, `_raster_mask_to_loops`, `_shoelace_signed`,
  `_merge_collinear`, `_has_self_intersection`, `_rdp`,
  `_simplify_closed_polygon`, `grid_corners`, `build_exact_boundary`.
  В отличие от `"hull"`, метод `"exact"` берёт только САМУЮ КРУПНУЮ связную
  компоненту кластера (не строит отдельный контур на каждую) — если у одного
  `cluster_id` несколько физически разнесённых областей, мелкие останутся без
  контура. НЕ ПРОВЕРЕНО на реальных данных.
- **Добавлен третий вариант — вообще без границы** (`draw_envelope=False`):
  легенда Cyclones/Anticyclones теперь рисуется НЕЗАВИСИМО от `draw_envelope`
  (раньше была внутри того же `if`, что и отрисовка контура) — то есть в
  варианте "без оболочки" звёзды экстремумов и легенда с их числом всё равно
  показываются, просто без линии контура.
- **Скрипт теперь генерирует ВСЕ три варианта автоматически на каждом
  запуске** (по просьбе пользователя "все рисуй при запуске"), а не один
  выбранный: для каждой комбинации (`data_type`/`eps`, `size_filter`)
  сохраняется по одному PNG на вариант `hull` / `exact` / `none`, с суффиксом
  `_env-{variant}` в имени файла — то есть общее число картинок за запуск
  ТРОИТСЯ по сравнению с предыдущими раундами. Флаг `--no-envelope` УБРАН;
  вместо него — `--envelope-variants hull exact none` (можно сузить список,
  по умолчанию все три) и новый `--max-contour-points` (только для `exact`).
  Конфигурация вариантов — словарь `ENVELOPE_VARIANTS` рядом с `MAP_EXTENT`.

Раунд 7 (тоже 2026-09-05)
--------------------------
- **Контуры (граница оболочки) теперь рисуются для ВСЕХ вихрей на сетке, а
  не только для тех, у кого есть точка внутри показанного extent.** Раньше
  список кластеров для отрисовки контура (`unique_clusters`) вычислялся из
  той же extent-отфильтрованной выборки, что и счётчики Ц/АЦ в легенде
  (Раунд 4) — а звёзды глобальных экстремумов (`center_cluster`) рисовались
  БЕЗ такого ограничения. Из-за этого расхождения на карте могла быть видна
  звезда экстремума без контура вокруг неё — по-видимому, это и была жалоба
  пользователя "рисуй контуры вокруг всех вихрей, а не только тех, что
  попали в extent". Теперь контур строится по `all_clusters` — ВСЕМ
  ненулевым id на всей сетке, без ограничения по extent (см. `plot_R2D()`).
  Всё, что физически лежит за пределами `ax.set_extent(...)`, matplotlib/
  cartopy всё равно обрежут при рендере, так что лишних видимых линий это
  не добавит. Счётчики Ц/АЦ в легенде/заголовке ПО-ПРЕЖНЕМУ считаются по
  extent-отфильтрованной выборке (`unique_clusters`, решение Раунда 4 не
  тронуто) — то есть теперь возможна ситуация, когда контуров на карте
  видно немного больше/иначе, чем цифра в легенде: легенда считает только
  вихри, реально показанные внутри extent, а не все, для кого нарисован
  контур. НЕ ПРОВЕРЕНО на реальных данных.
- **Порядок вариантов границы по умолчанию изменён**: было
  `hull → exact → none`, стало `none → hull → exact` (по прямой просьбе
  пользователя) — словарь `ENVELOPE_VARIANTS` переставлен, `main()` без
  изменений (порядок берётся из `list(ENVELOPE_VARIANTS.keys())`). Внутри
  каждого варианта порядок типов данных как раньше: ERA5, затем LoRes, затем
  HiRes (LoRes/HiRes — оба WRF, ERA5 — reanalysis, по сути и есть то
  разделение "LoRes/ERA5, HiRes", о котором просил пользователь).

Опорный факт для объединения трёх типов данных: несмотря на разные форматы
хранения (см. ниже), переменные внутри .nc называются ОДИНАКОВО что для ERA5
(6_grid_runner_TE_ERA5.py / 1_create_Nodes_from_DBSCAN.py), что для LoRes/HiRes
(article_2026_FAO.ipynb, PMC_map_pic.ipynb): `cluster`, `center_cluster`,
`local_extr_cluster`, `local_extr_crit`, `rad_eff`, `local_extr_rad_eff`, поле
критерия — `R2D`. Отличаются только: имена координат (ERA5 — 1D `longitude`/
`latitude`; LoRes/HiRes — 2D `XLONG`/`XLAT`, нативная сетка WRF) и раскладка
файлов на диске (см. ниже).

Формат хранения по типам данных
--------------------------------
- **ERA5**: один .nc на дату из пайплайна 6_grid_runner_TE_ERA5.py (Stage A),
  поле R2D и кластеры DBSCAN уже в одном файле — как в предыдущей версии
  скрипта, без изменений (`stage_a_nc_path()`).
- **LoRes / HiRes** (WRF NAAD): поле R2D и результат DBSCAN лежат в РАЗНЫХ
  файлах на диске (см. article_2026_FAO.ipynb, ячейки 2-3, 6-8):
    * DBSCAN (кластеры/центры) — ЕЖЕДНЕВНЫЙ .nc:
      `{WRF_BASE}/{data_type}/{data_type}/DBSCAN_02-04-10_smoothing_sigma_{sigma}_daily/{year}/sigma_{sigma}_DBSCAN_{data_type}_level_12_{date}.nc`
    * R2D (поле критерия) — ЕЖЕМЕСЯЧНЫЙ .nc:
      `{WRF_BASE}/{data_type}/{data_type}/R2D_{data_type}_level_12_smoothing_sigma_{sigma}/sigma_{sigma}_R2D_{data_type}_level_12_{year}-{month}.nc`
  Оба файла делят одну и ту же сетку WRF, поэтому при отрисовке (как и в
  ноутбуке) координаты `XLONG`/`XLAT` берутся из DBSCAN-файла, а значения
  поля R2D — из отдельного R2D-файла (`plot_wrf()` ниже).

  ВНИМАНИЕ, НЕ ПРОВЕРЕНО НА РЕАЛЬНЫХ ФАЙЛАХ: пути для LoRes/HiRes скопированы
  из ноутбука дословно, но в самом article_2026_FAO.ipynb / PMC_map_pic.ipynb
  для них же встречаются ЗАКОММЕНТИРОВАННЫЕ альтернативные варианты пути
  (разное число уровней вложенности `{data_type}`, наличие/отсутствие
  `_daily/{year}`) — то есть даже в исходном ноутбуке раскладка была не до
  конца стабильна. Ни у облачной песочницы, ни у Cowork-моста нет доступа к
  `/storage/thalassa/...`, поэтому пути здесь — лучшее предположение, а НЕ
  протестированный факт. Если при первом реальном запуске путь не совпадёт —
  поправьте `wrf_dbscan_nc_path()` / `wrf_r2d_nc_path()` ниже (или передайте
  готовые пути напрямую через `--lores-cluster-nc/--lores-r2d-nc` и
  `--hires-cluster-nc/--hires-r2d-nc`, они имеют приоритет над шаблоном).
  Также не проверено: реальное имя dim'а уровня в WRF-файлах (в коде
  использовано общее решение — squeeze по первому найденному из
  `LEVEL_DIM_CANDIDATES`, т.к. в имени файла уже зашит `level_12`, т.е. в
  файле должен быть только один уровень).

Пример запуска
--------------
python plot_R2D_maps.py                                    # LoRes+HiRes+ERA5 x size_filter=10/25/49 x envelope=hull/exact/none, 2010-08-28 12:00
python plot_R2D_maps.py --data-types ERA5                  # только ERA5, вся сетка eps x size_filter x envelope-variants
python plot_R2D_maps.py --data-types LoRes HiRes
python plot_R2D_maps.py --size-filter 10                   # без пост-фильтрации, только базовый пул
python plot_R2D_maps.py --date 2010-08-28 --hour 12
python plot_R2D_maps.py --envelope-variants hull            # только один вариант границы, вместо всех трёх
python plot_R2D_maps.py --envelope-variants none            # без границы вообще (только звёзды + легенда)
python plot_R2D_maps.py --max-contour-points 200            # для envelope_method=exact — детальнее контур
python plot_R2D_maps.py --out-dir /storage/.../my_maps

Каждая карта — один PNG (суффикс env-{variant}: hull/exact/none, см. Раунд 6):
  ERA5:          R2D_map_ERA5_eps{eps}_sf{size_filter}_env-{variant}_{date}_{hour}00.png
  LoRes / HiRes: R2D_map_{data_type}_sigma_{sigma}_sf{size_filter}_env-{variant}_{date}_{hour}00.png
"""

from __future__ import annotations

import argparse
import calendar
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter

from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
grid_runner = importlib.import_module("6_grid_runner_TE_ERA5")

# === CONFIG ERA5 (переиспользуем из 6_grid_runner_TE_ERA5.py — единая точка правды) ===
PATH_INIT = grid_runner.PATH_INIT
DATA_TYPE_ERA5 = grid_runner.DATA_TYPE
SIGMA_ERA5 = grid_runner.SIGMA
LEVEL_HPA = grid_runner.LEVEL_HPA
REGION_NAME = grid_runner.REGION_NAME
MIN_SAMPLES = grid_runner.MIN_SAMPLES
EPS_VALUES = grid_runner.EPS_VALUES
SIZE_FILTER_VALUES = grid_runner.SIZE_FILTER_VALUES
# SOURCE_SIZE_FILTER == 10: Stage A всегда на этом пуле (см. докстринг модуля)
SOURCE_SIZE_FILTER = grid_runner.stage_b_global_local.SOURCE_SIZE_FILTER

# === CONFIG LoRes/HiRes (WRF NAAD) — см. докстринг модуля про непроверенность ===
WRF_BASE = "/storage/thalassa/users/vkoshkina/data"
WRF_DATA_TYPES = ("LoRes", "HiRes")
WRF_SIGMA_DEFAULT = 2
LEVEL_DIM_CANDIDATES = ("level", "bottom_top", "num_metgrid_levels")

# Раскладка каталогов на диске у LoRes и HiRes РАЗНАЯ (подтверждено реальным
# запуском 2026-09-05): у LoRes имя типа данных повторяется дважды
# ({WRF_BASE}/LoRes/LoRes/...), у HiRes — один раз ({WRF_BASE}/HiRes/...).
# И article_2026_FAO.ipynb, и PMC_map_pic.ipynb внутренне последовательны в
# этом (все ячейки с data_type='HiRes' используют одинарный путь), поэтому
# это не опечатка в одной ячейке, а системное отличие между типами данных.
WRF_DOUBLE_TYPE_DIR = {"LoRes": True, "HiRes": False}

# Экстенты — дословно из article_2026_FAO.ipynb (LoRes/HiRes). По просьбе
# пользователя (2026-09-05) ERA5 теперь использует ТОТ ЖЕ экстент, что и
# LoRes, вместо рабочего домена ERA5-пайплайна NA_for_TC (-110, 17, 0, 75).
MAP_EXTENT = {
    "LoRes": (-79, -11, 5, 81),
    "HiRes": (-77, -13, 8, 79),
    "ERA5": (-79, -11, 5, 81),
}

# Варианты отрисовки границы кластера, генерируются АВТОМАТИЧЕСКИ на каждом
# запуске (Раунд 6, по просьбе пользователя): "hull" — ConvexHull по
# компонентам связности (Раунд 2), "exact" — точная растровая граница,
# портированная из save_eddy_boundary_combined_exact.py (Раунд 6), "none" —
# без границы вообще (легенда Cyclones/Anticyclones всё равно рисуется).
# Значение — (draw_envelope, envelope_method); для "none" envelope_method
# не используется (draw_envelope=False).
#
# Порядок ключей ниже — это и порядок отрисовки по умолчанию (Раунд 7, по
# просьбе пользователя, 2026-09-05): сначала "none" по ВСЕМ типам данных
# (LoRes/ERA5, HiRes), затем "hull" по всем типам данных, затем "exact" —
# см. цикл `for variant in args.envelope_variants:` в main() ниже, где
# вариант — внешний цикл, а типы данных — внутренний.
ENVELOPE_VARIANTS: dict[str, tuple[bool, str]] = {
    "none": (False, "hull"),
    "hull": (True, "hull"),
    "exact": (True, "exact"),
}

DEFAULT_DATE = "2010-08-28"
DEFAULT_HOUR = 0
DEFAULT_DATA_TYPES = ["LoRes", "HiRes", "ERA5"]
DEFAULT_OUT_DIR = Path(PATH_INIT) / "TempestExtremes" / "R2D_maps_article_style"

# ВРЕМЕННО отключена отрисовка заголовка (ax.set_title) по просьбе
# пользователя, 2026-09-05 — сам текст заголовка по-прежнему считается (см.
# plot_R2D()), просто не рисуется. Верните True, чтобы включить обратно.
DRAW_TITLE = False


# ============================================================================
# Пути к файлам
# ============================================================================

def stage_a_nc_path(eps: int, date: str) -> Path:
    """ERA5 Stage A .nc — без изменений относительно предыдущей версии скрипта.
    Всегда на пуле SOURCE_SIZE_FILTER=10, независимо от целевого size_filter
    (см. докстринг модуля и fix_stale_nodes_txt.py::stage_a_nc_dir)."""
    nc_dir = (
        Path(PATH_INIT) / "ERA5"
        / f"DBSCAN_{eps:02d}-{MIN_SAMPLES:02d}-{SOURCE_SIZE_FILTER:02d}_{REGION_NAME}_sigma_{SIGMA_ERA5}_rad"
    )
    return nc_dir / f"sigma_{SIGMA_ERA5}_DBSCAN_{DATA_TYPE_ERA5}_{date}.nc"


def _wrf_type_base(data_type: str) -> Path:
    base = Path(WRF_BASE) / data_type
    if WRF_DOUBLE_TYPE_DIR.get(data_type, True):
        base = base / data_type
    return base


def wrf_dbscan_nc_path(data_type: str, sigma: int, date: str) -> Path:
    """LoRes/HiRes: ЕЖЕДНЕВНЫЙ .nc с результатом DBSCAN (кластеры/центры).
    Раскладка — см. докстринг модуля и WRF_DOUBLE_TYPE_DIR выше."""
    year = date.split("-")[0]
    return (
        _wrf_type_base(data_type)
        / f"DBSCAN_02-04-10_smoothing_sigma_{sigma}_daily" / year
        / f"sigma_{sigma}_DBSCAN_{data_type}_level_12_{date}.nc"
    )


def wrf_r2d_nc_path(data_type: str, sigma: int, date: str) -> Path:
    """LoRes/HiRes: ЕЖЕМЕСЯЧНЫЙ .nc с полем R2D (тот же критерий, другая
    раскладка по времени — см. докстринг модуля)."""
    year, month, _ = date.split("-")
    return (
        _wrf_type_base(data_type)
        / f"R2D_{data_type}_level_12_smoothing_sigma_{sigma}"
        / f"sigma_{sigma}_R2D_{data_type}_level_12_{year}-{month}.nc"
    )


# ============================================================================
# filter_clusters_by_size — портировано без изменений из ERA5_plot_test.ipynb
# (см. предыдущую версию скрипта). Применимо только к ERA5-сетке eps x
# size_filter — LoRes/HiRes используют фиксированные параметры DBSCAN,
# зашитые в путь к файлу, без пост-фильтрации.
# ============================================================================

def filter_clusters_by_size(ds: xr.Dataset, min_points: int) -> xr.Dataset:
    """Оставляет в 2D-срезе (lat, lon) только кластеры с размером >= min_points,
    переиндексируя id кластеров с 1. Обновляет cluster/center/center_cluster/
    rad_eff/local_extr_crit/local_extr_cluster/local_extr_rad_eff."""
    ds_filtered = ds.copy(deep=True)

    cluster_ids = np.unique(ds["cluster"].values)
    cluster_ids = cluster_ids[cluster_ids != 0]

    keep = {cid for cid in cluster_ids if int(np.sum(ds["cluster"].values == cid)) >= min_points}

    new_mask = np.zeros_like(ds["cluster"].values, dtype=np.int16)
    id_mapping = {}
    # Циклоны (id > 0) и антициклоны (id < 0) перенумеровываются РАЗДЕЛЬНО,
    # каждые со своего 1, с сохранением знака.
    pos_ids = sorted(cid for cid in keep if cid > 0)
    neg_ids = sorted((cid for cid in keep if cid < 0), reverse=True)
    for new_id, old_id in enumerate(pos_ids, start=1):
        id_mapping[old_id] = new_id
        new_mask[ds["cluster"].values == old_id] = new_id
    for new_id, old_id in enumerate(neg_ids, start=1):
        id_mapping[old_id] = -new_id
        new_mask[ds["cluster"].values == old_id] = -new_id

    ds_filtered["cluster"] = (ds["cluster"].dims, new_mask)

    if "center" in ds_filtered:
        center_values = ds_filtered["center"].values.copy()
        center_values[new_mask == 0] = 0
        ds_filtered["center"] = (ds["center"].dims, center_values)

    if "center_cluster" in ds_filtered:
        center_cluster_values = ds_filtered["center_cluster"].values.copy()
        center_cluster_values[new_mask == 0] = 0
        for old_id, new_id in id_mapping.items():
            center_cluster_values[ds["center_cluster"].values == old_id] = new_id
        ds_filtered["center_cluster"] = (ds["center_cluster"].dims, center_cluster_values)

    if "rad_eff" in ds_filtered:
        rad_eff_values = ds_filtered["rad_eff"].values.copy()
        rad_eff_values[new_mask == 0] = np.nan
        ds_filtered["rad_eff"] = (ds["rad_eff"].dims, rad_eff_values)

    for var, zero_fill in [
        ("local_extr_crit", 0), ("local_extr_cluster", 0), ("local_extr_rad_eff", np.nan),
    ]:
        if var not in ds_filtered:
            continue
        var_values = ds_filtered[var].values.copy()
        var_values[new_mask == 0] = zero_fill
        if var == "local_extr_cluster":
            for old_id, new_id in id_mapping.items():
                var_values[ds[var].values == old_id] = new_id
        ds_filtered[var] = (ds[var].dims, var_values)

    return ds_filtered


# ============================================================================
# ТОЧНАЯ (растровая) оболочка кластера — альтернатива ConvexHull, портировано
# практически без изменений из save_eddy_boundary_combined_exact.py
# (пользователь прислал этот файл 2026-09-05 как образец для порта). В
# отличие от ConvexHull, контур идёт РОВНО по границам ячеек сетки, входящих
# в кластер (crack-code трассировка) — повторяет реальную, в общем случае
# невыпуклую форму (полумесяцы, "гантели", вихри с перемычками), а не
# "раздувает" её выпуклой оболочкой. Внутренние дыры (ячейки не в кластере,
# но со всех сторон окружённые им) ЗАПОЛНЯЮТСЯ. Контур всегда простой (без
# самопересечений) многоугольник; если кластер после заполнения дыр всё
# равно распадается на несколько частей (соприкасаются только по диагонали),
# берётся самая большая по площади — в отличие от ConvexHull-варианта выше
# (Раунд 2), здесь части НЕ разбиваются на отдельные контуры: один контур на
# cluster_id, как в исходном скрипте пользователя.
#
# Выбирается через envelope_method="exact" в plot_R2D() / --envelope-variants
# в CLI (см. main()).
# ============================================================================

def _pixel_edges(i, j):
    """4 направленных ребра ячейки (i,j) в координатах узлов сетки (row,col)."""
    return (
        ((i + 1, j), (i + 1, j + 1)),   # низ
        ((i + 1, j + 1), (i, j + 1)),   # право
        ((i, j + 1), (i, j)),           # верх
        ((i, j), (i + 1, j)),           # лево
    )


def _raster_mask_to_loops(filled):
    """filled: 2D bool (ny, nx). Возвращает список замкнутых петель (каждая -
    (M+1,2) массив вершин (row,col) в координатах узлов сетки, первая точка =
    последней) методом crack-code трассировки: общее ребро двух соседних
    закрашенных ячеек получает противоположные направления и потому взаимно
    сокращается, остаются только настоящие граничные рёбра."""
    ys, xs = np.nonzero(filled)
    all_edges = set()
    for i, j in zip(ys.tolist(), xs.tolist()):
        for e in _pixel_edges(i, j):
            all_edges.add(e)

    boundary = {}
    for a, b in all_edges:
        if (b, a) not in all_edges:
            boundary[a] = b

    loops = []
    visited = set()
    for start in list(boundary.keys()):
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        cur = boundary[start]
        while cur != start:
            loop.append(cur)
            visited.add(cur)
            cur = boundary[cur]
        loop.append(start)
        loops.append(np.array(loop, dtype=np.int64))
    return loops


def _shoelace_signed(loop_rc):
    x = loop_rc[:, 1].astype(float)
    y = loop_rc[:, 0].astype(float)
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])


def _merge_collinear(loop_rc):
    """Убирает вершины, лежащие точно на прямой между соседями — без
    изменения формы контура (безопасно, самопересечений внести не может)."""
    pts = loop_rc[:-1]
    n = len(pts)
    if n <= 3:
        return loop_rc
    keep = []
    for k in range(n):
        prev = pts[k - 1]
        cur = pts[k]
        nxt = pts[(k + 1) % n]
        v1 = cur - prev
        v2 = nxt - cur
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross != 0:
            keep.append(cur)
    if len(keep) < 3:
        return loop_rc
    keep = np.array(keep)
    return np.vstack([keep, keep[0]])


def _has_self_intersection(poly_xy):
    """Векторизованная (numpy) проверка самопересечений замкнутого
    многоугольника — O(n^2), используется только как guard при упрощении
    (n там уже небольшое, <= max_points)."""
    n = len(poly_xy) - 1
    if n < 4:
        return False
    p1 = poly_xy[:-1]
    p2 = poly_xy[1:]

    def ccw(a, b, c):
        return (c[..., 1] - a[..., 1]) * (b[..., 0] - a[..., 0]) - (
            b[..., 1] - a[..., 1]
        ) * (c[..., 0] - a[..., 0])

    A1 = p1[:, None, :]
    A2 = p2[:, None, :]
    B1 = p1[None, :, :]
    B2 = p2[None, :, :]

    d1 = ccw(B1, B2, A1)
    d2 = ccw(B1, B2, A2)
    d3 = ccw(A1, A2, B1)
    d4 = ccw(A1, A2, B2)

    cross = ((d1 > 0) & (d2 < 0) | (d1 < 0) & (d2 > 0)) & (
        (d3 > 0) & (d4 < 0) | (d3 < 0) & (d4 > 0)
    )
    ii = np.arange(n)
    adjacent = (np.abs(ii[:, None] - ii[None, :]) <= 1) | (
        (ii[:, None] == 0) & (ii[None, :] == n - 1)
    ) | ((ii[None, :] == 0) & (ii[:, None] == n - 1))
    cross &= ~adjacent
    return bool(cross.any())


def _rdp(points, eps):
    """Классический Рамер-Дуглас-Пекер для открытой полилинии."""
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    d = end - start
    norm = np.hypot(*d)
    if norm < 1e-12:
        dists = np.hypot(*(points[1:-1] - start).T)
    else:
        cross = d[0] * (start[1] - points[1:-1, 1]) - d[1] * (start[0] - points[1:-1, 0])
        dists = np.abs(cross) / norm
    if len(dists) == 0:
        return points
    idx = np.argmax(dists)
    if dists[idx] <= eps:
        return np.array([start, end])
    left = _rdp(points[: idx + 2], eps)
    right = _rdp(points[idx + 1 :], eps)
    return np.vstack([left[:-1], right])


def _simplify_closed_polygon(loop_xy, max_points, eps_steps=40):
    """Упрощает замкнутый контур до <= max_points точек. Контур режется на 2
    полилинии по самой дальней паре вершин, каждая упрощается отдельно
    (Дуглас-Пекер), результат склеивается обратно. Итог принимается ТОЛЬКО
    если он остаётся простым многоугольником (без самопересечений) — иначе
    шаг упрощения делается тоньше; если так и не получилось — возвращается
    неупрощённый контур (корректность важнее лимита точек)."""
    pts = loop_xy[:-1]
    n = len(pts)
    if n <= max_points:
        return loop_xy

    sample = np.linspace(0, n - 1, min(n, 200), dtype=int)
    sub = pts[sample]
    d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
    ia, ib = np.unravel_index(np.argmax(d2), d2.shape)
    i0, i1 = sample[ia], sample[ib]
    if i0 > i1:
        i0, i1 = i1, i0

    chain1 = pts[i0 : i1 + 1]
    chain2 = np.vstack([pts[i1:], pts[: i0 + 1]])

    def simplify_at(eps):
        s1 = _rdp(chain1, eps)
        s2 = _rdp(chain2, eps)
        return np.vstack([s1[:-1], s2[:-1], s1[:1]])

    max_dist = max(float(np.hypot(*(pts.max(0) - pts.min(0)))), 1e-9)

    lo, hi = 0.0, max_dist
    coarsest = simplify_at(hi)
    if len(coarsest) - 1 > max_points:
        return coarsest if not _has_self_intersection(coarsest) else loop_xy

    for _ in range(eps_steps):
        eps = 0.5 * (lo + hi)
        if len(simplify_at(eps)) - 1 <= max_points:
            hi = eps
        else:
            lo = eps
    eps_needed = hi

    scale = 1.0
    for _ in range(25):
        eps = eps_needed * scale
        merged = simplify_at(eps)
        if len(merged) - 1 <= max_points and not _has_self_intersection(merged):
            return merged
        scale *= 1.15

    return loop_xy  # не удалось безопасно упростить — отдаём как есть


def grid_corners(centers):
    """centers: (ny, nx) координаты центров ячеек (lon ИЛИ lat). Возвращает
    (ny+1, nx+1) координаты узлов (углов ячеек): внутренние узлы — среднее 4
    соседних центров, крайние узлы — через дублирование крайних значений
    (тот же приём, что в pcolormesh(..., shading='auto')). Работает и для
    регулярной сетки ERA5, и для криволинейной 2D-сетки WRF (XLONG/XLAT) —
    расчёт локальный (соседи по индексу), глобальная регулярность не нужна."""
    padded = np.pad(centers, ((1, 1), (1, 1)), mode="edge")
    return 0.25 * (padded[:-1, :-1] + padded[:-1, 1:] + padded[1:, :-1] + padded[1:, 1:])


def build_exact_boundary(cluster_mask, cluster_id, corner_lon, corner_lat, max_points=150):
    """Точная растровая оболочка кластера cluster_id (вместо ConvexHull).

    corner_lon/corner_lat — координаты УЗЛОВ сетки (см. grid_corners),
    считаются один раз на карту (не на кластер) и передаются сюда для
    каждого cluster_id. Возвращает (K+1, 2) массив [lon, lat] (первая точка =
    последняя) или None, если контур построить не удалось (кластер пуст)."""
    ys, xs = np.where(cluster_mask == cluster_id)
    if len(ys) < 1:
        return None

    ny_full, nx_full = cluster_mask.shape
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    # рамка в 1 ячейку вокруг bbox — чтобы граница кластера не "прилипала"
    # к краю локального окна
    y0p, y1p = max(y0 - 1, 0), min(y1 + 1, ny_full)
    x0p, x1p = max(x0 - 1, 0), min(x1 + 1, nx_full)

    sub_mask = cluster_mask[y0p:y1p, x0p:x1p] == cluster_id
    filled = ndimage.binary_fill_holes(sub_mask)  # дыры внутри кластера включаются в оболочку

    loops = _raster_mask_to_loops(filled)
    if not loops:
        return None
    # если после заполнения дыр всё равно несколько петель (кластер распался
    # на части, соприкасающиеся только по диагонали) — берём самую большую
    # по площади как внешний контур (как в исходном скрипте пользователя)
    loop = max(loops, key=lambda l: abs(_shoelace_signed(l)))
    loop = _merge_collinear(loop)

    rows_g = loop[:, 0] + y0p
    cols_g = loop[:, 1] + x0p
    lon = corner_lon[rows_g, cols_g]
    lat = corner_lat[rows_g, cols_g]
    loop_xy = np.column_stack([lon, lat])

    loop_xy = _simplify_closed_polygon(loop_xy, max_points)
    return loop_xy


# ============================================================================
# plot_R2D — общая отрисовка карты: R2D-заливка + граница оболочки кластера
# (ConvexHull) + один глобальный экстремум на кластер (звезда). Работает
# одинаково для 2D-сетки WRF (XLONG/XLAT) и регулярной сетки ERA5
# (longitude/latitude, предварительно превращённой в meshgrid) — вызывающий
# код всегда передаёт lon2d/lat2d уже в 2D.
#
# Оформление (gridlines/LAND/coastlines/подпись в углу) — как в
# plot_DBSCAN_centers_map() из article_2026_FAO.ipynb.
# ============================================================================

def plot_R2D(
    ax,
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    field_2d: np.ndarray,
    cluster_ds_t: xr.Dataset,
    vmax: float,
    name: str,
    extent: tuple[float, float, float, float],
    cluster_var: str = "cluster",
    center_cluster_var: str = "center_cluster",
    draw_envelope: bool = True,
    envelope_method: str = "hull",
    max_contour_points: int = 150,
) -> None:
    """Отрисовка R2D с одним глобальным экстремумом на кластер (звезда) и
    (опционально) границей оболочки кластера, в оформлении article_2026_FAO.ipynb.

    field_2d — уже 2D-массив (lat, lon) со значениями R2D на срез (может
    приходить из ДРУГОГО .nc, чем cluster_ds_t — см. wrf_r2d_nc_path в
    докстринге модуля); lon2d/lat2d — координаты СЕТКИ CLUSTER_ds_t (общие
    для поля и кластеров, т.к. это один и тот же грид).

    envelope_method: "hull" — ConvexHull по связным компонентам (Раунд 2,
    2026-09-05); "exact" — точная растровая оболочка кластера (Раунд 6,
    портировано из save_eddy_boundary_combined_exact.py, повторяет реальную
    невыпуклую форму, дыры заполняются). Учитывается только если
    draw_envelope=True.
    """
    ax.set_global()
    gl = ax.gridlines(draw_labels=True, linewidth=1, color="grey", alpha=0.7, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlocator = mticker.MultipleLocator(15)
    gl.ylocator = mticker.MultipleLocator(10)

    ax.add_feature(cfeature.LAND, facecolor="#f0e0c0", alpha=0.3)
    ax.coastlines(color="k", alpha=0.7, lw=1)

    field_masked = np.where(field_2d == 0, np.nan, field_2d)
    ax.contourf(
        lon2d, lat2d, field_masked,
        cmap="PiYG", vmin=-vmax, vmax=vmax, transform=ccrs.PlateCarree(),
    )

    ax.set_extent(list(extent), ccrs.PlateCarree())
    ax.coastlines(color="k", alpha=0.7, lw=1)

    cluster_mask = cluster_ds_t[cluster_var].values

    # Число Ц/АЦ в легенде/заголовке считаем только по кластерам, у которых
    # ЕСТЬ хотя бы одна точка внутри показанного extent (Раунд 4, 2026-09-05)
    # — иначе счёт включал бы кластеры, целиком лежащие ЗА пределами видимой
    # карты (актуально для ERA5 после перехода на extent LoRes вместо более
    # широкого рабочего домена NA_for_TC).
    lon_min, lon_max, lat_min, lat_max = extent
    in_extent = (lon2d >= lon_min) & (lon2d <= lon_max) & (lat2d >= lat_min) & (lat2d <= lat_max)
    unique_clusters = np.unique(cluster_mask[in_extent])
    unique_clusters = unique_clusters[unique_clusters != 0]
    n_clusters = len(unique_clusters)

    # А ВОТ КОНТУРЫ (и звёзды экстремумов, см. ниже) рисуем для ВСЕХ
    # кластеров на всей сетке, БЕЗ ограничения по extent (Раунд 7, по
    # просьбе пользователя, 2026-09-05: "рисуй контуры вокруг всех вихрей, а
    # не только тех, что попали в extent"). Раньше контур рисовался только
    # для `unique_clusters` (см. выше) — то есть кластеров с хотя бы одной
    # точкой внутри показанного extent по НАИВНОЙ проверке диапазона
    # lon/lat; звёзды экстремумов (center_cluster) при этом УЖЕ рисовались
    # без такого ограничения (см. ниже), так что если наивная проверка по
    # каким-то причинам не совпадает с тем, что реально видно на карте
    # (например, из-за разной конвенции долготы 0..360 против -180..180
    # между ERA5 и WRF, или просто кластера у самого края extent), звезда
    # экстремума оказывалась на карте БЕЗ контура вокруг неё. Всё, что
    # физически лежит за пределами ax.set_extent(...), matplotlib/cartopy
    # всё равно обрежут при рендере — так что рисовать контур для
    # действительно невидимых кластеров безопасно (лишние вычисления, но не
    # лишние линии на итоговой картинке).
    all_clusters = np.unique(cluster_mask)
    all_clusters = all_clusters[all_clusters != 0]

    # ---------- ГРАНИЦА ОБОЛОЧКИ КЛАСТЕРА ("hull" ИЛИ "exact") ----------
    #
    # "hull" (Раунд 2, 2026-09-05): на LoRes/HiRes ConvexHull по ВСЕМ пикселям
    # cluster_id рисовался неверно, когда один id занимал НЕСКОЛЬКО физически
    # несвязанных областей на сетке — общая выпуклая оболочка через них
    # "перетягивала" контур через пустое пространство между разными вихрями.
    # Кластер сначала разбивается на связные компоненты В ИНДЕКСНОМ
    # пространстве сетки (4-связность, scipy.ndimage.label), оболочка
    # рисуется отдельно для каждой компоненты.
    #
    # "exact" (Раунд 6): точная растровая трассировка границы кластера
    # (build_exact_boundary, портировано из save_eddy_boundary_combined_exact.py,
    # см. докстринг блока выше) — контур повторяет реальную, в общем случае
    # невыпуклую форму, а не выпуклую оболочку.
    if draw_envelope and len(all_clusters) > 0:
        if envelope_method == "exact":
            corner_lon = grid_corners(lon2d)
            corner_lat = grid_corners(lat2d)
            for cluster_id in all_clusters:
                if cluster_id < 0:
                    contour_color, contour_label = "deeppink", f"Anticyclone {cluster_id}"
                else:
                    contour_color, contour_label = "limegreen", f"Cyclone {cluster_id}"

                loop_xy = build_exact_boundary(cluster_mask, cluster_id, corner_lon, corner_lat,
                                                max_contour_points)
                if loop_xy is None or len(loop_xy) < 4:
                    cluster_points = np.where(cluster_mask == cluster_id)
                    if len(cluster_points[0]) > 0:
                        ax.scatter(lon2d[cluster_points], lat2d[cluster_points], color=contour_color,
                                   s=10, alpha=0.5, transform=ccrs.PlateCarree())
                    continue

                ax.plot(loop_xy[:, 0], loop_xy[:, 1], color=contour_color, linewidth=2.5,
                       alpha=0.9, transform=ccrs.PlateCarree(), label=contour_label)
        else:  # "hull"
            for cluster_id in all_clusters:
                if cluster_id < 0:
                    contour_color, contour_label = "deeppink", f"Anticyclone {cluster_id}"
                else:
                    contour_color, contour_label = "limegreen", f"Cyclone {cluster_id}"

                id_mask = cluster_mask == cluster_id
                labeled, n_components = ndimage.label(id_mask)

                for comp_id in range(1, n_components + 1):
                    cluster_points = np.where(labeled == comp_id)
                    if len(cluster_points[0]) < 3:
                        if len(cluster_points[0]) > 0:
                            ax.scatter(lon2d[cluster_points], lat2d[cluster_points], color=contour_color,
                                       s=10, alpha=0.5, transform=ccrs.PlateCarree())
                        continue

                    cluster_lons = lon2d[cluster_points]
                    cluster_lats = lat2d[cluster_points]
                    points = np.column_stack((cluster_lons, cluster_lats))

                    if np.std(cluster_lats) < 1e-10 or np.std(cluster_lons) < 1e-10:
                        ax.scatter(cluster_lons, cluster_lats, color=contour_color, s=10, alpha=0.5,
                                   transform=ccrs.PlateCarree())
                        continue

                    try:
                        hull = ConvexHull(points)
                        hull_points = np.vstack([points[hull.vertices], points[hull.vertices][0]])
                        ax.plot(hull_points[:, 0], hull_points[:, 1], color=contour_color, linewidth=2.5,
                               alpha=0.9, transform=ccrs.PlateCarree(), label=contour_label)
                    except QhullError:
                        try:
                            noisy = points + np.random.normal(0, 1e-8, points.shape)
                            hull = ConvexHull(noisy)
                            hull_points = np.vstack([points[hull.vertices], points[hull.vertices][0]])
                            ax.plot(hull_points[:, 0], hull_points[:, 1], color=contour_color, linewidth=2.5,
                                   alpha=0.9, transform=ccrs.PlateCarree(), label=contour_label)
                        except Exception:
                            ax.scatter(cluster_lons, cluster_lats, color=contour_color, s=10, alpha=0.5,
                                       transform=ccrs.PlateCarree())

    # ---------- ОДИН ГЛОБАЛЬНЫЙ ЭКСТРЕМУМ НА КЛАСТЕР (звезда) ----------
    center_field = cluster_ds_t[center_cluster_var].values
    center_mask = center_field != 0

    if center_mask.any():
        n_centers_total = len(np.unique(center_field[center_mask]))
        cmap = mpl.colormaps["tab20"].resampled(max(n_centers_total, 1))
        ax.scatter(
            lon2d[center_mask], lat2d[center_mask],
            c=center_field[center_mask],
            s=70,
            cmap=cmap,
            marker="*",
            alpha=1.0,
            transform=ccrs.PlateCarree(),
            zorder=20,
            edgecolors="black",
            linewidth=0.5,
        )

    # Легенда с числом Ц/АЦ не зависит от draw_envelope — показывается всегда
    # (в т.ч. в варианте "без оболочки"), т.к. сами счётчики не связаны с тем,
    # рисуется ли контур кластера.
    if n_clusters > 0:
        n_clusters_neg = int(np.sum(unique_clusters < 0))
        n_clusters_pos = int(np.sum(unique_clusters > 0))
        legend_elements = [
            Patch(facecolor="deeppink", alpha=0.7, label=f"Anticyclones: {n_clusters_neg}"),
            Patch(facecolor="limegreen", alpha=0.7, label=f"Cyclones: {n_clusters_pos}"),
        ]
        legend = ax.legend(handles=legend_elements, loc="upper left", fontsize=9, framealpha=0.9)
        legend.set_zorder(25)

    # Подписи "{data_type} (XX км)" в углу больше нет (убрана по просьбе
    # пользователя, 2026-09-05) — тип данных виден только в заголовке.

    ax.tick_params(axis="both", which="both", direction="in", labelsize=9)

    date_str = None
    if "Time" in cluster_ds_t.coords:
        try:
            time_val = pd.Timestamp(cluster_ds_t["Time"].values)
            if time_val.year >= 1990:  # см. find_time_index — иначе это сырой номер шага
                date_str = time_val.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            date_str = None
    title = f"{n_clusters} CVSs" + (f" at {date_str}" if date_str else "") + f" for {name}"
    if DRAW_TITLE:  # временно отключено по просьбе пользователя, 2026-09-05 — см. докстринг модуля
        ax.set_title(title)


# ============================================================================
# Общие утилиты
# ============================================================================

def _days_in_month(date: str) -> int:
    year, month, _ = (int(x) for x in date.split("-"))
    return calendar.monthrange(year, month)[1]


def _decode_wrf_times_var(ds: xr.Dataset) -> pd.DatetimeIndex | None:
    """Некоторые WRF-файлы хранят календарное время не в числовой координате
    Time, а в отдельной строковой переменной Times/times (классический WRF-
    формат вида b'2010-08-28_12:00:00', dims (Time, DateStrLen)). Пробуем её,
    если числовая Time оказалась нерасшифровываемой (см. find_time_index)."""
    for name in ("Times", "times"):
        if name not in ds.variables:
            continue
        raw = ds[name].values
        try:
            decoded = []
            for row in raw:
                if isinstance(row, (bytes, np.bytes_)):
                    s = row.decode("utf-8")
                elif isinstance(row, str):
                    s = row
                elif hasattr(row, "dtype") and row.dtype.kind == "S":
                    s = b"".join(row).decode("utf-8")
                else:
                    s = "".join(str(x) for x in np.atleast_1d(row))
                decoded.append(s.replace("_", " ").strip())
            return pd.to_datetime(decoded)
        except Exception:
            continue
    return None


def find_time_index(ds: xr.Dataset, date: str, hour: int, samples_per_day: int | None = None) -> int:
    """Индекс среза, ближайшего к date+hour, по координате Time.

    ВАЖНО (обнаружено реальным запуском 2026-09-05, на LoRes И HiRes R2D .nc):
    координата Time в этих ежемесячных файлах оказалась НЕ календарным
    временем, а сырыми номерами шагов — pd.to_datetime() трактовала их как
    наносекунды с эпохи 1970 года и без ошибки выдавала физически
    бессмысленный ближайший срез в районе 1970-01-01. Порог детектирования
    year<1900 из первой версии фикса оказался НЕДОСТАТОЧНЫМ (1970 >= 1900) —
    поднят до year<1990 + добавлена проверка на подозрительно узкий диапазон
    значений. Есть два уровня отката:
      1) переменная Times/times (WRF-строки вида '2010-08-28_12:00:00'),
      2) чисто позиционный расчёт: (день_месяца - 1) * шагов_в_сутки + округление(hour / шаг_в_часах),
         где шагов_в_сутки = samples_per_day (если задан явно через
         --wrf-samples-per-day) иначе инферится как
         (число срезов в файле) / (число дней в месяце date) — это ПРЕДПОЛОЖЕНИЕ,
         что файл покрывает ровно один календарный месяц без пропусков;
         если это не так — передайте --wrf-samples-per-day явно.
    """
    target = pd.Timestamp(f"{date} {hour:02d}:00:00")

    times: pd.DatetimeIndex | None = None
    try:
        candidate = pd.to_datetime(ds["Time"].values)
        # Порог 1900 оказался НЕДОСТАТОЧНЫМ на реальных данных: сырые номера
        # шагов (0, 1, 2, ..., N), интерпретированные pd.to_datetime() как
        # наносекунды с эпохи, дают год 1970 — а 1970 >= 1900, проверка
        # проходила. В этом проекте нет и не будет данных раньше 1990-х, так
        # что порог сдвинут туда. Дополнительно проверяется, что диапазон
        # значений не подозрительно узкий (для месячного файла с несколькими
        # срезами в сутки реальный охват — минимум сутки).
        plausible_year = len(candidate) > 0 and candidate.min().year >= 1990
        plausible_span = (
            len(candidate) <= 1
            or (candidate.max() - candidate.min()) >= pd.Timedelta(hours=1)
        )
        if plausible_year and plausible_span:
            times = candidate
    except Exception:
        pass

    if times is None:
        times = _decode_wrf_times_var(ds)

    if times is None:
        n_total = ds.sizes["Time"]
        spd = samples_per_day or max(1, round(n_total / _days_in_month(date)))
        interval_hours = 24.0 / spd
        day_idx = int(date.split("-")[2]) - 1
        offset = int(round(hour / interval_hours)) % spd
        idx = min(day_idx * spd + offset, n_total - 1)
        print(f"  ВНИМАНИЕ: координата Time в {ds.encoding.get('source', '?')} не похожа на "
              f"календарные даты (сырые номера шагов?) — использован позиционный расчёт "
              f"(срезов/сутки={spd}, шаг={interval_hours:.2f}ч) -> индекс {idx}. Если результат "
              f"не тот — задайте --wrf-samples-per-day явно.")
        return idx

    idx = int(np.argmin(np.abs(times - target)))
    if times[idx] != target:
        print(f"  ВНИМАНИЕ: точного совпадения {target} нет в {ds.encoding.get('source', '?')}, "
              f"ближайший срез — {times[idx]}")
    return idx


def _squeeze_extra_dims(ds_t: xr.Dataset) -> xr.Dataset:
    """Убирает размерность уровня (какое бы имя она ни носила — level_12 уже
    зашит в имя файла, поэтому в самом файле должен остаться только 1
    уровень) без изменения смысла данных."""
    for dim in LEVEL_DIM_CANDIDATES:
        if dim in ds_t.dims and ds_t.sizes[dim] == 1:
            ds_t = ds_t.isel({dim: 0})
    return ds_t.squeeze()


# ============================================================================
# Рендер ERA5 — без изменений в загрузке данных относительно предыдущей
# версии скрипта, только вызов обновлённого plot_R2D().
# ============================================================================

def render_era5(eps: int, size_filter: int, date: str, hour: int, out_dir: Path,
                 draw_envelope: bool, envelope_method: str, vmax_frac: float,
                 max_contour_points: int) -> Path | None:
    nc_path = stage_a_nc_path(eps, date)
    if not nc_path.exists():
        print(f"[ERA5 eps={eps} size_filter={size_filter}] ПРОПУСК: нет файла {nc_path} (Stage A не посчитан)")
        return None

    ds = xr.open_dataset(nc_path)
    try:
        t_idx = find_time_index(ds, date, hour)
        ds_t = ds.isel(Time=t_idx)
        ds_t = _squeeze_extra_dims(ds_t)

        if size_filter != SOURCE_SIZE_FILTER:
            ds_t = filter_clusters_by_size(ds_t, min_points=size_filter)

        field_2d = ds_t["R2D"].values
        vmax = vmax_frac * float(np.nanmax(np.abs(field_2d)))
        lon2d, lat2d = np.meshgrid(ds_t["longitude"].values, ds_t["latitude"].values)
        # На самой картинке (заголовок/подпись в углу) технические параметры
        # eps/size_filter/hPa больше не показываются (по просьбе пользователя,
        # 2026-09-05) — они по-прежнему есть в имени файла для идентификации.
        label = "ERA5"

        out_dir.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(14, 10), dpi=200)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(central_latitude=45.0, central_longitude=-45))
        plot_R2D(
            ax, lon2d, lat2d, field_2d, ds_t, vmax, name=label,
            extent=MAP_EXTENT["ERA5"],
            draw_envelope=draw_envelope, envelope_method=envelope_method,
            max_contour_points=max_contour_points,
        )

        env_tag = envelope_method if draw_envelope else "none"
        out_path = out_dir / f"R2D_map_ERA5_eps{eps:02d}_sf{size_filter:02d}_env-{env_tag}_{date}_{hour:02d}00.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[ERA5 eps={eps} size_filter={size_filter} env={env_tag}] сохранено: {out_path}")
        return out_path
    finally:
        ds.close()


# ============================================================================
# Рендер LoRes/HiRes (WRF) — см. докстринг модуля про непроверенность путей.
# ============================================================================

def render_wrf(data_type: str, date: str, hour: int, sigma: int, size_filter: int, out_dir: Path,
                draw_envelope: bool, envelope_method: str, vmax_frac: float,
                max_contour_points: int,
                cluster_nc_override: Path | None, r2d_nc_override: Path | None,
                samples_per_day: int | None = None) -> Path | None:
    """size_filter: базовый пул LoRes/HiRes на диске — это "02-04-10", т.е.
    eps=2, min_samples=4, SOURCE_SIZE_FILTER=10 (см. докстринг модуля) — то
    же самое SOURCE_SIZE_FILTER, что и у ERA5 (переменные и конвенция
    именования идентичны). Для size_filter=25/49 (или любого другого,
    заданного через --size-filter) кластеры пост-фильтруются той же
    filter_clusters_by_size(), что и для ERA5 — новых .nc на диске для
    LoRes/HiRes при других size_filter НЕТ, это ровно тот же файл, что и для
    базового пула."""
    cluster_nc = cluster_nc_override or wrf_dbscan_nc_path(data_type, sigma, date)
    r2d_nc = r2d_nc_override or wrf_r2d_nc_path(data_type, sigma, date)

    if not cluster_nc.exists():
        print(f"[{data_type} size_filter={size_filter}] ПРОПУСК: нет файла с кластерами DBSCAN {cluster_nc}")
        return None
    if not r2d_nc.exists():
        print(f"[{data_type} size_filter={size_filter}] ПРОПУСК: нет файла с полем R2D {r2d_nc}")
        return None

    cluster_ds = xr.open_dataset(cluster_nc)
    r2d_ds = xr.open_dataset(r2d_nc)
    try:
        c_idx = find_time_index(cluster_ds, date, hour, samples_per_day=samples_per_day)
        r_idx = find_time_index(r2d_ds, date, hour, samples_per_day=samples_per_day)

        cluster_ds_t = _squeeze_extra_dims(cluster_ds.isel(Time=c_idx))
        r2d_ds_t = _squeeze_extra_dims(r2d_ds.isel(Time=r_idx))

        if size_filter != SOURCE_SIZE_FILTER:
            cluster_ds_t = filter_clusters_by_size(cluster_ds_t, min_points=size_filter)

        field_2d = np.asarray(r2d_ds_t["R2D"].values)
        lon2d = np.asarray(cluster_ds_t["XLONG"].values)
        lat2d = np.asarray(cluster_ds_t["XLAT"].values)

        if field_2d.shape != lon2d.shape:
            print(f"[{data_type} size_filter={size_filter}] ВНИМАНИЕ: форма поля R2D {field_2d.shape} "
                  f"не совпадает с формой сетки XLONG/XLAT {lon2d.shape} — проверьте, что оба файла "
                  f"на одной и той же сетке WRF (см. докстринг модуля).")

        vmax = vmax_frac * float(np.nanmax(np.abs(field_2d)))
        label = data_type

        out_dir.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(14, 10), dpi=200)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(central_latitude=45.0, central_longitude=-45))
        plot_R2D(
            ax, lon2d, lat2d, field_2d, cluster_ds_t, vmax, name=label,
            extent=MAP_EXTENT[data_type],
            draw_envelope=draw_envelope, envelope_method=envelope_method,
            max_contour_points=max_contour_points,
        )

        env_tag = envelope_method if draw_envelope else "none"
        out_path = out_dir / f"R2D_map_{data_type}_sigma_{sigma}_sf{size_filter:02d}_env-{env_tag}_{date}_{hour:02d}00.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[{data_type} size_filter={size_filter} env={env_tag}] сохранено: {out_path}")
        return out_path
    finally:
        cluster_ds.close()
        r2d_ds.close()


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Карты R2D в стиле article_2026_FAO.ipynb (1 глобальный экстремум + "
                     "граница оболочки, без радиусов) для LoRes/HiRes/ERA5"
    )
    parser.add_argument("--date", default=DEFAULT_DATE, help=f"YYYY-MM-DD, по умолчанию {DEFAULT_DATE}")
    parser.add_argument("--hour", type=int, default=DEFAULT_HOUR, help=f"по умолчанию {DEFAULT_HOUR}")
    parser.add_argument("--data-types", nargs="+", choices=["LoRes", "HiRes", "ERA5"],
                         default=DEFAULT_DATA_TYPES, help=f"по умолчанию {DEFAULT_DATA_TYPES}")

    # eps — сетка только для ERA5 (у LoRes/HiRes eps=2 зашит в путь к файлу,
    # см. wrf_dbscan_nc_path/wrf_r2d_nc_path). size_filter — общий параметр:
    # базовый пул на диске у ВСЕХ трёх типов — SOURCE_SIZE_FILTER (=10), для
    # остальных значений (25/49 по умолчанию) кластеры пост-фильтруются той
    # же filter_clusters_by_size() что для ERA5, что для LoRes/HiRes.
    parser.add_argument("--eps", type=int, nargs="+", default=EPS_VALUES,
                         help=f"только для ERA5, по умолчанию {EPS_VALUES}")
    parser.add_argument("--size-filter", type=int, nargs="+", default=SIZE_FILTER_VALUES,
                         help=f"для всех типов данных, по умолчанию {SIZE_FILTER_VALUES}")

    # LoRes/HiRes-специфичные параметры
    parser.add_argument("--wrf-sigma", type=int, default=WRF_SIGMA_DEFAULT,
                         help=f"sigma сглаживания для LoRes/HiRes, по умолчанию {WRF_SIGMA_DEFAULT}")
    parser.add_argument("--lores-cluster-nc", type=Path, default=None,
                         help="явный путь к .nc с DBSCAN-кластерами LoRes (обходит шаблон пути)")
    parser.add_argument("--lores-r2d-nc", type=Path, default=None,
                         help="явный путь к .nc с полем R2D LoRes (обходит шаблон пути)")
    parser.add_argument("--hires-cluster-nc", type=Path, default=None,
                         help="явный путь к .nc с DBSCAN-кластерами HiRes (обходит шаблон пути)")
    parser.add_argument("--hires-r2d-nc", type=Path, default=None,
                         help="явный путь к .nc с полем R2D HiRes (обходит шаблон пути)")
    parser.add_argument("--wrf-samples-per-day", type=int, default=None,
                         help="сколько срезов в сутки в LoRes/HiRes .nc (для позиционного расчёта "
                              "индекса времени, если координата Time не расшифровывается как "
                              "календарная дата — см. find_time_index); по умолчанию инферится "
                              "как (число срезов в файле) / (число дней в месяце)")

    parser.add_argument("--envelope-variants", nargs="+", choices=list(ENVELOPE_VARIANTS.keys()),
                         default=list(ENVELOPE_VARIANTS.keys()),
                         help="какие варианты границы кластера рисовать за один запуск — "
                              "hull (ConvexHull по компонентам связности), exact (точная "
                              "растровая граница, портирована из "
                              "save_eddy_boundary_combined_exact.py), none (без границы, но "
                              "с легендой Cyclones/Anticyclones). По умолчанию — ВСЕ три "
                              "варианта сразу (по просьбе пользователя, 2026-09-05): для "
                              "каждой комбинации (data_type/eps, size_filter) сохраняется "
                              "по одной картинке на вариант, с суффиксом _env-{variant} в "
                              "имени файла.")
    parser.add_argument("--max-contour-points", type=int, default=150,
                         help="макс. число точек упрощённого контура для envelope_method="
                              "'exact' (см. build_exact_boundary/_simplify_closed_polygon); "
                              "не влияет на 'hull'. По умолчанию 150.")
    parser.add_argument("--vmax-frac", type=float, default=0.2,
                         help="доля от max|R2D| для границ цветовой шкалы — чем меньше, тем "
                              "ярче/контрастнее заливка (значения быстрее упираются в края "
                              "палитры PiYG). По умолчанию 0.2 (как в самом "
                              "article_2026_FAO.ipynb; было 0.5 в предыдущей версии скрипта — "
                              "уменьшено по просьбе пользователя, 2026-09-05, чтобы картинки "
                              "были ярче).")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    print(f"Дата/время: {args.date} {args.hour:02d}:00, типы данных: {args.data_types}")
    print(f"Папка вывода: {args.out_dir}")
    print(f"Варианты границы кластера: {args.envelope_variants}; радиусы вихрей: нет (убраны)")

    saved: list[Path] = []

    for variant in args.envelope_variants:
        draw_envelope, envelope_method = ENVELOPE_VARIANTS[variant]

        if "ERA5" in args.data_types:
            for eps in args.eps:
                for size_filter in args.size_filter:
                    path = render_era5(eps, size_filter, args.date, args.hour, args.out_dir,
                                        draw_envelope, envelope_method, args.vmax_frac,
                                        args.max_contour_points)
                    if path is not None:
                        saved.append(path)

        if "LoRes" in args.data_types:
            for size_filter in args.size_filter:
                path = render_wrf("LoRes", args.date, args.hour, args.wrf_sigma, size_filter, args.out_dir,
                                   draw_envelope, envelope_method, args.vmax_frac,
                                   args.max_contour_points,
                                   args.lores_cluster_nc, args.lores_r2d_nc,
                                   samples_per_day=args.wrf_samples_per_day)
                if path is not None:
                    saved.append(path)

        if "HiRes" in args.data_types:
            for size_filter in args.size_filter:
                path = render_wrf("HiRes", args.date, args.hour, args.wrf_sigma, size_filter, args.out_dir,
                                   draw_envelope, envelope_method, args.vmax_frac,
                                   args.max_contour_points,
                                   args.hires_cluster_nc, args.hires_r2d_nc,
                                   samples_per_day=args.wrf_samples_per_day)
                if path is not None:
                    saved.append(path)

    print(f"\nГотово: {len(saved)} карт(а) сохранено в {args.out_dir}")


if __name__ == "__main__":
    main()
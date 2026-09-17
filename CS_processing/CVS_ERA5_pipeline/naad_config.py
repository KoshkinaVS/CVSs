"""
Общие константы и вспомогательные функции для пайплайна TempestExtremes на
данных NAAD (WRF, LoRes/HiRes) — используются 1_create_Nodes_from_DBSCAN_NAAD.py
и 2_run_StitchNodes_by_year_NAAD.py.

Собрано по аналогии с ERA5-пайплайном в этой же папке (1_create_Nodes_from_DBSCAN.py,
2_run_StitchNodes_by_year.py, 3_create_csv_tracks_from_StitchNodes.py), но под
WRF-специфику NAAD:

- Координаты в исходных Stage-A (DBSCAN identification) nc — НЕ регулярная
  лат/лон сетка, а нативная сетка WRF (south_north x west_east), с широтой/
  долготой в 2D-переменных XLAT/XLONG (значение зависит от ОБОИХ индексов
  сразу, не по отдельности, как lat[i]/lon[j] у ERA5). Время — переменная
  XTIME при размерности Time (см. config_NAAD.json в репозитории
  KoshkinaVS/CVSs: time_name="Time", time_unit="XTIME").
- Сетка WRF NAAD физически ~равномерна в метрах по всему домену (проекция),
  но НЕ равномерна в градусах — поэтому вместо ERA5-поправки на cos(lat)
  (см. plot_R2D_maps.py, project-doc era5-tracking-latlon-adaptation.md) для
  NAAD используется фиксированный шаг сетки в метрах (DIST_M_BY_TYPE),
  одинаковый по всему домену для данного data_type.
- eps/min_samples/size_filter для Stage A (DBSCAN identification — уже
  посчитан отдельно, НЕ этим пайплайном) сейчас одинаковы для LoRes и HiRes
  (2/4/10, см. config_NAAD.json) — то есть DBSCAN ищет соседей в пределах
  2 ЯЧЕЕК СЕТКИ независимо от разрешения. Физически (в км) это разный радиус
  для LoRes (2 x 77.8 ≈ 156 км) и HiRes (2 x 13.9 ≈ 28 км) — по словам
  пользователя (2026-09-05) это осознанный выбор, не то, что нужно чинить
  здесь.

2026-09-05: создано по задаче "аналог TempestExtremes-пайплайна для NAAD
LoRes/HiRes" (Stage B/C, до CSV на трек — Stage E'/сравнение сознательно вне
скоупа этой задачи). Код НЕ прогонялся на реальных данных (нет доступа к
серверу /storage/thalassa/users/vkoshkina/data из этой сессии) — пути и
разбор XTIME/XLAT/XLONG нужно проверить на первом реальном запуске, см. TODO
в 1_create_Nodes_from_DBSCAN_NAAD.py.
"""

from __future__ import annotations

# === Физический шаг сетки (метры) — для перевода индексных величин Stage A
# (eps, rad_eff — обе в ЯЧЕЙКАХ) в физические км, см. rad_km() ниже.
# Значения от пользователя (2026-09-05): LoRes — как в config_NAAD.json
# (dist_m=77824.23), HiRes — отдельное значение (в репозитории на момент
# написания не нашлось config_NAAD_HiRes.json/аналога, откуда его можно было
# бы подтвердить автоматически).
DIST_M_BY_TYPE = {
    "LoRes": 77824.23,
    "HiRes": 13897.18,
}

# === Где лежат уже посчитанные Stage A (DBSCAN identification) nc-файлы.
# ВНИМАНИЕ: пути НЕСИММЕТРИЧНЫ (у LoRes два вложенных "LoRes/LoRes", у HiRes —
# без "HiRes/HiRes") — это реальная структура на сервере (от пользователя,
# 2026-09-05), не опечатка, оставлено как есть намеренно.
# {eps}/{min_samples}/{size_filter} — параметры DBSCAN (Stage A), {sigma} —
# сглаживание, {year} — подпапка года (файлы внутри неё — по одному nc в
# день, как у ERA5, но без плоской раскладки — сразу разложены по годам).
NC_DIR_TEMPLATE_BY_TYPE = {
    "LoRes": (
        "/storage/thalassa/users/vkoshkina/data/LoRes/LoRes/"
        "DBSCAN_{eps:02d}-{min_samples:02d}-{size_filter:02d}_smoothing_sigma_{sigma}_daily/{year}"
    ),
    "HiRes": (
        "/storage/thalassa/users/vkoshkina/data/HiRes/"
        "DBSCAN_{eps:02d}-{min_samples:02d}-{size_filter:02d}_smoothing_sigma_{sigma}_daily/{year}"
    ),
}

# Куда Stage B (этот пайплайн) кладёт узлы/треки — та же схема папок, что и у
# ERA5 (1_create_Nodes_from_DBSCAN.py / 2_run_StitchNodes_by_year.py), чтобы
# 2_run_StitchNodes_by_year.py и 3_create_csv_tracks_from_StitchNodes.py можно
# было использовать БЕЗ ИЗМЕНЕНИЙ — они уже параметризованы по
# data_type/region_name/sigma/eps/... и ничего не знают про формат исходных
# nc, только про формат Nodes-txt, который этот пайплайн производит в том же
# виде, что и ERA5-версия.
PATH_INIT = "/storage/thalassa/users/vkoshkina/data"

# У NAAD нет деления на регионы (в отличие от ERA5 NA_for_TC_850hPa) — домен
# фиксирован самой WRF-сеткой. "full_domain" — предварительное имя папки,
# ничего не проверяет и не фильтрует, просто занимает место region_name в уже
# существующей схеме путей. Переименуйте на своё усмотрение до первого
# реального запуска — это только имя папки на диске, менять код не нужно
# (передаётся через --region-name).
DEFAULT_REGION_NAME = "full_domain"

# Stage A уже посчитан с этими eps/min_samples всегда с SOURCE_SIZE_FILTER=10
# (см. config_NAAD.json) — как и у ERA5, целевой size_filter (10/25/49...)
# применяется поверх этого как пост-фильтр по числу точек в кластере (см.
# get_valid_cluster_ids() в 1_create_Nodes_from_DBSCAN_NAAD.py), без
# повторного запуска DBSCAN.
DEFAULT_EPS = 2
DEFAULT_MIN_SAMPLES = 4
SOURCE_SIZE_FILTER = 10
DEFAULT_SIGMA = 2

# 2026-09-05: временное разрешение NAAD — 3 часа (не 1 час, как ERA5), со
# слов пользователя. Используется как шаг по умолчанию для fallback-разбора
# времени (см. TODO про XTIME в 1_create_Nodes_from_DBSCAN_NAAD.py) и чтобы
# выбрать разумный maxgap ниже.
DEFAULT_TIMESTEP_HOURS = 3

# StitchNodes параметры по умолчанию. search_range/mintime — как в
# ERA5-пайплайне (2_run_StitchNodes_by_year.py: search_range по макс.
# скорости смещения EX <=140 км/ч (Bernhardt & DeGaetano, 2012; Lodise et
# al., 2022), mintime=18ч по Han & Ullrich, 2025) — сами физические критерии
# НЕ пересчитаны и не перепроверены отдельно для NAAD, только перенесены по
# аналогии. maxgap увеличен относительно ERA5 (3ч), чтобы не оказаться МЕНЬШЕ
# одного нативного шага NAAD (3ч) — с maxgap=3ч единственный пропущенный срез
# уже рвал бы трек; здесь допущен 1 пропущенный срез (2 x 3ч = 6ч).
DEFAULT_SEARCH_RANGE_DEG = 1.5
DEFAULT_MINTIME_HOURS = 18
DEFAULT_MAXGAP_HOURS = 6


def rad_km(rad_cells: float, data_type: str) -> float:
    """Перевод rad_eff (в ЯЧЕЙКАХ сетки WRF) в физические км.

    В отличие от ERA5 (нужна поправка на cos(lat), см. plot_R2D_maps.py и
    project-doc era5-tracking-latlon-adaptation.md), сетка WRF NAAD (проекция,
    шаг ~постоянен в метрах по всему домену) поправки на широту не требует —
    просто фиксированный шаг DIST_M_BY_TYPE[data_type].

    НЕ проверено на реальных данных — разумное предположение по dist_m из
    config_NAAD.json, но конкретная проекция WRF-домена NAAD (Lambert
    conformal? polar stereographic?) в этой сессии нигде не подтверждена.
    """
    if data_type not in DIST_M_BY_TYPE:
        raise ValueError(f"Неизвестный data_type={data_type!r}, ожидается один из {list(DIST_M_BY_TYPE)}")
    return rad_cells * DIST_M_BY_TYPE[data_type] / 1000.0


def nc_dir_for_year(
    data_type: str,
    year: int,
    eps: int = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    size_filter: int = SOURCE_SIZE_FILTER,
    sigma: int = DEFAULT_SIGMA,
) -> str:
    """Путь к папке с Stage-A DBSCAN nc-файлами NAAD за конкретный год."""
    if data_type not in NC_DIR_TEMPLATE_BY_TYPE:
        raise ValueError(f"Неизвестный data_type={data_type!r}, ожидается один из {list(NC_DIR_TEMPLATE_BY_TYPE)}")
    template = NC_DIR_TEMPLATE_BY_TYPE[data_type]
    return template.format(eps=eps, min_samples=min_samples, size_filter=size_filter, sigma=sigma, year=year)

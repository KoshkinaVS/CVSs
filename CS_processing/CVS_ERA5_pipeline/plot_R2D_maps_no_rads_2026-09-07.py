"""
2026-09-07: копия plot_R2D_maps.py, урезанная до ОДНОГО варианта отрисовки —
"_no_rads" (только контуры кластеров ConvexHull, без окружностей радиусов
вихрей и подписей км, см. RENDER_VARIANTS ниже) — плюс увеличенная легенда на
русском вида "Ц: NN" / "АЦ: MM" вместо "Cyclones: NN" / "Anticyclones: MM"
(см. plot_R2D: блок legend_elements/ax.legend). Остальная логика не менялась —
см. plot_R2D_maps.py как основной, поддерживаемый вариант скрипта.

Отрисовка карт R2D (Stage A: DBSCAN "сырые" вихри, ДО TempestExtremes-трекинга)
— портировано из plot_R2D() в ERA5_plot_test.ipynb (положен пользователем в
папку TempestExtremes/) для визуального сравнения всей сетки eps x size_filter
(6_grid_runner_TE_ERA5.py: EPS_VALUES x SIZE_FILTER_VALUES).

Задача
------
Один и тот же временной срез (по умолчанию 2010-08-28 12:00) — 6 отдельных
PNG "_no_rads": по одному на комбинацию (eps=1/2) x (size_filter=10/25/49) —
только контуры кластеров ConvexHull (циклоны — limegreen, антициклоны —
deeppink) и звёздочки центров, без окружностей радиусов вихрей и подписей
км; легенда — крупная, на русском ("Ц: NN АЦ: MM").

extr_type сюда не входит: R2D/DBSCAN (Stage A) единый для всех extr_type —
last делится на "global"/"local"/"geom" только на Stage B, при извлечении
узлов TempestExtremes из уже готового .nc (см. докстринг 6_grid_runner_TE_ERA5.py
"Экономия на Stage E'" и 1_create_Nodes_from_DBSCAN.py). Эта отрисовка — карта
самого Stage A, до этого разделения.

size_filter — пост-фильтр, не отдельный .nc
--------------------------------------------
Stage A всегда считает и хранит ТОЛЬКО пул size_filter=SOURCE_SIZE_FILTER (10)
в .nc (см. докстринги fix_stale_nodes_txt.py / 1_create_Nodes_from_DBSCAN.py) —
у size_filter=25/49 нет своих .nc-файлов, они получаются фильтрацией уже
загруженного .nc в памяти (filter_clusters_by_size(), портировано из ноутбука
без изменений). Поэтому скрипт открывает ОДИН .nc на eps (пул size_filter=10)
и для size_filter=25/49 просто фильтрует уже загруженный датасет — Stage A/B
заново не считаются, TempestExtremes не запускается.

Отличия от исходного plot_R2D() в ноутбуке
-------------------------------------------
- В ноутбуке plot_R2D индексировала массивы двумя РАЗНЫМИ позиционными
  индексами (our_time для center/local_extr_cluster, our_level — для
  cluster/R2D/rad_eff/center_cluster) поверх датасета, из которого размерность
  Time то ли была, то ли не была убрана вызывающим кодом заранее — в разных
  ячейках ноутбука это делалось непоследовательно (типичная путаница
  интерактивного ноутбука, не production-код). Реальные размерности данных
  (см. 1_create_Nodes_from_DBSCAN.py: `ds['cluster'].values  # (time, level,
  lat, lon)`) — (time, level, lat, lon), level всегда размера 1 (папка Stage A
  и так на один hPa-уровень). Здесь вместо позиционных индексов один раз явно
  делается `ds.isel(Time=t_idx, level=0).squeeze()` ДО вызова plot_R2D — сама
  функция получает уже чистый 2D-срез (lat, lon), без своих индексов по
  времени/уровню.
- cmaps.MPL_PiYG (внешний пакет cmaps, NCL-палитры) заменён на встроенный в
  matplotlib 'PiYG' — та же самая цветовая схема (MPL_PiYG в пакете cmaps —
  это и есть реэкспорт matplotlib PiYG под NCL-именем), без лишней внешней
  зависимости.
- Параметр `crit`/ветка 'R2D_DB' убраны — в текущем пайплайне такого значения
  extr_type нет (см. выше), эта ветка была нужна в другой, более старой схеме
  именования.
- shapely (Polygon/unary_union) — был импортирован в ноутбуке, но не
  использовался внутри plot_R2D; здесь не импортируется.

Пример запуска
--------------
python plot_R2D_maps_no_rads_2026-09-07.py                              # вся сетка (2x3=6 PNG), 2010-08-28 12:00
python plot_R2D_maps_no_rads_2026-09-07.py --date 2010-08-28 --hour 12
python plot_R2D_maps_no_rads_2026-09-07.py --eps 1 --size-filter 10 25   # сузить сетку
python plot_R2D_maps_no_rads_2026-09-07.py --out-dir /storage/.../my_maps
python plot_R2D_maps_no_rads_2026-09-07.py --center-name center         # звёздочки = глобальный центр кластера
                                                        # (по умолчанию — local_extr_cluster,
                                                        # как в последнем состоянии ноутбука)

Единственный вариант отрисовки (см. RENDER_VARIANTS):
  R2D_eps{eps}_sf{size_filter}_{date}_{hour}00_no_rads.png   — только контуры кластеров, легенда "Ц: NN АЦ: MM"
"""

from __future__ import annotations

import argparse
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
from matplotlib.patches import Patch

import cartopy.crs as ccrs

from scipy.spatial import ConvexHull, QhullError

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
grid_runner = importlib.import_module("6_grid_runner_TE_ERA5")

# === CONFIG (переиспользуем из 6_grid_runner_TE_ERA5.py — единая точка правды) ===
PATH_INIT = grid_runner.PATH_INIT
DATA_TYPE = grid_runner.DATA_TYPE
SIGMA = grid_runner.SIGMA
LEVEL_HPA = grid_runner.LEVEL_HPA
REGION_NAME = grid_runner.REGION_NAME
MIN_SAMPLES = grid_runner.MIN_SAMPLES
EPS_VALUES = grid_runner.EPS_VALUES
SIZE_FILTER_VALUES = grid_runner.SIZE_FILTER_VALUES
# SOURCE_SIZE_FILTER == 10: Stage A всегда на этом пуле (см. докстринг модуля)
SOURCE_SIZE_FILTER = grid_runner.stage_b_global_local.SOURCE_SIZE_FILTER

DEFAULT_DATE = "2010-08-28"
DEFAULT_HOUR = 12
DEFAULT_OUT_DIR = Path(PATH_INIT) / "TempestExtremes" / DATA_TYPE / f"R2D_maps_{REGION_NAME}_sigma_{SIGMA}"


def stage_a_nc_path(eps: int, date: str) -> Path:
    """Тот же .nc, что использует Stage A/Stage B — ВСЕГДА на пуле
    SOURCE_SIZE_FILTER=10, независимо от целевого size_filter (см. докстринг
    модуля и fix_stale_nodes_txt.py::stage_a_nc_dir)."""
    nc_dir = (
        Path(PATH_INIT) / "ERA5"
        / f"DBSCAN_{eps:02d}-{MIN_SAMPLES:02d}-{SOURCE_SIZE_FILTER:02d}_{REGION_NAME}_sigma_{SIGMA}_rad"
    )
    return nc_dir / f"sigma_{SIGMA}_DBSCAN_{DATA_TYPE}_{date}.nc"


# ============================================================================
# filter_clusters_by_size — портировано без изменений (по сути) из
# ERA5_plot_test.ipynb: size_filter=25/49 как пост-фильтр над одним и тем же
# .nc (size_filter=10). Здесь работает над уже time/level-срезанным 2D-
# датасетом (без размерности Time) — то же самое, что notebook делала в
# process_all_times() поэлементно по времени.
# ============================================================================

def filter_clusters_by_size(ds: xr.Dataset, min_points: int) -> xr.Dataset:
    """Оставляет в 2D-срезе (lat, lon) только кластеры с размером >= min_points,
    переиндексируя id кластеров с 1. Обновляет cluster/center/center_cluster/
    rad_eff/local_extr_crit/local_extr_cluster/local_extr_rad_eff — те же поля,
    что фильтровала версия из ноутбука."""
    ds_filtered = ds.copy(deep=True)

    cluster_ids = np.unique(ds["cluster"].values)
    cluster_ids = cluster_ids[cluster_ids != 0]

    keep = {cid for cid in cluster_ids if int(np.sum(ds["cluster"].values == cid)) >= min_points}

    new_mask = np.zeros_like(ds["cluster"].values, dtype=np.int16)
    id_mapping = {}
    # Циклоны (id > 0) и антициклоны (id < 0) перенумеровываются РАЗДЕЛЬНО,
    # каждые со своего 1, с сохранением знака — иначе enumerate() по общему
    # sorted(keep) отображал все id (в т.ч. отрицательные АЦ) в новые
    # ПОЛОЖИТЕЛЬНЫЕ new_id, и на карте все кластеры при size_filter=25/49
    # превращались в циклоны (зелёный цвет), теряя деление Ц/АЦ.
    pos_ids = sorted(cid for cid in keep if cid > 0)
    neg_ids = sorted((cid for cid in keep if cid < 0), reverse=True)  # -1, -2, ... по |id|
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
# plot_R2D — портировано из ERA5_plot_test.ipynb (см. докстринг модуля про
# отличия: чистый 2D-срез вместо our_time/our_level, cmaps -> matplotlib 'PiYG',
# без ветки crit == 'R2D_DB').
# ============================================================================

def plot_R2D(ax, cluster_ds_t: xr.Dataset, vmax: float, center_name: str, name: str,
             cluster_var: str = "cluster", center_cluster_var: str = "center_cluster",
             rad_name: str = "rad_eff", draw_contours: bool = True, show_radius: bool = True) -> None:
    """Отрисовка R2D с кластерами, их контурами (ConvexHull) и радиусами вихрей.

    cluster_ds_t — уже 2D-срез (lat, lon) на один timestamp (см. render_one:
    ds.isel(Time=t_idx, level=0).squeeze()).
    center_name — какая переменная используется для звёздочек-центров:
        'center'              — один глобальный центр на кластер
        'local_extr_cluster'  — все локальные экстремумы внутри кластера
    Контуры/радиусы вихрей всегда рисуются по cluster_var/center_cluster_var/
    rad_name (глобальная геометрия кластера) — как в исходном ноутбуке,
    независимо от center_name.
    """
    ax.set_global()
    gl = ax.gridlines(draw_labels=True, linewidth=2, color="k", alpha=0.6, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    clusters = np.where(cluster_ds_t["R2D"].values == 0, np.nan, cluster_ds_t["R2D"].values)

    mc = ax.contourf(
        cluster_ds_t["longitude"], cluster_ds_t["latitude"], clusters,
        cmap="PiYG", vmin=-vmax, vmax=vmax, transform=ccrs.PlateCarree(),
    )

    xx, yy = np.meshgrid(cluster_ds_t["longitude"].values, cluster_ds_t["latitude"].values)
    cluster_mask = cluster_ds_t[cluster_var].values

    unique_clusters = np.unique(cluster_mask)
    unique_clusters = unique_clusters[unique_clusters != 0]
    n_clusters = len(unique_clusters)

    # ---------- КОНТУРЫ КЛАСТЕРОВ (ConvexHull) ----------
    if draw_contours and n_clusters > 0:
        for cluster_id in unique_clusters:
            if cluster_id < 0:
                contour_color, contour_label = "deeppink", f"Anticyclone {cluster_id}"
            else:
                contour_color, contour_label = "limegreen", f"Cyclone {cluster_id}"

            cluster_points = np.where(cluster_mask == cluster_id)
            if len(cluster_points[0]) < 3:
                continue

            cluster_lons = xx[cluster_points]
            cluster_lats = yy[cluster_points]
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

    # NA_for_TC (см. REGION_NAME) — тот же экстент, что и в ноутбуке
    ax.set_extent([-110, 17, 0, 75], ccrs.PlateCarree())

    # ---------- РАДИУСЫ ВИХРЕЙ ----------
    if show_radius and rad_name in cluster_ds_t and center_cluster_var in cluster_ds_t:
        radius_data = cluster_ds_t[rad_name].values
        cluster_mask_int = cluster_ds_t[cluster_var].values.astype(np.int16)
        center_mask_int = cluster_ds_t[center_cluster_var].values.astype(np.int16)
        unique_centers = np.unique(center_mask_int)
        unique_centers = unique_centers[unique_centers != 0]

        cluster_centers = {}
        for cid in unique_centers:
            cluster_points = np.where(cluster_mask_int == cid)
            if len(cluster_points[0]) < 3:
                continue
            geo_lat = np.mean(yy[cluster_points])
            geo_lon = np.mean(xx[cluster_points])
            center_pos = np.where(center_mask_int == cid)
            if len(center_pos[0]) == 0:
                continue
            radius_val = radius_data[center_pos[0][0], center_pos[1][0]]
            radius_val = radius_val * np.cos(np.radians(geo_lat)) * 0.25 * 111
            if np.isfinite(radius_val) and radius_val > 0:
                cluster_centers[cid] = (geo_lon, geo_lat, radius_val)

        if cluster_centers:
            x0, x1, y0, y1 = ax.get_extent(crs=ccrs.PlateCarree())
            for lon, lat, rad_km in cluster_centers.values():
                rad_deg = rad_km / 111.0
                lon_correction = 1.0 / np.cos(np.radians(lat))
                angles = np.linspace(0, 2 * np.pi, 50)
                circle_lons = lon + rad_deg * np.cos(angles) * lon_correction
                circle_lats = lat + rad_deg * np.sin(angles)

                ax.plot(circle_lons, circle_lats, color="k", linewidth=2.5, alpha=0.8,
                       transform=ccrs.PlateCarree(), zorder=10)
                ax.plot(lon, lat, "o", color="k", markersize=6, transform=ccrs.PlateCarree(), zorder=11)

                if x0 <= lon <= x1 and y0 <= lat <= y1:
                    offset = rad_deg * 0.3
                    ax.text(lon + offset, lat + offset, f"{rad_km:.1f} km", transform=ccrs.PlateCarree(),
                           fontsize=7, fontweight="bold", ha="left", va="bottom",
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="gray"),
                           zorder=12)

    # ---------- ЦЕНТРЫ (звёздочки) ----------
    clusters_centers = np.where(cluster_ds_t[center_name].values == 0, np.nan, cluster_ds_t[center_name].values)
    n_clusters_neg = int(-np.nanmin(clusters_centers)) if np.any(clusters_centers < 0) else 0
    n_clusters_pos = int(np.nanmax(clusters_centers)) if np.any(clusters_centers > 0) else 0
    n_clusters_total = n_clusters_neg + n_clusters_pos
    cmap = mpl.colormaps["tab20"].resampled(max(n_clusters_total, 1))

    ax.scatter(xx, yy, c=clusters_centers, s=5, cmap=cmap, marker="*", alpha=1.0,
              label="center", zorder=15, transform=ccrs.PlateCarree())

    ax.tick_params(axis="both", which="both", direction="in", labelsize=9)
    ax.coastlines(color="k", alpha=0.9, lw=1.5)

    if draw_contours and n_clusters > 0:
        # Крупная легенда на русском ("Ц: NN АЦ: MM") — см. докстринг модуля:
        # это единственное отличие от plot_R2D_maps.py в отрисовке.
        legend_elements = [
            Patch(facecolor="limegreen", alpha=0.7, label=f"Ц: {n_clusters_pos}"),
            Patch(facecolor="deeppink", alpha=0.7, label=f"АЦ: {n_clusters_neg}"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=20, framealpha=0.9,
                  handlelength=2.5, handleheight=1.8, borderpad=0.8, labelspacing=0.8)

    date_str = pd.to_datetime(cluster_ds_t["Time"].values).strftime("%Y-%m-%d %H:%M:%S")
    ax.set_title(f"{n_clusters_total} CVSs at {date_str} for {name}")


# ============================================================================
# CLI / рендер сетки
# ============================================================================

def find_time_index(ds: xr.Dataset, date: str, hour: int) -> int:
    target = pd.Timestamp(f"{date} {hour:02d}:00:00")
    times = pd.to_datetime(ds["Time"].values)
    idx = int(np.argmin(np.abs(times - target)))
    if times[idx] != target:
        print(f"  ВНИМАНИЕ: точного совпадения {target} нет в {ds.encoding.get('source', '?')}, "
              f"ближайший срез — {times[idx]}")
    return idx


# (show_radius, suffix для имени файла) — здесь ТОЛЬКО "no_rads" (см.
# докстринг модуля): draw_contours у plot_R2D всегда True (контуры кластеров
# остаются), убирается show_radius (окружности радиусов вихрей + подписи км).
RENDER_VARIANTS = [
    (False, "_no_rads"),
]


def render_one(eps: int, size_filter: int, date: str, hour: int, out_dir: Path, center_name: str) -> list[Path]:
    nc_path = stage_a_nc_path(eps, date)
    if not nc_path.exists():
        print(f"[eps={eps} size_filter={size_filter}] ПРОПУСК: нет файла {nc_path} (Stage A не посчитан)")
        return []

    ds = xr.open_dataset(nc_path)
    try:
        t_idx = find_time_index(ds, date, hour)
        ds_t = ds.isel(Time=t_idx)
        if "level" in ds_t.dims:
            ds_t = ds_t.isel(level=0)
        ds_t = ds_t.squeeze()

        if size_filter != SOURCE_SIZE_FILTER:
            ds_t = filter_clusters_by_size(ds_t, min_points=size_filter)

        vmax = 0.5 * float(np.nanmax(np.abs(ds_t["R2D"].values)))
        label = f"ERA5 {LEVEL_HPA} hPa, eps={eps}, size_filter>={size_filter}"

        out_dir.mkdir(parents=True, exist_ok=True)
        out_paths = []
        for show_radius, suffix in RENDER_VARIANTS:
            fig = plt.figure(figsize=(14, 10), dpi=200)
            ax = fig.add_subplot(111, projection=ccrs.LambertConformal(central_latitude=45.0, central_longitude=-45))
            plot_R2D(ax, ds_t, vmax, center_name=center_name, name=label, show_radius=show_radius)

            out_path = out_dir / f"R2D_eps{eps:02d}_sf{size_filter:02d}_{date}_{hour:02d}00{suffix}.png"
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"[eps={eps} size_filter={size_filter}] сохранено: {out_path}")
            out_paths.append(out_path)
        return out_paths
    finally:
        ds.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Карты R2D (Stage A, с радиусами) для сетки eps x size_filter на один timestamp"
    )
    parser.add_argument("--date", default=DEFAULT_DATE, help=f"YYYY-MM-DD, по умолчанию {DEFAULT_DATE}")
    parser.add_argument("--hour", type=int, default=DEFAULT_HOUR, help=f"по умолчанию {DEFAULT_HOUR}")
    parser.add_argument("--eps", type=int, nargs="+", default=EPS_VALUES, help=f"по умолчанию {EPS_VALUES}")
    parser.add_argument("--size-filter", type=int, nargs="+", default=SIZE_FILTER_VALUES,
                         help=f"по умолчанию {SIZE_FILTER_VALUES}")
    parser.add_argument("--center-name", choices=["center", "local_extr_cluster"], default="local_extr_cluster",
                         help="какая переменная для звёздочек-центров (по умолчанию local_extr_cluster, "
                              "как в последнем состоянии ноутбука)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    print(f"Дата/время: {args.date} {args.hour:02d}:00, сетка eps={args.eps} x size_filter={args.size_filter}")
    print(f"Папка вывода: {args.out_dir}")

    saved = []
    for eps in args.eps:
        for size_filter in args.size_filter:
            paths = render_one(eps, size_filter, args.date, args.hour, args.out_dir, args.center_name)
            saved.extend(paths)

    n_combos = len(args.eps) * len(args.size_filter)
    n_expected = n_combos * len(RENDER_VARIANTS)
    print(f"\nГотово: {len(saved)}/{n_expected} карт сохранено в {args.out_dir} "
          f"({n_combos} комбинаций eps x size_filter x {len(RENDER_VARIANTS)} варианта)")


if __name__ == "__main__":
    main()

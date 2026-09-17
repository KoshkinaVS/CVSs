#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Плотность треков, число уникальных треков через ячейку и распределения
параметров вдоль трека (скорость ветра, R2D/рортекс, радиус, скорость
смещения) — по годовым txt-файлам StitchNodes (TempestExtremes).

Место в пайплайне
------------------
Читает выход Stage C (2_run_StitchNodes_by_year.py):
    {sigma_dir}/{eps}-{min_samples}-{size_filter}_{extr_type}/
        Tracks_R2D_txt_files{postfix}/{data_type}_TC_tracks_{year}.txt

Формат файла (см. 1_create_Nodes_from_DBSCAN.py, строка записи узла):
    start\t<n_points>\t<year>\t<month>\t<day>\t<hour>
    \t<i>\t<j>\t<lon>\t<lat>\t<rad>\t<r2d>\t<wspd>\t<year>\t<month>\t<day>\t<hour>
    ...(n_points строк на трек)...
Колонки rad/r2d/wspd — см. 3_create_csv_tracks_from_StitchNodes.py /
6_grid_runner_TE_ERA5.py (STAGE_D_VARIABLE_NAMES = ["rad", "r2d", "wspd"]):
    rad  — rad_eff кластера DBSCAN, в ячейках сетки (см. 1_create_Nodes_from_DBSCAN.py,
           переводится в км так же, как в plot_R2D_maps.py: rad * cos(lat) * grid_res * 111)
    r2d  — критерий R2D (Рортекс-2D), уже в физических единицах, с^-1
    wspd — модуль скорости ветра в точке узла, м/с

Что считается
-------------
1. Плотность треков на карте (--cell-size°, по умолчанию 2°):
   - "точечная" плотность: число прохождений точек трека через ячейку
     (без дедупликации — трек, задержавшийся в ячейке, даёт много точек);
   - времявзвешенная плотность: то же, но каждая точка берётся с весом —
     число часов до следующей точки этого же трека (последняя точка трека —
     медианный шаг по треку). Это нужно, чтобы честно сравнивать треки с
     РАЗНЫМ шагом трекинга по времени (например local 3h vs global 9h из
     задачи) — иначе точечная плотность 3h-файла будет систематически втрое
     выше только из-за более частой записи, а не из-за реальной статистики.
2. Число уникальных треков через ячейку: каждый трек учитывается в ячейке
   не более одного раза, даже если прошёл через неё несколько раз подряд.
3. Распределения параметров вдоль трека (по всем точкам всех треков):
   скорость ветра (wspd, м/с), R2D/рортекс (r2d, с^-1), радиус (rad, км),
   и дополнительно — скорость смещения самого трека (translation speed,
   км/ч, из haversine-расстояния между соседними точками трека и dt) —
   выключается флагом --no-translation-speed, если нужны только 3
   запрошенных параметра.

Можно передать НЕСКОЛЬКО файлов/наборов с подписями через --input LABEL=PATH
(PATH может быть glob-шаблоном на несколько лет, например ".../*_tracks_*.txt")
— тогда карты считаются для каждого набора отдельно и рисуются рядом для
сравнения, а распределения накладываются друг на друга на одних осях.

Пример запуска — ровно случай из задачи (local 3h vs global 9h, 2010)
----------------------------------------------------------------------
python plot_track_density_and_distributions.py \\
    --input local="/data/TempestExtremes/ERA5/R2D_ERA5_NA_for_TC_850hPa_sigma_2/01-04-49_local/Tracks_R2D_txt_files_range_1_5_24h_3h/ERA5_TC_tracks_2010.txt" \\
    --input global="/data/TempestExtremes/ERA5/R2D_ERA5_NA_for_TC_850hPa_sigma_2/02-04-49_global/Tracks_R2D_txt_files_range_1_5_24h_9h/ERA5_TC_tracks_2010.txt" \\
    --out-dir ./pics/track_density_2010 \\
    --cell-size 2.0

Зависимости: numpy, pandas, matplotlib, cartopy.
"""

from __future__ import annotations

import argparse
import glob
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except ImportError:
    HAVE_CARTOPY = False

EARTH_R_KM = 6371.0088

# Домен по умолчанию — тот же North Atlantic bbox, что уже используется в
# tracks_comparison_lib.py (ax.set_extent([-110, 15, 0, 73], ...))
DEFAULT_EXTENT = (-110.0, 15.0, 0.0, 73.0)

COLUMN_NAMES = ["_blank", "i", "j", "lon", "lat", "rad", "r2d", "wspd",
                "year", "month", "day", "hour"]

# Разные подписи для до 6 наборов данных (local/global и т.п.)
COLOR_CYCLE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


# ---------------------------------------------------------------------------
# Парсинг StitchNodes txt
# ---------------------------------------------------------------------------

def _read_one_file(path: Path, track_id_offset: int) -> tuple[pd.DataFrame, int]:
    """Читает один годовой txt StitchNodes. Возвращает (df, n_tracks_в_файле).

    Быстрый двухпроходный разбор: сначала находим строки-заголовки треков
    ('start\\t...') обычным Python-циклом (дёшево, это только startswith),
    затем все строки с данными парсятся ОДНИМ вызовом pandas.read_csv
    (C-парсер) — на файле ~836k строк это в ~10-20 раз быстрее, чем
    построчный float() в чистом Python.
    """
    with open(path, "r") as f:
        raw_lines = f.readlines()

    lines = [ln.rstrip("\n") for ln in raw_lines if ln.strip()]
    if not lines:
        return pd.DataFrame(), 0

    is_start = np.fromiter((ln.startswith("start") for ln in lines),
                            dtype=bool, count=len(lines))
    # track_id локальный для файла: 0, 1, 2, ... по порядку встречи 'start'
    local_track_id = np.cumsum(is_start) - 1

    data_lines = [ln for ln, s in zip(lines, is_start) if not s]
    data_track_id = local_track_id[~is_start]

    if not data_lines:
        return pd.DataFrame(), int(is_start.sum())

    buf = io.StringIO("\n".join(data_lines))
    df = pd.read_csv(buf, sep="\t", header=None, names=COLUMN_NAMES,
                      dtype={"i": np.int32, "j": np.int32,
                             "lon": np.float64, "lat": np.float64,
                             "rad": np.float64, "r2d": np.float64,
                             "wspd": np.float64,
                             "year": np.int32, "month": np.int32,
                             "day": np.int32, "hour": np.int32})
    df = df.drop(columns="_blank")
    df["track_id"] = data_track_id.astype(np.int64) + track_id_offset

    n_tracks = int(is_start.sum())
    return df, n_tracks


def load_track_set(path_or_glob: str) -> pd.DataFrame:
    """Читает один файл или несколько (glob-шаблон, например по годам) и
    склеивает их в один DataFrame со сквозной нумерацией track_id (чтобы
    треки из разных файлов/лет никогда не путались)."""
    paths = sorted(glob.glob(path_or_glob)) if any(ch in path_or_glob for ch in "*?[") \
        else [path_or_glob]
    paths = [Path(p) for p in paths]
    if not paths:
        raise FileNotFoundError(f"Не найдено ни одного файла по шаблону: {path_or_glob}")

    frames = []
    offset = 0
    for p in paths:
        if not p.exists():
            print(f"  [!] файл не найден, пропуск: {p}", file=sys.stderr)
            continue
        df, n_tracks = _read_one_file(p, offset)
        if len(df) == 0:
            print(f"  [!] пустой файл (0 точек): {p}", file=sys.stderr)
            continue
        df["source_file"] = p.name
        frames.append(df)
        offset += n_tracks
        print(f"  {p.name}: {n_tracks} треков, {len(df)} точек")

    if not frames:
        raise ValueError(f"Ни один файл не дал данных: {path_or_glob}")

    out = pd.concat(frames, ignore_index=True)
    out["time"] = pd.to_datetime(dict(year=out.year, month=out.month,
                                       day=out.day, hour=out.hour))
    # долгота на всякий случай в [-180, 180] (если сервер отдаст 0..360)
    out["lon"] = np.where(out["lon"] > 180.0, out["lon"] - 360.0, out["lon"])
    out = out.sort_values(["track_id", "time"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Производные величины вдоль трека
# ---------------------------------------------------------------------------

def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def add_derived_columns(df: pd.DataFrame, grid_res_deg: float) -> pd.DataFrame:
    """Добавляет: rad_km (радиус в км), dt_h_fwd (часы до следующей точки
    трека — вес для времявзвешенной плотности), speed_kmh (скорость
    смещения трека)."""
    df = df.copy()

    # rad_eff (ячейки сетки) -> км: та же формула, что в plot_R2D_maps.py
    # (radius_val * cos(lat) * grid_res * 111 — км по долготе на широте lat).
    df["rad_km"] = df["rad"] * np.cos(np.radians(df["lat"])) * grid_res_deg * 111.0

    g = df.groupby("track_id", sort=False)
    next_time = g["time"].shift(-1)
    next_lon = g["lon"].shift(-1)
    next_lat = g["lat"].shift(-1)

    dt_h_fwd = (next_time - df["time"]).dt.total_seconds() / 3600.0
    dist_km = haversine_km(df["lon"].values, df["lat"].values,
                            next_lon.values, next_lat.values)

    # последняя точка каждого трека: нет "следующего" шага — подставляем
    # медианный шаг ПО ЭТОМУ ЖЕ треку (а если трек из одной точки — медиану
    # по всему набору), чтобы не терять точку в весе плотности.
    df["dt_h_fwd"] = dt_h_fwd.values
    med_by_track = df.groupby("track_id")["dt_h_fwd"].transform("median")
    global_med = float(np.nanmedian(df["dt_h_fwd"])) if np.isfinite(df["dt_h_fwd"]).any() else 1.0
    df["dt_h_fwd"] = df["dt_h_fwd"].fillna(med_by_track).fillna(global_med)
    df.loc[df["dt_h_fwd"] <= 0, "dt_h_fwd"] = global_med

    with np.errstate(invalid="ignore", divide="ignore"):
        speed_kmh = np.where(dt_h_fwd.values > 0, dist_km / dt_h_fwd.values, np.nan)
    df["speed_kmh"] = speed_kmh  # NaN на последней точке каждого трека — это нормально

    return df


# ---------------------------------------------------------------------------
# Сетка и плотности
# ---------------------------------------------------------------------------

@dataclass
class Grid:
    lon_edges: np.ndarray
    lat_edges: np.ndarray

    @property
    def lon_centers(self):
        return 0.5 * (self.lon_edges[:-1] + self.lon_edges[1:])

    @property
    def lat_centers(self):
        return 0.5 * (self.lat_edges[:-1] + self.lat_edges[1:])


def make_grid(extent: tuple[float, float, float, float], cell_size: float) -> Grid:
    lon0, lon1, lat0, lat1 = extent
    lon_edges = np.arange(lon0, lon1 + cell_size * 0.5, cell_size)
    lat_edges = np.arange(lat0, lat1 + cell_size * 0.5, cell_size)
    return Grid(lon_edges=lon_edges, lat_edges=lat_edges)


def point_density(df: pd.DataFrame, grid: Grid, weight_col: str | None = None) -> np.ndarray:
    """Число прохождений точек трека через ячейку (или сумма весов, если
    weight_col задан). Возвращает массив (n_lon, n_lat), как histogram2d."""
    weights = df[weight_col].values if weight_col else None
    h, _, _ = np.histogram2d(df["lon"].values, df["lat"].values,
                              bins=[grid.lon_edges, grid.lat_edges],
                              weights=weights)
    return h


def track_count(df: pd.DataFrame, grid: Grid) -> np.ndarray:
    """Число УНИКАЛЬНЫХ треков, прошедших через ячейку (каждый трек — не
    более одного раза на ячейку, даже если прошёл через неё многократно)."""
    lon_idx = np.digitize(df["lon"].values, grid.lon_edges) - 1
    lat_idx = np.digitize(df["lat"].values, grid.lat_edges) - 1
    n_lon = len(grid.lon_edges) - 1
    n_lat = len(grid.lat_edges) - 1
    valid = (lon_idx >= 0) & (lon_idx < n_lon) & (lat_idx >= 0) & (lat_idx < n_lat)

    tmp = pd.DataFrame({
        "track_id": df["track_id"].values[valid],
        "lon_idx": lon_idx[valid],
        "lat_idx": lat_idx[valid],
    }).drop_duplicates()

    counts = tmp.groupby(["lon_idx", "lat_idx"]).size()
    grid_arr = np.zeros((n_lon, n_lat))
    for (li, lj), c in counts.items():
        grid_arr[li, lj] = c
    return grid_arr


# ---------------------------------------------------------------------------
# Отрисовка
# ---------------------------------------------------------------------------

def new_map_axes(fig, subplot_spec, extent):
    if HAVE_CARTOPY:
        ax = fig.add_subplot(*subplot_spec, projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="#eef6fb", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5, zorder=3)
        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.4,
                           color="gray", alpha=0.6)
        gl.top_labels = False
        gl.right_labels = False
        return ax
    else:
        # запасной вариант без cartopy (для тестов/окружений без geos/proj) —
        # тот же bbox, только без береговой линии.
        ax = fig.add_subplot(*subplot_spec)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.grid(linestyle="--", linewidth=0.4, alpha=0.6)
        return ax


def pcolor_kwargs():
    return dict(transform=ccrs.PlateCarree()) if HAVE_CARTOPY else {}


def plot_density_map(ax, grid: Grid, values: np.ndarray, title: str,
                      cbar_label: str, cmap: str = "YlOrRd", vmax=None):
    masked = np.ma.masked_where(values <= 0, values)
    mesh = ax.pcolormesh(grid.lon_edges, grid.lat_edges, masked.T,
                          cmap=cmap, vmin=0, vmax=vmax, alpha=0.9,
                          shading="flat", **pcolor_kwargs())
    ax.set_title(title, fontsize=10)
    plt.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.75, pad=0.02,
                 label=cbar_label)
    return mesh


def plot_all_maps(datasets: dict[str, dict], grid: Grid, extent, out_dir: Path,
                   per_year: bool):
    """Для каждого набора данных — 3 карты (точечная плотность,
    времявзвешенная плотность, число уникальных треков), плюс общая фигура
    для сравнения всех наборов бок о бок по каждой из трёх метрик."""

    metrics = [
        ("density_points", "число прохождений точек трека / ячейку" +
         (" / год" if per_year else ""), "YlOrRd"),
        ("density_hours", "занятость ячейки, часы" + (" / год" if per_year else ""), "OrRd"),
        ("track_count", "число уникальных треков / ячейку" + (" / год" if per_year else ""), "viridis"),
    ]

    # общий vmax по каждой метрике — чтобы сравнение local/global было честным
    vmax_by_metric = {}
    for key, _, _ in metrics:
        all_vals = [d[key] for d in datasets.values()]
        vmax_by_metric[key] = max(float(np.max(v)) for v in all_vals) if all_vals else None

    # --- отдельная фигура на набор (3 панели) ---
    for label, d in datasets.items():
        fig = plt.figure(figsize=(18, 6))
        for k, (key, cbar_label, cmap) in enumerate(metrics):
            ax = new_map_axes(fig, (1, 3, k + 1), extent)
            plot_density_map(ax, grid, d[key],
                              title=f"{label}: {cbar_label.split(',')[0]}",
                              cbar_label=cbar_label, cmap=cmap,
                              vmax=vmax_by_metric[key])
        fig.suptitle(f"Треки — {label}  (n_tracks={d['n_tracks']}, "
                     f"n_points={d['n_points']}, n_years={d['n_years']})")
        fig.tight_layout()
        out_path = out_dir / f"track_maps_{label}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  сохранено: {out_path}")

    # --- сравнение всех наборов бок о бок, по одной метрике на строку ---
    if len(datasets) > 1:
        n_sets = len(datasets)
        fig = plt.figure(figsize=(6 * n_sets, 6 * len(metrics)))
        for row, (key, cbar_label, cmap) in enumerate(metrics):
            for col, (label, d) in enumerate(datasets.items()):
                idx = row * n_sets + col + 1
                ax = new_map_axes(fig, (len(metrics), n_sets, idx), extent)
                plot_density_map(ax, grid, d[key], title=f"{label}",
                                  cbar_label=cbar_label, cmap=cmap,
                                  vmax=vmax_by_metric[key])
        fig.tight_layout()
        out_path = out_dir / "track_maps_comparison.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  сохранено: {out_path}")

        # разностная карта (только если ровно 2 набора — иначе неоднозначно,
        # какой из какого вычитать)
        if n_sets == 2:
            (label_a, d_a), (label_b, d_b) = list(datasets.items())
            fig = plt.figure(figsize=(6 * len(metrics), 6))
            for k, (key, cbar_label, _) in enumerate(metrics):
                ax = new_map_axes(fig, (1, len(metrics), k + 1), extent)
                diff = d_b[key] - d_a[key]
                vlim = np.max(np.abs(diff)) if np.any(diff) else 1.0
                masked = np.ma.masked_where(diff == 0, diff)
                mesh = ax.pcolormesh(grid.lon_edges, grid.lat_edges, masked.T,
                                      cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                                      alpha=0.9, shading="flat", **pcolor_kwargs())
                ax.set_title(f"{label_b} − {label_a}: {cbar_label.split(',')[0]}", fontsize=10)
                plt.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.75, pad=0.02,
                             label=cbar_label)
            fig.tight_layout()
            out_path = out_dir / f"track_maps_diff_{label_b}_minus_{label_a}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  сохранено: {out_path}")


def plot_distributions(dfs: dict[str, pd.DataFrame], out_dir: Path,
                        include_translation_speed: bool = True,
                        out_name: str = "track_param_distributions.png"):
    """Гистограммы (плотность вероятности) параметров вдоль трека,
    наборы данных наложены друг на друга для сравнения."""

    panels = [
        ("wspd", "Скорость ветра в узле трека, м/с", (0, None)),
        ("r2d", "R2D (рортекс), с$^{-1}$", (0, None)),
        ("rad_km", "Радиус кластера, км", (0, None)),
    ]
    if include_translation_speed:
        panels.append(("speed_kmh", "Скорость смещения трека, км/ч", (0, None)))

    n = len(panels)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (col, xlabel, (xmin, xmax)) in zip(axes, panels):
        for i, (label, df) in enumerate(dfs.items()):
            vals = df[col].dropna().values
            vals = vals[np.isfinite(vals)]
            if xmax is not None:
                vals = vals[(vals >= xmin) & (vals <= xmax)]
            if len(vals) == 0:
                continue
            color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
            ax.hist(vals, bins=60, density=True, histtype="step",
                    linewidth=1.8, color=color, label=f"{label} (n={len(vals)})")
            ax.axvline(np.median(vals), color=color, linestyle=":", linewidth=1.2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("плотность вероятности")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle("Распределения параметров вдоль трека")
    fig.tight_layout()
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  сохранено: {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_input_arg(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--input нужно в формате LABEL=PATH (получено: {spec!r})")
    label, path = spec.split("=", 1)
    return label.strip(), path.strip()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Плотность/число треков на карте + распределения параметров вдоль трека "
                     "по txt-файлам StitchNodes (TempestExtremes).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", action="append", required=True, type=parse_input_arg,
                    metavar="LABEL=PATH",
                    help="Набор треков с подписью, можно указывать несколько раз. "
                         "PATH может быть glob-шаблоном на несколько лет.")
    p.add_argument("--out-dir", type=Path, default=Path("./track_density_out"),
                    help="Куда сохранять PNG (по умолчанию ./track_density_out)")
    p.add_argument("--cell-size", type=float, default=2.0,
                    help="Размер ячейки сетки в градусах (по умолчанию 2.0)")
    p.add_argument("--extent", type=float, nargs=4, metavar=("LON0", "LON1", "LAT0", "LAT1"),
                    default=list(DEFAULT_EXTENT),
                    help=f"Границы карты lon0 lon1 lat0 lat1 (по умолчанию {DEFAULT_EXTENT}, "
                         "тот же North Atlantic bbox, что в tracks_comparison_lib.py)")
    p.add_argument("--grid-res-deg", type=float, default=0.25,
                    help="Разрешение исходной сетки ERA5 в градусах — нужно, чтобы перевести "
                         "'rad' (в ячейках) в километры (по умолчанию 0.25, как ERA5)")
    p.add_argument("--no-per-year", dest="per_year", action="store_false",
                    help="Не нормировать карты плотности на число лет в данных "
                         "(по умолчанию нормируются — 'в среднем за год')")
    p.add_argument("--no-translation-speed", dest="translation_speed", action="store_false",
                    help="Не считать/не рисовать скорость смещения трека (км/ч), "
                         "оставить только скорость ветра, рортекс и радиус")
    p.add_argument("--save-parsed", action="store_true",
                    help="Дополнительно сохранить распарсенные точки треков в parquet "
                         "рядом с out-dir (для повторного использования без пере-парсинга)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not HAVE_CARTOPY:
        print("[!] cartopy не найден в окружении — карты будут нарисованы БЕЗ береговой "
              "линии (просто прямоугольные оси). Установите cartopy для полноценных карт.",
              file=sys.stderr)

    extent = tuple(args.extent)
    grid = make_grid(extent, args.cell_size)

    dfs: dict[str, pd.DataFrame] = {}
    datasets: dict[str, dict] = {}

    for label, path_spec in args.input:
        print(f"[{label}] чтение: {path_spec}")
        df = load_track_set(path_spec)
        df = add_derived_columns(df, grid_res_deg=args.grid_res_deg)
        dfs[label] = df

        n_years = int(df["year"].nunique())
        norm = float(n_years) if (args.per_year and n_years > 0) else 1.0

        dens_pts = point_density(df, grid) / norm
        dens_hrs = point_density(df, grid, weight_col="dt_h_fwd") / norm
        cnt = track_count(df, grid) / norm

        datasets[label] = dict(
            density_points=dens_pts,
            density_hours=dens_hrs,
            track_count=cnt,
            n_tracks=int(df["track_id"].nunique()),
            n_points=int(len(df)),
            n_years=n_years,
        )

        if args.save_parsed:
            out_parquet = args.out_dir / f"parsed_points_{label}.parquet"
            df.to_parquet(out_parquet)
            print(f"  сохранены разобранные точки: {out_parquet}")

    print("Отрисовка карт плотности/числа треков...")
    plot_all_maps(datasets, grid, extent, args.out_dir, per_year=args.per_year)

    print("Отрисовка распределений параметров вдоль трека...")
    plot_distributions(dfs, args.out_dir, include_translation_speed=args.translation_speed)

    print("Готово.")


if __name__ == "__main__":
    main()

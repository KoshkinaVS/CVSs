#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Упрощённая версия plot_track_density_and_distributions.py: ОДИН вариант
трекинга на входе, ровно ДВЕ картинки на выходе — без времявзвешенной
плотности (она нужна только для честного сравнения нескольких файлов с
разным шагом трекинга по времени; для одного файла это лишняя картинка).

Использует тот же парсер/сетку/формулы, что и
plot_track_density_and_distributions.py (импортируется как модуль — держите
оба файла в одной папке).

Что рисуется
------------
1. track_maps_<label>.png — 2 карты рядом (cartopy, PlateCarree, LAND/OCEAN/
   COASTLINE/BORDERS, тот же North Atlantic bbox, что в tracks_comparison_lib.py):
   - точечная плотность (число прохождений точек трека через ячейку);
   - число уникальных треков через ячейку.
2. track_param_distributions_<label>.png — гистограммы параметров вдоль
   трека: скорость ветра (wspd, м/с), R2D/рортекс (r2d, с^-1), радиус
   кластера (rad -> км), и опционально скорость смещения трека (км/ч,
   выключается --no-translation-speed).

Пример запуска
--------------
python plot_track_density_single.py \\
    "/data/TempestExtremes/ERA5/R2D_ERA5_NA_for_TC_850hPa_sigma_2/01-04-49_local/Tracks_R2D_txt_files_range_1_5_24h_3h/ERA5_TC_tracks_2010.txt" \\
    --label local_3h --out-dir ./pics/track_density_local_2010 --cell-size 2.0

Зависимости: numpy, pandas, matplotlib, cartopy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_track_density_and_distributions import (
    DEFAULT_EXTENT,
    HAVE_CARTOPY,
    Grid,
    add_derived_columns,
    load_track_set,
    make_grid,
    new_map_axes,
    plot_density_map,
    plot_distributions,
    point_density,
    track_count,
)


def plot_two_maps(grid: Grid, dens_pts: np.ndarray, cnt: np.ndarray, extent,
                   label: str, per_year: bool, stats: dict, out_dir: Path) -> Path:
    suffix = " / год" if per_year else ""
    fig = plt.figure(figsize=(13, 6))

    ax1 = new_map_axes(fig, (1, 2, 1), extent)
    plot_density_map(ax1, grid, dens_pts,
                      title="Плотность точек трека",
                      cbar_label=f"число прохождений точек / ячейку{suffix}",
                      cmap="YlOrRd")

    ax2 = new_map_axes(fig, (1, 2, 2), extent)
    plot_density_map(ax2, grid, cnt,
                      title="Число уникальных треков",
                      cbar_label=f"число треков / ячейку{suffix}",
                      cmap="viridis")

    fig.suptitle(f"{label}  (n_tracks={stats['n_tracks']}, "
                 f"n_points={stats['n_points']}, n_years={stats['n_years']})")
    fig.tight_layout()
    out_path = out_dir / f"track_maps_{label}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Плотность точек трека + число уникальных треков на карте, и "
                     "распределения параметров вдоль трека — для ОДНОГО варианта "
                     "трекинга (без времявзвешенной плотности).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", type=str,
                    help="Путь к txt-файлу StitchNodes (или glob-шаблон на несколько лет, "
                         "например '.../ERA5_TC_tracks_*.txt')")
    p.add_argument("--label", type=str, default=None,
                    help="Подпись для файлов/заголовков (по умолчанию — имя входного файла)")
    p.add_argument("--out-dir", type=Path, default=Path("./track_density_out"),
                    help="Куда сохранять PNG (по умолчанию ./track_density_out)")
    p.add_argument("--cell-size", type=float, default=2.0,
                    help="Размер ячейки сетки в градусах (по умолчанию 2.0)")
    p.add_argument("--extent", type=float, nargs=4, metavar=("LON0", "LON1", "LAT0", "LAT1"),
                    default=list(DEFAULT_EXTENT),
                    help=f"Границы карты lon0 lon1 lat0 lat1 (по умолчанию {DEFAULT_EXTENT})")
    p.add_argument("--grid-res-deg", type=float, default=0.25,
                    help="Разрешение исходной сетки ERA5 в градусах — для перевода "
                         "'rad' (в ячейках) в километры (по умолчанию 0.25, как ERA5)")
    p.add_argument("--no-per-year", dest="per_year", action="store_false",
                    help="Не нормировать карты плотности на число лет в данных "
                         "(по умолчанию нормируются — 'в среднем за год')")
    p.add_argument("--no-translation-speed", dest="translation_speed", action="store_false",
                    help="Не считать/не рисовать скорость смещения трека (км/ч), "
                         "оставить только скорость ветра, рортекс и радиус")
    p.add_argument("--save-parsed", action="store_true",
                    help="Дополнительно сохранить разобранные точки треков в parquet")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not HAVE_CARTOPY:
        print("[!] cartopy не найден в окружении — карты будут нарисованы БЕЗ береговой "
              "линии (просто прямоугольные оси). Установите cartopy для полноценных карт.",
              file=sys.stderr)

    label = args.label or Path(args.input.split("*")[0]).stem or "tracks"
    extent = tuple(args.extent)
    grid = make_grid(extent, args.cell_size)

    print(f"[{label}] чтение: {args.input}")
    df = load_track_set(args.input)
    df = add_derived_columns(df, grid_res_deg=args.grid_res_deg)

    n_years = int(df["year"].nunique())
    norm = float(n_years) if (args.per_year and n_years > 0) else 1.0

    dens_pts = point_density(df, grid) / norm
    cnt = track_count(df, grid) / norm
    stats = dict(n_tracks=int(df["track_id"].nunique()), n_points=int(len(df)), n_years=n_years)

    if args.save_parsed:
        out_parquet = args.out_dir / f"parsed_points_{label}.parquet"
        df.to_parquet(out_parquet)
        print(f"  сохранены разобранные точки: {out_parquet}")

    print("Отрисовка карт плотности/числа треков...")
    out_path = plot_two_maps(grid, dens_pts, cnt, extent, label, args.per_year, stats, args.out_dir)
    print(f"  сохранено: {out_path}")

    print("Отрисовка распределений параметров вдоль трека...")
    plot_distributions({label: df}, args.out_dir,
                        include_translation_speed=args.translation_speed,
                        out_name=f"track_param_distributions_{label}.png")

    print("Готово.")


if __name__ == "__main__":
    main()

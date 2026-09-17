"""
Общая библиотека для сравнения треков КВС этого пайплайна
(DBSCAN_tracking_ERA5_latlon.py, эта копия — CVS_latlon_tracking/) с
внешними референсными базами треков тропических циклонов — IBTrACS и
SyCLoPS (TC) — через huracanpy.assess.match/pod (Probability of Detection).

По аналогии с
../CVS_alt_tracking/TempestExtremes/tracks_comparison_lib.py: та же идея
разделения — то, что специфично для ИСТОЧНИКА (как читать файлы, откуда
брать track_id/lon/lat/time), в load_*_tracks() ниже; всё остальное
(matching/POD/статистика/графики) работает только с track_id/lon/lat/time и
не знает, откуда взялись данные.

Что взято оттуда (секции 2, 3, 5, 6 оригинала) практически дословно, с одной
системной заменой имени "TE_NA" -> OWN_NAME = "OWN" (это наши треки, не
TempestExtremes) — dataframe_to_huracanpy/build_csv_for_huracanpy, SyCLoPS-
загрузчик (normalize_lon_360_to_180/find_syclops_files/load_syclops_tracks),
matching/POD/трек-статистика (calculate_matches/calculate_pod/
calculate_track_statistics/summarize_track_statistics), графики
(plot_reference_track/plot_all_reference_tracks/plot_pod_bar/
plot_pod_vs_overlap_scatter).

Что НЕ переносилось:
  - секция 1 оригинала (grid_run_log.csv/combo — сетка StitchNodes-
    гиперпараметров TempestExtremes) — у этого пайплайна нет такой сетки,
    "конфигурация" — это просто (tracking_type, CVS_speed) из
    DBSCAN_tracking_ERA5_latlon.py; вместо неё здесь load_own_tracks() ниже
    и поиск конфигураций в compare_with_reference.py (find_configs);
  - секция 4 оригинала (EddyClicker, ручная разметка) — не относится к
    сравнению с треками ТЦ.

ВАЖНО (честно, как и в оригинале): в этой песочнице нет доступа ни к
huracanpy/cartopy, ни к серверу. load_own_tracks() (своя часть, без
huracanpy) и normalize_lon_360_to_180/find_syclops_files (парсинг SyCLoPS)
прогнаны на синтетических данных — см. CHANGELOG.md. Всё, что реально
вызывает huracanpy/cartopy (calculate_matches/calculate_pod, сам
huracanpy.load(source='csv'/'ibtracs'), графики) — перенесено из уже
работавшего tracks_comparison_lib.py практически без изменений по сути,
но здесь не перезапускалось. Прогоните --dry-run, затем один короткий
период, перед боевым прогоном на всю историю (см. README.md).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

OWN_NAME = "OWN"                    # имя наших треков в huracanpy.assess.match(names=[...])
OWN_ID_COLUMN = f"id_{OWN_NAME}"    # так huracanpy называет колонку id в matches (id_<name>)

SYCLOPS_TYPES = ["TC", "SS", "PL"]

# huracanpy.assess.match: макс. расстояние между точками (км), среднее
# расстояние по треку (км), минимум совпадающих точек — те же значения,
# что уже использовались для TE (tracks_comparison_lib.py), не CLI-флаги
# и там тоже — правьте здесь, если нужно другое.
MAX_DIST = 200
MEAN_DIST = 120
MIN_OVERLAP = 3


# ============================================================================
# 1. Наши CSV-треки (save_track_csv, step_of_tracking.py) -> huracanpy
# ============================================================================

def load_own_tracks(months, path_tracks_dir):
    """
    Треки этого пайплайна (CSV из save_track_csv) за месяцы `months`
    (список pandas.Period, freq='M') из папки ОДНОЙ конфигурации
    (tracking_type, CVS_speed) — path_tracks_dir, как в
    plot_tracking_results.py (".../tracks_C"). Каждый CSV-файл — один трек,
    track_id присваивается по порядку файлов. Возвращает huracanpy Dataset
    (как load_syclops_tracks) либо None, если треков за период нет.
    """
    from plot_tracking_results import load_period_tracks

    frames = load_period_tracks(months, path_tracks_dir)
    if not frames:
        return None

    for track_id, df in enumerate(frames):
        df['track_id'] = track_id
    combined = pd.concat(frames, ignore_index=True)

    return dataframe_to_huracanpy(
        combined, track_id_col='track_id', lon_col='lon', lat_col='lat', time_col='datetime',
        extra_cols=['rad', 'crit'],
    )


# ============================================================================
# 2. Произвольный табличный источник -> huracanpy (см. докстринг выше;
#    huracanpy.load(path, source="csv") — официальный маршрут построения
#    xarray Dataset, https://huracanpy.readthedocs.io/en/stable/examples/load_csv.html)
# ============================================================================

def build_csv_for_huracanpy(df: pd.DataFrame, track_id_col: str, lon_col: str, lat_col: str, time_col: str,
                             extra_cols: list[str] | None = None) -> pd.DataFrame:
    """Чистая (без huracanpy/IO) часть dataframe_to_huracanpy — DataFrame в
    точности того вида, который уходит во временный CSV для huracanpy.load."""
    cols = [track_id_col, time_col, lon_col, lat_col] + list(extra_cols or [])
    work = df[cols].copy()
    work = work.rename(columns={track_id_col: "track_id", lon_col: "lon", lat_col: "lat"})
    t = pd.to_datetime(work[time_col])
    work["year"] = t.dt.year
    work["month"] = t.dt.month
    work["day"] = t.dt.day
    work["hour"] = t.dt.hour
    return work.drop(columns=[time_col])


def dataframe_to_huracanpy(df: pd.DataFrame, track_id_col: str, lon_col: str, lat_col: str, time_col: str,
                            extra_cols: list[str] | None = None):
    import huracanpy
    work = build_csv_for_huracanpy(df, track_id_col, lon_col, lat_col, time_col, extra_cols)
    fd, tmp_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        work.to_csv(tmp_path, index=False)
        return huracanpy.load(tmp_path, source="csv")
    finally:
        os.remove(tmp_path)


# ============================================================================
# 3. SyCLoPS — читает RAW CSV из <syclops-root>/SyCLoPS_{type}_{region}_*_{year}_tracks.csv
# ============================================================================

def normalize_lon_360_to_180(lon) -> float:
    """SyCLoPS хранит LON в [0, 360) — переводим в [-180, 180), как у ERA5."""
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def find_syclops_files(syclops_type: str, year: int, region: str, root) -> list[Path]:
    pattern = f"SyCLoPS_{syclops_type}_{region}_*_{year}_tracks.csv"
    return sorted(Path(root).glob(pattern))


def load_syclops_tracks(syclops_type: str, years: list[int], region: str, root,
                         on_multiple: str = "newest"):
    """on_multiple: 'newest' (по умолчанию) — если для (type,region,year)
    нашлось несколько файлов, берёт самый свежий по mtime; 'all' объединяет
    все найденные (используйте, только если точно знаете, что дублей нет)."""
    frames = []
    for year in years:
        files = find_syclops_files(syclops_type, year, region, root)
        if not files:
            print(f"SyCLoPS {syclops_type}/{region}: {year} не найден в {root}, пропуск")
            continue
        if len(files) > 1:
            print(f"SyCLoPS {syclops_type}/{region}/{year}: найдено {len(files)} файлов: "
                  f"{[f.name for f in files]}")
            if on_multiple == "newest":
                files = [max(files, key=lambda f: f.stat().st_mtime)]
                print(f"  -> беру самый свежий: {files[0].name} (on_multiple='newest')")
        for f in files:
            df = pd.read_csv(f, parse_dates=["ISOTIME"])
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"Не найдено ни одного файла SyCLoPS {syclops_type}/{region} за {years} в {root}")

    combined = pd.concat(frames, ignore_index=True)
    combined["lon_180"] = normalize_lon_360_to_180(combined["LON"])
    combined = combined.rename(columns={"TID": "track_id_syclops"})

    return dataframe_to_huracanpy(
        combined, track_id_col="track_id_syclops", lon_col="lon_180", lat_col="LAT", time_col="ISOTIME",
        extra_cols=["MSLP", "WS"],
    )


# ============================================================================
# 4. Generic matching / POD / трек-статистика (не знает про источник)
# ============================================================================

def calculate_matches(reference_tracks, own_tracks, reference_name):
    import huracanpy
    return huracanpy.assess.match(
        [reference_tracks, own_tracks], names=[reference_name, OWN_NAME],
        max_dist=MAX_DIST, mean_dist=MEAN_DIST, min_overlap=MIN_OVERLAP, tracks1_is_ref=True,
    )


def calculate_pod(matches, reference_tracks, reference_name):
    import huracanpy
    return float(huracanpy.assess.pod(matches, ref=reference_tracks, ref_name=reference_name))


def select_best_matches(matches, reference_id_column):
    if len(matches) == 0:
        return matches.copy()
    return (
        matches.sort_values([reference_id_column, "temp", "dist"], ascending=[True, False, True])
        .drop_duplicates(subset=reference_id_column, keep="first")
        .reset_index(drop=True)
    )


def calculate_temporal_overlap(reference_track, own_track):
    ref = pd.to_datetime(reference_track.time.values)
    own = pd.to_datetime(own_track.time.values)
    if len(ref) == 0 or len(own) == 0:
        return {"overlap_start": pd.NaT, "overlap_end": pd.NaT, "overlap_duration_hours": 0, "overlap_duration_fraction": 0}
    ref_start, ref_end = ref.min(), ref.max()
    own_start, own_end = own.min(), own.max()
    duration = (ref_end - ref_start).total_seconds() / 3600
    start, end = max(ref_start, own_start), min(ref_end, own_end)
    overlap = 0 if start > end else (end - start).total_seconds() / 3600
    fraction = overlap / duration if duration > 0 else 0
    return {"overlap_start": start, "overlap_end": end, "overlap_duration_hours": overlap, "overlap_duration_fraction": fraction}


def calculate_track_statistics(reference_tracks, own_tracks, matches, reference_id_column,
                                own_id_column: str = OWN_ID_COLUMN):
    reference_ids = np.unique(reference_tracks.track_id.values)
    best = select_best_matches(matches, reference_id_column)
    best_dict = {r[reference_id_column]: r for _, r in best.iterrows()}
    rows = []

    for ref_id in reference_ids:
        ref_track = reference_tracks.where(reference_tracks.track_id == ref_id, drop=True).sortby("time")
        times = pd.to_datetime(ref_track.time.values)
        n = len(times)
        duration = ((times.max() - times.min()).total_seconds() / 3600) if n > 1 else 0

        if ref_id not in best_dict:
            rows.append({
                "reference_ID": ref_id, "detected": False, f"{OWN_NAME}_ID_best": np.nan,
                "reference_n_points": n, "reference_duration_hours": duration,
                "matched_n_points": 0, "overlap_fraction": 0, "overlap_percent": 0,
                "mean_distance_km": np.nan, "overlap_start": pd.NaT, "overlap_end": pd.NaT,
                "overlap_duration_hours": 0, "overlap_duration_fraction": 0, "overlap_duration_percent": 0,
            })
            continue

        row = best_dict[ref_id]
        own_id = row[own_id_column]
        matched = row["temp"]
        dist = row["dist"]
        frac = matched / n if n else np.nan
        own_track = own_tracks.where(own_tracks.track_id == own_id, drop=True).sortby("time")
        temporal = calculate_temporal_overlap(ref_track, own_track)

        rows.append({
            "reference_ID": ref_id, "detected": True, f"{OWN_NAME}_ID_best": own_id,
            "reference_n_points": n, "reference_duration_hours": duration,
            "matched_n_points": matched, "overlap_fraction": frac, "overlap_percent": 100 * frac,
            "mean_distance_km": dist, "overlap_start": temporal["overlap_start"], "overlap_end": temporal["overlap_end"],
            "overlap_duration_hours": temporal["overlap_duration_hours"],
            "overlap_duration_fraction": temporal["overlap_duration_fraction"],
            "overlap_duration_percent": 100 * temporal["overlap_duration_fraction"],
        })

    return pd.DataFrame(rows)


def summarize_track_statistics(track_stats: pd.DataFrame, pod: float, n_reference: int, n_own) -> dict:
    d = track_stats[track_stats.detected] if len(track_stats) else track_stats
    if len(d):
        vals = {
            "mean_overlap_percent": d.overlap_percent.mean(), "median_overlap_percent": d.overlap_percent.median(),
            "mean_temporal_overlap_percent": d.overlap_duration_percent.mean(),
            "median_temporal_overlap_percent": d.overlap_duration_percent.median(),
            "mean_distance_km": d.mean_distance_km.mean(), "median_distance_km": d.mean_distance_km.median(),
        }
    else:
        vals = {k: np.nan for k in [
            "mean_overlap_percent", "median_overlap_percent", "mean_temporal_overlap_percent",
            "median_temporal_overlap_percent", "mean_distance_km", "median_distance_km",
        ]}
    return {
        "n_reference_tracks": n_reference, "n_matched": len(d), "n_unmatched": n_reference - len(d),
        f"n_{OWN_NAME}_tracks": n_own, "POD": pod, **vals,
    }


# ============================================================================
# 5. Графики (generic — reference_label только для подписей)
# ============================================================================

def plot_reference_track(reference_track, own_tracks, reference_id, reference_label, output_filename, year=None,
                          config_label="", title_extra=""):
    fig = plt.figure(figsize=(12, 8), dpi=150)
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent([-110, 15, 0, 73], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
    ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)

    ax.plot(reference_track.lon, reference_track.lat, color="black", linewidth=3, zorder=2,
            transform=ccrs.PlateCarree(), label=f"{reference_label} {reference_id}")
    ax.plot(reference_track.lon[0], reference_track.lat[0], "go", markersize=8, zorder=2,
            transform=ccrs.PlateCarree(), label="Genesis")
    ax.plot(reference_track.lon[-1], reference_track.lat[-1], "rs", markersize=8, zorder=2,
            transform=ccrs.PlateCarree(), label="Lysis")

    own_ids = [] if own_tracks is None else np.unique(own_tracks.track_id.values)
    own_colors = ["#FF1493", "#FF0000", "#CC00FF", "#FF4500", "#FFA500", "#FF69B4", "#AD1457", "#8B008B", "#FF6347", "#FFD700"]
    for i, own_id in enumerate(own_ids):
        own_track = own_tracks.where(own_tracks.track_id == own_id, drop=True).sortby("time")
        ax.plot(own_track.lon, own_track.lat, color=own_colors[i % len(own_colors)], linewidth=1.8, alpha=0.9, zorder=5,
                transform=ccrs.PlateCarree(), label=f"{OWN_NAME} {own_id}")

    title = config_label if config_label else ""
    ref_line = f"{reference_label} {reference_id}"
    if year is not None:
        ref_line += f" - {year}"
    title = f"{title}\n{ref_line}" if title else ref_line
    title += f"\nBlack: {reference_label} | Colored: {OWN_NAME} ({len(own_ids)} tracks)"
    if title_extra:
        title += f"\n{title_extra}"
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_all_reference_tracks(reference_tracks, own_tracks, output_dir, reference_label, reference_id_column, matches,
                               max_matched_tracks: int = 10, config_label: str = ""):
    """Одна карта на референсный трек — референс чёрным + все совпавшие с ним
    свои треки (каждый своим цветом). Референсный трек с >= max_matched_tracks
    совпадениями пропускается (почти наверняка вырожденный матчинг, а не
    читаемая картина одного шторма) — см. tracks_comparison_lib.py, откуда
    взята эта защита."""
    os.makedirs(output_dir, exist_ok=True)
    ids = np.unique(reference_tracks.track_id.values)
    matched_ids = set(matches[reference_id_column].values) if len(matches) else set()
    n_skipped_too_many = 0

    for i, ref_id in enumerate(ids, start=1):
        ref_track = reference_tracks.where(reference_tracks.track_id == ref_id, drop=True).sortby("time")
        year = int(pd.Timestamp(ref_track.time.values[0]).year)

        if ref_id in matched_ids:
            rows = matches[matches[reference_id_column] == ref_id]
            own_ids = np.unique(rows[OWN_ID_COLUMN].values)
            if len(own_ids) >= max_matched_tracks:
                print(f"  [{reference_label} {ref_id}] карта пропущена: {len(own_ids)} совпавших треков "
                      f">= {max_matched_tracks} - похоже на вырожденный матчинг.")
                n_skipped_too_many += 1
                continue
            own_for_ref = own_tracks.where(own_tracks.track_id.isin(list(own_ids)), drop=True)
        else:
            own_for_ref = None

        filename = f"{output_dir}/{reference_label}_{ref_id}_{year}_track{i}.png"
        plot_reference_track(ref_track, own_for_ref, ref_id, reference_label, filename, year=year,
                             config_label=config_label)

    if n_skipped_too_many:
        print(f"  Итого пропущено карт (>= {max_matched_tracks} совпадений на референсный трек): {n_skipped_too_many}")


def plot_pod_bar(summary_df: pd.DataFrame, output_filename: Path, title: str, top_n: int = 30) -> None:
    df = summary_df.sort_values("POD", ascending=False).head(top_n)
    labels = list(df["config_label"])
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(df))))
    ax.barh(labels[::-1], (100 * df["POD"])[::-1], color="tab:blue")
    ax.set_xlabel("POD (%)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {output_filename}")


def plot_pod_vs_overlap_scatter(summary_df: pd.DataFrame, output_filename: Path, title: str) -> None:
    tracking_types = sorted(summary_df["tracking_type"].unique())
    colors = dict(zip(tracking_types, plt.cm.tab10.colors))
    speeds = sorted(summary_df["speed"].unique())
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X"]
    markers = dict(zip(speeds, marker_cycle))

    fig, ax = plt.subplots(figsize=(9, 7))
    for tt in tracking_types:
        for sp in speeds:
            sub = summary_df[(summary_df["tracking_type"] == tt) & (summary_df["speed"] == sp)]
            if sub.empty:
                continue
            ax.scatter(100 * sub["POD"], sub["median_temporal_overlap_percent"], c=[colors[tt]],
                       marker=markers[sp], s=90, alpha=0.75,
                       edgecolors="black", linewidths=0.4, label=f"{tt}, {sp}")

    ax.set_xlabel("POD (%)")
    ax.set_ylabel("Median temporal overlap (%)")
    ax.set_title(f"{title}\nцвет=tracking_type, маркер=speed")
    ax.grid(linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=7, ncol=1)
    plt.tight_layout()
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {output_filename}")

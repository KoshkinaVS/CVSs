#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
filter_syclops_tracks.py
========================

Универсальное выделение типов треков из SyCLoPS:

    --type TC
    --type SS
    --type PL

Критерий типа ТРЕКА:
    тип определяется по наличию соответствующей стадии
    в Adjusted_Label на протяжении ВСЕГО TID:

        TC -> Adjusted_Label == "TC"
        SS -> Adjusted_Label == "SS(STLC)"
        PL -> Adjusted_Label == "PL(PTLC)"

Приоритет типов для одного TID:
    TC > SS > PL

То есть TID, в котором встречается TC, сохраняется только
в TC; TID с SS, но без TC — только в SS; TID с PL,
но без TC и SS — только в PL.

Track_Info не используется как самостоятельный критерий
классификации TID, поскольку поиск подстроки "TC"/"SS"/"PL"
может относить к типу дополнительные переходные треки.

Регионы:

    --region NA
        0-71°N
        -110...15°E

    --region Arctic
        65-80°N
        0...105°E

ВАЖНО:
SyCLoPS хранит LON в диапазоне [0, 360).
Поэтому пользователь может задавать долготы в обычном диапазоне
[-180, 180], а скрипт автоматически переводит их в [0, 360).

Для NA:
    -110° -> 250°
     15°  -> 15°

и корректно формирует маску, проходящую через 0°/360°.

Примеры
--------

TC в Северной Атлантике за 2010:

    python filter_type_for_year_and_region.py \
    --input /storage/thalassa/users/vkoshkina/data/SyCLoPS/SyCLoPS_classified_ERA5_1940_2025_3hr.parquet \
    --outdir /storage/thalassa/users/vkoshkina/data/SyCLoPS/tracks_types_csv \
    --type TC \
    --region NA \
    --year 2010 \
    --full-tracks \
    --plot-map

SS:

    python filter_type_for_year_and_region.py \
        --input /storage/thalassa/users/vkoshkina/data/SyCLoPS/SyCLoPS_classified_ERA5_1940_2025_3hr.parquet \
        --outdir /storage/thalassa/users/vkoshkina/data/SyCLoPS/tracks_types_csv \
        --type SS \
        --region NA \
        --year 2010 \
        --full-tracks \
        --plot-map

PL в Северной Атлантике:

    python filter_type_for_year_and_region.py \
            --input /storage/thalassa/users/vkoshkina/data/SyCLoPS/SyCLoPS_classified_ERA5_1940_2025_3hr.parquet \
            --outdir /storage/thalassa/users/vkoshkina/data/SyCLoPS/tracks_types_csv \
            --type PL \
            --region NA \
            --year 2019 \
            --full-tracks \
            --plot-map
            
PL в Арктике:

    python filter_type_for_year_and_region.py \
            --input /storage/thalassa/users/vkoshkina/data/SyCLoPS/SyCLoPS_classified_ERA5_1940_2025_3hr.parquet \
            --outdir /storage/thalassa/users/vkoshkina/data/SyCLoPS/tracks_types_csv \
            --type PL \
            --region Arctic \
            --year 2019 \
            --full-tracks \
            --plot-map

Зависимости:
    pandas
    pyarrow

Для карт:
    matplotlib
    cartopy
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# OPTIONAL PLOTTING
# ============================================================================

try:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    HAS_MPL = True

except ImportError:
    HAS_MPL = False


try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True

except ImportError:
    HAS_CARTOPY = False


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger(__name__)


# ============================================================================
# TYPE CONFIGURATION
# ============================================================================

TYPE_CONFIG = {
    "TC": {
        "node_label": "TC",
        "description": "Tropical Cyclone",
    },

    "SS": {
        "node_label": "SS(STLC)",
        "description": "Subtropical Cyclone",
    },

    "PL": {
        "node_label": "PL(PTLC)",
        "description": "Polar Low / Polar Tropical-Like Cyclone",
    },
}


# ============================================================================
# REGION CONFIGURATION
# ============================================================================

REGION_CONFIG = {
    "NA": {
        "lat_min": 0.0,
        "lat_max": 71.0,

        # Задаём долготу в привычной системе -180...180.
        "lon_min": -110.0,
        "lon_max": 15.0,

        "description": "North Atlantic",
    },

    "Arctic": {
        "lat_min": 65.0,
        "lat_max": 80.0,

        "lon_min": 0.0,
        "lon_max": 105.0,

        "description": "Arctic",
    },
}



# ============================================================================
# TC GENESIS / NORTH ATLANTIC BASIN FILTER
# ============================================================================

NA_GENESIS_LON_MIN = -110.0
NA_GENESIS_LON_MAX = 15.0
NA_GENESIS_LAT_MIN = 0.0
NA_GENESIS_LAT_MAX = 71.0


def normalize_lon_180(lon):
    """SyCLoPS [0, 360) -> [-180, 180)."""
    return ((float(lon) + 180.0) % 360.0) - 180.0


def is_na_genesis_point(lon, lat):
    """Preliminary geographic-window test for North Atlantic genesis."""
    lon = normalize_lon_180(lon)
    lat = float(lat)

    return (
        NA_GENESIS_LAT_MIN <= lat <= NA_GENESIS_LAT_MAX
        and NA_GENESIS_LON_MIN <= lon <= NA_GENESIS_LON_MAX
    )


def build_land_geometry():
    """Load Natural Earth land geometry via Cartopy, if available."""
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.ops import unary_union

        shp = shpreader.natural_earth(
            resolution="110m",
            category="physical",
            name="land",
        )
        return unary_union(
            list(shpreader.Reader(shp).geometries())
        )
    except Exception as exc:
        log.warning(
            "Не удалось загрузить Natural Earth land geometry: %s",
            exc,
        )
        return None


def is_oceanic_na_genesis(lon, lat, land_geometry=None):
    """
    True if genesis is inside the NA window and not on land.

    If Natural Earth geometry is unavailable, falls back to the
    geographic NA window.
    """
    if not is_na_genesis_point(lon, lat):
        return False

    if land_geometry is None:
        return True

    try:
        from shapely.geometry import Point

        return not land_geometry.contains(
            Point(normalize_lon_180(lon), float(lat))
        )
    except Exception:
        return True


# ============================================================================
# REQUIRED COLUMNS
# ============================================================================

REQUIRED_COLUMNS = [
    "TID",
    "ISOTIME",
    "LON",
    "LAT",
    "MSLP",
    "WS",
    "Short_Label",
    "Adjusted_Label",
    "Tropical_Flag",
    "Transition_Zone",
    "Track_Info",
    "LPSAREA",
]


# ============================================================================
# ARGUMENTS
# ============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Универсальная фильтрация SyCLoPS по типу трека "
            "TC / SS / PL, региону и году."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Путь к SyCLoPS parquet-файлу.",
    )

    parser.add_argument(
        "--outdir",
        default=Path("."),
        type=Path,
        help="Каталог для результатов.",
    )

    parser.add_argument(
        "--type",
        required=True,
        choices=["TC", "SS", "PL"],
        help="Тип трека: TC, SS или PL.",
    )

    parser.add_argument(
        "--region",
        choices=list(REGION_CONFIG.keys()),
        default="NA",
        help="Регион: NA или Arctic. Для NA TC дополнительно фильтруются по доле трека в регионе.",
    )

    parser.add_argument(
        "--year",
        type=int,
        default=2010,
        help="Год.",
    )

    parser.add_argument(
        "--lat-min",
        type=float,
        default=None,
        help="Переопределить минимальную широту.",
    )

    parser.add_argument(
        "--lat-max",
        type=float,
        default=None,
        help="Переопределить максимальную широту.",
    )

    parser.add_argument(
        "--lon-min",
        type=float,
        default=None,
        help="Переопределить минимальную долготу (-180...180).",
    )

    parser.add_argument(
        "--lon-max",
        type=float,
        default=None,
        help="Переопределить максимальную долготу (-180...180).",
    )

    parser.add_argument(
        "--min-nodes",
        type=int,
        default=1,
        help=(
            "Минимальное число узлов выбранного трека "
            "внутри региона и года."
        ),
    )

    parser.add_argument(
        "--full-tracks",
        action="store_true",
        help=(
            "Сохранить полную историю выбранных треков, "
            "включая узлы вне региона и года."
        ),
    )

    parser.add_argument(
        "--plot-map",
        action="store_true",
        help="Построить карту выбранных треков.",
    )

    return parser.parse_args()


# ============================================================================
# COORDINATES
# ============================================================================

def lon_to_360(lon: float) -> float:
    """
    Перевод долготы из [-180, 180] в [0, 360).
    """

    return lon % 360.0


def make_longitude_mask(
    lon: pd.Series,
    lon_min: float,
    lon_max: float,
) -> pd.Series:
    """
    Создаёт корректную маску долготы для SyCLoPS.

    SyCLoPS:
        LON ∈ [0, 360)

    Пользователь:
        может задавать границы в [-180, 180].

    Например:

        -110 ... 15

    превращается в:

        250 ... 15

    и означает:

        250...360 + 0...15
    """

    lon_min_360 = lon_to_360(lon_min)
    lon_max_360 = lon_to_360(lon_max)

    lon = lon.astype(float)

    # Обычный интервал, например:
    # 0...105
    if lon_min_360 <= lon_max_360:

        return (
            (lon >= lon_min_360)
            & (lon <= lon_max_360)
        )

    # Интервал пересекает 0°:
    #
    # 250...360
    # +
    # 0...15
    else:

        return (
            (lon >= lon_min_360)
            | (lon <= lon_max_360)
        )


# ============================================================================
# LOAD DATA
# ============================================================================

def load_data(
    path: Path,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    log.info("Чтение parquet:")
    log.info("  %s", path)

    log.info(
        "Колонки: %s",
        ", ".join(REQUIRED_COLUMNS),
    )

    df_all = pd.read_parquet(
        path,
        columns=REQUIRED_COLUMNS,
    )

    log.info(
        "Всего узлов в каталоге: %d",
        len(df_all),
    )

    if not pd.api.types.is_datetime64_any_dtype(
        df_all["ISOTIME"]
    ):
        df_all["ISOTIME"] = pd.to_datetime(
            df_all["ISOTIME"]
        )

    # ------------------------------------------------------------
    # Latitude
    # ------------------------------------------------------------

    lat_mask = (
        (df_all["LAT"] >= lat_min)
        &
        (df_all["LAT"] <= lat_max)
    )

    # ------------------------------------------------------------
    # Longitude
    # ------------------------------------------------------------

    lon_mask = make_longitude_mask(
        df_all["LON"],
        lon_min,
        lon_max,
    )

    # ------------------------------------------------------------
    # Time
    # ------------------------------------------------------------

    time_mask = (
        df_all["ISOTIME"].dt.year == year
    )

    mask = (
        lat_mask
        &
        lon_mask
        &
        time_mask
    )

    df_reg = df_all.loc[mask].copy()

    log.info(
        "Пространственно-временной фильтр:"
    )

    log.info(
        "  latitude: %.1f ... %.1f°",
        lat_min,
        lat_max,
    )

    log.info(
        "  longitude: %.1f ... %.1f° "
        "(входной диапазон [-180,180])",
        lon_min,
        lon_max,
    )

    log.info(
        "  longitude в SyCLoPS: %.1f ... %.1f°",
        lon_to_360(lon_min),
        lon_to_360(lon_max),
    )

    log.info(
        "  year: %d",
        year,
    )

    log.info(
        "После фильтра: %d узлов, %d треков",
        len(df_reg),
        df_reg["TID"].nunique(),
    )

    return df_all, df_reg


# ============================================================================
# TRACK SELECTION
# ============================================================================

def determine_primary_track_type(
    df_all: pd.DataFrame,
    region: str,
) -> pd.DataFrame:
    """
    Определяет один приоритетный тип для каждого TID.

    Приоритет:
        TC > SS > PL

    Для TC в выбранном регионе применяется простой и быстрый
    критерий бассейна:

    1. ВСЕ точки, имеющие Adjusted_Label == "TC", должны находиться
       внутри выбранного регионального диапазона.

    2. Не менее 50% ВСЕХ точек полного TID должны находиться
       внутри выбранного регионального диапазона.

    Это позволяет исключить Pacific-origin TC, которые лишь
    позднее заходят в North Atlantic, без медленной проверки
    Natural Earth / shapely.

    Важно:
    классификация выполняется по полному TID ДО регионального
    фильтра.
    """

    cfg = REGION_CONFIG[region]

    # ------------------------------------------------------------
    # Векторный региональный фильтр для ВСЕХ точек.
    # ------------------------------------------------------------

    lon = (
        (df_all["LON"].astype(float) + 180.0) % 360.0
    ) - 180.0

    lat = df_all["LAT"].astype(float)

    in_region = (
        lat.between(
            cfg["lat_min"],
            cfg["lat_max"],
            inclusive="both",
        )
        & lon.between(
            cfg["lon_min"],
            cfg["lon_max"],
            inclusive="both",
        )
    )

    work = pd.DataFrame(
        {
            "TID": df_all["TID"].values,
            "in_region": in_region.values,
            "is_TC": (
                df_all["Adjusted_Label"]
                .astype("string")
                .eq("TC")
                .values
            ),
            "is_SS": (
                df_all["Adjusted_Label"]
                .astype("string")
                .eq("SS(STLC)")
                .values
            ),
            "is_PL": (
                df_all["Adjusted_Label"]
                .astype("string")
                .eq("PL(PTLC)")
                .values
            ),
        }
    )

    # ------------------------------------------------------------
    # Все необходимые статистики считаются vectorized groupby.
    # Никаких groupby.apply(), shapely или Natural Earth.
    # ------------------------------------------------------------

    total_points = (
        work.groupby("TID", sort=False)
        .size()
        .rename("n_total")
    )

    points_in_region = (
        work.loc[work["in_region"]]
        .groupby("TID", sort=False)
        .size()
        .rename("n_in_region")
    )

    tc_points = (
        work.loc[work["is_TC"]]
        .groupby("TID", sort=False)
        .size()
        .rename("n_TC")
    )

    tc_points_in_region = (
        work.loc[
            work["is_TC"] & work["in_region"]
        ]
        .groupby("TID", sort=False)
        .size()
        .rename("n_TC_in_region")
    )

    has_tc = (
        work.groupby("TID", sort=False)["is_TC"]
        .any()
        .rename("has_TC")
    )

    has_ss = (
        work.groupby("TID", sort=False)["is_SS"]
        .any()
        .rename("has_SS")
    )

    has_pl = (
        work.groupby("TID", sort=False)["is_PL"]
        .any()
        .rename("has_PL")
    )

    result = pd.concat(
        [
            total_points,
            points_in_region,
            tc_points,
            tc_points_in_region,
            has_tc,
            has_ss,
            has_pl,
        ],
        axis=1,
    ).fillna(0)

    result.index.name = "TID"

    # Boolean columns были заполнены 0 через fillna.
    result["has_TC"] = result["has_TC"].astype(bool)
    result["has_SS"] = result["has_SS"].astype(bool)
    result["has_PL"] = result["has_PL"].astype(bool)

    result["n_total"] = result["n_total"].astype(int)
    result["n_in_region"] = result["n_in_region"].astype(int)
    result["n_TC"] = result["n_TC"].astype(int)
    result["n_TC_in_region"] = result["n_TC_in_region"].astype(int)

    result["fraction_in_region"] = (
        result["n_in_region"]
        / result["n_total"]
    )

    # ------------------------------------------------------------
    # Критерий для TC:
    #
    #   ВСЕ TC-точки внутри региона
    #   +
    #   >= 50% ВСЕХ точек TID внутри региона
    #
    # Для TID без TC значение не имеет смысла.
    # ------------------------------------------------------------

    result["all_TC_in_region"] = (
        result["n_TC_in_region"]
        == result["n_TC"]
    )

    result["TC_pass_region_filter"] = (
        result["has_TC"]
        & result["all_TC_in_region"]
        & (result["fraction_in_region"] >= 0.50)
    )

    # ------------------------------------------------------------
    # Приоритет TC > SS > PL.
    #
    # Для NA TC дополнительно должен пройти бассейновый критерий.
    # Для Arctic/других регионов сохраняем TC без этого специального
    # ограничения: региональный отбор далее выполняется как раньше.
    # ------------------------------------------------------------

    primary_type = pd.Series(
        None,
        index=result.index,
        dtype="object",
    )

    if region == "NA":
        tc_allowed = result["TC_pass_region_filter"]
    else:
        tc_allowed = result["has_TC"]

    primary_type.loc[tc_allowed] = "TC"

    ss_allowed = (
        result["has_SS"]
        & primary_type.isna()
    )
    primary_type.loc[ss_allowed] = "SS"

    pl_allowed = (
        result["has_PL"]
        & primary_type.isna()
    )
    primary_type.loc[pl_allowed] = "PL"

    result["primary_type"] = primary_type

    # ------------------------------------------------------------
    # Сохраняем TID как обычную колонку.
    # ------------------------------------------------------------

    result = result.reset_index()

    # ------------------------------------------------------------
    # Диагностика
    # ------------------------------------------------------------

    if region == "NA":
        n_tc = int(result["has_TC"].sum())
        n_tc_ok = int(result["TC_pass_region_filter"].sum())
        n_tc_rejected = n_tc - n_tc_ok

        log.info(
            "NA TC: %d всего TID с TC",
            n_tc,
        )
        log.info(
            "NA TC: %d прошли фильтр "
            "(100%% TC-точек в NA + >=50%% всех точек в NA)",
            n_tc_ok,
        )
        log.info(
            "NA TC: %d исключено по бассейновому критерию",
            n_tc_rejected,
        )

    return result


# ============================================================================
# NODE INFORMATION
# ============================================================================

def get_node_statistics(
    df_selected: pd.DataFrame,
    track_type: str,
) -> pd.DataFrame:

    config = TYPE_CONFIG[track_type]

    node_label = config["node_label"]

    if df_selected.empty:

        return pd.DataFrame(
            columns=[
                "TID",
                "n_nodes",
                "n_type_nodes",
                "fraction_type_nodes",
                "n_unique_adjusted_labels",
                "adjusted_labels",
            ]
        )

    records = []

    for tid, group in df_selected.groupby("TID"):

        labels = group["Adjusted_Label"]

        n_nodes = len(group)

        n_type_nodes = (
            labels == node_label
        ).sum()

        records.append(
            {
                "TID": tid,
                "n_nodes": n_nodes,
                "n_type_nodes": int(n_type_nodes),
                "fraction_type_nodes": (
                    n_type_nodes / n_nodes
                    if n_nodes > 0
                    else np.nan
                ),
                "n_unique_adjusted_labels": (
                    labels.nunique()
                ),
                "adjusted_labels": (
                    "|".join(
                        sorted(
                            labels
                            .dropna()
                            .astype(str)
                            .unique()
                        )
                    )
                ),
            }
        )

    return pd.DataFrame(
        records
    ).sort_values("TID")


# ============================================================================
# ALL TYPE NODES
# ============================================================================

def select_type_nodes(
    df: pd.DataFrame,
    track_type: str,
) -> pd.DataFrame:
    """
    Из уже выбранных треков выделяет узлы,
    непосредственно находящиеся в соответствующей стадии.

    TC -> TC
    SS -> SS(STLC)
    PL -> PL(PTLC)
    """

    node_label = TYPE_CONFIG[
        track_type
    ]["node_label"]

    return df[
        df["Adjusted_Label"] == node_label
    ].copy()


# ============================================================================
# FULL TRACKS
# ============================================================================

def expand_to_full_tracks(
    df_all: pd.DataFrame,
    selected_tid: pd.Index,
) -> pd.DataFrame:

    df_full = df_all[
        df_all["TID"].isin(selected_tid)
    ].copy()

    log.info(
        "Полные треки: %d узлов, %d TID",
        len(df_full),
        df_full["TID"].nunique(),
    )

    return df_full


# ============================================================================
# SUMMARY
# ============================================================================

def build_summary(
    df_region: pd.DataFrame,
    df_tracks: pd.DataFrame,
    df_type_nodes: pd.DataFrame,
    selected_tid: pd.Index,
    track_type: str,
    region: str,
    year: int,
) -> pd.DataFrame:

    config = TYPE_CONFIG[track_type]

    rows = []

    # ------------------------------------------------------------
    # Общая информация
    # ------------------------------------------------------------

    rows.append(
        {
            "region": region,
            "year": year,
            "type": track_type,
            "description": config["description"],
            "selection_method": "Adjusted_Label with priority TC > SS > PL",
            "n_tracks": len(selected_tid),
            "n_nodes_selected_tracks": len(df_tracks),
            "n_type_stage_nodes": len(df_type_nodes),
        }
    )

    return pd.DataFrame(rows)


def build_monthly_summary(
    df_tracks: pd.DataFrame,
    track_type: str,
    year: int,
) -> pd.DataFrame:

    if df_tracks.empty:

        return pd.DataFrame(
            columns=[
                "year",
                "type",
                "month",
                "n_nodes",
                "n_tracks",
            ]
        )

    monthly = (
        df_tracks
        .groupby(
            df_tracks["ISOTIME"].dt.month
        )
        .agg(
            n_nodes=("TID", "size"),
            n_tracks=("TID", "nunique"),
        )
        .reset_index()
        .rename(
            columns={"ISOTIME": "month"}
        )
    )

    monthly["year"] = year
    monthly["type"] = track_type

    return monthly[
        [
            "year",
            "type",
            "month",
            "n_nodes",
            "n_tracks",
        ]
    ]


# ============================================================================
# LONGITUDE UNWRAPPING FOR PLOTS
# ============================================================================

def normalize_longitude_for_plot(
    lon: np.ndarray,
) -> np.ndarray:
    """
    Переводит долготы SyCLoPS из [0, 360) в [-180, 180).

    ВАЖНО:
    функция НЕ делает unwrap и не добавляет 360° к последующим
    точкам. Это позволяет отдельно обрабатывать разрыв через
    границу карты.
    """
    lon = np.asarray(lon, dtype=float)

    return (lon + 180.0) % 360.0 - 180.0


def iter_plot_segments(
    lon: np.ndarray,
    lat: np.ndarray,
    labels: np.ndarray,
):
    """
    Возвращает короткие последовательные сегменты трека.

    Если последовательные точки после перевода в [-180, 180)
    отличаются более чем на 180°, это переход через границу
    180°/-180° и соответствующий отрезок не рисуется.

    Это предотвращает длинную линию через всю карту для full tracks.
    """

    for i in range(1, len(lon)):
        lon1 = lon[i - 1]
        lon2 = lon[i]

        # Разрыв через -180/180
        if abs(lon2 - lon1) > 180.0:
            continue

        yield (
            i,
            lon1,
            lon2,
            lat[i - 1],
            lat[i],
            labels[i],
        )


# ============================================================================
# PLOT
# ============================================================================

def plot_tracks(
    df: pd.DataFrame,
    output_path: Path,
    track_type: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    title: str,
) -> None:

    if not HAS_MPL:

        log.warning(
            "matplotlib отсутствует — карта не построена."
        )

        return

    if df.empty:

        log.warning(
            "Нет данных для карты."
        )

        return

    config = TYPE_CONFIG[
        track_type
    ]

    node_label = config["node_label"]

    fig = plt.figure(
        figsize=(10, 10),
        dpi=150,
    )

    # ------------------------------------------------------------
    # Cartopy
    # ------------------------------------------------------------

    if HAS_CARTOPY:

        central_lon = (
            lon_min + lon_max
        ) / 2.0

        projection = (
            ccrs.NorthPolarStereo(
                central_longitude=central_lon
            )
        )

        data_crs = ccrs.PlateCarree()

        ax = plt.axes(
            projection=projection
        )

        ax.set_extent(
            [
                lon_min - 5,
                lon_max + 5,
                max(-90, lat_min - 3),
                min(90, lat_max + 3),
            ],
            crs=data_crs,
        )

        ax.add_feature(
            cfeature.LAND,
            facecolor="#e8e8e0",
            zorder=0,
        )

        ax.add_feature(
            cfeature.OCEAN,
            facecolor="#eef4fb",
            zorder=0,
        )

        ax.coastlines(
            resolution="50m",
            linewidth=0.6,
        )

        gridlines = ax.gridlines(
            draw_labels=True,
            linewidth=0.4,
            alpha=0.6,
            linestyle="--",
        )

        gridlines.top_labels = False
        gridlines.right_labels = False

        plot_kwargs = {
            "transform": data_crs
        }

    # ------------------------------------------------------------
    # Matplotlib fallback
    # ------------------------------------------------------------

    else:

        ax = plt.axes()

        ax.set_xlim(
            lon_min - 5,
            lon_max + 5,
        )

        ax.set_ylim(
            lat_min - 3,
            lat_max + 3,
        )

        ax.set_xlabel(
            "Longitude, °"
        )

        ax.set_ylabel(
            "Latitude, °"
        )

        ax.grid(
            True,
            linestyle="--",
            linewidth=0.4,
            alpha=0.6,
        )

        mean_lat = (
            lat_min + lat_max
        ) / 2

        ax.set_aspect(
            1.0 /
            max(
                math.cos(
                    math.radians(
                        mean_lat
                    )
                ),
                0.15,
            )
        )

        plot_kwargs = {}

    # ------------------------------------------------------------
    # Draw tracks
    # ------------------------------------------------------------

    n_tracks = 0

    for tid, group in (
        df
        .sort_values(
            ["TID", "ISOTIME"]
        )
        .groupby("TID")
    ):

        if len(group) == 0:
            continue

        lon = normalize_longitude_for_plot(
            group["LON"].to_numpy()
        )

        lat = (
            group["LAT"]
            .to_numpy()
        )

        labels = (
            group["Adjusted_Label"]
            .astype(str)
            .to_numpy()
        )

        # --------------------------------------------------------
        # Segments
        #
        # Не соединяем точки, если трек пересекает границу
        # -180/180. Для NA это устраняет артефакт при переходе
        # через 0°/360° в исходной системе SyCLoPS.
        # --------------------------------------------------------

        for (
            i,
            lon1,
            lon2,
            lat1,
            lat2,
            label,
        ) in iter_plot_segments(
            lon,
            lat,
            labels,
        ):

            if label == node_label:
                color = "#9400D3"
            else:
                color = "#1f4fd8"

            ax.plot(
                [lon1, lon2],
                [lat1, lat2],
                color=color,
                linewidth=1.8,
                alpha=0.9,
                solid_capstyle="round",
                **plot_kwargs,
            )

        # --------------------------------------------------------
        # Nodes
        # --------------------------------------------------------

        node_colors = [
            "#9400D3"
            if label == node_label
            else "#1f4fd8"
            for label in labels
        ]

        ax.scatter(
            lon,
            lat,
            c=node_colors,
            s=10,
            linewidths=0,
            zorder=3,
            **plot_kwargs,
        )

        # --------------------------------------------------------
        # Start / end
        # --------------------------------------------------------

        ax.scatter(
            lon[0],
            lat[0],
            c="#00A651",
            s=55,
            marker="o",
            edgecolors="black",
            linewidths=0.6,
            zorder=4,
            **plot_kwargs,
        )

        ax.scatter(
            lon[-1],
            lat[-1],
            c="#E4111C",
            s=55,
            marker="o",
            edgecolors="black",
            linewidths=0.6,
            zorder=4,
            **plot_kwargs,
        )

        n_tracks += 1

    # ------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------

    handles = [
        Line2D(
            [0],
            [0],
            color="#9400D3",
            lw=2.5,
            label=f"{track_type} stage: {node_label}",
        ),

        Line2D(
            [0],
            [0],
            color="#1f4fd8",
            lw=2.5,
            label="Other stages",
        ),

        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#00A651",
            markeredgecolor="black",
            markersize=8,
            label="Track start",
        ),

        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#E4111C",
            markeredgecolor="black",
            markersize=8,
            label="Track end",
        ),
    ]

    ax.legend(
        handles=handles,
        loc="lower left",
        fontsize=8,
        framealpha=0.9,
    )

    ax.set_title(
        title,
        fontsize=11,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    log.info(
        "Карта сохранена: %s",
        output_path,
    )


# ============================================================================
# SAVE
# ============================================================================

def save_csv(
    df: pd.DataFrame,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
    )

    log.info(
        "Сохранено: %s (%d строк)",
        path,
        len(df),
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    args = parse_args()

    # ------------------------------------------------------------
    # Check input
    # ------------------------------------------------------------

    if not args.input.exists():

        log.error(
            "Файл не найден: %s",
            args.input,
        )

        return 1

    # ------------------------------------------------------------
    # Region
    # ------------------------------------------------------------

    region_cfg = REGION_CONFIG[
        args.region
    ]

    lat_min = (
        args.lat_min
        if args.lat_min is not None
        else region_cfg["lat_min"]
    )

    lat_max = (
        args.lat_max
        if args.lat_max is not None
        else region_cfg["lat_max"]
    )

    lon_min = (
        args.lon_min
        if args.lon_min is not None
        else region_cfg["lon_min"]
    )

    lon_max = (
        args.lon_max
        if args.lon_max is not None
        else region_cfg["lon_max"]
    )

    # ------------------------------------------------------------
    # Type
    # ------------------------------------------------------------

    type_cfg = TYPE_CONFIG[
        args.type
    ]

    log.info("=" * 70)
    log.info(
        "SyCLoPS track selection"
    )
    log.info("=" * 70)

    log.info(
        "Type: %s (%s)",
        args.type,
        type_cfg["description"],
    )

    log.info(
        "Region: %s (%s)",
        args.region,
        region_cfg["description"],
    )

    log.info(
        "Year: %d",
        args.year,
    )

    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------

    df_all, df_region = load_data(
        args.input,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        args.year,
    )

    # ------------------------------------------------------------
    # Global classification of TID
    #
    # ВАЖНО: делаем ДО регионально-временного выбора.
    # Один TID получает только один приоритетный тип:
    # TC > SS > PL.
    # ------------------------------------------------------------

    track_types = determine_primary_track_type(
        df_all,
        region=args.region,
    )

    log.info(
        "Глобальная классификация TID по Adjusted_Label:"
    )

    for typ in ["TC", "SS", "PL"]:
        n = (
            track_types["primary_type"] == typ
        ).sum()

        log.info(
            "  %s: %d TID",
            typ,
            n,
        )

    # ------------------------------------------------------------
    # Select tracks
    # ------------------------------------------------------------

    df_tracks, selected_tid = select_tracks(
        df_region,
        track_types,
        args.type,
        args.min_nodes,
    )

    # ------------------------------------------------------------
    # Direct type-stage nodes
    # ------------------------------------------------------------

    df_type_nodes = select_type_nodes(
        df_tracks,
        args.type,
    )

    log.info(
        "Непосредственно %s-ноды (%s): %d",
        args.type,
        type_cfg["node_label"],
        len(df_type_nodes),
    )

    # ------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------

    statistics = get_node_statistics(
        df_tracks,
        args.type,
    )

    # ------------------------------------------------------------
    # Full tracks
    # ------------------------------------------------------------

    if args.full_tracks:

        df_output = expand_to_full_tracks(
            df_all,
            selected_tid,
        )

    else:

        df_output = df_tracks.copy()

    # ------------------------------------------------------------
    # Output stem
    # ------------------------------------------------------------

    stem = (
        f"SyCLoPS_{args.type}_"
        f"{args.region}_"
        f"{lat_min:g}-{lat_max:g}N_"
        f"{lon_min:g}-{lon_max:g}E_"
        f"{args.year}"
    )

    # ------------------------------------------------------------
    # Main track data
    # ------------------------------------------------------------

    save_csv(
        df_output.sort_values(
            ["TID", "ISOTIME"]
        ),
        args.outdir / f"{stem}_tracks.csv",
    )

    # ------------------------------------------------------------
    # Type-stage nodes
    # ------------------------------------------------------------

    save_csv(
        df_type_nodes.sort_values(
            ["TID", "ISOTIME"]
        ),
        args.outdir / f"{stem}_type_nodes.csv",
    )

    # ------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------

    save_csv(
        statistics,
        args.outdir / f"{stem}_track_statistics.csv",
    )

    # ------------------------------------------------------------
    # Basin / global classification
    # ------------------------------------------------------------

    selected_genesis = track_types[
        track_types["TID"].isin(selected_tid)
    ].copy()

    save_csv(
        selected_genesis,
        args.outdir / f"{stem}_basin_classification.csv",
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    summary = build_summary(
        df_region=df_region,
        df_tracks=df_tracks,
        df_type_nodes=df_type_nodes,
        selected_tid=selected_tid,
        track_type=args.type,
        region=args.region,
        year=args.year,
    )

    save_csv(
        summary,
        args.outdir / f"{stem}_summary.csv",
    )

    # ------------------------------------------------------------
    # Monthly statistics
    # ------------------------------------------------------------

    monthly = build_monthly_summary(
        df_tracks,
        args.type,
        args.year,
    )

    save_csv(
        monthly,
        args.outdir / f"{stem}_monthly_summary.csv",
    )

    # ------------------------------------------------------------
    # Map
    # ------------------------------------------------------------

    if args.plot_map:

        plot_tracks(
            df_output,
            args.outdir / f"{stem}_map.png",
            track_type=args.type,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            title=(
                f"SyCLoPS {args.type} tracks — "
                f"{args.region}, {args.year} "
                f"({len(selected_tid)} tracks)"
            ),
        )

    # ------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------

    log.info("=" * 70)
    log.info("ГОТОВО")
    log.info("=" * 70)

    log.info(
        "Тип: %s",
        args.type,
    )

    log.info(
        "Отобрано треков: %d",
        len(selected_tid),
    )

    log.info(
        "Узлов выбранных треков в регионе/году: %d",
        len(df_tracks),
    )

    log.info(
        "Нодов непосредственно стадии %s: %d",
        type_cfg["node_label"],
        len(df_type_nodes),
    )

    if args.full_tracks:

        log.info(
            "Сохранены полные истории выбранных треков."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
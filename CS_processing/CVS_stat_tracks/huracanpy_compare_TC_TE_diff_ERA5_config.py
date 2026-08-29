import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from itertools import product


import huracanpy
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

# Период анализа
START_YEAR = 2010
END_YEAR = 2010

# TempestExtremes matching parameters
MAX_DIST = 200       # km
MEAN_DIST = 120      # km
MIN_OVERLAP = 3      # minimum number of matching points

# Base data directory
path_dir_data = f'/storage/thalassa/users/vkoshkina/data'

# Output root directory
OUTPUT_ROOT = (
    f"{path_dir_data}/TempestExtremes/ERA5/"
    f"TC_comparison_huracanpy/"
)


# ============================================================
# 1. LOAD TEMPEST EXTREMES TRACKS
# ============================================================

def load_te_tracks(
    level_hPa,
    size_filter,
    extr_type,
    start_year,
    end_year,
):
    """
    Загружает TempestExtremes tracks за заданный период.

    Parameters
    ----------
    level_hPa : int
        Pressure level, e.g. 500 or 850.

    size_filter : int
        Minimum number of points.

    extr_type : str
        '' or '_global'.

    Returns
    -------
    tracks_all_years
    tracks_by_year
    """

    base_folder = (
        f"{path_dir_data}/TempestExtremes/ERA5/"
        f"R2D_ERA5_NA_for_TC_{level_hPa}hPa_sigma_2/"
        f"Tracks_R2D_txt_files_range_1_5_18h_12h_2010_"
        f"{size_filter}points{extr_type}"
    )

    tracks_by_year = {}
    all_tracks = []

    years = np.arange(
        start_year,
        end_year + 1,
    )

    for year in tqdm(
        years,
        desc=f"Loading TE {level_hPa} hPa "
             f"{size_filter} points {extr_type}",
    ):

        filename = (
            f"{base_folder}/"
            f"ERA5_TC_tracks_{year}.txt"
        )

        try:

            tracks_year = huracanpy.load(
                filename,
                source="tempestextremes",
                variable_names=[
                    "rad",
                    "r2d",
                    "wspd",
                ],
            )

            tracks_by_year[year] = tracks_year
            all_tracks.append(tracks_year)

        except FileNotFoundError:

            print(
                f"Year {year} not found, skipping..."
            )

    if len(all_tracks) == 0:
        raise FileNotFoundError(
            f"No TE tracks found in {base_folder}"
        )

    tracks_all_years = (
        huracanpy.concat_tracks(all_tracks)
    )

    return (
        tracks_all_years,
        tracks_by_year,
    )


# ============================================================
# 2. LOAD IBTRACS
# ============================================================

def load_ibtracs(
    start_year,
    end_year,
):
    """
    Загружает IBTrACS и оставляет North Atlantic
    за заданный период.
    """

    all_tracks_TC = huracanpy.load(
        source="ibtracs",
    )

    all_tracks_TC = all_tracks_TC.where(
        (
            all_tracks_TC.time.dt.year >= start_year
        )
        &
        (
            all_tracks_TC.time.dt.year <= end_year
        ),
        drop=True,
    )

    na_tracks = all_tracks_TC.where(
        all_tracks_TC.basin == "NA",
        drop=True,
    )

    return na_tracks


# ============================================================
# 3. MATCHING
# ============================================================

def calculate_matches(
    ibtracs_tracks,
    te_tracks,
):
    """
    Выполняет matching IBTrACS vs TempestExtremes.
    """

    matches = huracanpy.assess.match(
        [
            ibtracs_tracks,
            te_tracks,
        ],
        names=[
            "IBTrACS",
            "TE_NA",
        ],
        max_dist=MAX_DIST,
        mean_dist=MEAN_DIST,
        min_overlap=MIN_OVERLAP,
        tracks1_is_ref=True,
    )

    return matches


# ============================================================
# 4. POD
# ============================================================

def calculate_pod(
    matches,
    ibtracs_tracks,
):
    """
    POD через huracanpy — оставляем именно этот расчет.
    """

    POD = huracanpy.assess.pod(
        matches,
        ref=ibtracs_tracks,
        ref_name="IBTrACS",
    )

    return float(POD)


# ============================================================
# 5. SELECT BEST MATCH FOR EACH IBTRACS TRACK
# ============================================================

def select_best_matches(matches):
    """
    Если одному IBTrACS треку соответствует несколько
    TE-треков, выбираем лучшее соответствие.

    Критерии:
        1. максимальное количество matching points (temp)
        2. минимальное среднее расстояние (dist)
    """

    if len(matches) == 0:
        return matches.copy()

    best_matches = (
        matches
        .sort_values(
            [
                "id_IBTrACS",
                "temp",
                "dist",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .drop_duplicates(
            subset="id_IBTrACS",
            keep="first",
        )
        .reset_index(drop=True)
    )

    return best_matches


# ============================================================
# 6. TEMPORAL OVERLAP
# ============================================================

def calculate_temporal_overlap(
    ib_track,
    te_track,
):
    """
    Считает временное перекрытие двух треков.

    overlap_duration_hours:
        длительность пересечения временных интервалов.

    overlap_duration_fraction:
        overlap_duration / duration IBTrACS.
    """

    ib_times = pd.to_datetime(
        ib_track.time.values
    )

    te_times = pd.to_datetime(
        te_track.time.values
    )

    if len(ib_times) == 0 or len(te_times) == 0:

        return {
            "overlap_start": pd.NaT,
            "overlap_end": pd.NaT,
            "overlap_duration_hours": 0,
            "overlap_duration_fraction": 0,
        }

    ib_start = ib_times.min()
    ib_end = ib_times.max()

    te_start = te_times.min()
    te_end = te_times.max()

    ib_duration = (
        ib_end - ib_start
    ).total_seconds() / 3600

    overlap_start = max(
        ib_start,
        te_start,
    )

    overlap_end = min(
        ib_end,
        te_end,
    )

    if overlap_start > overlap_end:

        overlap_duration = 0

    else:

        overlap_duration = (
            overlap_end - overlap_start
        ).total_seconds() / 3600

    if ib_duration > 0:

        fraction = (
            overlap_duration
            / ib_duration
        )

    else:

        fraction = 0

    return {
        "overlap_start": overlap_start,
        "overlap_end": overlap_end,
        "overlap_duration_hours": (
            overlap_duration
        ),
        "overlap_duration_fraction": (
            fraction
        ),
    }


# ============================================================
# 7. TRACK-LEVEL STATISTICS
# ============================================================

def calculate_track_statistics(
    ibtracs_tracks,
    te_tracks,
    matches,
):
    """
    Создает DataFrame:

    одна строка = один IBTrACS track.

    Включает:
        detected
        TE_ID_best
        matched_n_points
        overlap_percent
        mean_distance_km
        temporal overlap
    """

    # --------------------------------------------------------
    # Все IBTrACS IDs
    # --------------------------------------------------------

    ib_ids = np.unique(
        ibtracs_tracks.track_id.values
    )

    rows = []

    # --------------------------------------------------------
    # Лучшие matches
    # --------------------------------------------------------

    best_matches = select_best_matches(
        matches
    )

    # Словарь IBTrACS -> TE
    best_match_dict = {}

    for _, row in best_matches.iterrows():

        best_match_dict[
            row["id_IBTrACS"]
        ] = row

    # --------------------------------------------------------
    # Каждый IBTrACS track
    # --------------------------------------------------------

    for ib_id in ib_ids:

        ib_track = ibtracs_tracks.where(
            ibtracs_tracks.track_id == ib_id,
            drop=True,
        ).sortby("time")

        ib_times = pd.to_datetime(
            ib_track.time.values
        )

        n_ib_points = len(ib_times)

        if n_ib_points > 1:

            ib_duration_hours = (
                ib_times.max()
                - ib_times.min()
            ).total_seconds() / 3600

        else:

            ib_duration_hours = 0

        # ----------------------------------------------------
        # Нет matching
        # ----------------------------------------------------

        if ib_id not in best_match_dict:

            rows.append({

                "IBTrACS_ID": ib_id,

                "detected": False,

                "TE_ID_best": np.nan,

                "IBTrACS_n_points": (
                    n_ib_points
                ),

                "IBTrACS_duration_hours": (
                    ib_duration_hours
                ),

                "matched_n_points": 0,

                "overlap_fraction": 0,

                "overlap_percent": 0,

                "mean_distance_km": np.nan,

                "overlap_start": pd.NaT,

                "overlap_end": pd.NaT,

                "overlap_duration_hours": 0,

                "overlap_duration_fraction": 0,

                "overlap_duration_percent": 0,
            })

            continue

        # ----------------------------------------------------
        # Matching найден
        # ----------------------------------------------------

        match_row = best_match_dict[
            ib_id
        ]

        te_id = match_row[
            "id_TE_NA"
        ]

        matched_points = match_row[
            "temp"
        ]

        mean_distance = match_row[
            "dist"
        ]

        # ----------------------------------------------------
        # Point overlap
        # ----------------------------------------------------

        overlap_fraction = (
            matched_points
            / n_ib_points
            if n_ib_points > 0
            else np.nan
        )

        # ----------------------------------------------------
        # TE track
        # ----------------------------------------------------

        te_track = te_tracks.where(
            te_tracks.track_id == te_id,
            drop=True,
        ).sortby("time")

        # ----------------------------------------------------
        # Temporal overlap
        # ----------------------------------------------------

        temporal = (
            calculate_temporal_overlap(
                ib_track,
                te_track,
            )
        )

        rows.append({

            "IBTrACS_ID": ib_id,

            "detected": True,

            "TE_ID_best": te_id,

            "IBTrACS_n_points": (
                n_ib_points
            ),

            "IBTrACS_duration_hours": (
                ib_duration_hours
            ),

            "matched_n_points": (
                matched_points
            ),

            "overlap_fraction": (
                overlap_fraction
            ),

            "overlap_percent": (
                100 * overlap_fraction
            ),

            "mean_distance_km": (
                mean_distance
            ),

            "overlap_start": (
                temporal["overlap_start"]
            ),

            "overlap_end": (
                temporal["overlap_end"]
            ),

            "overlap_duration_hours": (
                temporal[
                    "overlap_duration_hours"
                ]
            ),

            "overlap_duration_fraction": (
                temporal[
                    "overlap_duration_fraction"
                ]
            ),

            "overlap_duration_percent": (
                100
                * temporal[
                    "overlap_duration_fraction"
                ]
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# 8. PLOT ONE IBTRACS TRACK
# ============================================================

def plot_track_comparison(
    ibtracs_track,
    te_tracks,
    ibtracs_id,
    year,
    output_filename,
):
    """
    Рисует IBTrACS-трек и все TE-треки,
    соответствующие данному IBTrACS.
    """

    # --------------------------------------------------------
    # TE tracks
    # --------------------------------------------------------

    if te_tracks is None:
        te_ids = []
    else:
        te_ids = np.unique(
            te_tracks.track_id.values
        )
    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(12, 8),
        dpi=150,
    )

    ax = fig.add_subplot(
        111,
        projection=ccrs.PlateCarree(),
    )

    # --------------------------------------------------------
    # Map
    # --------------------------------------------------------

    ax.set_extent(
        [-110, 15, 0, 73],
        ccrs.PlateCarree(),
    )

    ax.add_feature(
        cfeature.LAND,
        facecolor="lightgray",
        alpha=0.5,
    )

    ax.add_feature(
        cfeature.OCEAN,
        facecolor="lightblue",
        alpha=0.3,
    )

    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=0.5,
    )

    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.3,
        alpha=0.5,
    )

    ax.gridlines(
        draw_labels=True,
        linestyle="--",
        alpha=0.5,
    )

    # --------------------------------------------------------
    # IBTrACS
    # --------------------------------------------------------

    ax.plot(
        ibtracs_track.lon,
        ibtracs_track.lat,
        color="black",
        linewidth=3,
        transform=ccrs.PlateCarree(),
        label="IBTrACS",
    )

    ax.plot(
        ibtracs_track.lon[0],
        ibtracs_track.lat[0],
        "go",
        markersize=8,
        transform=ccrs.PlateCarree(),
        label="Genesis",
    )

    ax.plot(
        ibtracs_track.lon[-1],
        ibtracs_track.lat[-1],
        "rs",
        markersize=8,
        transform=ccrs.PlateCarree(),
        label="Lysis",
    )

    # --------------------------------------------------------
    # TE colors
    # --------------------------------------------------------

    map_colors = [
        "#FF1493",
        "#CC00FF",
        "#00CC00",
        "#FF0000",
        "#FF8C00",
        "#FFD700",
        "#FF4500",
        "#7FFF00",
        "#FF6347",
        "#FF00FF",
        "#FFA500",
        "#ADFF2F",
    ]

    colors = (
        map_colors
        * (
            len(te_ids)
            // len(map_colors)
            + 1
        )
    )

    # --------------------------------------------------------
    # TE tracks
    # --------------------------------------------------------

    for i, te_id in enumerate(te_ids):

        te_track = te_tracks.where(
            te_tracks.track_id == te_id,
            drop=True,
        ).sortby("time")

        ax.plot(
            te_track.lon,
            te_track.lat,
            color=colors[i],
            linewidth=1.5,
            alpha=0.8,
            transform=ccrs.PlateCarree(),
            label=f"TE {te_id}",
        )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    storm_name = (
        ibtracs_track.attrs.get(
            "name",
            "Unknown",
        )
        if hasattr(
            ibtracs_track,
            "attrs",
        )
        else "Unknown"
    )

    ax.set_title(
        f"IBTrACS {ibtracs_id} "
        f"({storm_name}) - {year}\n"
        f"Black: IBTrACS | "
        f"Colored: TempestExtremes "
        f"({len(te_ids)} tracks)"
    )

    ax.legend(
        loc="upper left",
        fontsize=8,
    )

    plt.tight_layout()

    os.makedirs(
        os.path.dirname(output_filename),
        exist_ok=True,
    )

    plt.savefig(
        output_filename,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

def plot_configuration_summary(
    summary_df,
    output_filename=None,
):
    """
    Визуализация результатов для всех конфигураций.

    Графики:
        (a) POD
        (b) Median temporal overlap
        (c) Number of TE tracks
        (d) Median distance between matched tracks

    Визуальная кодировка:

        Цвет:
            500 hPa -> blue
            850 hPa -> green

        Hatch:
            local  -> ///
            global -> no hatch

        Насыщенность:
            10 points -> бледный
            25 points -> насыщенный

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Таблица с результатами всех конфигураций.

    output_filename : str or None
        Путь для сохранения рисунка.

    Returns
    -------
    pandas.DataFrame
        Копия summary_df с дополнительными служебными колонками.
    """

    # ========================================================
    # 1. COPY DATAFRAME
    # ========================================================

    df = summary_df.copy()

    # ========================================================
    # 2. CHECK REQUIRED COLUMNS
    # ========================================================

    required_columns = [
        "level_hPa",
        "size_filter",
        "extr_type",
        "POD",
        "median_overlap_percent",
        "n_TE_tracks",
        "n_IBTrACS",
        "median_distance_km",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns in summary_df: "
            f"{missing_columns}"
        )

    # ========================================================
    # 3. NORMALIZE EXTRACTION TYPE
    # ========================================================

    df["extr_label"] = (
        df["extr_type"]
        .astype(str)
        .replace({
            "local": "local",
            "_local": "local",
            "global": "global",
            "_global": "global",
        })
    )

    # ========================================================
    # 4. SORT CONFIGURATIONS
    #
    # 500 hPa:
    #     10 local
    #     10 global
    #     25 local
    #     25 global
    #
    # 850 hPa:
    #     10 local
    #     10 global
    #     25 local
    #     25 global
    # ========================================================

    extraction_order = {
        "local": 0,
        "global": 1,
    }

    df["_extr_order"] = (
        df["extr_label"]
        .map(extraction_order)
    )

    df = df.sort_values(
        [
            "level_hPa",
            "size_filter",
            "_extr_order",
        ]
    ).reset_index(drop=True)

    df = df.drop(
        columns="_extr_order"
    )

    # ========================================================
    # 5. X AXIS
    # ========================================================

    x = np.arange(len(df))

    labels = [
        f"{int(level)}\n"
        f"{int(size)} / {extr}"
        for level, size, extr in zip(
            df["level_hPa"],
            df["size_filter"],
            df["extr_label"],
        )
    ]

    # ========================================================
    # 6. VISUAL SETTINGS
    # ========================================================

    # Pressure level -> color
    level_colors = {
        500: "tab:blue",
        850: "tab:green",
    }

    # Extraction type -> hatch
    hatch_types = {
        "local": "///",
        "global": "",
    }

    # Size filter -> transparency
    size_alpha = {
        10: 0.35,
        25: 1.0,
    }

    # ========================================================
    # 7. CREATE FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 10),
    )

    # ========================================================
    # 8. HELPER FUNCTION
    # ========================================================

    def draw_bars(
        ax,
        values,
        ylabel,
        title,
        log_scale=False,
    ):
        """
        Рисует столбцы с общей визуальной кодировкой.
        """

        bars = []

        for i, (
            value,
            level,
            size,
            extr,
        ) in enumerate(
            zip(
                values,
                df["level_hPa"],
                df["size_filter"],
                df["extr_label"],
            )
        ):

            level = int(level)
            size = int(size)

            color = level_colors[level]
            alpha = size_alpha[size]
            hatch = hatch_types[extr]

            bar = ax.bar(
                i,
                value,
                width=0.7,
                color=color,
                alpha=alpha,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.8,
            )[0]

            bars.append(bar)

        ax.set_ylabel(ylabel)

        ax.set_title(
            title,
            fontsize=12,
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            labels,
            fontsize=9,
        )

        if log_scale:
            ax.set_yscale("log")

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.35,
        )

        ax.set_axisbelow(True)

        # Автоматический запас сверху для подписей
        finite_values = np.asarray(values)[
            np.isfinite(values)
        ]
    
        if len(finite_values) > 0:
    
            max_value = np.max(finite_values)
    
            if log_scale:
                ax.set_ylim(
                    bottom=max(
                        1,
                        np.min(finite_values) * 0.8
                    ),
                    top=max_value * 1.5,
                )
    
            else:
                ax.set_ylim(
                    bottom=0,
                    top=max_value * 1.20,
                )

        return bars

    # ========================================================
    # 9. POD
    # ========================================================

    ax = axes[0, 0]

    pod_percent = (
        100 * df["POD"]
    )

    bars = draw_bars(
        ax=ax,
        values=pod_percent,
        ylabel="POD (%)",
        title="(a) Probability of Detection",
    )


    for bar, value in zip(
        bars,
        pod_percent,
    ):

        if np.isfinite(value):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + 1,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # ========================================================
    # 10. MEDIAN TEMPORAL OVERLAP
    # ========================================================

    ax = axes[0, 1]

    overlap = df[
        "median_overlap_percent"
    ]

    bars = draw_bars(
        ax=ax,
        values=overlap,
        ylabel="Median overlap (%)",
        title="(b) Median Overlap",
    )


    for bar, value in zip(
        bars,
        overlap,
    ):

        if np.isfinite(value):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + 1,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # ========================================================
    # 11. NUMBER OF TE TRACKS
    # ========================================================

    ax = axes[1, 0]

    n_te = df[
        "n_TE_tracks"
    ]

    bars = draw_bars(
        ax=ax,
        values=n_te,
        ylabel="Number of TE tracks",
        title="(c) Number of TempestExtremes Tracks",
        log_scale=True,
    )

    for bar, value in zip(
        bars,
        n_te,
    ):

        if (
            np.isfinite(value)
            and value > 0
        ):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value * 1.08,
                f"{int(value):,}",
                ha="center",
                va="bottom",
                fontsize=8,
                # rotation=90,
            )

    # ========================================================
    # 12. MEDIAN DISTANCE
    # ========================================================

    ax = axes[1, 1]

    median_distance = df[
        "median_distance_km"
    ]

    bars = draw_bars(
        ax=ax,
        values=median_distance,
        ylabel="Median distance (km)",
        title="(d) Median Distance Between Matched Tracks",
    )

    for bar, value in zip(
        bars,
        median_distance,
    ):

        if np.isfinite(value):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + 1,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # ========================================================
    # 13. SEPARATE 500 AND 850 hPa
    # ========================================================

    for ax in axes.flat:

        ax.axvline(
            3.5,
            linestyle="--",
            linewidth=1,
            color="gray",
            alpha=0.6,
        )

    # ========================================================
    # 14. LEGEND
    # ========================================================

    from matplotlib.patches import Patch

    legend_elements = [

        # Pressure levels
        Patch(
            facecolor="tab:blue",
            edgecolor="black",
            label="500 hPa",
        ),

        Patch(
            facecolor="tab:green",
            edgecolor="black",
            label="850 hPa",
        ),

        # Extraction
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch="///",
            label="Local extraction",
        ),

        Patch(
            facecolor="white",
            edgecolor="black",
            label="Global extraction",
        ),

        # Size filter
        Patch(
            facecolor="gray",
            edgecolor="black",
            alpha=0.35,
            label="10 points",
        ),

        Patch(
            facecolor="gray",
            edgecolor="black",
            alpha=1.0,
            label="25 points",
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            0.005,
        ),
        ncol=3,
        frameon=False,
        fontsize=10,
    )

    # ========================================================
    # 15. MAIN TITLE
    # ========================================================

    fig.suptitle(
        "TempestExtremes vs IBTrACS: "
        "Sensitivity to pressure level, "
        "size filter and extraction type",
        fontsize=15,
        y=0.98,
    )

    # ========================================================
    # 16. LAYOUT
    # ========================================================

    plt.tight_layout(
        rect=[
            0,
            0.09,
            1,
            0.95,
        ]
    )

    # ========================================================
    # 17. SAVE
    # ========================================================

    if output_filename is not None:

        output_dir = os.path.dirname(
            output_filename
        )

        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True,
            )

        plt.savefig(
            output_filename,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"Saved figure:\n"
            f"{output_filename}"
        )

    # ========================================================
    # 18. SHOW
    # ========================================================

    plt.show()

    return df

# ============================================================
# 9. RUN ONE CONFIGURATION
# ============================================================

def run_configuration(
    level_hPa,
    size_filter,
    extr_type,
    ibtracs_all,
):
    """
    Полностью обрабатывает одну конфигурацию.
    """

    config_name = (
        f"{level_hPa}hPa_"
        f"{size_filter}points"
        f"{extr_type}"
    )

    print("\n")
    print("=" * 80)
    print(
        f"CONFIGURATION: {config_name}"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    config_dir = (
        f"{OUTPUT_ROOT}/{config_name}"
    )

    tracks_output_dir = (
        f"{config_dir}/tracks"
    )

    os.makedirs(
        tracks_output_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load TE
    # --------------------------------------------------------

    te_tracks, tracks_by_year = (
        load_te_tracks(
            level_hPa=level_hPa,
            size_filter=size_filter,
            extr_type=extr_type,
            start_year=START_YEAR,
            end_year=END_YEAR,
        )
    )

    # --------------------------------------------------------
    # IBTrACS for this period
    # --------------------------------------------------------

    ibtracs = ibtracs_all.where(
        (
            ibtracs_all.time.dt.year
            >= START_YEAR
        )
        &
        (
            ibtracs_all.time.dt.year
            <= END_YEAR
        ),
        drop=True,
    )

    print(
        f"IBTrACS tracks: "
        f"{ibtracs.track_id.hrcn.nunique()}"
    )

    print(
        f"TE tracks: "
        f"{te_tracks.track_id.hrcn.nunique()}"
    )

    # --------------------------------------------------------
    # Matching
    # --------------------------------------------------------

    matches = calculate_matches(
        ibtracs_tracks=ibtracs,
        te_tracks=te_tracks,
    )

    # Save raw matching
    matches.to_csv(
        f"{config_dir}/matches_raw.csv",
        index=False,
    )

    # --------------------------------------------------------
    # POD
    # --------------------------------------------------------

    POD = calculate_pod(
        matches=matches,
        ibtracs_tracks=ibtracs,
    )

    print(
        f"POD = {POD:.4f}"
    )

    # --------------------------------------------------------
    # Track-level statistics
    # --------------------------------------------------------

    track_stats = (
        calculate_track_statistics(
            ibtracs_tracks=ibtracs,
            te_tracks=te_tracks,
            matches=matches,
        )
    )

    # --------------------------------------------------------
    # Save track statistics
    # --------------------------------------------------------

    track_stats.to_csv(
        f"{config_dir}/matching_statistics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    n_ibtracs = len(track_stats)

    n_matched = int(
        track_stats.detected.sum()
    )

    n_unmatched = (
        n_ibtracs
        - n_matched
    )

    detected = track_stats[
        track_stats.detected
    ]

    if len(detected) > 0:

        mean_overlap = (
            detected[
                "overlap_percent"
            ].mean()
        )

        median_overlap = (
            detected[
                "overlap_percent"
            ].median()
        )

        mean_temporal_overlap = (
            detected[
                "overlap_duration_percent"
            ].mean()
        )

        median_temporal_overlap = (
            detected[
                "overlap_duration_percent"
            ].median()
        )

        mean_distance = (
            detected[
                "mean_distance_km"
            ].mean()
        )

        median_distance = (
            detected[
                "mean_distance_km"
            ].median()
        )

    else:

        mean_overlap = np.nan
        median_overlap = np.nan

        mean_temporal_overlap = np.nan
        median_temporal_overlap = np.nan

        mean_distance = np.nan
        median_distance = np.nan

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)

    print(
        f"IBTrACS tracks : {n_ibtracs}"
    )

    print(
        f"Matched        : {n_matched}"
    )

    print(
        f"Unmatched      : {n_unmatched}"
    )

    print(
        f"POD            : {POD:.3f}"
    )

    print(
        f"Mean overlap   : "
        f"{mean_overlap:.1f}%"
    )

    print(
        f"Median overlap : "
        f"{median_overlap:.1f}%"
    )

    print(
        f"Mean temporal overlap   : "
        f"{mean_temporal_overlap:.1f}%"
    )

    print(
        f"Median temporal overlap : "
        f"{median_temporal_overlap:.1f}%"
    )

    print(
        f"Mean distance  : "
        f"{mean_distance:.1f} km"
    )

    # --------------------------------------------------------
    # Plot every IBTrACS track
    # --------------------------------------------------------

    print("\nCreating maps...")

    matched_ib_ids = set(
        matches["id_IBTrACS"].values
    )

    # Все IBTrACS tracks, а не только matched
    ib_ids = np.unique(
        ibtracs.track_id.values
    )

    for i, ib_id in enumerate(
        ib_ids,
        start=1,
    ):
    
        try:
    
            ib_track = ibtracs.where(
                ibtracs.track_id == ib_id,
                drop=True,
            ).sortby("time")
    
            year = int(
                pd.Timestamp(
                    ib_track.time.values[0]
                ).year
            )
    
            # ----------------------------------------------------
            # Найти все TE tracks,
            # соответствующие этому IBTrACS
            # ----------------------------------------------------
    
            matching_rows = matches[
                matches["id_IBTrACS"] == ib_id
            ]
    
            te_ids = set(
                matching_rows["id_TE_NA"].values
            )
    
            # ----------------------------------------------------
            # TE tracks для данного IBTrACS
            # ----------------------------------------------------
    
            if len(te_ids) > 0:
    
                te_tracks_for_ib = (
                    te_tracks.where(
                        te_tracks.track_id.isin(
                            list(te_ids)
                        ),
                        drop=True,
                    )
                )
    
            else:
    
                te_tracks_for_ib = None
    
            # ----------------------------------------------------
            # Filename
            # ----------------------------------------------------
    
            filename = (
                f"{tracks_output_dir}/"
                f"IBTrACS_{ib_id}_"
                f"{year}_track{i}.png"
            )
    
            # ----------------------------------------------------
            # Plot
            # ----------------------------------------------------
    
            plot_track_comparison(
                ibtracs_track=ib_track,
                te_tracks=te_tracks_for_ib,
                ibtracs_id=ib_id,
                year=year,
                output_filename=filename,
            )
    
            print(
                f"  [{i}/{len(ib_ids)}] "
                f"IBTrACS {ib_id}: "
                f"{len(te_ids)} TE matches"
            )
    
        except Exception as e:
    
            print(
                f"  ERROR plotting "
                f"IBTrACS {ib_id}: {e}"
            )
    
            continue

    # --------------------------------------------------------
    # Return summary
    # --------------------------------------------------------

    return {
        "level_hPa": level_hPa,
        "size_filter": size_filter,
        "extr_type": (
            extr_type
            if extr_type != ""
            else "local"
        ),

        "n_IBTrACS": n_ibtracs,
        "n_matched": n_matched,
        "n_unmatched": n_unmatched,

        "POD": POD,

        "mean_overlap_percent": (
            mean_overlap
        ),

        "median_overlap_percent": (
            median_overlap
        ),

        "mean_temporal_overlap_percent": (
            mean_temporal_overlap
        ),

        "median_temporal_overlap_percent": (
            median_temporal_overlap
        ),

        "mean_distance_km": (
            mean_distance
        ),

        "median_distance_km": (
            median_distance
        ),

        "n_TE_tracks": int(
            te_tracks.track_id.hrcn.nunique()
        ),
    }


# ============================================================
# 10. MAIN LOOP — ALL 8 CONFIGURATIONS
# ============================================================

# Загружаем IBTrACS ОДИН раз
ibtracs_all = load_ibtracs(
    start_year=START_YEAR,
    end_year=END_YEAR,
)



levels_hPa = [500, 850]
size_filters = [10, 25]
extr_types = ["_local", "_global"]

# size_filters = [25]
# extr_types = ["_global"]

configurations = list(
    product(
        levels_hPa,
        size_filters,
        extr_types,
    )
)


all_summary = []


for (
    level_hPa,
    size_filter,
    extr_type,
) in configurations:

    try:

        summary = run_configuration(
            level_hPa=level_hPa,
            size_filter=size_filter,
            extr_type=extr_type,
            ibtracs_all=ibtracs_all,
        )

        all_summary.append(
            summary
        )

    except Exception as e:

        print("\n")
        print("!" * 80)

        print(
            f"ERROR for configuration:"
            f" {level_hPa} hPa, "
            f"{size_filter} points, "
            f"{extr_type}"
        )

        print(e)

        print("!" * 80)

        continue


# ============================================================
# 11. SAVE SUMMARY OF ALL CONFIGURATIONS
# ============================================================

summary_df = pd.DataFrame(
    all_summary
)

summary_filename = (
    f"{OUTPUT_ROOT}/"
    f"summary_all_configurations.csv"
)

summary_df.to_csv(
    summary_filename,
    index=False,
)

summary_plot_filename = (
    f"{OUTPUT_ROOT}/"
    f"summary_all_configurations.png"
)

summary_df = plot_configuration_summary(
    summary_df,
    output_filename=summary_plot_filename,
)

print("\n")
print("=" * 80)
print("ALL CONFIGURATIONS FINISHED")
print("=" * 80)

print(
    summary_df.to_string(
        index=False
    )
)

print("\nSaved:")
print(summary_filename)
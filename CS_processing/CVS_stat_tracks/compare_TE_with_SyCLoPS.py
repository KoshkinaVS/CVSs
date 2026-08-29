import os
import numpy as np
import pandas as pd

import huracanpy
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

START_YEAR = 2010
END_YEAR = 2010

# TempestExtremes matching parameters
MAX_DIST = 200       # km
MEAN_DIST = 120      # km
MIN_OVERLAP = 3      # minimum number of matching points

path_dir_data = "/storage/thalassa/users/vkoshkina/data"

# Directory with the four SyCLoPS track files
SYCLoPS_ROOT = (
    f"{path_dir_data}/TempestExtremes/SyCLoPS"
)

# Output directory
OUTPUT_ROOT = (
    f"{path_dir_data}/TempestExtremes/"
    f"SyCLoPS_comparison_huracanpy"
)


# ============================================================
# TE CONFIGURATIONS
# ============================================================

levels_hPa = [500, 850]
size_filters = [10, 25]
extr_types = ["_local", "_global"]


# ============================================================
# SYCLOPS TYPES
# ============================================================

SYCLOPS_TYPES = [
    "TC",
    "STLC",
    "PTLC",
    "MS",
]


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
    """

    base_folder = (
        f"{path_dir_data}/TempestExtremes/ERA5/"
        f"R2D_ERA5_NA_for_TC_{level_hPa}hPa_sigma_2/"
        f"Tracks_R2D_txt_files_range_1_5_18h_12h_2010_"
        f"{size_filter}points{extr_type}"
    )

    all_tracks = []

    years = np.arange(
        start_year,
        end_year + 1,
    )

    for year in tqdm(
        years,
        desc=(
            f"Loading TE {level_hPa} hPa "
            f"{size_filter} points {extr_type}"
        ),
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

            all_tracks.append(tracks_year)

        except FileNotFoundError:

            print(
                f"Year {year} not found, skipping..."
            )

    if not all_tracks:
        raise FileNotFoundError(
            f"No TE tracks found in {base_folder}"
        )

    return huracanpy.concat_tracks(all_tracks)


# ============================================================
# 2. LOAD SYCLOPS TRACKS
# ============================================================

def load_syclops_tracks(
    syclops_type,
    start_year,
    end_year,
):
    """
    Загружает один тип SyCLoPS, сохранённый в формате
    TempestExtremes.

    Ожидается структура:

        SyCLoPS/
            TC/
                SyCLoPS_TC_tracks_2010.txt
            STLC/
                SyCLoPS_STLC_tracks_2010.txt
            PTLC/
                SyCLoPS_PTLC_tracks_2010.txt
            MS/
                SyCLoPS_MS_tracks_2010.txt
    """

    folder = (
        f"{SYCLoPS_ROOT}/{syclops_type}"
    )

    all_tracks = []

    years = np.arange(
        start_year,
        end_year + 1,
    )

    for year in years:

        filename = (
            f"{folder}/"
            f"SyCLoPS_{syclops_type}_tracks_{year}.txt"
        )

        try:

            tracks_year = huracanpy.load(
                filename,
                source="tempestextremes",
                variable_names=[
                    "MSLP",
                    "WS",
                ],
            )

            all_tracks.append(tracks_year)

        except FileNotFoundError:

            print(
                f"SyCLoPS {syclops_type}: "
                f"year {year} not found, skipping..."
            )

    if not all_tracks:
        raise FileNotFoundError(
            f"No SyCLoPS {syclops_type} tracks found "
            f"in {folder}"
        )

    return huracanpy.concat_tracks(all_tracks)


# ============================================================
# 3. MATCHING
# ============================================================

def calculate_matches(
    reference_tracks,
    te_tracks,
):
    """
    Matching SyCLoPS (reference) vs TE.

    SyCLoPS является reference, поэтому POD означает:

        доля SyCLoPS-треков, которые были обнаружены TE.
    """

    matches = huracanpy.assess.match(
        [
            reference_tracks,
            te_tracks,
        ],
        names=[
            "SyCLoPS",
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
    reference_tracks,
):
    """
    POD через huracanpy.
    """

    POD = huracanpy.assess.pod(
        matches,
        ref=reference_tracks,
        ref_name="SyCLoPS",
    )

    return float(POD)


# ============================================================
# 5. SELECT BEST MATCH
# ============================================================

def select_best_matches(matches):
    """
    Если одному SyCLoPS-треку соответствует несколько
    TE-треков, выбирается лучшее соответствие:

        1. максимальное количество matching points;
        2. при равенстве — минимальное расстояние.
    """

    if len(matches) == 0:
        return matches.copy()

    return (
        matches
        .sort_values(
            [
                "id_SyCLoPS",
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
            subset="id_SyCLoPS",
            keep="first",
        )
        .reset_index(drop=True)
    )


# ============================================================
# 6. TEMPORAL OVERLAP
# ============================================================

def calculate_temporal_overlap(
    reference_track,
    te_track,
):
    """
    Временное перекрытие двух треков.

    overlap_duration_fraction:
        длительность пересечения / длительность
        reference SyCLoPS track.
    """

    ref_times = pd.to_datetime(
        reference_track.time.values
    )

    te_times = pd.to_datetime(
        te_track.time.values
    )

    if len(ref_times) == 0 or len(te_times) == 0:

        return {
            "overlap_start": pd.NaT,
            "overlap_end": pd.NaT,
            "overlap_duration_hours": 0,
            "overlap_duration_fraction": 0,
        }

    ref_start = ref_times.min()
    ref_end = ref_times.max()

    te_start = te_times.min()
    te_end = te_times.max()

    ref_duration = (
        ref_end - ref_start
    ).total_seconds() / 3600

    overlap_start = max(
        ref_start,
        te_start,
    )

    overlap_end = min(
        ref_end,
        te_end,
    )

    if overlap_start > overlap_end:

        overlap_duration = 0

    else:

        overlap_duration = (
            overlap_end - overlap_start
        ).total_seconds() / 3600

    if ref_duration > 0:
        fraction = (
            overlap_duration
            / ref_duration
        )
    else:
        fraction = 0

    return {
        "overlap_start": overlap_start,
        "overlap_end": overlap_end,
        "overlap_duration_hours": overlap_duration,
        "overlap_duration_fraction": fraction,
    }


# ============================================================
# 7. TRACK-LEVEL STATISTICS
# ============================================================

def calculate_track_statistics(
    reference_tracks,
    te_tracks,
    matches,
):
    """
    Одна строка = один SyCLoPS track.

    Колонки:
        detected
        TE_ID_best
        matched_n_points
        overlap_percent
        mean_distance_km
        temporal overlap
    """

    reference_ids = np.unique(
        reference_tracks.track_id.values
    )

    rows = []

    best_matches = select_best_matches(matches)

    best_match_dict = {}

    for _, row in best_matches.iterrows():

        best_match_dict[
            row["id_SyCLoPS"]
        ] = row

    for ref_id in reference_ids:

        ref_track = reference_tracks.where(
            reference_tracks.track_id == ref_id,
            drop=True,
        ).sortby("time")

        ref_times = pd.to_datetime(
            ref_track.time.values
        )

        n_ref_points = len(ref_times)

        if n_ref_points > 1:

            ref_duration_hours = (
                ref_times.max()
                - ref_times.min()
            ).total_seconds() / 3600

        else:

            ref_duration_hours = 0

        # ----------------------------------------------------
        # No match
        # ----------------------------------------------------

        if ref_id not in best_match_dict:

            rows.append({
                "SyCLoPS_ID": ref_id,
                "detected": False,
                "TE_ID_best": np.nan,
                "SyCLoPS_n_points": n_ref_points,
                "SyCLoPS_duration_hours": ref_duration_hours,
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
        # Match
        # ----------------------------------------------------

        match_row = best_match_dict[ref_id]

        te_id = match_row["id_TE_NA"]

        matched_points = match_row["temp"]

        mean_distance = match_row["dist"]

        overlap_fraction = (
            matched_points / n_ref_points
            if n_ref_points > 0
            else np.nan
        )

        te_track = te_tracks.where(
            te_tracks.track_id == te_id,
            drop=True,
        ).sortby("time")

        temporal = calculate_temporal_overlap(
            ref_track,
            te_track,
        )

        rows.append({
            "SyCLoPS_ID": ref_id,
            "detected": True,
            "TE_ID_best": te_id,
            "SyCLoPS_n_points": n_ref_points,
            "SyCLoPS_duration_hours": ref_duration_hours,
            "matched_n_points": matched_points,
            "overlap_fraction": overlap_fraction,
            "overlap_percent": (
                100 * overlap_fraction
            ),
            "mean_distance_km": mean_distance,
            "overlap_start": temporal["overlap_start"],
            "overlap_end": temporal["overlap_end"],
            "overlap_duration_hours": (
                temporal["overlap_duration_hours"]
            ),
            "overlap_duration_fraction": (
                temporal["overlap_duration_fraction"]
            ),
            "overlap_duration_percent": (
                100
                * temporal["overlap_duration_fraction"]
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# 8. RUN ONE SYCLOPS TYPE
# ============================================================

def run_one_syclops_type(
    level_hPa,
    size_filter,
    extr_type,
    syclops_type,
    syclops_tracks,
):
    """
    Сравнивает одну конфигурацию TE с одним типом SyCLoPS.
    """

    config_name = (
        f"{level_hPa}hPa_"
        f"{size_filter}points"
        f"{extr_type}"
    )

    output_dir = (
        f"{OUTPUT_ROOT}/"
        f"{config_name}/"
        f"{syclops_type}"
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    print("\n" + "-" * 70)
    print(
        f"{config_name} vs SyCLoPS {syclops_type}"
    )
    print("-" * 70)

    # --------------------------------------------------------
    # Load TE
    # --------------------------------------------------------

    te_tracks = load_te_tracks(
        level_hPa=level_hPa,
        size_filter=size_filter,
        extr_type=extr_type,
        start_year=START_YEAR,
        end_year=END_YEAR,
    )

    n_ref = int(
        syclops_tracks.track_id.hrcn.nunique()
    )

    n_te = int(
        te_tracks.track_id.hrcn.nunique()
    )

    print(f"SyCLoPS tracks: {n_ref}")
    print(f"TE tracks:      {n_te}")

    # --------------------------------------------------------
    # Matching
    # --------------------------------------------------------

    matches = calculate_matches(
        reference_tracks=syclops_tracks,
        te_tracks=te_tracks,
    )

    matches.to_csv(
        f"{output_dir}/matches_raw.csv",
        index=False,
    )

    # --------------------------------------------------------
    # POD
    # --------------------------------------------------------

    POD = calculate_pod(
        matches=matches,
        reference_tracks=syclops_tracks,
    )

    # --------------------------------------------------------
    # Track statistics
    # --------------------------------------------------------

    track_stats = calculate_track_statistics(
        reference_tracks=syclops_tracks,
        te_tracks=te_tracks,
        matches=matches,
    )

    track_stats.to_csv(
        f"{output_dir}/matching_statistics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    detected = track_stats[
        track_stats["detected"]
    ]

    n_matched = len(detected)

    n_unmatched = n_ref - n_matched

    if len(detected) > 0:

        median_overlap = detected[
            "overlap_percent"
        ].median()

        median_temporal_overlap = detected[
            "overlap_duration_percent"
        ].median()

        median_distance = detected[
            "mean_distance_km"
        ].median()

        mean_overlap = detected[
            "overlap_percent"
        ].mean()

        mean_temporal_overlap = detected[
            "overlap_duration_percent"
        ].mean()

        mean_distance = detected[
            "mean_distance_km"
        ].mean()

    else:

        median_overlap = np.nan
        median_temporal_overlap = np.nan
        median_distance = np.nan

        mean_overlap = np.nan
        mean_temporal_overlap = np.nan
        mean_distance = np.nan

    print(f"POD:                    {POD:.4f}")
    print(f"Matched SyCLoPS:        {n_matched}")
    print(f"Unmatched SyCLoPS:      {n_unmatched}")
    print(f"Median overlap:         {median_overlap:.1f}%")
    print(
        "Median temporal overlap: "
        f"{median_temporal_overlap:.1f}%"
    )
    print(
        f"Median distance:        {median_distance:.1f} km"
    )

    return {
        "level_hPa": level_hPa,
        "size_filter": size_filter,
        "extr_type": (
            extr_type
            if extr_type
            else "local"
        ),
        "SyCLoPS_type": syclops_type,
        "n_SyCLoPS_tracks": n_ref,
        "n_matched": n_matched,
        "n_unmatched": n_unmatched,
        "n_TE_tracks": n_te,
        "POD": POD,
        "mean_overlap_percent": mean_overlap,
        "median_overlap_percent": median_overlap,
        "mean_temporal_overlap_percent": (
            mean_temporal_overlap
        ),
        "median_temporal_overlap_percent": (
            median_temporal_overlap
        ),
        "mean_distance_km": mean_distance,
        "median_distance_km": median_distance,
    }


# ============================================================
# 9. MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load SyCLoPS ONCE
    # --------------------------------------------------------

    print("=" * 80)
    print("LOADING SyCLoPS")
    print("=" * 80)

    syclops_all = {}

    for syclops_type in SYCLOPS_TYPES:

        syclops_all[syclops_type] = (
            load_syclops_tracks(
                syclops_type=syclops_type,
                start_year=START_YEAR,
                end_year=END_YEAR,
            )
        )

        n_tracks = int(
            syclops_all[
                syclops_type
            ].track_id.hrcn.nunique()
        )

        print(
            f"{syclops_type:5s}: "
            f"{n_tracks} tracks"
        )

    # --------------------------------------------------------
    # Run all 8 × 4 comparisons
    # --------------------------------------------------------

    configurations = [
        (level, size, extr)
        for level in levels_hPa
        for size in size_filters
        for extr in extr_types
    ]

    all_summary = []

    for (
        level_hPa,
        size_filter,
        extr_type,
    ) in configurations:

        for syclops_type in SYCLOPS_TYPES:

            print("\n" + "=" * 80)
            print(
                f"CONFIGURATION: "
                f"{level_hPa} hPa | "
                f"{size_filter} points | "
                f"{extr_type or 'local'} | "
                f"SyCLoPS {syclops_type}"
            )
            print("=" * 80)

            try:

                summary = run_one_syclops_type(
                    level_hPa=level_hPa,
                    size_filter=size_filter,
                    extr_type=extr_type,
                    syclops_type=syclops_type,
                    syclops_tracks=syclops_all[
                        syclops_type
                    ],
                )

                all_summary.append(summary)

            except Exception as e:

                print("\n" + "!" * 80)
                print(
                    f"ERROR: "
                    f"{level_hPa} hPa | "
                    f"{size_filter} points | "
                    f"{extr_type} | "
                    f"SyCLoPS {syclops_type}"
                )
                print(e)
                print("!" * 80)

                continue

    # --------------------------------------------------------
    # Save global summary
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        all_summary
    )

    summary_filename = (
        f"{OUTPUT_ROOT}/"
        f"summary_syclops_all_configurations.csv"
    )

    summary_df.to_csv(
        summary_filename,
        index=False,
    )

    print("\n" + "=" * 80)
    print("ALL SyCLoPS COMPARISONS FINISHED")
    print("=" * 80)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(summary_filename)


if __name__ == "__main__":
    main()

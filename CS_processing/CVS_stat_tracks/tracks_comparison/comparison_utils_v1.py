import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import huracanpy
from tqdm import tqdm

PATH_DIR_DATA = "/storage/thalassa/users/vkoshkina/data"
TE_ROOT = f"{PATH_DIR_DATA}/TempestExtremes/ERA5"
SYCLOPS_ROOT = f"{PATH_DIR_DATA}/TempestExtremes/SyCLoPS"
IBTRACS_OUTPUT_ROOT = f"{PATH_DIR_DATA}/TempestExtremes/ERA5/TC_comparison_huracanpy"
SYCLOPS_OUTPUT_ROOT = f"{PATH_DIR_DATA}/TempestExtremes/ERA5/SyCLoPS_comparison_huracanpy"

MAX_DIST = 200
MEAN_DIST = 120
MIN_OVERLAP = 3
LEVELS_HPA = [500, 850]
SIZE_FILTERS = [10, 25]
EXTR_TYPES = ["_local", "_global"]
SYCLOPS_TYPES = ["TC", "STLC", "PTLC", "MS"]


def configuration_name(level_hPa, size_filter, extr_type):
    return f"{level_hPa}hPa_{size_filter}points{extr_type}"


def configuration_list():
    return [(l, s, e) for l in LEVELS_HPA for s in SIZE_FILTERS for e in EXTR_TYPES]


def load_te_tracks(level_hPa, size_filter, extr_type, start_year, end_year):
    folder = (
        f"{TE_ROOT}/R2D_ERA5_NA_for_TC_{level_hPa}hPa_sigma_2/"
        f"Tracks_R2D_txt_files_range_1_5_18h_12h_2010_{size_filter}points{extr_type}"
    )
    tracks = []
    for year in tqdm(range(start_year, end_year + 1), desc=f"Loading TE {level_hPa} {size_filter} {extr_type}"):
        filename = f"{folder}/ERA5_TC_tracks_{year}.txt"
        try:
            tracks.append(huracanpy.load(
                filename, source="tempestextremes",
                variable_names=["rad", "r2d", "wspd"]
            ))
        except FileNotFoundError:
            print(f"TE: {year} not found, skipping...")
    if not tracks:
        raise FileNotFoundError(f"No TE tracks found in {folder}")
    return huracanpy.concat_tracks(tracks)


def load_ibtracs(start_year, end_year):
    tracks = huracanpy.load(source="ibtracs")
    tracks = tracks.where(
        (tracks.time.dt.year >= start_year) &
        (tracks.time.dt.year <= end_year), drop=True
    )
    return tracks.where(tracks.basin == "NA", drop=True)


def load_syclops_tracks(syclops_type, start_year, end_year):
    folder = f"{SYCLOPS_ROOT}/{syclops_type}"
    tracks = []
    for year in range(start_year, end_year + 1):
        filename = f"{folder}/SyCLoPS_{syclops_type}_tracks_{year}.txt"
        try:
            tracks.append(huracanpy.load(
                filename, source="tempestextremes",
                variable_names=["MSLP", "WS"]
            ))
        except FileNotFoundError:
            print(f"SyCLoPS {syclops_type}: {year} not found, skipping...")
    if not tracks:
        raise FileNotFoundError(f"No SyCLoPS {syclops_type} tracks found in {folder}")
    return huracanpy.concat_tracks(tracks)


def get_unique_track_ids(tracks):
    return np.unique(tracks.track_id.values)


def number_of_tracks(tracks):
    return int(tracks.track_id.hrcn.nunique())


def calculate_matches(reference_tracks, te_tracks, reference_name):
    return huracanpy.assess.match(
        [reference_tracks, te_tracks],
        names=[reference_name, "TE_NA"],
        max_dist=MAX_DIST, mean_dist=MEAN_DIST,
        min_overlap=MIN_OVERLAP, tracks1_is_ref=True,
    )


def calculate_pod(matches, reference_tracks, reference_name):
    return float(huracanpy.assess.pod(
        matches, ref=reference_tracks, ref_name=reference_name
    ))


def select_best_matches(matches, reference_id_column):
    if len(matches) == 0:
        return matches.copy()
    return (
        matches.sort_values(
            [reference_id_column, "temp", "dist"],
            ascending=[True, False, True]
        )
        .drop_duplicates(subset=reference_id_column, keep="first")
        .reset_index(drop=True)
    )


def calculate_temporal_overlap(reference_track, te_track):
    ref = pd.to_datetime(reference_track.time.values)
    te = pd.to_datetime(te_track.time.values)
    if len(ref) == 0 or len(te) == 0:
        return {"overlap_start": pd.NaT, "overlap_end": pd.NaT,
                "overlap_duration_hours": 0, "overlap_duration_fraction": 0}
    ref_start, ref_end = ref.min(), ref.max()
    te_start, te_end = te.min(), te.max()
    duration = (ref_end - ref_start).total_seconds() / 3600
    start, end = max(ref_start, te_start), min(ref_end, te_end)
    overlap = 0 if start > end else (end - start).total_seconds() / 3600
    fraction = overlap / duration if duration > 0 else 0
    return {"overlap_start": start, "overlap_end": end,
            "overlap_duration_hours": overlap,
            "overlap_duration_fraction": fraction}


def calculate_track_statistics(reference_tracks, te_tracks, matches,
                                reference_id_column, te_id_column="id_TE_NA"):
    reference_ids = get_unique_track_ids(reference_tracks)
    best = select_best_matches(matches, reference_id_column)
    best_dict = {r[reference_id_column]: r for _, r in best.iterrows()}
    rows = []

    for ref_id in reference_ids:
        ref_track = reference_tracks.where(
            reference_tracks.track_id == ref_id, drop=True
        ).sortby("time")
        times = pd.to_datetime(ref_track.time.values)
        n = len(times)
        duration = ((times.max() - times.min()).total_seconds() / 3600) if n > 1 else 0

        if ref_id not in best_dict:
            rows.append({
                "reference_ID": ref_id, "detected": False,
                "TE_ID_best": np.nan, "reference_n_points": n,
                "reference_duration_hours": duration,
                "matched_n_points": 0, "overlap_fraction": 0,
                "overlap_percent": 0, "mean_distance_km": np.nan,
                "overlap_start": pd.NaT, "overlap_end": pd.NaT,
                "overlap_duration_hours": 0,
                "overlap_duration_fraction": 0,
                "overlap_duration_percent": 0
            })
            continue

        row = best_dict[ref_id]
        te_id = row[te_id_column]
        matched = row["temp"]
        dist = row["dist"]
        frac = matched / n if n else np.nan
        te_track = te_tracks.where(
            te_tracks.track_id == te_id, drop=True
        ).sortby("time")
        temporal = calculate_temporal_overlap(ref_track, te_track)

        rows.append({
            "reference_ID": ref_id, "detected": True,
            "TE_ID_best": te_id, "reference_n_points": n,
            "reference_duration_hours": duration,
            "matched_n_points": matched,
            "overlap_fraction": frac,
            "overlap_percent": 100 * frac,
            "mean_distance_km": dist,
            "overlap_start": temporal["overlap_start"],
            "overlap_end": temporal["overlap_end"],
            "overlap_duration_hours": temporal["overlap_duration_hours"],
            "overlap_duration_fraction": temporal["overlap_duration_fraction"],
            "overlap_duration_percent": 100 * temporal["overlap_duration_fraction"]
        })

    return pd.DataFrame(rows)


def summarize_track_statistics(track_stats, pod, n_reference, n_te):
    d = track_stats[track_stats.detected]
    if len(d):
        vals = {
            "mean_overlap_percent": d.overlap_percent.mean(),
            "median_overlap_percent": d.overlap_percent.median(),
            "mean_temporal_overlap_percent": d.overlap_duration_percent.mean(),
            "median_temporal_overlap_percent": d.overlap_duration_percent.median(),
            "mean_distance_km": d.mean_distance_km.mean(),
            "median_distance_km": d.mean_distance_km.median(),
        }
    else:
        vals = {k: np.nan for k in [
            "mean_overlap_percent", "median_overlap_percent",
            "mean_temporal_overlap_percent", "median_temporal_overlap_percent",
            "mean_distance_km", "median_distance_km"
        ]}
    return {
        "n_reference_tracks": n_reference,
        "n_matched": len(d),
        "n_unmatched": n_reference - len(d),
        "n_TE_tracks": n_te,
        "POD": pod,
        **vals
    }


def plot_reference_track(reference_track, te_tracks, reference_id,
                          reference_label, output_filename, year=None,
                          title_extra=""):
    fig = plt.figure(figsize=(12, 8), dpi=150)
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent([-110, 15, 0, 73], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
    ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)

    # Reference is deliberately drawn first: TE is on top.
    ax.plot(reference_track.lon, reference_track.lat,
            color="black", linewidth=3, zorder=2,
            transform=ccrs.PlateCarree(), label=reference_label)
    ax.plot(reference_track.lon[0], reference_track.lat[0],
            "go", markersize=8, zorder=2,
            transform=ccrs.PlateCarree(), label="Genesis")
    ax.plot(reference_track.lon[-1], reference_track.lat[-1],
            "rs", markersize=8, zorder=2,
            transform=ccrs.PlateCarree(), label="Lysis")

    te_ids = [] if te_tracks is None else get_unique_track_ids(te_tracks)

    # Same-family warm colors for multiple TE tracks.
    te_colors = [
        "#FF1493", "#FF0000", "#CC00FF", "#FF4500", "#FFA500",
        "#FF69B4", "#AD1457", "#8B008B", "#FF6347", "#FFD700"
    ]

    for i, te_id in enumerate(te_ids):
        te_track = te_tracks.where(
            te_tracks.track_id == te_id, drop=True
        ).sortby("time")
        ax.plot(te_track.lon, te_track.lat,
                color=te_colors[i % len(te_colors)],
                linewidth=1.8, alpha=0.9, zorder=5,
                transform=ccrs.PlateCarree(), label=f"TE {te_id}")

    title = f"{reference_label} {reference_id}"
    if year is not None:
        title += f" - {year}"
    title += f"\nBlack: {reference_label} | Colored: TempestExtremes ({len(te_ids)} tracks)"
    if title_extra:
        title += f"\n{title_extra}"
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_all_reference_tracks(reference_tracks, te_tracks, output_dir,
                              reference_label, reference_id_column, matches):
    os.makedirs(output_dir, exist_ok=True)
    ids = get_unique_track_ids(reference_tracks)
    matched_ids = set(matches[reference_id_column].values) if len(matches) else set()

    for i, ref_id in enumerate(ids, start=1):
        ref_track = reference_tracks.where(
            reference_tracks.track_id == ref_id, drop=True
        ).sortby("time")
        year = int(pd.Timestamp(ref_track.time.values[0]).year)

        if ref_id in matched_ids:
            rows = matches[matches[reference_id_column] == ref_id]
            te_ids = np.unique(rows["id_TE_NA"].values)
            te_for_ref = te_tracks.where(
                te_tracks.track_id.isin(list(te_ids)), drop=True
            )
        else:
            te_for_ref = None

        filename = (
            f"{output_dir}/{reference_label}_{ref_id}_{year}_track{i}.png"
        )
        plot_reference_track(
            ref_track, te_for_ref, ref_id, reference_label,
            filename, year=year
        )


def plot_configuration_summary(summary_df, reference_type, output_filename):
    df = summary_df.copy()

    if reference_type.lower() == "ibtracs":
        ncol = "n_IBTrACS" if "n_IBTrACS" in df else "n_reference_tracks"
    else:
        ncol = "n_SyCLoPS_tracks" if "n_SyCLoPS_tracks" in df else "n_reference_tracks"

    df["reference_n"] = df[ncol]
    df["TE_tracks_per_reference"] = df["n_TE_tracks"] / df["reference_n"]
    df = df.sort_values(["level_hPa", "size_filter", "extr_type"])
    x = np.arange(len(df))

    colors = []
    for _, r in df.iterrows():
        base = np.array([31,119,180]) / 255 if int(r.level_hPa) == 500 else np.array([44,160,44]) / 255
        colors.append(0.45 * base + 0.55 * np.ones(3) if int(r.size_filter) == 10 else base)

    hatches = ["///" if str(r.extr_type).lower() == "local" else "" for _, r in df.iterrows()]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    metrics = [
        ("POD", "POD"),
        ("median_temporal_overlap_percent", "Median temporal overlap (%)"),
        ("TE_tracks_per_reference", "TE tracks / reference track"),
        ("median_distance_km", "Median distance (km)")
    ]

    labels = [
        f"{int(r.level_hPa)}\n{int(r.size_filter)}{'L' if str(r.extr_type).lower()=='local' else 'G'}"
        for _, r in df.iterrows()
    ]

    for ax, (col, ylabel) in zip(axes.flat, metrics):
        for i in range(len(df)):
            ax.bar(x[i], df.iloc[i][col], color=colors[i],
                   hatch=hatches[i], edgecolor="black",
                   linewidth=0.8, width=0.72)
        ax.set_ylabel(ylabel)
        ymax = df[col].max()
        if np.isfinite(ymax) and ymax > 0:
            ax.set_ylim(0, ymax * 1.15)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        for tick, (_, r) in zip(ax.get_xticklabels(), df.iterrows()):
            if int(r.size_filter) == 25:
                tick.set_fontweight("bold")
            else:
                tick.set_alpha(0.45)

    fig.suptitle(f"TE vs {reference_type} — configuration comparison", fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename, dpi=200, bbox_inches="tight")
    plt.close()

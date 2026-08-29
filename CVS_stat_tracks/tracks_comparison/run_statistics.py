import os
import argparse
import pandas as pd

# python run_statistics.py \
#     --reference ibtracs \
#     --start-year 2010 \
#     --end-year 2010

# python run_statistics.py \
#     --reference syclops \
#     --syclops-type all \
#     --start-year 2010 \
#     --end-year 2010


from comparison_utils import (
    configuration_list, configuration_name,
    load_te_tracks, load_ibtracs, load_syclops_tracks,
    calculate_matches, calculate_pod, calculate_track_statistics,
    summarize_track_statistics, number_of_tracks,
    IBTRACS_OUTPUT_ROOT, SYCLOPS_OUTPUT_ROOT, SYCLOPS_TYPES,
)


def run_one(level, size, extr, reference_tracks, reference_name,
            reference_id_column, output_root, start_year, end_year):
    cfg = configuration_name(level, size, extr)
    out = f"{output_root}/{cfg}"
    os.makedirs(out, exist_ok=True)

    print(f"\n{'='*80}\n{reference_name} vs TE: {cfg}\n{'='*80}")

    te = load_te_tracks(level, size, extr, start_year, end_year)
    matches = calculate_matches(reference_tracks, te, reference_name)

    matches.to_csv(f"{out}/matches_raw.csv", index=False)

    pod = calculate_pod(matches, reference_tracks, reference_name)

    stats = calculate_track_statistics(
        reference_tracks, te, matches, reference_id_column
    )

    # Keep both generic and familiar names.
    if reference_name == "IBTrACS":
        stats["IBTrACS_ID"] = stats["reference_ID"]
        stats["IBTrACS_n_points"] = stats["reference_n_points"]
        stats["IBTrACS_duration_hours"] = stats["reference_duration_hours"]
    else:
        stats["SyCLoPS_ID"] = stats["reference_ID"]
        stats["SyCLoPS_n_points"] = stats["reference_n_points"]
        stats["SyCLoPS_duration_hours"] = stats["reference_duration_hours"]

    stats.to_csv(f"{out}/matching_statistics.csv", index=False)

    summary = summarize_track_statistics(
        stats, pod, number_of_tracks(reference_tracks),
        number_of_tracks(te)
    )
    summary.update({
        "level_hPa": level,
        "size_filter": size,
        "extr_type": extr if extr else "local",
    })

    print(f"POD = {pod:.4f}")
    print(f"Matched = {summary['n_matched']}/{summary['n_reference_tracks']}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, choices=["ibtracs", "syclops"])
    p.add_argument("--syclops-type", default="all",
                   choices=["TC", "STLC", "PTLC", "MS", "all"])
    p.add_argument("--start-year", type=int, default=2010)
    p.add_argument("--end-year", type=int, default=2010)
    args = p.parse_args()

    if args.reference == "ibtracs":
        ref = load_ibtracs(args.start_year, args.end_year)
        root = IBTRACS_OUTPUT_ROOT
        rows = []
        for level, size, extr in configuration_list():
            try:
                rows.append(run_one(
                    level, size, extr, ref, "IBTrACS", "id_IBTrACS",
                    root, args.start_year, args.end_year
                ))
            except Exception as e:
                print(f"ERROR {level} {size} {extr}: {e}")
        pd.DataFrame(rows).to_csv(
            f"{root}/summary_all_configurations.csv", index=False
        )

    else:
        types = SYCLOPS_TYPES if args.syclops_type == "all" else [args.syclops_type]
        for typ in types:
            ref = load_syclops_tracks(typ, args.start_year, args.end_year)
            root = f"{SYCLOPS_OUTPUT_ROOT}/{typ}"
            rows = []
            for level, size, extr in configuration_list():
                try:
                    row = run_one(
                        level, size, extr, ref, "SyCLoPS", "id_SyCLoPS",
                        root, args.start_year, args.end_year
                    )
                    row["SyCLoPS_type"] = typ
                    rows.append(row)
                except Exception as e:
                    print(f"ERROR {typ} {level} {size} {extr}: {e}")
            pd.DataFrame(rows).to_csv(
                f"{root}/summary_all_configurations.csv", index=False
            )


if __name__ == "__main__":
    main()

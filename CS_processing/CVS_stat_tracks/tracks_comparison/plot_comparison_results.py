import os
import argparse
import pandas as pd

# python plot_comparison_results.py \
#     --reference syclops \
#     --syclops-type all \
#     --start-year 2010 \
#     --end-year 2010

from comparison_utils import (
    configuration_list, load_ibtracs, load_syclops_tracks, load_te_tracks,
    plot_all_reference_tracks, plot_configuration_summary,
    SYCLOPS_TYPES, IBTRACS_OUTPUT_ROOT, SYCLOPS_OUTPUT_ROOT,
)


def plot_dataset(reference_type, start_year, end_year, syclops_type=None):
    if reference_type == "ibtracs":
        reference = load_ibtracs(start_year, end_year)
        label = "IBTrACS"
        id_col = "id_IBTrACS"
        root = IBTRACS_OUTPUT_ROOT
        summary_name = "summary_all_configurations.csv"
    else:
        reference = load_syclops_tracks(syclops_type, start_year, end_year)
        label = f"SyCLoPS_{syclops_type}"
        id_col = "id_SyCLoPS"
        root = f"{SYCLOPS_OUTPUT_ROOT}/{syclops_type}"
        summary_name = "summary_all_configurations.csv"

    # --------------------------------------------------------
    # Maps: every reference track, all matched TE tracks
    # --------------------------------------------------------
    for level, size, extr in configuration_list():
        cfg = f"{level}hPa_{size}points{extr}"
        cfg_dir = f"{root}/{cfg}"
        raw_file = f"{cfg_dir}/matches_raw.csv"

        if not os.path.exists(raw_file):
            print(f"Missing: {raw_file}")
            continue

        matches = pd.read_csv(raw_file)

        te = load_te_tracks(
            level, size, extr, start_year, end_year
        )

        plot_all_reference_tracks(
            reference_tracks=reference,
            te_tracks=te,
            output_dir=f"{cfg_dir}/tracks",
            reference_label=label,
            reference_id_column=id_col,
            matches=matches,
        )

    # --------------------------------------------------------
    # One 4-panel statistical figure
    # --------------------------------------------------------
    summary_file = f"{root}/{summary_name}"
    if os.path.exists(summary_file):
        df = pd.read_csv(summary_file)
        plot_configuration_summary(
            df, label,
            f"{root}/configuration_summary.png"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, choices=["ibtracs", "syclops"])
    p.add_argument("--syclops-type", default="all",
                   choices=["TC", "STLC", "PTLC", "MS", "all"])
    p.add_argument("--start-year", type=int, default=2010)
    p.add_argument("--end-year", type=int, default=2010)
    args = p.parse_args()

    if args.reference == "ibtracs":
        plot_dataset("ibtracs", args.start_year, args.end_year)
    else:
        types = SYCLOPS_TYPES if args.syclops_type == "all" else [args.syclops_type]
        for typ in types:
            plot_dataset("syclops", args.start_year, args.end_year, typ)


if __name__ == "__main__":
    main()

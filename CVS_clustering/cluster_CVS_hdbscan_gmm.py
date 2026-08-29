#!/usr/bin/env python3
"""
Cluster mesoscale vortex maxima (CVS) using HDBSCAN (if present) and GMM for comparison.
Scans CSV files with pattern max_crit_day_CVS_YYYY-MM.csv inside a directory tree,
reads them, concatenates into a single DataFrame, selects features (excluding geometry/time),
scales, optionally reduces dimensionality, runs clustering, and saves outputs:
 - concatenated CSV with new columns: hdbscan_label (if used), gmm_label, gmm_prob_max
 - UMAP 2D scatter plot for each clustering
 - cluster summary CSVs (means, counts)
 - diagnostics (silhouette, cluster sizes)
"""

import os
import glob
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Optional imports
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# ----------------- User parameters (can be overridden by CLI) -----------------

data_type = 'LoRes'
path_init = f'/storage/thalassa/users/vkoshkina'
DEFAULT_DATA_DIR = f"{path_init}/data/TC_tracks/{data_type}/{data_type}_sigma_2/max_crit_day_data_ocean"

FILE_GLOB = "max_crit_day_CVS_*.csv"
OUTPUT_DIR = f"{path_init}/data/TC_tracks/{data_type}/{data_type}_sigma_2/outputs_CVS_clustering"
FEATURES_DEFAULT = [
    # typical dynamic / thermodynamic features from your preview
    "rad", "crit", "track_len", "msl_min", "mslhf_max",
    "pvo_max", "helicity_max", "updraft_helicity_max",
    "pw_max", "slp_delta", "duration", "vel", "wspd_max"
]
EXCLUDE_COLS = {"datetime", "x", "y", "lat", "lon", "path", "name", "start_stop"}


if data_type == 'LoRes':
    time_th = 8
    t_up = 'theta_median'
    t_surf = 't2_median'
    lh = 'mslhf_max'
    sh = 'msshf_max'

elif data_type == 'ERA5':
    time_th = 24
    t_up = 't_median'
    t_surf = 't2m_median'
    lh = 'mlhf_max'
    sh = 'mshf_max'
    
numeric_cols = [
    'rad', 'crit', 
    'track_len',
    'duration',
    'vel',
    'msl_min', 'wspd_max', 
    lh, sh, 
    t_surf, 'dT',
    'w_median'
]

if data_type == 'LoRes':
    numeric_cols = [
        'rad', 'crit', 
        'track_len',
        'duration',
        'vel',
        'msl_min', 'wspd_max', 
        lh, sh, 
        t_surf, 'dT',
        'w_median',
        'cape_2d_max', 
        'pvo_max', 'helicity_max', 
        'updraft_helicity_max', 'pw_max', 
        'slp_delta'
    ]
elif data_type == 'ERA5':
    numeric_cols.append('tp_median')

FEATURES_DEFAULT = numeric_cols

# -----------------------------------------------------------------------------
def find_csv_files(base_dir, pattern=FILE_GLOB):
    p = Path(base_dir)
    files = sorted([str(f) for f in p.rglob(pattern)])
    return files

def read_and_concat(files, verbose=True):
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["_source_file"] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: failed to read {f}: {e}")
    if not dfs:
        raise RuntimeError("No CSV files read. Check path/pattern.")
    big = pd.concat(dfs, ignore_index=True)
    if verbose:
        print(f"Concatenated {len(dfs)} files -> total rows: {len(big)}")
    return big

def select_features(df, requested_features=None):
    # Remove excluded columns automatically
    cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    # Keep numeric candidates
    numcols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if requested_features:
        # prefer intersection of requested_features and available numeric columns
        feats = [c for c in requested_features if c in numcols]
    else:
        feats = numcols
    # fallback: if none found, use top variance numeric columns
    if not feats:
        variances = pd.Series({c: df[c].var() for c in numcols}).sort_values(ascending=False)
        feats = variances.index.tolist()[:12]
    return feats

def clean_and_scale(df, features, dropna_thresh=0.5):
    # Drop columns with too many NaNs (more than dropna_thresh fraction)
    keep = []
    n = len(df)
    for c in features:
        if df[c].isna().sum() / n > (1.0 - dropna_thresh):
            print(f"Column {c} has many NaNs; dropping from features.")
        else:
            keep.append(c)
    features = keep
    # Fill remaining NaNs by column median (document this!)
    df_f = df[features].copy()
    med = df_f.median()
    df_f = df_f.fillna(med)
    scaler = StandardScaler()
    X = scaler.fit_transform(df_f.values)
    return X, features, scaler

def run_hdbscan(X, min_cluster_size=50, min_samples=None):
    if not HDBSCAN_AVAILABLE:
        print("HDBSCAN not available in the environment.")
        return None
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, prediction_data=True)
    labels = clusterer.fit_predict(X)
    return clusterer, labels

def run_gmm(X, n_components=7, random_state=0):
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=random_state)
    gmm.fit(X)
    labels = gmm.predict(X)
    probs = gmm.predict_proba(X).max(axis=1)
    return gmm, labels, probs

def compute_and_save_summaries(df, label_col, features, out_prefix):
    grp = df.groupby(label_col)
    summary = grp[features].mean().rename(columns=lambda c: f"mean_{c}")
    counts = grp.size().rename("count")
    summary = summary.join(counts)
    summary_path = f"{out_prefix}_cluster_summary.csv"
    summary.to_csv(summary_path)
    print(f"Wrote cluster summary to {summary_path}")
    return summary_path

def plot_umap(X, labels, outpath, title=None):
    if not UMAP_AVAILABLE:
        print("UMAP not installed; skipping UMAP plot.")
        return None
    reducer = umap.UMAP(n_components=2, random_state=0)
    emb = reducer.fit_transform(X)
    plt.figure(figsize=(8,6))
    unique = np.unique(labels)
    # Color -1 (noise) separately if present
    for lab in unique:
        mask = labels == lab
        plt.scatter(emb[mask,0], emb[mask,1], s=6, label=str(lab), alpha=0.6)
    plt.legend(markerscale=2, fontsize="small", ncol=2, bbox_to_anchor=(1.05,1))
    plt.title(title or "UMAP projection of features")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"Saved UMAP scatter to {outpath}")
    return outpath

def main(args):
    os.makedirs(args.output, exist_ok=True)
    files = find_csv_files(args.data_dir, pattern=args.pattern)
    if not files:
        raise RuntimeError(f"No files found with pattern {args.pattern} in {args.data_dir}")
    df = read_and_concat(files)
    # Try to ensure year/month present
    if "year" not in df.columns:
        if "datetime" in df.columns:
            try:
                df["event_datetime"] = pd.to_datetime(df["datetime"])
                df["year"] = df["event_datetime"].dt.year
                df["month"] = df["event_datetime"].dt.month
            except Exception:
                pass
    # Feature selection
    feats = select_features(df, requested_features=args.features)
    print("Selected features for clustering:", feats)
    # Clean/scale
    X, used_feats, scaler = clean_and_scale(df, feats, dropna_thresh=args.dropna_thresh)
    print(f"Final features used ({len(used_feats)}): {used_feats}")
    # Optional PCA pre-reduction (helpful for HDBSCAN/GMM if many features)
    if args.pca_components and args.pca_components < X.shape[1]:
        pca = PCA(n_components=args.pca_components, random_state=0)
        Xred = pca.fit_transform(X)
        print(f"PCA reduced from {X.shape[1]} to {Xred.shape[1]} components (explained var: {pca.explained_variance_ratio_.sum():.3f})")
    else:
        Xred = X

    # Run HDBSCAN if asked
    if args.use_hdbscan:
        if not HDBSCAN_AVAILABLE:
            print("HDBSCAN not installed; skipping HDBSCAN stage.")
            hdbscan_labels = None
            hdbscan_clusterer = None
        else:
            hdbscan_clusterer, hdbscan_labels = run_hdbscan(Xred, min_cluster_size=args.hdbscan_min_cluster_size, min_samples=args.hdbscan_min_samples)
            print("HDBSCAN cluster sizes:", pd.Series(hdbscan_labels).value_counts().sort_index().to_dict())
            # attach labels to df
            df["hdbscan_label"] = hdbscan_labels
            # silhouette (exclude noise -1)
            mask = hdbscan_labels != -1
            if mask.sum() >= 10:
                s = silhouette_score(Xred[mask], hdbscan_labels[mask])
                print(f"HDBSCAN silhouette (non-noise): {s:.3f}")
            else:
                print("Too few non-noise points for silhouette score calculation.")
            # save summary
            compute_and_save_summaries(df[df["hdbscan_label"]!=-1], "hdbscan_label", used_feats, os.path.join(args.output, "hdbscan"))

            # UMAP plot
            if UMAP_AVAILABLE:
                plot_umap(Xred, hdbscan_labels, os.path.join(args.output, "hdbscan_umap.png"), title="HDBSCAN clusters (UMAP)")
    else:
        hdbscan_labels = None

    # Run GMM
    gmm, gmm_labels, gmm_probs = run_gmm(Xred, n_components=args.gmm_components)
    df["gmm_label"] = gmm_labels
    df["gmm_prob_max"] = gmm_probs
    print("GMM cluster sizes:", pd.Series(gmm_labels).value_counts().sort_index().to_dict())
    # silhouette for GMM
    try:
        s_g = silhouette_score(Xred, gmm_labels)
        print(f"GMM silhouette: {s_g:.3f}")
    except Exception as e:
        print("Could not compute silhouette for GMM:", e)
    compute_and_save_summaries(df, "gmm_label", used_feats, os.path.join(args.output, "gmm"))
    if UMAP_AVAILABLE:
        plot_umap(Xred, gmm_labels, os.path.join(args.output, "gmm_umap.png"), title="GMM clusters (UMAP)")

    # Save annotated full CSV
    out_csv = os.path.join(args.output, "CVS_all_with_clusters.csv")
    df.to_csv(out_csv, index=False)
    print(f"Wrote annotated dataset to {out_csv}")

    # Save small diagnostics summary
    diag = {
        "n_files": len(files),
        "n_rows": len(df),
        "features_used": used_feats,
        "gmm_components": args.gmm_components,
        "hdbscan_used": args.use_hdbscan and HDBSCAN_AVAILABLE
    }
    pd.Series(diag).to_csv(os.path.join(args.output, "diag_summary.csv"))
    print("Done. Outputs in", args.output)

# ------------------ CLI entrypoint ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDBSCAN + GMM clustering for CVS maxima files")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Base directory with CSV files")
    parser.add_argument("--pattern", type=str, default=FILE_GLOB, help="Filename glob pattern")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--features", nargs="+", default=FEATURES_DEFAULT, help="List of feature columns to use (will intersect with available numeric cols)")
    parser.add_argument("--dropna-thresh", type=float, default=0.5, help="Fraction of non-NaN required to keep a feature column")
    parser.add_argument("--pca-components", type=int, default=8, help="If >0, reduce dimensions with PCA to this many components before clustering (recommended)")
    parser.add_argument("--use-hdbscan", action="store_true", default=False, help="Use HDBSCAN if installed")
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=50)
    parser.add_argument("--hdbscan-min-samples", type=int, default=None)
    parser.add_argument("--gmm-components", type=int, default=5)
    args = parser.parse_args()
    main(args)

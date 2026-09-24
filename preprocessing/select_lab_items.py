"""
Data-driven candidate selection for LAB_ITEM_IDS (the TABULAR_DIM=50 clinical
features that build_mimic_dataset.py::build_tabular_features() pulls out of
labevents.csv for each admission's 48h-before-t0 window).

WHY THIS EXISTS: LAB_ITEM_IDS in build_mimic_dataset.py is currently a
PLACEHOLDER -- fake itemids (90001..90008) that only exist in the synthetic
seed data (see the TODO comment right above it). No real MIMIC-IV itemid
list for the paper's 50 tabular features has ever been chosen; the
manuscript's own "Lab tests: 568" figure (Table 5) is an illustrative
dataset-statistic placeholder, not a concrete feature list. This script
proposes real candidates by measuring, on the actual downloaded cohort, how
often each real MIMIC-IV lab itemid is recorded inside the exact same
48h-before-chest-X-ray window that build_tabular_features() will look at --
so whichever items are finally chosen are ones that are actually well
populated for this cohort, rather than guessed from memory.

This is a RESEARCH DESIGN DECISION and is NOT auto-applied anywhere: run
this, review the ranked table (printed + written to lab_item_candidates.csv),
and only then manually update LAB_ITEM_IDS in build_mimic_dataset.py.

Usage (run from the repo root, same convention as the other preprocessing
scripts, e.g. `python preprocessing/download_mimic_subset.py ...` -- only
works once Step 5's labevents.csv has finished downloading):
    python preprocessing/select_lab_items.py                  # top 50 by coverage
    python preprocessing/select_lab_items.py --top-n 60 --min-coverage 0.05

Reuses build_mimic_dataset.py's own load_cxr_metadata / load_admissions /
map_study_to_admission / select_representative_view / _hosp_table_path
functions directly (imported, not reimplemented), so the admission
population and file-path logic can never drift out of sync with what
build_mimic_dataset.py itself will actually do with the chosen items.
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import Config
import build_mimic_dataset as bmd


def load_d_labitems():
    path = os.path.join(Config.MIMIC_IV_HOSP_DIR, "d_labitems.csv")
    if not os.path.exists(path):
        path = os.path.join(Config.MIMIC_IV_HOSP_DIR, "d_labitems.csv.gz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"d_labitems.csv(.gz) not found under {Config.MIMIC_IV_HOSP_DIR} -- "
            f"Step 5 downloads this dictionary file; run download_mimic_subset.py first."
        )
    return pd.read_csv(path, compression="infer")


def load_labevents_for_ranking():
    path = bmd._hosp_table_path("labevents")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- labevents.csv hasn't finished downloading yet "
            f"(Step 5). Run this script again once it has."
        )
    df = pd.read_csv(
        path, compression="infer", low_memory=False,
        usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
    )
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    return df


def build_candidate_admissions():
    """The same admission population build_mimic_dataset.py scores tabular
    coverage against: one representative-view CXR study per (subject_id,
    hadm_id), computed BEFORE the report/tabular eligibility filters -- so
    item selection isn't biased by which admissions happen to already have
    a downloaded report."""
    meta = bmd.load_cxr_metadata()
    adm = bmd.load_admissions()
    mapped = bmd.map_study_to_admission(meta, adm)
    mapped = bmd.select_representative_view(mapped)
    out = mapped[["subject_id", "hadm_id", "study_datetime"]].reset_index(drop=True)
    out["subject_id"] = out["subject_id"].astype("int64")
    out["hadm_id"] = out["hadm_id"].astype("int64")
    return out


def rank_lab_items(labevents: pd.DataFrame, candidates: pd.DataFrame, window_hours: int):
    """For every itemid: the fraction of candidate admissions that have at
    least one non-null valuenum reading for it inside [t0-window, t0) --
    exactly the "observed" definition build_tabular_features() uses. A
    single vectorized merge + time-window filter (not a per-admission
    Python loop), so this stays fast even at hundreds of thousands of rows."""
    lab = labevents.dropna(subset=["valuenum", "hadm_id", "charttime"]).copy()
    lab["subject_id"] = lab["subject_id"].astype("int64")
    lab["hadm_id"] = lab["hadm_id"].astype("int64")

    merged = lab.merge(candidates, on=["subject_id", "hadm_id"], how="inner")
    in_window = merged[
        (merged["charttime"] >= merged["study_datetime"] - pd.Timedelta(hours=window_hours))
        & (merged["charttime"] < merged["study_datetime"])
    ]
    # One "observed" credit per (admission, itemid) -- repeated draws of the
    # same test within the window shouldn't inflate an item's coverage.
    observed = in_window.drop_duplicates(subset=["subject_id", "hadm_id", "itemid"])
    counts = observed.groupby("itemid").size().rename("n_admissions_observed")

    total_candidates = len(candidates)
    stats = counts.to_frame()
    stats["coverage_frac"] = stats["n_admissions_observed"] / total_candidates
    stats["total_readings_in_cohort"] = lab.groupby("itemid").size()
    stats = stats.sort_values("coverage_frac", ascending=False)
    return stats, total_candidates


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top-n", type=int, default=Config.TABULAR_DIM,
                     help=f"How many itemids to shortlist (default: Config.TABULAR_DIM = {Config.TABULAR_DIM})")
    ap.add_argument("--min-coverage", type=float, default=0.0,
                     help="Drop candidates below this coverage fraction before ranking (0.0-1.0)")
    ap.add_argument("--out-csv", default="lab_item_candidates.csv")
    args = ap.parse_args()

    print("Loading d_labitems dictionary ...")
    d_labitems = load_d_labitems()

    print("Loading labevents.csv (filtered cohort) ...")
    labevents = load_labevents_for_ranking()
    print(f"  {len(labevents):,} lab rows, {labevents['itemid'].nunique():,} distinct itemids")

    print("Rebuilding the candidate admission population (same logic as build_mimic_dataset.py) ...")
    candidates = build_candidate_admissions()
    print(f"  {len(candidates):,} candidate admissions (one representative CXR study each)")

    print(f"Ranking itemids by coverage within the {Config.LAB_WINDOW_HOURS}h-before-t0 window ...")
    stats, total = rank_lab_items(labevents, candidates, Config.LAB_WINDOW_HOURS)

    stats = stats[stats["coverage_frac"] >= args.min_coverage]
    stats = stats.merge(
        d_labitems[["itemid", "label", "fluid", "category"]],
        left_index=True, right_on="itemid", how="left",
    )
    stats = stats.sort_values("coverage_frac", ascending=False).reset_index(drop=True)

    stats.to_csv(args.out_csv, index=False)
    print(f"\nFull ranked table written to {args.out_csv} "
          f"({len(stats)} itemids, out of {total} candidate admissions).")

    top = stats.head(args.top_n)
    print(f"\n=== Top {args.top_n} candidates by coverage (suggested LAB_ITEM_IDS) ===")
    print(top[["itemid", "label", "category", "fluid", "coverage_frac",
               "n_admissions_observed"]].to_string(index=False))

    ids = top["itemid"].astype(int).tolist()
    print("\n# Paste into build_mimic_dataset.py, replacing the placeholder LAB_ITEM_IDS:")
    print("LAB_ITEM_IDS = [")
    for i in range(0, len(ids), 10):
        print("    " + ", ".join(str(x) for x in ids[i:i + 10]) + ",")
    print("]")
    print(
        "\nNOTE: this is a data-driven shortlist by coverage only -- it does NOT know which "
        "items are clinically redundant (e.g. two itemids for the same analyte drawn from "
        "different specimen types) or which ones you'd rather force-include for clinical "
        "relevance even at lower coverage (Table 1's Albumin/Creatinine/BUN/AST/Bilirubin "
        f"panel, say). Review the full {args.out_csv} table and swap items in/out before "
        "finalizing -- this script proposes candidates, it does not decide the paper's "
        "feature panel."
    )


if __name__ == "__main__":
    main()

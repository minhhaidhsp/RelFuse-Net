"""
Links MIMIC-CXR-JPG (images + reports) with MIMIC-IV (hosp module: admissions,
diagnoses_icd, hcpcsevents, labevents) into the admission-level cohort described
in Section 4.1.1, and builds the training-set patient-similarity graph.

WHY MIMIC-IV, NOT MIMIC-III: MIMIC-CXR-JPG's subject_id/hadm_id namespace is
shared with MIMIC-IV, not MIMIC-III (the two were de-identified independently),
so MIMIC-IV is the only PhysioNet source that can be joined to MIMIC-CXR by ID.

This script assumes you have credentialed access to and have downloaded:
  - mimic-cxr-jpg/2.0.0  (metadata csv, chexpert csv, and the JPGs / reports)
  - mimic-iv/3.1/hosp    (admissions.csv.gz, diagnoses_icd.csv.gz,
                           hcpcsevents.csv.gz, labevents.csv.gz, d_labitems.csv.gz)

Local smoke-testing without real PhysioNet data: run
`preprocessing/generate_physionet_seed.py` first, point config.py at its
output (see config.py's _MIMIC_ROOT), and this script will run end-to-end
against that synthetic-but-realistically-shaped data. It prints the
LAB_ITEM_IDS placeholder value you need below -- keep it in sync with that
script's SEED_LAB_ITEM_IDS until you replace it with real curated itemids.

Several site-specific pieces are marked TODO because they depend on exactly how
you stored the free-text radiology reports and which lab/vital panel you curate
into the TABULAR_DIM=50 feature vector -- fill these in for your local copy.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import Config
from graph_utils import build_patient_graph
import torch

CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]

# TODO (before running on REAL data): replace with your own curated MIMIC-IV
# d_labitems itemid list (labs + vitals), in the exact order that should
# populate the TABULAR_DIM=50 feature vector -- this is a research design
# choice (which labs/vitals matter clinically), not something that can be
# auto-derived.
#
# The values below are PLACEHOLDERS that only exist in the synthetic seed
# data from preprocessing/generate_physionet_seed.py (SEED_LAB_ITEM_IDS in
# that script) -- they are intentionally far outside MIMIC-IV's real itemid
# range (which is 5-6 digits, e.g. 50912) so they can never be silently
# mistaken for real lab items. Swap this list out before running against a
# real MIMIC-IV download.
LAB_ITEM_IDS = [90001, 90002, 90003, 90004, 90005, 90006, 90007, 90008]
# e.g. for real data: [50912, 50971, 51006, ...]  (Creatinine, Potassium, BUN, ...)


def _read_gz_csv(path, **kwargs):
    return pd.read_csv(path, compression="infer", low_memory=False, **kwargs)


def _hosp_table_path(name: str) -> str:
    """Path to a MIMIC-IV hosp table, accepting either the DECOMPRESSED .csv
    (as written by preprocessing/download_mimic_subset.py and
    preprocessing/generate_physionet_seed.py, both of which decompress
    everything up front) or the original .csv.gz (if you downloaded/kept it
    compressed some other way)."""
    csv_path = os.path.join(Config.MIMIC_IV_HOSP_DIR, f"{name}.csv")
    if os.path.exists(csv_path):
        return csv_path
    return os.path.join(Config.MIMIC_IV_HOSP_DIR, f"{name}.csv.gz")


def load_cxr_metadata():
    meta = _read_gz_csv(Config.MIMIC_CXR_METADATA_CSV)
    meta["study_datetime"] = pd.to_datetime(
        meta["StudyDate"].astype(str) + meta["StudyTime"].astype(float).astype(int).astype(str).str.zfill(6),
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )
    return meta


def load_chexpert_labels():
    labels = _read_gz_csv(Config.MIMIC_CXR_CHEXPERT_CSV)
    # Standard "U-Zeros" convention: uncertain (-1) and missing -> 0. Document this
    # choice in the manuscript; "U-Ones" (uncertain -> 1) is an equally defensible
    # alternative used in some CheXpert papers -- pick one and be explicit about it.
    for col in CHEXPERT_LABELS:
        labels[col] = labels[col].fillna(0).replace(-1, 0)
    return labels


def load_admissions():
    adm = _read_gz_csv(_hosp_table_path("admissions"))
    adm["admittime"] = pd.to_datetime(adm["admittime"])
    adm["dischtime"] = pd.to_datetime(adm["dischtime"])
    return adm


def load_diagnoses():
    return _read_gz_csv(_hosp_table_path("diagnoses_icd"))


def load_hcpcs():
    hcpcs = _read_gz_csv(_hosp_table_path("hcpcsevents"))
    hcpcs["chartdate"] = pd.to_datetime(hcpcs["chartdate"])
    return hcpcs


def map_study_to_admission(meta: pd.DataFrame, adm: pd.DataFrame) -> pd.DataFrame:
    """For each CXR study, find the admission whose [admittime, dischtime] window
    contains the study's acquisition time -- the standard MIMIC-CXR/IV linkage
    heuristic used in prior work. Studies with no containing admission (e.g.
    outpatient imaging) are dropped, since ICD/CPT-based similarity and the
    48h lab window both require an admission context."""
    merged = meta.merge(adm[["subject_id", "hadm_id", "admittime", "dischtime"]], on="subject_id", how="inner")
    inside = merged[(merged["study_datetime"] >= merged["admittime"]) & (merged["study_datetime"] <= merged["dischtime"])]
    # If a study matches multiple admissions (overlapping stays), keep the tightest window.
    inside = inside.sort_values(by=["dicom_id"])
    inside = inside.drop_duplicates(subset=["dicom_id"], keep="first")
    return inside


def select_representative_view(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (subject_id, hadm_id): the earliest-eligible study, and within
    that study the view chosen by Config.VIEW_PRIORITY (PA > AP > LATERAL > LL)."""
    priority = {v: i for i, v in enumerate(Config.VIEW_PRIORITY)}
    df = df.copy()
    df["view_rank"] = df["ViewPosition"].map(priority).fillna(len(priority))
    df = df.sort_values(by=["subject_id", "hadm_id", "study_datetime", "view_rank"])
    return df.drop_duplicates(subset=["subject_id", "hadm_id"], keep="first")


def build_report_text_lookup(row) -> str:
    """MIMIC-CXR-JPG ships free-text reports as
    files/p<first2-of-subject>/p<subject_id>/s<study_id>.txt. Returning "" makes
    a row ineligible downstream (eligibility criterion 2)."""
    report_path = os.path.join(
        Config.MIMIC_CXR_JPG_DIR,
        "files",
        f"p{str(row['subject_id'])[:2]}",
        f"p{row['subject_id']}",
        f"s{row['study_id']}.txt",
    )
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    return ""


def build_historical_icd(diagnoses: pd.DataFrame, adm: pd.DataFrame, subject_id, current_admittime) -> list:
    prior_hadm_ids = adm[(adm["subject_id"] == subject_id) & (adm["dischtime"] < current_admittime)]["hadm_id"]
    codes = diagnoses[diagnoses["hadm_id"].isin(prior_hadm_ids)]["icd_code"].astype(str).tolist()
    return codes


def build_current_cpt(hcpcs: pd.DataFrame, hadm_id, t0) -> list:
    rows = hcpcs[(hcpcs["hadm_id"] == hadm_id) & (hcpcs["chartdate"] < t0)]
    return rows["hcpcs_cd"].astype(str).tolist()


def build_tabular_features(labevents: pd.DataFrame, hadm_id, t0, window_hours: int) -> (np.ndarray, np.ndarray):
    """Pivots LAB_ITEM_IDS into a fixed-length (values, observed_mask) pair, both
    length TABULAR_DIM; values default to 0.0 and observed_mask to 0 wherever a
    lab/vital was not recorded in the window."""
    values = np.zeros(Config.TABULAR_DIM, dtype=np.float32)
    observed = np.zeros(Config.TABULAR_DIM, dtype=np.float32)
    if not LAB_ITEM_IDS:
        return values, observed

    window_start = t0 - pd.Timedelta(hours=window_hours)
    rows = labevents[
        (labevents["hadm_id"] == hadm_id)
        & (labevents["charttime"] >= window_start)
        & (labevents["charttime"] < t0)
        & (labevents["itemid"].isin(LAB_ITEM_IDS))
    ]
    latest = rows.sort_values("charttime").drop_duplicates(subset=["itemid"], keep="last")
    for i, itemid in enumerate(LAB_ITEM_IDS[: Config.TABULAR_DIM]):
        match = latest[latest["itemid"] == itemid]
        if len(match) > 0:
            values[i] = float(match["valuenum"].iloc[0])
            observed[i] = 1.0
    return values, observed


def zscore_fit_apply(train_values, train_mask, *other_splits_values_masks):
    """Fit mean/std on TRAINING observed entries only; apply to all splits."""
    train_values = np.stack(train_values)
    train_mask = np.stack(train_mask)
    means, stds = [], []
    for d in range(train_values.shape[1]):
        col = train_values[train_mask[:, d] == 1, d]
        means.append(col.mean() if len(col) else 0.0)
        stds.append(col.std() if len(col) > 1 else 1.0)
    means, stds = np.array(means), np.array(stds)
    stds[stds == 0] = 1.0

    def _apply(values_list, mask_list):
        out = []
        for v, m in zip(values_list, mask_list):
            z = (v - means) / stds
            z = z * m  # keep naturally-missing entries at 0 after scaling
            out.append(z)
        return out

    results = [_apply(train_values, train_mask)]
    for vals, mask in other_splits_values_masks:
        results.append(_apply(vals, mask))
    return results, means, stds


def main():
    os.makedirs(Config.PROCESSED_DIR, exist_ok=True)

    print("Loading MIMIC-CXR metadata / labels ...")
    meta = load_cxr_metadata()
    labels_df = load_chexpert_labels()

    print("Loading MIMIC-IV hosp tables ...")
    adm = load_admissions()
    diagnoses = load_diagnoses()
    hcpcs = load_hcpcs()
    # labevents is very large in the real data; consider reading in chunks or via
    # DuckDB/parquet for a real run instead of pandas.read_csv directly.
    labevents_path = _hosp_table_path("labevents")
    labevents = pd.DataFrame(columns=["hadm_id", "itemid", "charttime", "valuenum"])
    if os.path.exists(labevents_path):
        labevents = _read_gz_csv(labevents_path, usecols=["hadm_id", "itemid", "charttime", "valuenum"])
        labevents["charttime"] = pd.to_datetime(labevents["charttime"])

    print("Mapping CXR studies to admissions ...")
    mapped = map_study_to_admission(meta, adm)
    mapped = select_representative_view(mapped)
    print(f"  {len(mapped)} admissions have at least one eligible, mapped CXR study.")

    records = []
    for _, row in mapped.iterrows():
        t0 = row["study_datetime"]
        report_text = build_report_text_lookup(row)
        if not report_text:
            continue  # eligibility criterion (2): complete corresponding textual report

        tab_values, tab_mask = build_tabular_features(labevents, row["hadm_id"], t0, Config.LAB_WINDOW_HOURS)
        if tab_mask.sum() == 0:
            continue  # eligibility criterion (3): tabular EHR within the 48h window

        icd_hist = build_historical_icd(diagnoses, adm, row["subject_id"], row["admittime"])
        cpt_codes = build_current_cpt(hcpcs, row["hadm_id"], t0)

        lbl_row = labels_df[(labels_df["subject_id"] == row["subject_id"]) & (labels_df["study_id"] == row["study_id"])]
        if len(lbl_row) == 0:
            continue
        label_vec = lbl_row.iloc[0][CHEXPERT_LABELS].values.astype(np.float32)

        records.append({
            "subject_id": row["subject_id"],
            "hadm_id": row["hadm_id"],
            "image_path": f"p{str(row['subject_id'])[:2]}/p{row['subject_id']}/s{row['study_id']}/{row['dicom_id']}.jpg",
            "report_text": report_text,
            "tabular_values": tab_values,
            "tabular_observed_mask": tab_mask,
            "icd_history": icd_hist,
            "cpt_codes": cpt_codes,
            "labels": label_vec,
        })

    print(f"Eligible admissions after full filtering: {len(records)}")
    if len(records) == 0:
        print("No eligible records -- check TODOs (report path, LAB_ITEM_IDS) before proceeding.")
        return

    df = pd.DataFrame(records)

    # --- Patient-level split (Section 4.1.1): 70/10/20, no subject in >1 split ---
    rng = np.random.RandomState(Config.SEED)
    subjects = df["subject_id"].unique()
    rng.shuffle(subjects)
    n = len(subjects)
    n_train = int(Config.TRAIN_RATIO * n)
    n_val = int(Config.VAL_RATIO * n)
    train_subj = set(subjects[:n_train])
    val_subj = set(subjects[n_train:n_train + n_val])
    test_subj = set(subjects[n_train + n_val:])

    df_train = df[df["subject_id"].isin(train_subj)].reset_index(drop=True)
    df_val = df[df["subject_id"].isin(val_subj)].reset_index(drop=True)
    df_test = df[df["subject_id"].isin(test_subj)].reset_index(drop=True)
    print(f"Split sizes -- train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")

    # --- Z-score normalization fit on TRAIN only, applied to all splits ---
    (train_vals, val_vals, test_vals), means, stds = zscore_fit_apply(
        df_train["tabular_values"].tolist(), df_train["tabular_observed_mask"].tolist(),
        (df_val["tabular_values"].tolist(), df_val["tabular_observed_mask"].tolist()),
        (df_test["tabular_values"].tolist(), df_test["tabular_observed_mask"].tolist()),
    )
    df_train["tabular_values"] = train_vals
    df_val["tabular_values"] = val_vals
    df_test["tabular_values"] = test_vals
    np.save(os.path.join(Config.PROCESSED_DIR, "tabular_zscore_stats.npy"), {"mean": means, "std": stds})

    def _serialize(df_split: pd.DataFrame) -> pd.DataFrame:
        out = df_split.copy()
        out["tabular_values"] = out["tabular_values"].apply(lambda a: ",".join(f"{x:.4f}" for x in a))
        out["tabular_observed_mask"] = out["tabular_observed_mask"].apply(lambda a: ",".join(str(int(x)) for x in a))
        out["icd_history"] = out["icd_history"].apply(lambda lst: ";".join(lst))
        out["cpt_codes"] = out["cpt_codes"].apply(lambda lst: ";".join(lst))
        out["labels"] = out["labels"].apply(lambda a: ",".join(str(int(x)) for x in a))
        return out

    _serialize(df_train).to_csv(Config.CSV_TRAIN, index=False)
    _serialize(df_val).to_csv(Config.CSV_VAL, index=False)
    _serialize(df_test).to_csv(Config.CSV_TEST, index=False)
    print(f"Wrote {Config.CSV_TRAIN}, {Config.CSV_VAL}, {Config.CSV_TEST}")

    # --- Training-set graph (Section 3.1.2), built ONCE, train admissions only ---
    print("Building the training-set patient-similarity graph ...")
    edge_index = build_patient_graph(
        df_train["icd_history"].tolist(), df_train["cpt_codes"].tolist(), Config.CPT_OVERLAP_THRESHOLD
    )
    torch.save(edge_index, Config.GRAPH_EDGES_TRAIN)
    print(f"  {edge_index.shape[1] // 2} undirected edges among {len(df_train)} training admissions.")
    print(f"  Saved to {Config.GRAPH_EDGES_TRAIN}")


if __name__ == "__main__":
    main()

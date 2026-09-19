"""
generate_physionet_seed.py
===========================
SEED / SMOKE-TEST DATA ONLY. Generates a small, entirely synthetic dataset laid
out EXACTLY like a real PhysioNet download of MIMIC-CXR-JPG + MIMIC-IV (hosp
module) -- same directory structure, same filenames, same CSV column names --
so you can run the REAL `preprocessing/build_mimic_dataset.py` pipeline
end-to-end right now, without waiting for PhysioNet to fix your account's
download access.

This is a stronger smoke test than `generate_smoke_test_data.py`: that script
skips preprocessing entirely and writes directly to the already-processed
train/val/test CSV schema (good for testing model.py / train.py). This script
instead exercises the actual linkage/eligibility/graph-construction code in
preprocessing/build_mimic_dataset.py -- the part that has never been run
against anything yet -- so it catches bugs in that logic specifically.

RESULTS PRODUCED FROM THIS DATA ARE NOT VALID EXPERIMENTAL FINDINGS. They
exist only to prove the preprocessing + training pipeline runs correctly,
with correct shapes and no crashes, end to end.

What it builds, under --out-dir (default ./data/mimic_seed):

    mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv
    mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-chexpert.csv
    mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv
    mimic-cxr-jpg/2.1.0/files/p1./p<subject_id>/s<study_id>/<dicom_id>.jpg   (real tiny JPGs)
    mimic-cxr-jpg/2.1.0/files/p1./p<subject_id>/s<study_id>.txt              (report text)
    mimiciv/3.1/hosp/admissions.csv
    mimiciv/3.1/hosp/diagnoses_icd.csv
    mimiciv/3.1/hosp/hcpcsevents.csv
    mimiciv/3.1/hosp/labevents.csv

This exactly mirrors what `preprocessing/download_mimic_subset.py` writes
under `./data/mimic_subset/`, on purpose: once PhysioNet access is fixed,
switching config.py from the seed paths to the real downloaded paths is a
one-line prefix change (./data/mimic_seed -> ./data/mimic_subset), nothing
else in the pipeline needs to change.

Design notes on WHY the synthetic cohort is structured the way it is (so the
generated graph and eligible-record counts are non-trivial, not degenerate):

  - Patients are assigned to one of a few "clinical clusters", each with its
    own small signature set of ICD codes and CPT/HCPCS codes. Admissions in
    the same cluster share codes with each other by construction, so both
    the ICD-based edge rule (>=1 shared HISTORICAL code) and the CPT-based
    edge rule (>=Config.CPT_OVERLAP_THRESHOLD shared current-admission
    codes) actually fire and produce a non-empty graph -- a purely random
    code assignment would very likely produce a graph with almost no edges.
  - Most patients get 2-3 admissions (not just 1), because the ICD-based
    edge rule only looks at HISTORICAL codes (from admissions that
    discharged before the current one) -- a patient with only one admission
    contributes no historical ICD codes at all.
  - Lab values are generated for the same placeholder itemids this script
    tells you to put in `preprocessing/build_mimic_dataset.py`'s
    LAB_ITEM_IDS (see the note printed at the end), inside the
    Config.LAB_WINDOW_HOURS window before each study's acquisition time --
    otherwise every record fails eligibility criterion (3) and you get zero
    rows out, which is a common first-run trap.

Usage:
    python preprocessing/generate_physionet_seed.py --n-patients 150
"""
import argparse
import os
import random
from datetime import datetime, timedelta

import pandas as pd
from PIL import Image, ImageDraw

CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]

VIEW_PRIORITY = ["PA", "AP", "LATERAL", "LL"]

# These MUST match the LAB_ITEM_IDS placeholder in preprocessing/build_mimic_dataset.py
# for the seed to produce any eligible records. Kept far outside real MIMIC-IV's
# actual itemid range (which is 5-6 digits, e.g. 50912) so they can never be
# mistaken for real lab items once you switch to real data.
SEED_LAB_ITEM_IDS = [90001, 90002, 90003, 90004, 90005, 90006, 90007, 90008]

# Three synthetic "clinical clusters": each has its own signature ICD codes
# (drives historical-ICD graph edges) and CPT/HCPCS codes (drives CPT-overlap
# graph edges), plus a CheXpert label bias so the smoke-test signal is at
# least weakly learnable (not required for the pipeline to run, but makes a
# later train.py smoke-test more meaningful than pure noise).
CLUSTERS = [
    {
        "name": "cardio",
        "icd": ["I509", "I2510", "I110"],
        "cpt": [f"C{n}" for n in range(1001, 1009)],  # 8 codes, threshold r=5
        "label_bias": {"Cardiomegaly": 0.7, "Enlarged Cardiomediastinum": 0.5, "No Finding": 0.05},
    },
    {
        "name": "pulm",
        "icd": ["J189", "J90", "J449"],
        "cpt": [f"C{n}" for n in range(2001, 2009)],
        "label_bias": {"Pneumonia": 0.6, "Consolidation": 0.4, "Pleural Effusion": 0.3, "No Finding": 0.05},
    },
    {
        "name": "healthy",
        "icd": ["Z000", "Z023"],
        "cpt": [f"C{n}" for n in range(3001, 3009)],
        "label_bias": {"No Finding": 0.85},
    },
]

REPORT_TEMPLATES = {
    "cardio": "The cardiac silhouette is enlarged with mild pulmonary vascular congestion. "
              "No focal consolidation is seen.",
    "pulm": "Patchy airspace opacity in the right lower lobe consistent with consolidation. "
            "Small pleural effusion is noted.",
    "healthy": "The lungs are clear bilaterally. No focal consolidation, pleural effusion, "
               "or pneumothorax. Cardiomediastinal silhouette is within normal limits.",
}


def make_jpg(path: str, cluster_name: str, rng: random.Random):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base = {"cardio": (180, 120, 120), "pulm": (120, 160, 180), "healthy": (150, 150, 150)}[cluster_name]
    img = Image.new("RGB", (256, 256), tuple(max(0, min(255, c + rng.randint(-15, 15))) for c in base))
    draw = ImageDraw.Draw(img)
    for _ in range(20):
        x, y = rng.randint(0, 255), rng.randint(0, 255)
        r = rng.randint(1, 4)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=tuple(rng.randint(0, 255) for _ in range(3)))
    img.save(path, quality=85)


def sample_labels(cluster: dict, rng: random.Random):
    labels = {name: 0 for name in CHEXPERT_LABELS}
    for name, p in cluster["label_bias"].items():
        labels[name] = 1 if rng.random() < p else 0
    return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="./data/mimic_seed")
    ap.add_argument("--cxr-jpg-version", default="2.1.0")
    ap.add_argument("--mimic-iv-version", default="3.1")
    ap.add_argument("--n-patients", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    out_dir = args.out_dir
    cxr_root = os.path.join(out_dir, "mimic-cxr-jpg", args.cxr_jpg_version)
    cxr_files = os.path.join(cxr_root, "files")
    hosp_root = os.path.join(out_dir, "mimiciv", args.mimic_iv_version, "hosp")
    os.makedirs(cxr_root, exist_ok=True)
    os.makedirs(hosp_root, exist_ok=True)

    metadata_rows, chexpert_rows, split_rows = [], [], []
    admissions_rows, diagnoses_rows, hcpcs_rows, labevents_rows = [], [], [], []

    subject_id_counter = 10000000
    study_id_counter = 50000000
    hadm_id_counter = 20000000

    base_time = datetime(2019, 1, 1)
    n_images_written = 0

    for _ in range(args.n_patients):
        subject_id = subject_id_counter
        subject_id_counter += 1
        cluster = rng.choice(CLUSTERS)

        n_admissions = rng.choices([1, 2, 3], weights=[0.25, 0.45, 0.30])[0]
        patient_start = base_time + timedelta(days=rng.randint(0, 2000))
        admittime = patient_start

        for adm_idx in range(n_admissions):
            hadm_id = hadm_id_counter
            hadm_id_counter += 1

            duration = timedelta(days=rng.randint(2, 8), hours=rng.randint(0, 23))
            dischtime = admittime + duration
            admissions_rows.append({
                "subject_id": subject_id, "hadm_id": hadm_id,
                "admittime": admittime.isoformat(sep=" "), "dischtime": dischtime.isoformat(sep=" "),
                "admission_type": "EW EMER.",
            })

            # This admission's own diagnoses (become "historical" ICD for the
            # patient's NEXT admission, if any) -- drawn from the cluster's
            # signature codes plus a little noise so it isn't perfectly clean.
            n_icd = rng.randint(1, 2)
            this_adm_icd = rng.sample(cluster["icd"], k=min(n_icd, len(cluster["icd"])))
            if rng.random() < 0.2:
                other_cluster = rng.choice([c for c in CLUSTERS if c is not cluster])
                this_adm_icd.append(rng.choice(other_cluster["icd"]))
            for code in this_adm_icd:
                diagnoses_rows.append({
                    "subject_id": subject_id, "hadm_id": hadm_id,
                    "icd_code": code, "icd_version": 10,
                })

            # Study time somewhere inside [admittime, dischtime].
            study_dt = admittime + timedelta(
                seconds=rng.randint(0, max(1, int(duration.total_seconds())))
            )
            t0 = study_dt

            # This admission's CPT/HCPCS codes, all recorded before t0 -- the
            # full cluster CPT set (>= CPT_OVERLAP_THRESHOLD=5 by construction)
            # plus a little noise.
            this_adm_cpt = list(cluster["cpt"])
            if rng.random() < 0.3:
                other_cluster = rng.choice([c for c in CLUSTERS if c is not cluster])
                this_adm_cpt.append(rng.choice(other_cluster["cpt"]))
            for code in this_adm_cpt:
                chart_dt = admittime + timedelta(
                    seconds=rng.randint(0, max(1, int((t0 - admittime).total_seconds())))
                )
                hcpcs_rows.append({
                    "subject_id": subject_id, "hadm_id": hadm_id,
                    "chartdate": chart_dt.date().isoformat(), "hcpcs_cd": code,
                })

            # Lab events within the LAB_WINDOW_HOURS window before t0, for the
            # placeholder SEED_LAB_ITEM_IDS -- without these every record is
            # dropped by eligibility criterion (3) in build_mimic_dataset.py.
            n_labs = rng.randint(4, len(SEED_LAB_ITEM_IDS))
            for itemid in rng.sample(SEED_LAB_ITEM_IDS, k=n_labs):
                lab_dt = t0 - timedelta(hours=rng.uniform(0.5, 47.0))
                bias = 1.0 if cluster["name"] != "healthy" else -1.0
                value = round(rng.gauss(0.0, 1.0) + bias, 3)
                labevents_rows.append({
                    "subject_id": subject_id, "hadm_id": hadm_id, "itemid": itemid,
                    "charttime": lab_dt.isoformat(sep=" "), "valuenum": value,
                })

            # CXR study: 1-3 candidate views (tests select_representative_view's
            # PA > AP > LATERAL > LL priority logic, not just a single pre-picked view).
            study_id = study_id_counter
            study_id_counter += 1
            n_views = rng.choice([1, 1, 2, 3])
            views = rng.sample(VIEW_PRIORITY, k=min(n_views, len(VIEW_PRIORITY)))
            study_date_str = study_dt.strftime("%Y%m%d")
            study_time_str = study_dt.strftime("%H%M%S") + ".000000"

            for view in views:
                dicom_id = f"{subject_id}-{study_id}-{view.lower()}-{rng.randint(1000,9999)}"
                metadata_rows.append({
                    "subject_id": subject_id, "study_id": study_id, "dicom_id": dicom_id,
                    "ViewPosition": view, "StudyDate": study_date_str, "StudyTime": study_time_str,
                })
                img_path = os.path.join(
                    cxr_files, f"p{str(subject_id)[:2]}", f"p{subject_id}", f"s{study_id}", f"{dicom_id}.jpg"
                )
                make_jpg(img_path, cluster["name"], rng)
                n_images_written += 1

            chexpert_row = {"subject_id": subject_id, "study_id": study_id}
            chexpert_row.update(sample_labels(cluster, rng))
            chexpert_rows.append(chexpert_row)

            split_name = rng.choices(["train", "validate", "test"], weights=[0.7, 0.1, 0.2])[0]
            for view, drow in zip(views, metadata_rows[-len(views):]):
                split_rows.append({
                    "dicom_id": drow["dicom_id"], "study_id": study_id,
                    "subject_id": subject_id, "split": split_name,
                })

            report_path = os.path.join(cxr_files, f"p{str(subject_id)[:2]}", f"p{subject_id}", f"s{study_id}.txt")
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(REPORT_TEMPLATES[cluster["name"]])

            # Next admission (if any) starts well after this one discharges,
            # so build_historical_icd's "prior admission" logic is exercised.
            admittime = dischtime + timedelta(days=rng.randint(10, 200))

    pd.DataFrame(metadata_rows).to_csv(
        os.path.join(cxr_root, "mimic-cxr-2.0.0-metadata.csv"), index=False)
    pd.DataFrame(chexpert_rows).to_csv(
        os.path.join(cxr_root, "mimic-cxr-2.0.0-chexpert.csv"), index=False)
    pd.DataFrame(split_rows).to_csv(
        os.path.join(cxr_root, "mimic-cxr-2.0.0-split.csv"), index=False)

    pd.DataFrame(admissions_rows).to_csv(os.path.join(hosp_root, "admissions.csv"), index=False)
    pd.DataFrame(diagnoses_rows).to_csv(os.path.join(hosp_root, "diagnoses_icd.csv"), index=False)
    pd.DataFrame(hcpcs_rows).to_csv(os.path.join(hosp_root, "hcpcsevents.csv"), index=False)
    pd.DataFrame(labevents_rows).to_csv(os.path.join(hosp_root, "labevents.csv"), index=False)

    n_patients_written = subject_id_counter - 10000000
    n_admissions_written = hadm_id_counter - 20000000
    print(f"Wrote seed data to {out_dir}:")
    print(f"  {n_patients_written} patients, {n_admissions_written} admissions, "
          f"{len(metadata_rows)} CXR view rows, {n_images_written} JPGs, "
          f"{len(labevents_rows)} lab events.")
    print()
    print("IMPORTANT -- one code change still needed before running "
          "preprocessing/build_mimic_dataset.py against this seed:")
    print(f"  Set LAB_ITEM_IDS = {SEED_LAB_ITEM_IDS} in preprocessing/build_mimic_dataset.py")
    print("  (placeholder values for this seed only -- replace with your own curated real")
    print("   MIMIC-IV d_labitems itemids before running against real downloaded data).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
download_mimic_subset.py
=========================
Download a LIGHT-WEIGHT, MANAGEABLE SUBSET of MIMIC-CXR-JPG + MIMIC-IV (hosp
module) directly from PhysioNet, using your own credentialed account --
without ever pulling the full corpora to local disk.

--------------------------------------------------------------------------
IMPORTANT: the PhysioNet page you were looking at
(https://physionet.org/content/mimic-cxr/2.1.0/, "Total uncompressed size:
4.7 TB") is the RAW DICOM release, project `mimic-cxr`. RelFuse-Net's
DenseNet-121 encoder does not need DICOM at all -- it needs the pre-rendered
JPEGs, which live in a SEPARATE credentialed PhysioNet project:
`mimic-cxr-jpg`. That corpus is still large (~570 GB for all ~227,000
studies) but this script never touches it in full: it samples a cohort
first, then fetches only the files that cohort needs.

You must have requested (and been granted) access to BOTH restricted
projects under your PhysioNet account before running this:
  - mimic-cxr-jpg   https://physionet.org/content/mimic-cxr-jpg/2.1.0/
  - mimiciv         https://physionet.org/content/mimiciv/3.1/
Your `mimic-cxr` (raw DICOM) access from the screenshot does NOT
automatically grant `mimic-cxr-jpg` -- PhysioNet treats them as separate
projects, each needing its own "Request access" click, even though a
single credentialing application/DUA covers both.
--------------------------------------------------------------------------

What it does, step by step
---------------------------
1. Downloads the small MIMIC-CXR-JPG metadata files in full (a few hundred
   MB total, NOT the images): the CheXpert label table and the official
   train/val/test split, keyed by subject_id/study_id.
2. Samples `--n-patients` subjects, trying to keep at least a few positive
   examples of every CheXpert label (a pure random sample of a few thousand
   patients can otherwise lose rare findings almost entirely).
3. Downloads the free-text radiology reports zip (135 MB, small enough to
   grab whole) and extracts only the reports belonging to your sampled
   subjects.
4. Downloads only the representative-view JPEG per study for your sampled
   subjects (PA > AP > LATERAL > LL, same priority as Config.VIEW_PRIORITY
   in this repo), never the full multi-view image set.
5. Streams the MIMIC-IV `hosp` tables you need (admissions, diagnoses_icd,
   procedures_icd, hcpcsevents, labevents) directly off PhysioNet while
   still gzip-compressed, keeping only rows for your sampled subjects and
   discarding the rest as it streams -- so e.g. `labevents.csv.gz` (tens of
   GB compressed for the full database) is NEVER written to disk in full.

Output layout (matches what config.py / preprocessing/build_mimic_dataset.py
in this repo expect, once you point their path constants here):

    <out-dir>/mimic-cxr-jpg/<version>/mimic-cxr-2.0.0-metadata.csv
    <out-dir>/mimic-cxr-jpg/<version>/mimic-cxr-2.0.0-chexpert.csv
    <out-dir>/mimic-cxr-jpg/<version>/mimic-cxr-2.0.0-split.csv
    <out-dir>/mimic-cxr-jpg/<version>/files/p1./p<subject_id>/s<study_id>/<dicom_id>.jpg
    <out-dir>/reports/...                       (only sampled subjects)
    <out-dir>/mimiciv/<version>/hosp/admissions.csv        (filtered)
    <out-dir>/mimiciv/<version>/hosp/diagnoses_icd.csv     (filtered)
    <out-dir>/mimiciv/<version>/hosp/procedures_icd.csv    (filtered)
    <out-dir>/mimiciv/<version>/hosp/hcpcsevents.csv       (filtered)
    <out-dir>/mimiciv/<version>/hosp/labevents.csv         (filtered)
    <out-dir>/mimiciv/<version>/hosp/d_*.csv               (full dictionaries, tiny)
    <out-dir>/cohort_manifest.csv

Usage
-----
    python download_mimic_subset.py --username hainguyen83 --n-patients 2000

You will be prompted for your PhysioNet password (or export
PHYSIONET_PASSWORD beforehand so it isn't typed interactively). The
password is only kept in memory for this run -- it is never written to
disk or printed.

Useful flags
------------
    --dry-run          sample the cohort and print size estimates, download nothing
    --skip-images       only fetch metadata + hosp tables, no JPEGs
    --skip-hosp         only fetch CXR metadata + images, no MIMIC-IV tables
    --hosp-tables       comma-separated subset, e.g. "admissions,diagnoses_icd"
                        (drop "labevents" here first if you want a quick trial run --
                        it is by far the slowest table to stream-filter)

Re-running is safe: any file that already exists locally with a non-zero
size is skipped, so an interrupted run can simply be restarted.

Requires only the Python standard library -- no extra `pip install` needed.
"""

import argparse
import base64
import csv
import getpass
import gzip
import io
import os
import random
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

PHYSIONET_HOST = "https://physionet.org"

CHEXPERT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax", "Support Devices",
]

VIEW_PRIORITY = ["PA", "AP", "LATERAL", "LL"]


# --------------------------------------------------------------------------
# Authenticated HTTP helpers (stdlib only)
# --------------------------------------------------------------------------
#
# NOTE: PhysioNet's file server returns a bare 403 Forbidden for an
# unauthenticated request to a restricted file -- it does NOT send back a
# 401 + "WWW-Authenticate: Basic" challenge first. Python's default
# HTTPBasicAuthHandler only attaches credentials *after* seeing that
# challenge, so it never fires here and every request 403s before your
# username/password are ever sent. `wget --user/--password` avoids this by
# sending the Basic-Auth header pre-emptively on the very first request --
# so we do the same thing manually below, instead of relying on urllib's
# reactive auth handler.

def make_auth_headers(username, password):
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}", "User-Agent": "download_mimic_subset.py"}


def open_url(url, headers, retries=3, backoff=2.0):
    """Open a URL for streaming reads, pre-emptively authenticated, with a
    few retries on transient errors. Raises with the HTTP status code on
    failure, so a wrong password (401/403) is distinguishable from a wrong
    filename (404)."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            return urllib.request.urlopen(req, timeout=60)
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (401, 403):
                # Wrong credentials or no access to this project -- retrying won't help.
                raise RuntimeError(
                    f"HTTP {e.code} for {url} -- check your PhysioNet username/password, "
                    f"and that your account has been granted access to this specific "
                    f"restricted project (not just a sibling project)."
                ) from e
            if attempt < retries:
                time.sleep(backoff * attempt)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff * attempt)
    raise RuntimeError(f"Failed to open {url}: {last_err}")


def download_file(headers, url, dest: Path, min_size=1, quiet=False):
    """Download a whole (small/medium) file. Skips if dest already exists non-empty."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_size:
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        resp = open_url(url, headers)
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        tmp.rename(dest)
        if not quiet:
            print(f"  downloaded {dest.name} ({dest.stat().st_size / 1e6:.2f} MB)")
        return True
    except Exception as e:
        if not quiet:
            print(f"  FAILED {url}: {e}", file=sys.stderr)
        if tmp.exists():
            tmp.unlink()
        return False


def download_and_gunzip(headers, url, dest_csv: Path, quiet=False):
    """Download a .csv.gz and write the decompressed .csv to dest_csv."""
    if dest_csv.exists() and dest_csv.stat().st_size > 0:
        return True
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_csv.with_suffix(dest_csv.suffix + ".part")
    try:
        resp = open_url(url, headers)
        with gzip.GzipFile(fileobj=resp) as gz, open(tmp, "wb") as out:
            while True:
                chunk = gz.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        tmp.rename(dest_csv)
        if not quiet:
            print(f"  decompressed {dest_csv.name} ({dest_csv.stat().st_size / 1e6:.2f} MB)")
        return True
    except Exception as e:
        if not quiet:
            print(f"  FAILED {url}: {e}", file=sys.stderr)
        if tmp.exists():
            tmp.unlink()
        return False


def stream_filter_gz_csv(headers, url, dest_csv: Path, subject_col, subject_ids, quiet=False):
    """
    Stream a remote .csv.gz row by row, writing only rows whose `subject_col`
    is in `subject_ids` to dest_csv. The full remote table is decompressed
    on the fly but never buffered or written to disk in full -- only the
    matching rows are kept.
    """
    if dest_csv.exists() and dest_csv.stat().st_size > 0:
        return True
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    subject_ids = {str(s) for s in subject_ids}
    fname = url.rsplit("/", 1)[-1]
    try:
        resp = open_url(url, headers)
    except Exception as e:
        print(f"  FAILED to open {url}: {e}", file=sys.stderr)
        return False

    kept, total = 0, 0
    try:
        with gzip.GzipFile(fileobj=resp) as gz:
            text = io.TextIOWrapper(gz, encoding="utf-8", newline="")
            reader = csv.reader(text)
            header = next(reader)
            try:
                col_idx = header.index(subject_col)
            except ValueError:
                raise RuntimeError(f"Column '{subject_col}' not found in {fname} header {header}")
            with open(dest_csv, "w", newline="", encoding="utf-8") as out:
                writer = csv.writer(out)
                writer.writerow(header)
                for row in reader:
                    total += 1
                    if len(row) > col_idx and row[col_idx] in subject_ids:
                        writer.writerow(row)
                        kept += 1
                    if not quiet and total % 2_000_000 == 0:
                        print(f"    ...scanned {total:,} rows of {fname}, kept {kept:,} so far")
        if not quiet:
            print(f"  filtered {dest_csv.name}: kept {kept:,} / {total:,} rows")
        return True
    except Exception as e:
        if not quiet:
            print(f"  FAILED {url}: {e}", file=sys.stderr)
        if dest_csv.exists():
            dest_csv.unlink()
        return False


# --------------------------------------------------------------------------
# Cohort sampling
# --------------------------------------------------------------------------

def sample_subjects_stratified(chexpert_csv: Path, n_patients: int, seed: int):
    """
    Pick `n_patients` subject_ids, trying to keep at least a small quota of
    positive examples for every CheXpert label instead of pure random
    sampling (rare findings can otherwise nearly disappear from a small
    subset).
    """
    rng = random.Random(seed)
    subject_labels = {}
    with open(chexpert_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["subject_id"]
            positives = {lbl for lbl in CHEXPERT_LABELS if row.get(lbl) in ("1.0", "1")}
            # a subject can have several studies; keep the union of positive labels
            subject_labels.setdefault(sid, set()).update(positives)

    all_subjects = list(subject_labels.keys())
    rng.shuffle(all_subjects)

    chosen = set()
    per_label_quota = max(10, n_patients // (4 * len(CHEXPERT_LABELS)))
    label_counts = defaultdict(int)

    # Pass 1: prioritize subjects that still help an under-quota label.
    for sid in all_subjects:
        if len(chosen) >= n_patients:
            break
        labels = subject_labels[sid]
        if labels and any(label_counts[l] < per_label_quota for l in labels):
            chosen.add(sid)
            for l in labels:
                label_counts[l] += 1

    # Pass 2: fill the remainder randomly.
    for sid in all_subjects:
        if len(chosen) >= n_patients:
            break
        chosen.add(sid)

    return sorted(chosen)


def cxr_relative_dir(subject_id: str) -> str:
    return f"files/p{subject_id[:2]}/p{subject_id}"


def pick_representative(rows):
    """rows: list of (study_id, dicom_id, view_position) for ONE study_id."""
    by_view = defaultdict(list)
    for study_id, dicom_id, view in rows:
        by_view[view].append((study_id, dicom_id))
    for v in VIEW_PRIORITY:
        if by_view.get(v):
            return by_view[v][0]
    return rows[0][:2] if rows else None


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--username", required=True, help="Your PhysioNet username")
    ap.add_argument("--n-patients", type=int, default=2000, help="Number of subjects to sample")
    ap.add_argument("--out-dir", default="./data/mimic_subset", help="Local output directory")
    ap.add_argument("--cxr-jpg-version", default="2.1.0")
    ap.add_argument("--mimic-iv-version", default="3.1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-images", action="store_true", help="Only fetch metadata + hosp tables")
    ap.add_argument("--skip-hosp", action="store_true", help="Only fetch CXR metadata + images")
    ap.add_argument(
        "--hosp-tables",
        default="admissions,diagnoses_icd,procedures_icd,hcpcsevents,labevents",
        help="Comma-separated subset of hosp tables to filter+download",
    )
    ap.add_argument("--dry-run", action="store_true", help="Sample cohort, print estimates, download nothing")
    ap.add_argument(
        "--check-access", action="store_true",
        help="Just test authentication against one small restricted file from each "
             "project (mimic-cxr-jpg and mimiciv) and exit -- no cohort sampling, "
             "no download. Prints a clear OK/403/401 verdict per project.",
    )
    ap.add_argument(
        "--password-file", default=None,
        help="Path to a local text file containing ONLY your PhysioNet password "
             "(one line, no quotes). Safer than PHYSIONET_PASSWORD across tools/"
             "terminals that don't share environment variables -- create this file "
             "once with a text editor (never through a chat/automation tool), and "
             "keep it out of version control (add it to .gitignore).",
    )
    args = ap.parse_args()

    if args.password_file:
        pw_path = Path(args.password_file)
        if not pw_path.exists():
            print(f"--password-file {pw_path} does not exist. Create it once with a text "
                  f"editor, containing only your PhysioNet password on a single line.",
                  file=sys.stderr)
            sys.exit(1)
        password = pw_path.read_text(encoding="utf-8").strip()
    else:
        password = os.environ.get("PHYSIONET_PASSWORD") or getpass.getpass(
            f"PhysioNet password for {args.username}: "
        )
    headers = make_auth_headers(args.username, password)

    if args.check_access:
        print("== Access check only (--check-access): no cohort sampling, no download ==")
        checks = [
            ("mimic-cxr-jpg", f"{PHYSIONET_HOST}/files/mimic-cxr-jpg/{args.cxr_jpg_version}/mimic-cxr-2.0.0-chexpert.csv.gz"),
            ("mimiciv", f"{PHYSIONET_HOST}/files/mimiciv/{args.mimic_iv_version}/hosp/d_labitems.csv.gz"),
        ]
        all_ok = True
        for label, url in checks:
            try:
                resp = open_url(url, headers)
                resp.read(1)  # just confirm we can read a byte
                print(f"  [OK]  {label}: authenticated successfully ({url})")
            except Exception as e:
                all_ok = False
                print(f"  [FAIL] {label}: {e}")
        sys.exit(0 if all_ok else 1)

    out_dir = Path(args.out_dir)
    cxr_base = f"{PHYSIONET_HOST}/files/mimic-cxr-jpg/{args.cxr_jpg_version}"
    cxr_local = out_dir / "mimic-cxr-jpg" / args.cxr_jpg_version
    reports_base = f"{PHYSIONET_HOST}/files/mimic-cxr/2.1.0"
    hosp_base = f"{PHYSIONET_HOST}/files/mimiciv/{args.mimic_iv_version}/hosp"
    hosp_local = out_dir / "mimiciv" / args.mimic_iv_version / "hosp"

    print("== Step 1/5: MIMIC-CXR-JPG metadata (small files only) ==")
    meta_files = {
        "mimic-cxr-2.0.0-metadata.csv.gz": cxr_local / "mimic-cxr-2.0.0-metadata.csv",
        "mimic-cxr-2.0.0-chexpert.csv.gz": cxr_local / "mimic-cxr-2.0.0-chexpert.csv",
        "mimic-cxr-2.0.0-split.csv.gz": cxr_local / "mimic-cxr-2.0.0-split.csv",
    }
    for fname, dest in meta_files.items():
        ok = download_and_gunzip(headers, f"{cxr_base}/{fname}", dest)
        if not ok:
            print(f"  NOTE: if this 404s, check the exact filename on "
                  f"https://physionet.org/content/mimic-cxr-jpg/{args.cxr_jpg_version}/ "
                  f"and re-run (already-downloaded files are skipped).")

    metadata_csv = meta_files["mimic-cxr-2.0.0-metadata.csv.gz"]
    chexpert_csv = meta_files["mimic-cxr-2.0.0-chexpert.csv.gz"]

    if not chexpert_csv.exists():
        print("CheXpert label file missing, cannot sample a cohort. Fix Step 1 and re-run.", file=sys.stderr)
        sys.exit(1)

    print("== Step 2/5: sampling a stratified cohort ==")
    subjects = sample_subjects_stratified(chexpert_csv, args.n_patients, args.seed)
    print(f"  sampled {len(subjects)} subjects")

    subject_set = set(subjects)
    studies_by_subject = defaultdict(lambda: defaultdict(list))  # subject -> study -> [(study,dicom,view)]
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["subject_id"]
            if sid in subject_set:
                studies_by_subject[sid][row["study_id"]].append(
                    (row["study_id"], row["dicom_id"], row.get("ViewPosition", ""))
                )

    manifest_rows = []
    for sid, studies in studies_by_subject.items():
        for study_id, rows in studies.items():
            rep = pick_representative(rows)
            if rep:
                manifest_rows.append((sid, study_id, rep[1]))

    print(f"  {len(manifest_rows)} representative-view studies to fetch")

    if args.dry_run:
        est_gb = len(manifest_rows) * 1.0e6 / 1e9  # ~1 MB/JPEG, rough estimate
        print(f"  DRY RUN -- would download roughly {est_gb:.2f} GB of JPEGs "
              f"plus small metadata/report files. Nothing written; hosp tables not estimated "
              f"(size depends entirely on how many admissions your sampled subjects have).")
        return

    print("== Step 3/5: radiology reports (extract sampled subjects only) ==")
    reports_zip = out_dir / "mimic-cxr-reports.zip"
    if download_file(headers, f"{reports_base}/mimic-cxr-reports.zip", reports_zip, min_size=1_000_000):
        reports_out = out_dir / "reports"
        reports_out.mkdir(parents=True, exist_ok=True)
        wanted_tokens = {f"p{sid}" for sid in subjects}
        with zipfile.ZipFile(reports_zip) as zf:
            extracted = 0
            for name in zf.namelist():
                parts = name.replace("\\", "/").split("/")
                stem_tokens = {p.split(".")[0] for p in parts}
                if wanted_tokens & stem_tokens:
                    zf.extract(name, reports_out)
                    extracted += 1
            print(f"  extracted {extracted} report files for the sampled cohort")

    if not args.skip_images:
        print("== Step 4/5: chest X-ray JPEGs (representative view only) ==")
        ok_n, fail_n = 0, 0
        for i, (sid, study_id, dicom_id) in enumerate(manifest_rows, 1):
            rel = f"{cxr_relative_dir(sid)}/s{study_id}/{dicom_id}.jpg"
            dest = cxr_local / rel
            if download_file(headers, f"{cxr_base}/{rel}", dest, min_size=1000, quiet=True):
                ok_n += 1
            else:
                fail_n += 1
            if i % 200 == 0:
                print(f"  ...{i}/{len(manifest_rows)} ({ok_n} ok, {fail_n} failed so far)")
        print(f"  done: {ok_n} images downloaded, {fail_n} failed")
    else:
        print("== Step 4/5: skipped (--skip-images) ==")

    if not args.skip_hosp:
        print("== Step 5/5: MIMIC-IV hosp tables (streamed + filtered to sampled subjects) ==")
        for fname in ["d_icd_diagnoses.csv.gz", "d_icd_procedures.csv.gz", "d_hcpcs.csv.gz", "d_labitems.csv.gz"]:
            download_and_gunzip(headers, f"{hosp_base}/{fname}", hosp_local / fname.replace(".gz", ""))

        wanted_tables = [t.strip() for t in args.hosp_tables.split(",") if t.strip()]
        for table in wanted_tables:
            fname = f"{table}.csv.gz"
            print(f"  streaming {fname} (this can take a while for labevents)...")
            stream_filter_gz_csv(
                headers, f"{hosp_base}/{fname}", hosp_local / f"{table}.csv",
                subject_col="subject_id", subject_ids=subjects,
            )
    else:
        print("== Step 5/5: skipped (--skip-hosp) ==")

    manifest_csv = out_dir / "cohort_manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "study_id", "dicom_id"])
        writer.writerows(manifest_rows)

    print(f"\nDone. Cohort manifest: {manifest_csv} "
          f"({len(manifest_rows)} studies across {len(subjects)} subjects)")
    print("Next: point config.py's MIMIC_CXR_JPG_DIR / MIMIC_CXR_METADATA_CSV / "
          "MIMIC_CXR_CHEXPERT_CSV / MIMIC_CXR_SPLIT_CSV / MIMIC_IV_HOSP_DIR at the paths "
          f"under {out_dir} shown in the module docstring, then run "
          "preprocessing/build_mimic_dataset.py.")


if __name__ == "__main__":
    main()

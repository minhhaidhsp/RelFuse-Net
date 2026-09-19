# RelFuse-Net: Inductive GraphSAGE with LLM for Multimodal EHR Diagnosis

Official implementation of **RelFuse-Net**.

## Status of this repository (read before running anything)

The version of this code you are looking at has been corrected to actually match
the methodology described in the manuscript. An earlier version of this repo was
a **proof-of-concept scaffold**: it trained on entirely synthetic data
(`generate_data.py`, now removed), used a cosine-similarity graph on tabular
features instead of real ICD/CPT codes, an orthogonality-loss proxy instead of
real vCLUB, last-token instead of mean-pooled text features, Focal Loss instead
of class-weighted BCE, and had no held-out test set. None of the numbers that
scaffold could have produced are valid experimental results, and none were
reported as such going forward.

This version fixes all of the above (see the docstrings in `model.py`,
`graph_utils.py`, `losses.py`, `train.py` for exactly what changed and why).
**No result should be reported in the paper until it comes from a real run of
this pipeline on real MIMIC-CXR-JPG + MIMIC-IV data.**

## Data Access & Privacy

This code runs on **MIMIC-CXR-JPG** and **MIMIC-IV** (hosp module) --
**not MIMIC-III**. MIMIC-CXR shares its `subject_id`/`hadm_id` namespace with
MIMIC-IV, not MIMIC-III (the two were de-identified independently and there is
no official PhysioNet crosswalk between them), so MIMIC-IV is the only source
that can be joined to MIMIC-CXR by patient/admission ID. If your manuscript
currently cites MIMIC-III for this experiment, update it to MIMIC-IV.

Due to HIPAA and PhysioNet's Data Use Agreement, the real data cannot be
included here. To reproduce:
1. Complete PhysioNet's CITI Data Privacy training.
2. Apply for credentialed access on [PhysioNet](https://physionet.org/).
3. Download `mimic-cxr-jpg/2.0.0` and `mimic-iv/3.1/hosp`.
4. Fill in the paths at the top of `config.py` and the two TODOs in
   `preprocessing/build_mimic_dataset.py` (report-file path convention and the
   `LAB_ITEM_IDS` list of MIMIC-IV `d_labitems` codes for your 50-feature
   tabular panel).

## Pipeline

```
1. preprocessing/build_mimic_dataset.py   # links CXR+IV, builds train/val/test CSVs + graph
2. train.py                               # MLTM pretrain -> joint training -> per-class test metrics
```

Before spending time on step 1, smoke-test the code path with fully synthetic
data (fast, no GPU/LLM needed, and NOT valid for reporting):
```bash
python generate_smoke_test_data.py
# then in config.py temporarily set USE_REAL_LLM = False and EPOCHS = 2
python train.py
```
If that runs end-to-end without errors, the real run (with `USE_REAL_LLM = True`
and real data from step 1) is the one whose numbers belong in the paper.

## Project Structure
* `config.py` -- all hyperparameters and paths, matching the manuscript.
* `preprocessing/build_mimic_dataset.py` -- links MIMIC-CXR-JPG + MIMIC-IV,
  applies the eligibility filter, splits by patient (70/10/20), builds the
  training-set ICD/CPT graph.
* `graph_utils.py` -- real ICD-based / CPT-based (r>=5) edge construction, and
  inductive attachment of validation/test admissions to the fixed training graph.
* `losses.py` -- class-weighted BCE, the real vCLUB mutual-information upper
  bound, and the MLTM masked-reconstruction loss.
* `model.py` -- DenseNet-121 (image), Medical-Llama3-8B+LoRA with mean-pooling
  (text), Transformer-based MLTM (tabular), 2-layer GraphSAGE, and the 4-way
  disentangled fusion (shared / image-specific / text-specific / tabular-specific).
* `data_loader.py` -- dataset class for the processed CSVs.
* `train.py` -- MLTM pretraining, GraphSAGE minibatch (neighbor-sampled) joint
  training, inductive evaluation, per-class + macro metrics, multi-seed runs.
* `generate_smoke_test_data.py` -- synthetic data for code smoke-testing only.

## Setup

```bash
pip install -r requirements.txt
```

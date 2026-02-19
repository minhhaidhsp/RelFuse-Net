# RelFuse-Net: Inductive GraphSAGE with LLM for Multimodal EHR Diagnosis

This repository contains the official PyTorch implementation of **RelFuse-Net**, as presented in the paper: *"RelFuse-Net: Multimodal Fusion of Electronic Health Records for Disease Prediction via Inductive Graph Learning and Large Language Models"*.

## ⚠️ Data Access & Privacy
This code is designed to run on the **MIMIC-CXR** and **MIMIC-III** datasets. 
Due to HIPAA regulations and PhysioNet's Data Use Agreement (DUA), **we cannot provide the real patient data in this repository.**

Researchers wishing to reproduce our results must:
1.  Complete the CITI Data Privacy training.
2.  Apply for credentialed access via [PhysioNet](https://physionet.org/).
3.  Download the datasets and preprocess them into the format described below.

## 📂 Project Structure
* `model.py`: Core architecture implementing **Medical-Llama3 (LoRA)**, **MLTM**, and **Inductive GraphSAGE**.
* `utils.py`: Contains the **Graph Construction** logic and **Disentanglement Loss**.
* `data_loader.py`: Handles loading and transformation of real MIMIC data.
* `config.py`: Hyperparameters matching the paper's experimental setup.
* `train.py`: Main training script.
* `data/mimic_master_sample.csv`: **A template file illustrating the required data format.**

## 🛠️ Setup Instructions

### 1. Install Dependencies
```bash

pip install -r requirements.txt

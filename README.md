# IPPS-MIMIC Project

This repository contains the implementation of the **Interpretable Prognosis Prediction System (IPPS)** for mortality prediction using Electronic Health Records (EHR) from MIMIC datasets.

The goal of this project is to provide an accurate **and interpretable** clinical risk prediction framework by integrating disease-specific expert modules with temporal sequence models (e.g., Mamba backbone).

---

## 📂 Core Files

| File | Role |
|------|------|
| `my_dataset_mimicIV_3_mamba_mimiciii.py` | **Main execution script** — runs training/inference for mortality prediction |
| `process_mimic_iv_v1.py` | **Data preprocessing script** — prepares MIMIC dataset for modeling |

**Execution Order**

`process_mimic_iv_v1.py` → `my_dataset_mimicIV_3_mamba_mimiciii.py`


---

## ⚙️ Environment Requirements

- Python **3.12.0**
- PyTorch **2.7.0**

Recommended hardware:

- NVIDIA GPU (**strongly recommended**)
- CUDA-enabled environment



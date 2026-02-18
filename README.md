# IPPS-MIMIC Project

本仓库包含 **Interpretable Prognosis Prediction System (IPPS)** 在 MIMIC 数据集上的实现，用于基于 EHR（电子病历）进行 **可解释的死亡风险预测**。

---

## 📂 核心文件

| File | Role |
|------|------|
| `my_dataset_mimicIV_3_mamba_mimiciii.py` | **主运行脚本**：运行训练/推理流程（mortality prediction） |
| `process_mimic_iv_v1.py` | **数据处理脚本**：对 MIMIC 数据进行预处理，生成模型可用的数据格式 |

**运行顺序：**

`process_mimic_iv_v1.py` → `my_dataset_mimicIV_3_mamba_mimiciii.py`

---

## ⚙️ 环境要求

- Python: **3.12.0**
- PyTorch: **2.7.0**（GPU 版本推荐）

建议硬件环境：

- NVIDIA GPU（强烈推荐）
- CUDA 可用



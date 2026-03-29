
# 🚀 TransXplainRRG: Train, Test, and Evaluation Guide

This document provides a **step-by-step reproducible pipeline** for training, testing, and evaluating the **TransXplainRRG** framework.

---

## ⚙️ Environment Setup

```bash
git clone <your-repo-url>
cd TransXplainRRG

conda create -n transxplainrrg python=3.9 -y
conda activate transxplainrrg

pip install -r requirements.txt
pip install nlg-eval
pip install RadEval
```
## 📂 Dataset Preparation

### 🔹 MIMIC-CXR

```bash
python tools/report_extractor.py
Use CheXpert labels as ground truth
```

### 🔹 IU-CXR (Open-I Dataset)
```bash
wget https://raw.githubusercontent.com/ZexinYan/Medical-Report-Generation/master/data/new_data/captions.json
```

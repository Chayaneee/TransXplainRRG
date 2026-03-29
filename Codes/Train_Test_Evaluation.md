
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
```
Use CheXpert labels as ground truth

### 🔹 IU-CXR (Open-I Dataset)
```bash
wget https://raw.githubusercontent.com/ZexinYan/Medical-Report-Generation/master/data/new_data/captions.json
```
Place the file in: data/IU-CXR/

### 📌 Vocabulary Setting
Build your vocabulary model with SentencePiece 
```bash
python tools/vocab_builder.py 
```
#### 🔹 MIMIC-CXR Vocabulary (5000 tokens)

- 4500 high-frequency words  
- 500 SentencePiece unigram tokens  

#### 🔹 IU-CXR Vocabulary (1000 tokens)

- 900 high-frequency words  
- 100 SentencePiece unigram tokens  

Prebuilt vocabularies are available in:

```bash
Vocabulary/*.model
```

## 🧠 Training Pipeline

### Step 1: Train Transformer-Based Text Classifier
```bash
python train_text.py
```
Configuration:
```python
PHASE = 'TRAIN'
RELOAD = False
```
Predicts 14 disease labels 
Used for: 
- Supervision 
- Clinical evaluation proxy


### Step 2: Evaluate Text Classifier
Configuration:
PHASE = 'TEST'
RELOAD = True
Run:
python train_text.py

### Step 3: Train TransXplainRRG (Swin-Based)
python train_swin.py
Configuration:
PHASE = 'TRAIN'
RELOAD = False

### Step 4: Generate Reports (Inference)
Configuration:
PHASE = 'INFER'
RELOAD = True
Run:
python train_swin.py
Output:
outputs/raw_reports.txt

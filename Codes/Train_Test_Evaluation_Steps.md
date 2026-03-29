
# 🚀 TransXplainRRG: Train, Test, and Evaluation Guide

This document provides a **step-by-step reproducible pipeline** for training, testing, and evaluating the **TransXplainRRG** framework.

---

## ⚙️ Environment Setup

```bash
git clone <your-repo-url>
cd TransXplainRRG/Codes

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
```python
PHASE = 'TEST'
RELOAD = True
```
Run:
```bash
python train_text.py
```

### Step 3: Train TransXplainRRG (Swin-Based)
```bash
python train_swin.py
```
Configuration:
```python
PHASE = 'TRAIN'
RELOAD = False
```

### Step 4: Generate Reports (Inference)
Configuration:
```python
PHASE = 'INFER'
RELOAD = True
```

Run:
```bash
python train_swin.py
```

Output:
```bash
outputs/raw_reports.txt
```

## 📊 Evaluation Pipeline

### 🔹 Stage 1: Direct Evaluation
#### 1. Language Quality Metrics
```bash
   nlg-eval --hypothesis=outputs/*hyp.txt --references=outputs/*ref.txt
```
Metrics:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- METEOR
- ROUGE-L
- CIDEr

#### 2. Clinical Evaluation Metrics
```bash
python eval_text.py
```
CLinical Efficiency Metrics (Micro & Macro):
- AUC
- F1-score
- Precision
- Recall
- Accuracy 

#### 3. Radiology-Specific Evaluation (Rad-Eval)
Use RadEval to compute clinical correctness and semantic metrics directly from generated reports and ground truth reports.
```bash
cd Report_Evaluation/Metrics/
#Convert text to CSV:   `
python txt2csv.py
#Run evaluation:
python rad_evaluation.py
```
Metrics:
- F1CheXbert
- F1 RadGraph
- GREEN
- BLEU-4
- ROUGE-L
- BERTScore
- RadCliQ
Note: GREEN and RadCliQ require high GPU resources.
See https://github.com/jbdel/RadEval for more details.

###🔹 Stage 2: LLM-Based Report Refinement
After generating raw reports, we apply LLM-based restructuring to improve clinical readability and interpretability.
Objective:
Transform generated reports into:
- Inside-Out structured format
- Abnormal Findings-focused summary

#### LLM Prompting Strategy
Each generated report is passed through an LLM with prompts designed to:
- Reorganize content anatomically (Inside-Out approach)
- Highlight clinically relevant abnormalities 
- Remove redundancy and improve clarity

#### Inside-Out Report Generation
```bash
cd Codes/Inside-Out/Codes
#For GPT-API
python inside-out-GPT.py
#For Llama
python inside-out-llama3-8b.py
```
#### Abnormal Findings Extraction
```bash
cd Codes/Abnormal_Findings/Codes
#For GPT-API
python Abnormal-FIndings-GPT.py
#For Llama
python Abnormal_Findings_Llama3.py

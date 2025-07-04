# TransXplainRRG
A Clinically-Validated LVLM for Chest Radiograph Report Generation

# Introduction: 
This is the official repository of our proposed TransXplainRRG model details. Chest X-ray imaging is crucial for diagnosing and treating thoracic diseases, but the process of examining and generating reports for these images can be challenging. There is a shortage of experienced radiologists, and report generation is time-consuming, reducing effectiveness in clinical settings. To address this issue and advance clinical automation, researchers have been working on automated systems for radiology report generation. However, existing systems have limitations, such as disregarding clinical workflow, ignoring clinical context, and lacking explainability. This paper introduces a novel model for automatic chest X-ray report generation based entirely on transformers, integrating LLM. The model focuses on clinical accuracy while improving other text-generation metrics. Our proposed approach, TransXplainRRG, utilises an off-the-shelf Swin Transformer model along with a transformer-based text encoder that incorporates patient medical history to generate a radiology report. Further, we explore an expert-guided `inside-out' approach and extract only abnormal findings for radiology report refinement. Thus, this study bridges the gap between high-performance automation and the interpretability critical for clinical practice by combining state-of-the-art transformer-based vision encoders, text encoders, and LLMs. The model is trained and tested on the large-scale MIMIC-CXR dataset and further evaluated on the unseen IU X-Ray dataset to demonstrate its generalizability and robustness. It demonstrates promising results regarding word overlap, clinical accuracy, and radiology evaluation metrics like F1Radgraph. We employed a newly proposed GREEN metric to assess clinical accuracy, which can also analyse the generated report. Additionally, we introduce qualitative evaluation metrics developed from radiologists' viewpoints to evaluate the clinical relevance of the generated reports in practical settings. The qualitative results using Grad-CAM showcase disease location for better understanding by radiologists. The proposed model embraces radiologists' workflow, aiming to improve explainability, transparency, and trustworthiness for their benefit.

# Proposed Pipeline
![Block_Diagram](https://github.com/user-attachments/assets/0100a072-0d41-4387-82c9-e5cf3db32867)

# Data used for Experiments: 

We have used three datasets for this experiment.
  - [MIMIC-CXR](https://physionet.org/content/mimiciii-demo/1.4/)
  - [IU X-ray](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)

# Evaluation Metrics 
1. Word Overlap Metrics: BLEU-score, METEOR, ROUGE-L, CIDER
2. Clinical Efficiency (CE) Metrics: AUC, F1-score, Precision, Recall, Accuracy, F1ChexBert
3. Radiology Evaluation Metrics: F1Radgraph, GREEN
4. Clinical Safety Metrics based on Radiologists' Evaluation: Immediate Risk, Long Term Risk, Combined Risk, No Risk

# Qualitative Analysis
<img width="868" alt="Image" src="https://github.com/user-attachments/assets/a29a4008-878a-4e31-85ed-32d62f346435" />

# Qualitative Results (Radiologist Viewpoint)
![Image](https://github.com/user-attachments/assets/32909112-19a3-4074-b59f-f92d008d8f28)

# 🧬 A Clinically Interpretable Deep Learning Framework for Accurate Prediction of Breast Cancer Missense Variant Pathogenicity

> A clinically-driven deep learning pipeline for robust, high-performance, and interpretable classification of breast cancer missense variants.

---

## 📌 Overview

This repository hosts the complete code and results of our study:

> **"Interpretable Deep Learning Models for Breast Cancer Missense Variant Pathogenicity Prediction"**

The project benchmarks **7 deep learning architectures** on breast cancer missense variants, offering a unified pipeline with:
- Multi-seed evaluation
- Confidence interval metrics
- Statistical testing
- LIME + Permutation interpretability
- Clinical relevance alignment

---

## 🧠 Key Features

- ✅ **Single Script Simplicity**: Everything runs from `main.py`
- 🧬 7 Deep Learning Models: MLP, CNN, DNN, RNN, LSTM, GRU, Transformer
- 📈 Multi-metric Evaluation: AUC, F1, MCC, Specificity, Kappa, etc.
- 🧪 Statistical Testing: Z-test, Shapiro-Wilk, Levene, ANOVA
- 📊 ROC and PR curves with 95% CI
- 🔍 LIME explanations for TP, TN, FP, FN samples
- 📌 Seed-wise best model selection + CV metric table
- 📁 All outputs saved to Google Drive

---

## 📁 Project Structure

```bash
.
├── main.py                         # Complete pipeline: training, evaluation, interpretation
├── cleaned_dataset.csv             # Input dataset
├── results/
│   ├── roc_curve_with_ci.png
│   ├── metrics_with_ci.xlsx
│   ├── lime_visuals/
│   │   ├── GRU_TP_lime.png
│   │   ├── GRU_TN_lime.png
│   │   ├── GRU_FP_lime.png
│   │   └── GRU_FN_lime.png
├── requirements.txt
├── README.md
└── LICENSE

---

⚙️ Installation
# Clone this repository
git clone https://github.com/yourusername/breast-cancer-pathogenicity-dl.git
cd breast-cancer-pathogenicity-dl

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate      # For Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py


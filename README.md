# 🧬 A Clinically Interpretable Deep Learning Framework for Breast Cancer Missense Variant Pathogenicity Prediction

This repository implements a **validated and clinically interpretable deep learning framework** to predict the pathogenicity of **missense variants** in **breast cancer genes**. It incorporates leakage-controlled preprocessing, recursive feature elimination, seven deep learning architectures, multi-seed evaluation, probability calibration, and case-level interpretability via **LIME and Permutation Feature Importance**.

📝 **Associated Manuscript**:  
*"Clinically Interpretable Deep Learning for Breast Cancer Missense Variant Pathogenicity Prediction"*  
📄 Status: Under review  
📎 [GitHub Repository](https://github.com/rahafahmad89/DL-clinically-interpretable-framework)

---

## 📂 Project Structure

```
.
├── dl_pipeline.py       # End-to-end pipeline: training, evaluation, calibration, interpretability
├── requirements.txt
├── LICENSE
└── README.md
```

Output figures and tables are written to the directory specified by `--outdir` (default: `outputs/`). Trained models are saved to `--modeldir` (default: `saved_models/`). Neither directory is committed to the repository.

---

## 📌 Highlights

- Seven deep learning architectures: MLP, DNN, CNN, LSTM, RNN, GRU, Transformer
- Leakage-controlled pipeline: data split before any preprocessing; RFE and scaling fit on training data only
- Circular features excluded: `clinvar_id` and `ClinPred` removed prior to feature selection
- Multi-seed training with composite-score model selection
- External validation on an independent held-out variant set
- 95% bootstrap confidence intervals for all reported metrics
- Isotonic recalibration with ECE and Brier score decomposition
- DeLong pairwise AUC significance testing
- LIME explanations for TP, TN, FP, and FN cases
- Permutation Feature Importance across all seven models

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/rahafahmad89/DL-clinically-interpretable-framework.git
cd DL-clinically-interpretable-framework
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare your data
Provide two VEP-annotated CSV files with a `CLIN_SIG` label column:
- A training file (pathogenic/benign labelled variants)
- An external test file (independent held-out variants)

Labels are normalised automatically: `pathogenic` and `likely_pathogenic` → 1; `benign` and `likely_benign` → 0. VUS, conflicting, and `not_provided` entries are dropped.

### 4. Run the pipeline
```bash
python dl_pipeline.py \
  --data path/to/training_data.csv \
  --test path/to/external_test_data.csv \
  --outdir outputs \
  --modeldir saved_models
```

This will:
- Split data before any preprocessing (leakage-free)
- Remove correlated features and apply RFE (fit on training fold only)
- Train all seven architectures across five random seeds
- Select the best seed per model and retrain final models
- Evaluate on internal and external test sets
- Generate all figures and tables
- Save trained models as `.h5` files

---

## 🔄 Using Pretrained Models

Trained model weights are available for direct inference without retraining.

### Download
Download the pretrained `.h5` model files and the preprocessor from the [Releases](https://github.com/rahafahmad89/DL-clinically-interpretable-framework/releases) page.

Expected files:
```
saved_models/
├── MLP_best.h5
├── DNN_best.h5
├── CNN_best.h5
├── LSTM_best.h5
├── RNN_best.h5
├── GRU_best.h5
├── Transformer_best.h5
└── preprocessor.pkl
```

### Run inference only
Place the downloaded files in a local `saved_models/` folder, then run:

```bash
python dl_pipeline.py \
  --test path/to/your_variants.csv \
  --modeldir saved_models \
  --outdir results
```

> **Note:** The `preprocessor.pkl` file must be present in `--modeldir` alongside the `.h5` files. It contains the RFE feature list, imputer, and scaler fitted on the original training data. Your input file must be VEP-annotated with the same annotation columns used during training.

---

## 🧠 Deep Learning Architectures

| Model | Architecture |
|-------|-------------|
| MLP | Multi-layer perceptron with LayerNorm and LeakyReLU |
| DNN | Deep feedforward network with BatchNorm |
| CNN | 1D convolutional network |
| LSTM | Conv1D + stacked LSTM layers |
| RNN | Vanilla recurrent network |
| GRU | Stacked GRU with dense head |
| Transformer | Multi-head self-attention with residual connections |

---

## 📊 Output Artifacts

| File | Description |
|------|-------------|
| `Fig_ROC_PR_Calibration.png` | ROC, PR, and calibration curves — internal and external |
| `Fig_statistical_tests.png` | Z-test, Shapiro-Wilk, Levene, F-test per model |
| `Fig_correlation_heatmap.png` | Correlation heatmap of RFE-selected features |
| `Fig_PMI_all_models.png` | Permutation Feature Importance — all seven models |
| `Fig_LIME_all_models.png` | LIME explanations — representative pathogenic case |
| `Fig_LIME_best_model_TP_TN_FP_FN.png` | LIME — best model TP/TN/FP/FN cases |
| `Fig_external_ROC_benchmark.png` | External ROC: DL models vs standalone predictors |
| `Fig_seed_AUC_boxplot.png` | AUC distribution across five seeds |
| `Fig_isotonic_calibration.png` | Calibration before and after isotonic recalibration |
| `Fig_DeLong_heatmap.png` | DeLong pairwise AUC significance heatmap |
| `Table_seed_AUC.xlsx` | AUC per seed and best-seed summary |
| `Table_internal_metrics_CI.xlsx` | Full internal test metrics with 95% bootstrap CI |
| `Table_external_metrics_CI.xlsx` | Full external test metrics with 95% bootstrap CI |
| `Table_benchmark_all.xlsx` | All seven DL models vs eleven standalone predictors |
| `Table_delong_pvalues.xlsx` | Pairwise DeLong p-values |
| `Table_calibration_ECE_Brier.xlsx` | ECE and Brier scores before and after recalibration |
| `Table_feature_descriptions.xlsx` | RFE-selected features with sources and roles |
| `Table_leakage_flags.xlsx` | Feature-level leakage audit |
| `Table_composite_scores.xlsx` | Weighted composite scores for model ranking |
| `Table_statistical_tests.xlsx` | Full statistical test results |
| `predictions_internal.xlsx` | Per-variant predictions on internal test set |
| `predictions_external.xlsx` | Per-variant predictions on external test set with triage labels |

---

## 📈 Evaluation Strategy

- **Multi-seed evaluation**: all models trained across five seeds (42, 101, 202, 303, 404); best seed selected per model based on internal AUC
- **Composite scoring**: model selection uses a weighted combination of AUC, Precision, Recall, F1, Specificity, Sensitivity, MCC, and Kappa
- **External validation**: final models evaluated on an independent held-out set not seen during training or feature selection
- **Stratified split**: single 80/20 stratified train/test split with 1000-iteration bootstrap CIs
- **Statistical testing**: Z-test, Shapiro-Wilk, Levene, ANOVA, Mann-Whitney U, and Kruskal-Wallis used to assess score distributions across models
- **Calibration**: isotonic regression recalibration evaluated via ECE and Brier score decomposition
- **AUC comparison**: DeLong et al. (1988) method for pairwise significance testing

---

## 📚 Citation

```bibtex
@article{ahmad2025interpretableDL,
  author    = {Ahmad, Rahaf M. and Al Dhaheri, Noura and Mohamad, Mohd Saberi and Ali, Bassam R.},
  title     = {Clinically Interpretable Deep Learning for Breast Cancer Missense Variant Pathogenicity Prediction},
  year      = {2025},
  journal   = {To be updated upon acceptance},
  url       = {https://github.com/rahafahmad89/DL-clinically-interpretable-framework}
}
```

---

## 🔐 License

Distributed under the MIT License. See `LICENSE` for more details.

---

## 👩‍💻 Author

**Rahaf M. Ahmad**  
Postdoctoral Researcher — Genetics & Machine Learning  
United Arab Emirates University  
ORCID: [0000-0002-7531-5264](https://orcid.org/0000-0002-7531-5264)

---

## 🤝 Acknowledgements

This work is part of an ongoing effort to develop interpretable AI tools for clinical genomics and support evidence-based variant interpretation in precision oncology.

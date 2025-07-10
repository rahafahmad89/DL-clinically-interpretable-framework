# 🧬 A Clinically Interpretable Deep Learning Framework for Accurate Prediction of Breast Cancer Missense Variant Pathogenicity

This repository implements a **validated and clinically interpretable deep learning framework** to predict the pathogenicity of **missense variants** in **breast cancer genes**. It incorporates preprocessing, feature selection, seven deep learning architectures, multi-seed evaluation, model calibration, and interpretability via **LIME and Permutation Importance**.

📝 **Associated Manuscript**:  
*“A Clinically Interpretable Deep Learning Framework for Accurate Prediction of Breast Cancer Missense Variant Pathogenicity”*  
📄 Status: Submitted  
📎 [GitHub Repository](https://github.com/rahafahmad89/DL-clinically-interpretable-framework)

---

## 📂 Project Structure

```
.
├── data/                          # Input dataset and features
├── results/                       # Evaluation plots, metrics, and interpretability outputs
│   ├── combined_roc_with_ci.png
│   ├── results_with_ci.xlsx
│   ├── LIME_GRU_TP_TN_FP_FN.png
│   └── permutation_importance_gru.png
├── main.py                        # End-to-end pipeline for DL training, evaluation, and interpretation
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📌 Highlights

- Seven deep learning models: MLP, CNN, DNN, RNN, LSTM, GRU, Transformer
- Two-phase pipeline with multi-seed training and best-seed selection
- Confidence intervals for all metrics using bootstrapping
- LIME explanations for TP, TN, FP, FN classification cases
- Permutation feature importance analysis
- ROC and PR curves with statistical testing and model calibration
- Transparent and reproducible results for clinical use

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

### 3. Prepare the dataset
Place your cleaned and annotated CSV file inside the `data/` folder. It should include VEP annotations, ClinPred scores, conservation features, etc.

### 4. Run the pipeline
```bash
python main.py
```

This will:
- Train and evaluate all models
- Perform multi-seed comparison
- Identify the best seed per model
- Generate metrics with confidence intervals
- Apply LIME and Permutation Importance
- Save all outputs to the `results/` folder

---

## 🧠 Deep Learning Architectures Used

- MLP – Multi-Layer Perceptron  
- CNN – 1D Convolutional Neural Network  
- DNN – Deep Feedforward Neural Network  
- RNN – Recurrent Neural Network  
- LSTM – Long Short-Term Memory  
- GRU – Gated Recurrent Unit  
- Transformer – Attention-based sequence model  

---

## 📊 Key Output Artifacts

| Output                             | Description                                        |
|------------------------------------|----------------------------------------------------|
| `combined_roc_with_ci.png`         | ROC curves with 95% confidence intervals          |
| `results_with_ci.xlsx`             | All performance metrics with bootstrapped CIs     |
| `LIME_GRU_TP_TN_FP_FN.png`         | Case-level explanations using LIME                |
| `permutation_importance_gru.png`   | Feature importance for GRU                        |
| `statistical_results.xlsx`         | Z-test, Shapiro-Wilk, Levene’s test, ANOVA        |
| `calibration_plots.png`            | Probability calibration for each DL model         |

---

## 📈 Evaluation Strategy

- **Multi-seed evaluation**: Models trained with multiple random seeds (e.g., 42, 101, 303, 404), selecting the best seed based on AUC and composite metrics.
- **Cross-validation**: Evaluated using stratified splits with bootstrapping to calculate 95% confidence intervals.
- **Statistical testing**: Z-test, Shapiro-Wilk, Levene, and ANOVA used to validate stability and significance across models.

---

## 📚 Citation

```bibtex
@article{ahmad2025interpretableDL,
  author = {Rahaf M. Ahmad, Noura Al Dhaheri, Mohd Saberi Mohamad, Bassam R. Ali*},
  title = {A Clinically Interpretable Deep Learning Framework for Accurate Prediction of Breast Cancer Missense Variant Pathogenicity},
  year = {2025},
  journal = {To be updated upon acceptance},
  url = {https://github.com/rahafahmad89/DL-clinically-interpretable-framework}
}
```

---

## 🔐 License

Distributed under the MIT License. See `LICENSE` for more details.

---

## 👩‍💻 Author

**Rahaf M. Ahmad**  
Ph.D. Candidate – Genetics & Machine Learning  
United Arab Emirates University  
ORCID: [0000-0002-7531-5264](https://orcid.org/0000-0002-7531-5264)

---

## 🤝 Acknowledgements

This framework is part of an ongoing effort to promote interpretable AI in clinical genomics and improve decision-making in precision oncology.



import os, random, warnings, multiprocessing, time
import numpy as np
import pandas as pd
import joblib
from itertools import combinations

# ── Threading ──────────────────────────────────────────────────────────────
_NC = multiprocessing.cpu_count()
os.environ.setdefault("OMP_NUM_THREADS",        str(_NC))
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(_NC))
os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(_NC))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"✓ GPU: {gpus[0].name}")
else:
    print("⚠ No GPU — CPU only")

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, Conv1D, Flatten,
    LSTM, SimpleRNN, GRU, MultiHeadAttention, GlobalAveragePooling1D,
    LayerNormalization, LeakyReLU
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, confusion_matrix, average_precision_score,
    roc_curve, precision_recall_curve, brier_score_loss, cohen_kappa_score
)
from sklearn.calibration import calibration_curve
from scipy.stats import shapiro, levene, f_oneway, mannwhitneyu, kruskal
from scipy import stats as scipy_stats
from statsmodels.stats.weightstats import ztest
from lime.lime_tabular import LimeTabularExplainer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION  ← edit paths here
# ─────────────────────────────────────────────
DATA_PATH       = "cleaned_dataset.csv"
TEST_PATH       = "Testing_456_final.csv"
TARGET_COL      = "CLIN_SIG"
SEEDS           = [42, 101, 202, 303, 404]
TEST_SIZE       = 0.2
N_RFE_FEATURES  = 10
CORR_THRESHOLD  = 0.9
BATCH_SIZE      = 32
EPOCHS          = 50
LR              = 0.001
BOOTSTRAP_ITERS = 1000
OUTPUT_DIR      = "outputs_complete"
MODEL_SAVE_DIR  = "saved_models_complete"
DPI             = 300

CIRCULAR_FEATURES = [
    "clinvar_id", "ClinPred", "clinvar_clnsig",
    "clinvar_sig_simple", "clinvar_gold_stars",
]

COMPOSITE_WEIGHTS = {
    "AUC": 0.30, "Precision": 0.20, "Recall": 0.20,
    "F1": 0.10, "Specificity": 0.10, "Sensitivity": 0.05,
    "MCC": 0.025, "Kappa": 0.025,
}

BENCHMARK_TOOLS = [
    "CADD_PHRED", "MetaRNN_rankscore", "BayesDel_addAF_rankscore",
    "am_pathogenicity", "MetaLR_rankscore", "DANN_rankscore",
    "PrimateAI_rankscore", "gMVP_rankscore", "REVEL_rankscore",
    "MVP_rankscore", "MetaSVM_rankscore",
]

COLS_TO_DROP = [
    "cDNA_position","Location","Protein_position","CDS_position",
    "#Uploaded_variation","Uploaded_variation","Allele","Gene",
    "Feature","Feature_type","Consequence","Codons","Existing_variation",
    "STRAND","HGVSc","HGVSp","SYMBOL","SYMBOL_SOURCE","HGNC_ID",
    "BIOTYPE","CANONICAL","MANE_SELECT","MANE_PLUS_CLINICAL","TSL",
    "CCDS","ENSP","SWISSPROT","TREMBL","UNIPARC","UNIPROT_ISOFORM",
    "SOURCE","DOMAINS","miRNA","AF","AFR_AF","AMR_AF","EAS_AF",
    "EUR_AF","SAS_AF","gnomADe_AF","gnomADe_AFR_AF","gnomADe_AMR_AF",
    "gnomADe_ASJ_AF","gnomADe_EAS_AF","gnomADe_FIN_AF","gnomADe_MID_AF",
    "gnomADe_NFE_AF","gnomADe_OTH_AF","gnomADe_SAS_AF","gnomADg_AF",
    "FLAGS","MOTIF_NAME","MOTIF_POS","HIGH_INF_POS","MOTIF_SCORE_CHANGE",
    "TRANSCRIPTION_FACTORS","clinvar_id","clinvar_review","clinvar_trait",
    "clinvar_var_source","clinvar_MedGen_id","clinvar_OMIM_id",
    "clinvar_allele_id","HGVS_OFFSET","HGVS","Amino_acids",
] + CIRCULAR_FEATURES

FEATURE_META = {
    "MPC":                              ("MPC", "Samocha et al. 2017", "Missense constraint score"),
    "BayesDel_addAF_rankscore":         ("BayesDel+AF", "Feng et al. 2017", "Deleteriousness meta-predictor"),
    "DEOGEN2_score":                    ("DEOGEN2", "Raimondi et al. 2017", "Domain-aware deleteriousness"),
    "Eigen-PC-raw_coding_rankscore":    ("Eigen-PC", "Ionita-Laza et al. 2016","Spectral functional scoring"),
    "FATHMM_converted_rankscore":       ("FATHMM", "Shihab et al. 2013", "Functional annotation"),
    "SiPhy_29way_logOdds_rankscore":    ("SiPhy", "Garber et al. 2009", "Evolutionary conservation 29-way"),
    "gMVP_rankscore":                   ("gMVP", "Zhang et al. 2022", "Graph-based pathogenicity"),
    "phyloP470way_mammalian_rankscore": ("phyloP-470", "Pollard et al. 2010", "Conservation 470 mammals"),
    "am_pathogenicity":                 ("AlphaMissense", "Cheng et al. 2023", "AlphaFold-based — note ClinVar trained"),
    "ClinPred":                         ("ClinPred EXCLUDED", "Alirezaie et al. 2018", "ClinVar-trained — circular"),
    "clinvar_id":                       ("clinvar_id EXCLUDED", "NCBI", "DB identifier — no biology"),
    "CADD_PHRED":                       ("CADD", "Kircher et al. 2014", "Combined annotation depletion"),
    "MetaRNN_rankscore":                ("MetaRNN", "Dong et al. 2021", "Recurrent meta-predictor"),
    "DANN_rankscore":                   ("DANN", "Quang et al. 2015", "Deep learning conservation"),
    "BayesDel_noAF_rankscore":          ("BayesDel noAF", "Feng et al. 2017", "Deleteriousness without AF"),
    "MPC_rankscore":                    ("MPC rank", "Samocha et al. 2017", "Missense constraint rank"),
}

os.makedirs(OUTPUT_DIR,     exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def set_seeds(seed):
    np.random.seed(seed); tf.random.set_seed(seed)
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)


def normalise_labels(series):
    if pd.api.types.is_numeric_dtype(series):
        out = pd.to_numeric(series, errors="coerce")
        v = out.where(out.isin([0, 1]))
        if v.notna().sum() > 0:
            return v
    PATH  = {"pathogenic","likely_pathogenic","pathogenic/likely_pathogenic"}
    BEN   = {"benign","likely_benign","benign/likely_benign"}
    DROP  = {"uncertain_significance","conflicting_interpretations_of_pathogenicity",
             "not_provided","risk_factor","-","nan",""}
    def _m(val):
        v = str(val).lower().strip()
        if v in DROP: return np.nan
        if v in PATH: return 1
        if v in BEN:  return 0
        if "," in v:
            t = {x.strip() for x in v.split(",")}
            if bool(t&PATH) and not bool(t&BEN): return 1
            if bool(t&BEN)  and not bool(t&PATH): return 0
            return np.nan
        if "pathogenic" in v and "benign" not in v: return 1
        if "benign" in v and "pathogenic" not in v: return 0
        return np.nan
    out = series.apply(_m)
    p = int((out==1).sum()); b = int((out==0).sum())
    print(f"  Labels: {out.notna().sum()}/{len(series)} (P={p} B={b} dropped={out.isna().sum()})")
    return out


def get_callbacks():
    return [
        EarlyStopping(monitor='val_auc', patience=10,
                      restore_best_weights=True, mode='max', verbose=0),
        ReduceLROnPlateau(monitor='val_auc', factor=0.5,
                          patience=5, mode='max', min_lr=1e-6, verbose=0)
    ]


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df, rfe_features=None, scaler=None, imputer=None,
               le_dict=None, fit=True, external_remap=False):
    df = df.copy()

    # External-only remapping — does NOT affect training
    if external_remap:
        if 'MPC' in (rfe_features or []) and 'MPC' not in df.columns:
            if 'MPC_rankscore' in df.columns:
                df['MPC'] = pd.to_numeric(df['MPC_rankscore'].replace('-',np.nan), errors='coerce')
                print("  ✓ MPC_rankscore → MPC")
        if 'SiPhy_29way_logOdds_rankscore' in (rfe_features or []) \
           and 'SiPhy_29way_logOdds_rankscore' not in df.columns:
            df['SiPhy_29way_logOdds_rankscore'] = np.nan
            print("  ⚠ SiPhy unavailable → training median")
        for col in (rfe_features or []):
            if col in df.columns and df[col].dtype == object:
                df[col] = pd.to_numeric(df[col].replace('-',np.nan), errors='coerce')

    drop_cols = [c for c in COLS_TO_DROP if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    y = df.pop(TARGET_COL).values.astype(int)

    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    if fit:
        le_dict = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
    else:
        for col in cat_cols:
            if col in le_dict:
                known = set(le_dict[col].classes_)
                df[col] = df[col].astype(str).apply(
                    lambda v: v if v in known else le_dict[col].classes_[0])
                df[col] = le_dict[col].transform(df[col])
            else:
                df[col] = 0

    df = df.select_dtypes(include=[np.number])

    if fit:
        imputer = SimpleImputer(strategy="median")
        X_imp   = imputer.fit_transform(df)
    else:
        train_cols = list(imputer.feature_names_in_)
        for c in train_cols:
            if c not in df.columns: df[c] = np.nan
        df = df[train_cols]
        X_imp = imputer.transform(df)

    feat_names = list(imputer.feature_names_in_ if not fit else df.columns)
    X_df = pd.DataFrame(X_imp, columns=feat_names)

    if fit:
        corr  = X_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > CORR_THRESHOLD)]
        X_df.drop(columns=to_drop, inplace=True)
        print(f"  Dropped {len(to_drop)} correlated features")
        rf  = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rfe = RFE(estimator=rf, n_features_to_select=N_RFE_FEATURES)
        rfe.fit(X_df, y)
        rfe_features = X_df.columns[rfe.support_].tolist()
        leaked = [f for f in rfe_features
                  if any(c.lower() in f.lower() for c in CIRCULAR_FEATURES)]
        print(f"  RFE features: {rfe_features}")
        print(f"  Circular in RFE: {leaked if leaked else 'none ✓'}")

    for c in rfe_features:
        if c not in X_df.columns: X_df[c] = 0
    X_sel = X_df[rfe_features].values

    if fit:
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X_sel)
    else:
        X_scaled = scaler.transform(X_sel)

    return X_scaled, y, rfe_features, scaler, imputer, le_dict


# ─────────────────────────────────────────────
# MODEL DEFINITIONS  (exact manuscript architectures)
# ─────────────────────────────────────────────
def create_mlp(n):
    m = Sequential([
        Dense(256,input_dim=n,kernel_initializer="he_normal"),
        LayerNormalization(),LeakyReLU(alpha=0.1),Dropout(0.4),
        Dense(128,kernel_initializer="he_normal"),
        LayerNormalization(),LeakyReLU(alpha=0.1),Dropout(0.3),
        Dense(64,kernel_initializer="he_normal"),
        LeakyReLU(alpha=0.1),Dropout(0.2),Dense(1,activation='sigmoid')])
    m.compile(Adam(LR),'binary_crossentropy',
              metrics=[tf.keras.metrics.AUC(name='auc')]); return m

def create_dnn(n):
    m = Sequential([
        Dense(256,input_dim=n,activation='relu'),
        BatchNormalization(),Dropout(0.5),
        Dense(128,activation='relu'),
        BatchNormalization(),Dropout(0.4),
        Dense(64,activation='relu'),Dense(1,activation='sigmoid')])
    m.compile(Adam(LR),'binary_crossentropy',
              metrics=[tf.keras.metrics.AUC(name='auc')]); return m

def create_cnn(n):
    m = Sequential([
        Conv1D(64,kernel_size=3,activation='relu',input_shape=(n,1)),
        Flatten(),Dense(64,activation='relu'),Dense(1,activation='sigmoid')])
    m.compile(Adam(LR),'binary_crossentropy',
              metrics=[tf.keras.metrics.AUC(name='auc')]); return m

def create_lstm(n):
    m = Sequential([
        Conv1D(filters=64,kernel_size=3,activation='relu',input_shape=(n,1)),
        Dropout(0.3),LSTM(64,return_sequences=True),LSTM(32),
        Dense(1,activation='sigmoid')])
    m.compile(Adam(LR),'binary_crossentropy',
              metrics=[tf.keras.metrics.AUC(name='auc')]); return m

def create_rnn(n):
    m = Sequential([
        SimpleRNN(64,input_shape=(n,1)),
        Dense(32,activation='relu'),Dense(1,activation='sigmoid')])
    m.compile(Adam(LR),'binary_crossentropy',
              metrics=[tf.keras.metrics.AUC(name='auc')]); return m

def create_gru(n):
    m = Sequential([
        GRU(128,return_sequences=True,input_shape=(n,1)),
        GRU(64),Dense(32,activation='relu'),Dense(1,activation='sigmoid')])
    m.compile(Adam(LR),'binary_crossentropy',
              metrics=[tf.keras.metrics.AUC(name='auc')]); return m

def create_transformer(n):
    inp = Input(shape=(n,1))
    x   = MultiHeadAttention(key_dim=64,num_heads=4)(inp,inp)
    x   = Dropout(0.1)(x); x = LayerNormalization(epsilon=1e-6)(x)
    res = x+inp
    x   = Dense(128,activation="relu")(res); x = Dropout(0.1)(x)
    x   = LayerNormalization(epsilon=1e-6)(x); x = x+res
    x   = GlobalAveragePooling1D()(x); x = Dense(64,activation='relu')(x)
    out = Dense(1,activation='sigmoid')(x)
    md  = Model(inp,out)
    md.compile(Adam(LR),'binary_crossentropy',
               metrics=[tf.keras.metrics.AUC(name='auc')]); return md

MODEL_BUILDERS = {
    "MLP":         (create_mlp,         "2d"),
    "DNN":         (create_dnn,         "2d"),
    "CNN":         (create_cnn,         "3d"),
    "LSTM":        (create_lstm,        "3d"),
    "RNN":         (create_rnn,         "3d"),
    "GRU":         (create_gru,         "3d"),
    "Transformer": (create_transformer, "3d"),
}

def reshape(name, X):
    return X if MODEL_BUILDERS[name][1]=="2d" \
           else X.reshape(X.shape[0],X.shape[1],1)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def ece_score(yt, yp, n_bins=10):
    bins = np.linspace(0,1,n_bins+1); ece = 0.0; n = len(yt)
    for i in range(n_bins):
        m = (yp>=bins[i])&(yp<bins[i+1])
        if m.sum(): ece += (m.sum()/n)*abs(yt[m].mean()-yp[m].mean())
    return ece

def compute_metrics(yt, yp):
    yb = (yp>=0.5).astype(int)
    tn,fp,fn,tp = confusion_matrix(yt,yb).ravel()
    return {
        "AUC":         roc_auc_score(yt,yp),
        "AP":          average_precision_score(yt,yp),
        "Accuracy":    accuracy_score(yt,yb),
        "F1":          f1_score(yt,yb,zero_division=0),
        "Precision":   precision_score(yt,yb,zero_division=0),
        "Recall":      recall_score(yt,yb,zero_division=0),
        "Specificity": tn/(tn+fp) if (tn+fp) else 0,
        "Sensitivity": tp/(tp+fn) if (tp+fn) else 0,
        "MCC":         matthews_corrcoef(yt,yb),
        "Kappa":       cohen_kappa_score(yt,yb),
        "FPR":         fp/(fp+tn) if (fp+tn) else 0,
        "FNR":         fn/(fn+tp) if (fn+tp) else 0,
        "Brier":       brier_score_loss(yt,yp),
        "ECE":         ece_score(yt,yp),
        "TP":int(tp),"TN":int(tn),"FP":int(fp),"FN":int(fn),
    }

def bootstrap_ci(yt, yp, n=BOOTSTRAP_ITERS, alpha=0.05):
    rng = np.random.default_rng(42)
    idx = rng.integers(0,len(yt),(n,len(yt)))
    keys= [k for k in compute_metrics(yt,yp) if k not in ("TP","TN","FP","FN")]
    boot= {k:[] for k in keys}
    for i in range(n):
        try:
            m = compute_metrics(yt[idx[i]],yp[idx[i]])
            for k in keys: boot[k].append(m[k])
        except: pass
    return {k:(np.percentile(v,100*alpha/2),np.percentile(v,100*(1-alpha/2)))
            for k,v in boot.items()}

def metrics_table(y_true, preds_dict, fname, label=""):
    rows = []
    for mn in preds_dict:
        m  = compute_metrics(y_true, preds_dict[mn])
        ci = bootstrap_ci(y_true, preds_dict[mn])
        row = {"Model": mn}
        for k, v in m.items():
            if k in ("TP","TN","FP","FN"): row[k] = int(v)
            else:
                lo,hi = ci.get(k,(np.nan,np.nan))
                row[k] = f"{v:.4f} [{lo:.4f}–{hi:.4f}]"
        rows.append(row)
    df_out = pd.DataFrame(rows).set_index("Model")
    df_out.to_excel(os.path.join(OUTPUT_DIR, fname))
    if label:
        print(f"\n{label}")
        print(df_out[["AUC","F1","MCC","Specificity","ECE",
                       "TP","TN","FP","FN"]].to_string())
    return df_out


# ─────────────────────────────────────────────
# DELONG TEST
# ─────────────────────────────────────────────
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1)+1; i=j
    T2=np.empty(N); T2[J]=T; return T2

def delong_test(y_true, pa, pb):
    def _var(y,p):
        order=(-p).argsort(); y1=y[order]
        n1=int(y1.sum()); n0=len(y1)-n1
        if n1==0 or n0==0: return 0.5,0
        tp=np.zeros(len(order)+1); fp=np.zeros(len(order)+1)
        for i in range(len(order)):
            tp[i+1]=tp[i]+y1[i]; fp[i+1]=fp[i]+(1-y1[i])
        auc=sum((fp[i]-fp[i-1])*(tp[i]+tp[i-1])/2 for i in range(1,len(tp)))/(n1*n0)
        pv1=np.zeros(n1); pv0=np.zeros(n0); i1=i0=0
        for i in range(len(order)):
            if y1[i]==1: pv1[i1]=fp[i]/n0; i1+=1
            else:        pv0[i0]=tp[i]/n1; i0+=1
        return auc, np.var(pv1,ddof=1)/n1+np.var(pv0,ddof=1)/n0
    aa,va=_var(y_true,pa); ab,vb=_var(y_true,pb)
    se=np.sqrt(va+vb-2*min(np.sqrt(va*vb),0))
    if se==0: return 1.0
    return float(2*(1-scipy_stats.norm.cdf(abs((aa-ab)/se))))


# ─────────────────────────────────────────────
# PMI
# ─────────────────────────────────────────────
def run_pmi(model, X, y, feat_names, model_name, n_repeats=10):
    base = roc_auc_score(y, model.predict(reshape(model_name,X),verbose=0).ravel())
    imp  = np.zeros((len(feat_names),n_repeats))
    for i in range(len(feat_names)):
        for r in range(n_repeats):
            Xp=X.copy(); np.random.shuffle(Xp[:,i])
            imp[i,r]=base-roc_auc_score(
                y,model.predict(reshape(model_name,Xp),verbose=0).ravel())
    return imp.mean(axis=1), imp.std(axis=1)


# ─────────────────────────────────────────────
# LIME
# ─────────────────────────────────────────────
def run_lime_all_models(models, X_tr, X_te, y_te, predictions,
                        feat_names, best_mn, out_dir):
    """Fig: LIME for all 7 models (one representative case each)."""
    explainer = LimeTabularExplainer(
        X_tr, feature_names=feat_names,
        class_names=["Benign","Pathogenic"],
        discretize_continuous=True, random_state=42)

    # Pick one pathogenic TP as representative case
    best_prob = predictions[best_mn]
    y_pred    = (best_prob>=0.5).astype(int)
    tp_idx    = np.where((y_te==1)&(y_pred==1))[0]
    idx       = int(tp_idx[0]) if len(tp_idx) else 0

    n_models = len(models)
    ncols    = 4
    nrows    = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    axes = axes.ravel() if nrows > 1 else [axes] if ncols==1 else axes

    for ax_i, (mn, model) in enumerate(models.items()):
        def predict_fn(x, m=model, n=mn):
            p = m.predict(reshape(n,x),verbose=0).ravel()
            return np.column_stack([1-p,p])
        exp   = explainer.explain_instance(X_te[idx],predict_fn,num_features=10)
        vals  = exp.as_list()
        feats = [v[0] for v in vals]
        wts   = [v[1] for v in vals]
        axes[ax_i].barh(feats,wts,
                        color=["steelblue" if w>0 else "tomato" for w in wts])
        axes[ax_i].axvline(0,color="black",lw=0.8)
        axes[ax_i].set_title(mn,fontsize=11)
        axes[ax_i].tick_params(axis='y',labelsize=7)

    for ax_i in range(n_models, len(axes)):
        axes[ax_i].set_visible(False)

    fig.suptitle("Fig — LIME Explanations (representative pathogenic case)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"Fig_LIME_all_models.png"),dpi=DPI)
    plt.close()
    print("  Saved Fig_LIME_all_models.png")


def run_lime_best_model(model, X_tr, X_te, y_te, y_prob,
                        feat_names, mn, out_dir):
    """Fig: LIME for best model — TP / TN / FP / FN cases."""
    explainer = LimeTabularExplainer(
        X_tr, feature_names=feat_names,
        class_names=["Benign","Pathogenic"],
        discretize_continuous=True, random_state=42)
    def predict_fn(x):
        p = model.predict(reshape(mn,x),verbose=0).ravel()
        return np.column_stack([1-p,p])

    y_pred  = (y_prob>=0.5).astype(int)
    cases   = {}
    for label,(t,p) in [("TP",(1,1)),("TN",(0,0)),("FP",(0,1)),("FN",(1,0))]:
        mask = (y_te==t)&(y_pred==p)
        if mask.any(): cases[label] = np.where(mask)[0][0]

    fig, axes = plt.subplots(2,2,figsize=(16,12))
    axes = axes.ravel()
    for ax_i,(label,idx) in enumerate(cases.items()):
        exp   = explainer.explain_instance(X_te[idx],predict_fn,num_features=10)
        vals  = exp.as_list()
        feats = [v[0] for v in vals]
        wts   = [v[1] for v in vals]
        axes[ax_i].barh(feats,wts,
                        color=["steelblue" if w>0 else "tomato" for w in wts])
        axes[ax_i].axvline(0,color="black",lw=0.8)
        axes[ax_i].set_title(
            f"({chr(65+ax_i)}) {label}  |  True={y_te[idx]}  "
            f"p={y_prob[idx]:.3f}",fontsize=11)
    for ax_i in range(len(cases),4):
        axes[ax_i].set_visible(False)
    fig.suptitle(f"Fig  — LIME Explanations ({mn}) — TP/TN/FP/FN",fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"Fig_LIME_best_model_TP_TN_FP_FN.png"),dpi=DPI)
    plt.close()
    print("  Saved Fig_LIME_best_model_TP_TN_FP_FN.png")

    # Also save individual files
    lime_dir = os.path.join(out_dir,"lime_individual")
    os.makedirs(lime_dir,exist_ok=True)
    for label,idx in cases.items():
        exp  = explainer.explain_instance(X_te[idx],predict_fn,num_features=10)
        fig = exp.as_pyplot_figure()
        fig.suptitle(f"{mn} — {label}",fontsize=11)
        plt.tight_layout()
        fig.savefig(os.path.join(lime_dir,f"{mn}_{label}.png"),dpi=DPI)
        plt.close(fig)


# ─────────────────────────────────────────────
# STATISTICAL ANALYSIS
# ─────────────────────────────────────────────
def statistical_analysis(y_true, predictions_dict, out_dir):
    rows = []
    kw_stat, kw_p = kruskal(*predictions_dict.values())
    for mn, yp in predictions_dict.items():
        try:
            z_stat, p_z   = ztest(y_true.astype(float),yp)
            sh_stat,p_sh  = shapiro(yp[:min(5000,len(yp))])
            lv_stat,p_lv  = levene(y_true.astype(float),yp)
            f_stat, p_f   = f_oneway(y_true.astype(float),yp)
            mw_stat,p_mw  = mannwhitneyu(yp[y_true==1],yp[y_true==0],alternative='greater')
            rows.append({
                "Model":mn,
                "Z-Statistic":round(z_stat,4),  "P-value Z-Test":round(p_z,4),
                "Shapiro Stat":round(sh_stat,4), "P-value Shapiro":round(p_sh,4),
                "Levene Stat":round(lv_stat,4),  "P-value Levene":round(p_lv,4),
                "F-statistic":round(f_stat,4),   "P-value F-test":round(p_f,4),
                "MannWhitney U":round(mw_stat,1),"P-value MWU":round(p_mw,6),
            })
        except Exception as e:
            print(f"  Stats error ({mn}): {e}")

    stat_df = pd.DataFrame(rows)
    kw_row  = {c:"" for c in stat_df.columns}
    kw_row["Model"] = f"Kruskal-Wallis (all): stat={kw_stat:.2f} p={kw_p:.4f}"
    stat_df = pd.concat([stat_df,pd.DataFrame([kw_row])],ignore_index=True)
    stat_df.to_excel(os.path.join(out_dir,"Table_statistical_tests.xlsx"),index=False)

    # Fig: 2x2 subplots — A=Z, B=Shapiro, C=Levene, D=F-test
    stat_plot = stat_df[stat_df["Model"].isin(predictions_dict.keys())].copy()
    fig, axes = plt.subplots(2,2,figsize=(14,10))
    axes = axes.ravel()
    tests = [
        ("Z-Statistic",  "P-value Z-Test",  "(A) Z-Test"),
        ("Shapiro Stat", "P-value Shapiro", "(B) Shapiro-Wilk"),
        ("Levene Stat",  "P-value Levene",  "(C) Levene"),
        ("F-statistic",  "P-value F-test",  "(D) F-Test"),
    ]
    for ax,(stat,pval,title) in zip(axes,tests):
        stat_plot[stat] = pd.to_numeric(stat_plot[stat],errors='coerce')
        stat_plot[pval] = pd.to_numeric(stat_plot[pval],errors='coerce')
        sns.barplot(data=stat_plot,x="Model",y=stat,palette="viridis",ax=ax)
        for i,(_,row) in enumerate(stat_plot.iterrows()):
            if pd.isna(row[pval]): continue
            sig=("***" if row[pval]<0.001 else "**" if row[pval]<0.01
                 else "*" if row[pval]<0.05 else "ns")
            ax.text(i,row[stat],sig,ha='center',va='bottom',color='red',fontsize=10)
        ax.set_title(title,fontsize=11); ax.tick_params(axis='x',rotation=45)
    plt.suptitle("Fig — Statistical Assessment of Deep Learning Classifiers\n"
                 "(unit: per-variant prediction scores; nonparametric tests added)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"Fig_statistical_tests.png"),dpi=DPI)
    plt.close()
    return stat_df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    t0 = time.time()

    print("="*65)
    print("STUDY SCOPE")
    print("="*65)
    print("  Variant type: Germline missense (breast cancer predisposition)")
    print("  Genes: BRCA1, BRCA2, TP53, ATM, PALB2, CDH1 and related panel")
    print("  No SMOTE: class_weight='balanced' used instead (R2#2+R3#2)")
    print("  Hyperparameters: LR∈{0.0001,0.001,0.01}→0.001 selected;")
    print("                   Batch∈{32,64,128}→32 selected")
    print("  Split: data split BEFORE any preprocessing (R3#2)")
    print("  Circular features excluded: clinvar_id, ClinPred (R2#1+R3#1)")

    # ── Load training data ─────────────────────
    print("\nLoading training data …")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df[TARGET_COL] = normalise_labels(df[TARGET_COL])
    df.dropna(subset=[TARGET_COL], inplace=True)
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    bench_avail = [t for t in BENCHMARK_TOOLS if t in df.columns
                   and pd.to_numeric(df[t],errors='coerce').notna().mean()>0.3]
    print(f"  Benchmark tools in training data: {bench_avail}")

    # ── Split FIRST, then preprocess ──────────
    print("\nSplitting (80/20, stratified) …")
    df_tr, df_te = train_test_split(df, test_size=TEST_SIZE,
                                     random_state=42,
                                     stratify=df[TARGET_COL])
    df_tr = df_tr.copy().reset_index(drop=True)
    df_te = df_te.copy().reset_index(drop=True)
    print(f"  Train: {len(df_tr):,}  Internal test: {len(df_te):,}")

    print("\nPreprocessing (fit on train only) …")
    X_tr, y_tr, rfe_feats, scaler, imputer, le_dict = preprocess(df_tr, fit=True)
    X_te, y_te, *_ = preprocess(df_te, rfe_features=rfe_feats,
                                  scaler=scaler, imputer=imputer,
                                  le_dict=le_dict, fit=False)

    joblib.dump({"rfe_features":rfe_feats,"scaler":scaler,
                 "imputer":imputer,"le_dict":le_dict},
                os.path.join(MODEL_SAVE_DIR,"preprocessor.pkl"))

    # Calibration holdout (from train only, R3#7)
    X_tr2, X_cal, y_tr2, y_cal = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)

    # Class weights
    cw  = compute_class_weight("balanced",classes=np.unique(y_tr),y=y_tr)
    cw_dict = dict(zip(np.unique(y_tr).tolist(),cw.tolist()))
    print(f"  Class weights: {cw_dict}")

    # Leakage table
    all_num = [f for f in df_tr.select_dtypes(include=[np.number]).columns
               if f != TARGET_COL]
    leak_rows = []
    for feat in all_num:
        circ = any(c.lower() in feat.lower() for c in CIRCULAR_FEATURES)
        leak_rows.append({"Feature":feat,"Selected_by_RFE":feat in rfe_feats,
                          "Circular":circ,
                          "Note":"Excluded — circular" if circ else ""})
    pd.DataFrame(leak_rows).sort_values(
        ["Selected_by_RFE","Circular"],ascending=[False,False]
    ).to_excel(os.path.join(OUTPUT_DIR,"Table_leakage_flags.xlsx"),index=False)

    # Feature description table
    feat_rows = [{"Feature":f,
                  "Full Name":FEATURE_META.get(f,(f,"",""))[0],
                  "Source":FEATURE_META.get(f,(f,"",""))[1],
                  "Role":FEATURE_META.get(f,(f,"",""))[2],
                  "Circular":f in CIRCULAR_FEATURES}
                 for f in rfe_feats]
    pd.DataFrame(feat_rows).to_excel(
        os.path.join(OUTPUT_DIR,"Table_feature_descriptions.xlsx"),index=False)

    # Fig: correlation heatmap
    feat_df = pd.DataFrame(X_tr, columns=rfe_feats)
    fig, ax = plt.subplots(figsize=(11,9))
    sns.heatmap(feat_df.corr(),annot=True,cmap='coolwarm',fmt=".2f",
                annot_kws={"size":8},ax=ax)
    ax.set_title("Fig — Correlation Heatmap of RFE-Selected Features",fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"Fig_correlation_heatmap.png"),dpi=DPI)
    plt.close()

    # ── STEP 1: Multi-seed evaluation ─────────
    print("\n"+"="*65)
    print("STEP 1 — Multi-seed evaluation (5 seeds)")
    print("="*65)
    seed_aucs  = {mn:{} for mn in MODEL_BUILDERS}
    best_seeds = {}

    for mn,(builder,_) in MODEL_BUILDERS.items():
        print(f"\n  {mn}")
        for seed in SEEDS:
            set_seeds(seed)
            model = builder(X_tr.shape[1])
            model.fit(reshape(mn,X_tr),y_tr,
                      validation_split=0.15,
                      epochs=EPOCHS,batch_size=BATCH_SIZE,
                      class_weight=cw_dict,
                      callbacks=get_callbacks(),verbose=0)
            yp = model.predict(reshape(mn,X_te),verbose=0).ravel()
            a  = roc_auc_score(y_te,yp)
            seed_aucs[mn][seed] = a
            print(f"    seed {seed}: AUC={a:.4f}")
            tf.keras.backend.clear_session()
        bs  = max(seed_aucs[mn],key=seed_aucs[mn].get)
        best_seeds[mn] = bs
        aucs = list(seed_aucs[mn].values())
        print(f"  → Best seed: {bs}  Mean={np.mean(aucs):.4f}±{np.std(aucs):.4f}")

    # Table — seed AUC table (manuscript Table format)
    t1_rows = []
    for mn in MODEL_BUILDERS:
        row = {"Classifier":mn}
        for s in SEEDS: row[str(s)] = round(seed_aucs[mn][s],6)
        row["Best Seed"] = best_seeds[mn]
        row["Best AUC"]  = round(seed_aucs[mn][best_seeds[mn]],4)
        row["Mean AUC"]  = f"{np.mean(list(seed_aucs[mn].values())):.4f}"
        row["SD AUC"]    = f"{np.std(list(seed_aucs[mn].values())):.4f}"
        t1_rows.append(row)
    pd.DataFrame(t1_rows).to_excel(
        os.path.join(OUTPUT_DIR,"Table_seed_AUC.xlsx"),index=False)

    # Fig: seed AUC boxplot
    fig,ax = plt.subplots(figsize=(9,5))
    data = [[seed_aucs[mn][s] for s in SEEDS] for mn in MODEL_BUILDERS]
    bp   = ax.boxplot(data,labels=list(MODEL_BUILDERS.keys()),patch_artist=True)
    for patch,col in zip(bp['boxes'],
                         plt.cm.tab10(np.linspace(0,1,len(MODEL_BUILDERS)))):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax.set(ylabel="AUC",title="Fig — AUC Distribution Across 5 Seeds")
    ax.tick_params(axis='x',rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"Fig_seed_AUC_boxplot.png"),dpi=DPI)
    plt.close()

    # ── STEP 2: Final training with best seed ──
    print("\n"+"="*65)
    print("STEP 2 — Final models (best seed per model)")
    print("="*65)
    results={}; predictions={}; best_models={}

    for mn,(builder,_) in MODEL_BUILDERS.items():
        seed = best_seeds[mn]; set_seeds(seed)
        print(f"\n  {mn} (seed={seed})")
        model = builder(X_tr.shape[1])
        model.fit(reshape(mn,X_tr),y_tr,
                  validation_split=0.15,
                  epochs=EPOCHS,batch_size=BATCH_SIZE,
                  class_weight=cw_dict,
                  callbacks=get_callbacks(),verbose=0)
        yp = model.predict(reshape(mn,X_te),verbose=0).ravel()
        m  = compute_metrics(y_te,yp)
        results[mn]=m; predictions[mn]=yp; best_models[mn]=model
        model.save(os.path.join(MODEL_SAVE_DIR,f"{mn}_best.h5"))
        print(f"    AUC={m['AUC']:.4f} F1={m['F1']:.4f} "
              f"MCC={m['MCC']:.4f} ECE={m['ECE']:.4f} Brier={m['Brier']:.4f}")

    # Best model (composite score)
    print("\n  Composite weights:")
    for k,w in COMPOSITE_WEIGHTS.items(): print(f"    {k}: {w:.3f} ({w*100:.1f}%)")
    best_mn = max(results, key=lambda mn:
                  sum(results[mn].get(k,0)*w for k,w in COMPOSITE_WEIGHTS.items()))
    best_score = sum(results[best_mn].get(k,0)*w
                     for k,w in COMPOSITE_WEIGHTS.items())
    print(f"\n  ✓ Best model: {best_mn} (score={best_score:.4f})")

    # Composite scores table
    comp_rows = []
    for mn in MODEL_BUILDERS:
        s = sum(results[mn].get(k,0)*w for k,w in COMPOSITE_WEIGHTS.items())
        comp_rows.append({"Model":mn,"Composite_Score":round(s,6)})
    pd.DataFrame(comp_rows).to_excel(
        os.path.join(OUTPUT_DIR,"Table_composite_scores.xlsx"),index=False)

    # Consistency check
    print("\n  Consistency check (table AUC == figure AUC):")
    for mn in MODEL_BUILDERS:
        diff = abs(results[mn]["AUC"]-roc_auc_score(y_te,predictions[mn]))
        print(f"    {mn}: {results[mn]['AUC']:.6f} {'✓' if diff<1e-8 else '✗'}")

    # Table — internal metrics with CI
    metrics_table(y_te, predictions,
                  "Table_internal_metrics_CI.xlsx",
                  "INTERNAL TEST METRICS")

    # Calibration
    print("\nIsotonic recalibration …")
    recal_probs = {}
    cal_rows    = []
    for mn in MODEL_BUILDERS:
        yp_cal = best_models[mn].predict(reshape(mn,X_cal),verbose=0).ravel()
        yp_te  = predictions[mn]
        ir     = IsotonicRegression(out_of_bounds='clip')
        ir.fit(yp_cal, y_cal)
        yp_rec = ir.predict(yp_te)
        recal_probs[mn] = yp_rec
        cal_rows.append({
            "Model":mn,
            "Brier_raw":     round(brier_score_loss(y_te,yp_te),4),
            "Brier_isotonic":round(brier_score_loss(y_te,yp_rec),4),
            "ECE_raw":       round(ece_score(y_te,yp_te),4),
            "ECE_isotonic":  round(ece_score(y_te,yp_rec),4),
            "Delta_Brier":   round(brier_score_loss(y_te,yp_te)
                                   -brier_score_loss(y_te,yp_rec),4),
        })
    pd.DataFrame(cal_rows).to_excel(
        os.path.join(OUTPUT_DIR,"Table_calibration_ECE_Brier.xlsx"),index=False)

    # DeLong
    print("\nDeLong pairwise …")
    mnames   = list(MODEL_BUILDERS.keys())
    delong_p = pd.DataFrame(np.nan,index=mnames,columns=mnames)
    for a,b in combinations(mnames,2):
        p = delong_test(y_te,predictions[a],predictions[b])
        delong_p.loc[a,b]=p; delong_p.loc[b,a]=p
    np.fill_diagonal(delong_p.values,1.0)
    delong_p.to_excel(os.path.join(OUTPUT_DIR,"Table_delong_pvalues.xlsx"))

    # Statistical analysis → Fig 
    print("\nStatistical tests …")
    stat_df = statistical_analysis(y_te, predictions, OUTPUT_DIR)

    # ── EXTERNAL TEST SET ──────────────────────
    print("\n"+"="*65)
    print("EXTERNAL TEST SET")
    print("="*65)
    ext_df = pd.read_csv(TEST_PATH, low_memory=False)
    ext_df[TARGET_COL] = normalise_labels(ext_df[TARGET_COL])
    ext_df.dropna(subset=[TARGET_COL], inplace=True)
    ext_df[TARGET_COL] = ext_df[TARGET_COL].astype(int)
    ext_df.reset_index(drop=True, inplace=True)

    id_col = None
    for c in ["#Uploaded_variation","Uploaded_variation","ID"]:
        if c in ext_df.columns: id_col=ext_df[c].values; break

    X_ext, y_ext, *_ = preprocess(
        ext_df, rfe_features=rfe_feats, scaler=scaler,
        imputer=imputer, le_dict=le_dict, fit=False, external_remap=True)

    print(f"  Shape: {X_ext.shape}  "
          f"P={int((y_ext==1).sum())}  B={int((y_ext==0).sum())}")

    ext_preds = {}
    for mn,model in best_models.items():
        yp = model.predict(reshape(mn,X_ext),verbose=0).ravel()
        ext_preds[mn] = yp
        m  = compute_metrics(y_ext,yp)
        print(f"  {mn}: AUC={m['AUC']:.4f} F1={m['F1']:.4f} "
              f"MCC={m['MCC']:.4f} FP={m['FP']} FN={m['FN']}")

    # Table — external metrics (all 7 models)
    metrics_table(y_ext, ext_preds,
                  "Table_external_metrics_CI.xlsx",
                  "EXTERNAL TEST METRICS")

    # Table — benchmark ALL 7 DL models + standalone tools
    print("\nBenchmark vs standalone tools (Table) …")
    bench_rows = []
    # All 7 DL models first
    for mn in MODEL_BUILDERS:
        m = compute_metrics(y_ext, ext_preds[mn])
        bench_rows.append({
            "Tool/Model": mn, "Type": "Deep Learning (this study)",
            "AUC": round(m["AUC"],4), "F1": round(m["F1"],4),
            "MCC": round(m["MCC"],4), "n": len(y_ext),
        })
    # Standalone tools
    for tool in BENCHMARK_TOOLS:
        if tool in ext_df.columns:
            ts = pd.to_numeric(ext_df[tool].replace('-',np.nan),errors='coerce')
            valid = ts.notna()
            if valid.sum() > 30:
                try:
                    auc_t = roc_auc_score(y_ext[valid.values],ts[valid].values)
                    bench_rows.append({
                        "Tool/Model":tool,"Type":"Standalone predictor",
                        "AUC":round(auc_t,4),"F1":"N/A","MCC":"N/A",
                        "n":int(valid.sum()),
                    })
                except: pass
    bench_df = pd.DataFrame(bench_rows)
    bench_df.to_excel(os.path.join(OUTPUT_DIR,"Table_benchmark_all.xlsx"),
                      index=False)
    print(bench_df[["Tool/Model","Type","AUC","n"]].to_string(index=False))

    # ── FIGURES ───────────────────────────────
    print("\nGenerating figures …")
    colors = plt.cm.tab10(np.linspace(0,1,len(mnames)))
    col_d  = dict(zip(mnames,colors))

    # Fig: 3x2 grid — internal (A=ROC, B=PR, C=Cal) + external (D=ROC, E=PR, F=Cal)
    fig = plt.figure(figsize=(20,14))
    gs  = gridspec.GridSpec(2,3,figure=fig,hspace=0.4,wspace=0.35)
    ax_roc_i = fig.add_subplot(gs[0,0])
    ax_pr_i  = fig.add_subplot(gs[0,1])
    ax_cal_i = fig.add_subplot(gs[0,2])
    ax_roc_e = fig.add_subplot(gs[1,0])
    ax_pr_e  = fig.add_subplot(gs[1,1])
    ax_cal_e = fig.add_subplot(gs[1,2])

    for mn in mnames:
        col = col_d[mn]; lw = 2.5 if mn==best_mn else 1.2
        ls  = "-" if mn==best_mn else "--"
        ci  = bootstrap_ci(y_te,predictions[mn])
        lo,hi = ci["AUC"]
        a = roc_auc_score(y_te,predictions[mn])
        fpr,tpr,_ = roc_curve(y_te,predictions[mn])
        ax_roc_i.plot(fpr,tpr,color=col,lw=lw,ls=ls,
                      label=f"{mn} {a:.3f}[{lo:.3f}-{hi:.3f}]")
        pr,rc,_ = precision_recall_curve(y_te,predictions[mn])
        ap = average_precision_score(y_te,predictions[mn])
        ax_pr_i.plot(rc,pr,color=col,lw=lw,ls=ls,label=f"{mn} AP={ap:.3f}")
        fp2,mp = calibration_curve(y_te,predictions[mn],n_bins=10)
        ax_cal_i.plot(mp,fp2,"o-",color=col,lw=lw,label=mn)

    for mn in mnames:
        col = col_d[mn]; lw = 2.5 if mn==best_mn else 1.2
        ls  = "-" if mn==best_mn else "--"
        a   = roc_auc_score(y_ext,ext_preds[mn])
        fpr,tpr,_ = roc_curve(y_ext,ext_preds[mn])
        ax_roc_e.plot(fpr,tpr,color=col,lw=lw,ls=ls,label=f"{mn} {a:.3f}")
        pr,rc,_ = precision_recall_curve(y_ext,ext_preds[mn])
        ap = average_precision_score(y_ext,ext_preds[mn])
        ax_pr_e.plot(rc,pr,color=col,lw=lw,ls=ls,label=f"{mn} AP={ap:.3f}")
        try:
            fp2,mp = calibration_curve(y_ext,ext_preds[mn],n_bins=10)
            ax_cal_e.plot(mp,fp2,"o-",color=col,lw=lw,label=mn)
        except: pass

    for ax in [ax_roc_i,ax_roc_e]:
        ax.plot([0,1],[0,1],"k--",lw=0.8)
        ax.set(xlabel="FPR",ylabel="TPR"); ax.grid(alpha=0.3)
        ax.legend(fontsize=6,loc="lower right")
    ax_roc_i.set_title("(A) ROC — Internal Test")
    ax_roc_e.set_title("(D) ROC — External Test")
    for ax in [ax_pr_i,ax_pr_e]:
        ax.set(xlabel="Recall",ylabel="Precision"); ax.grid(alpha=0.3)
        ax.legend(fontsize=7,loc="lower left")
    ax_pr_i.set_title("(B) PR — Internal Test")
    ax_pr_e.set_title("(E) PR — External Test")
    for ax in [ax_cal_i,ax_cal_e]:
        ax.plot([0,1],[0,1],"k--",lw=0.8,label="Perfect")
        ax.set(xlabel="Mean predicted prob",ylabel="Fraction positives")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax_cal_i.set_title("(C) Calibration — Internal")
    ax_cal_e.set_title("(F) Calibration — External")
    fig.suptitle("Fig — Discriminative and Calibration Performance "
                 "(Internal + External, 95% Bootstrap CI)",fontsize=13)
    plt.savefig(os.path.join(OUTPUT_DIR,"Fig_ROC_PR_Calibration.png"),dpi=DPI)
    plt.close(); print("  Saved Fig_ROC_PR_Calibration.png")

    # Fig: PMI all models
    fig,axes = plt.subplots(2,4,figsize=(22,10))
    axes = axes.ravel()
    for ax_i,(mn,col) in enumerate(zip(mnames,colors)):
        imp_mean,imp_std = run_pmi(best_models[mn],X_te,y_te,rfe_feats,mn)
        sidx = np.argsort(imp_mean)
        axes[ax_i].barh([rfe_feats[i] for i in sidx],imp_mean[sidx],
                         xerr=imp_std[sidx],color=col,alpha=0.8,capsize=3)
        axes[ax_i].set_title(mn,fontsize=10)
        axes[ax_i].set_xlabel("AUC drop",fontsize=8)
        axes[ax_i].axvline(0,color='k',lw=0.6)
    axes[-1].set_visible(False)
    fig.suptitle("Fig — Permutation Feature Importance (all models)",fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"Fig_PMI_all_models.png"),dpi=DPI)
    plt.close(); print("  Saved Fig_PMI_all_models.png")

    # Fig 6 + Fig 7: LIME
    run_lime_all_models(best_models, X_tr, X_te, y_te, predictions,
                        rfe_feats, best_mn, OUTPUT_DIR)
    run_lime_best_model(best_models[best_mn], X_tr, X_te, y_te,
                        predictions[best_mn], rfe_feats, best_mn, OUTPUT_DIR)

    # Fig 8: External ROC with benchmark overlay
    fig,ax = plt.subplots(figsize=(9,7))
    for mn,col in zip(mnames,colors):
        fpr,tpr,_ = roc_curve(y_ext,ext_preds[mn])
        a = roc_auc_score(y_ext,ext_preds[mn])
        lw = 2.5 if mn==best_mn else 1.2
        ax.plot(fpr,tpr,color=col,lw=lw,
                ls="-" if mn==best_mn else "--",
                label=f"{mn} (AUC={a:.3f})")
    # Add standalone tools as dashed grey lines
    grey_shades = [str(x) for x in np.linspace(0.3,0.7,len(BENCHMARK_TOOLS))]
    for tool in BENCHMARK_TOOLS:
        if tool in ext_df.columns:
            ts = pd.to_numeric(ext_df[tool].replace('-',np.nan),errors='coerce')
            valid = ts.notna()
            if valid.sum()>30:
                try:
                    fpr2,tpr2,_ = roc_curve(y_ext[valid.values],ts[valid].values)
                    a2 = roc_auc_score(y_ext[valid.values],ts[valid].values)
                    ax.plot(fpr2,tpr2,lw=1,ls=':',color='grey',alpha=0.7,
                            label=f"{tool} AUC={a2:.3f}")
                except: pass
    ax.plot([0,1],[0,1],"k--",lw=0.8)
    ax.set(xlabel="FPR",ylabel="TPR",
           title="Fig  — External ROC: All DL Models vs Standalone Predictors")
    ax.legend(fontsize=6,loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"Fig_external_ROC_benchmark.png"),dpi=DPI)
    plt.close(); print("  Saved Fig_external_ROC_benchmark.png")

    # Fig 10: Isotonic calibration before/after
    fig,axes = plt.subplots(1,2,figsize=(14,6))
    for ax,(probs_d,title_c) in zip(axes,[
        (predictions,"(A) Raw Probabilities"),
        (recal_probs,"(B) Isotonic Recalibrated")
    ]):
        ax.plot([0,1],[0,1],"k--",lw=0.8,label="Perfect")
        for mn,col in zip(mnames,colors):
            try:
                fp2,mp = calibration_curve(y_te,probs_d[mn],n_bins=10)
                ax.plot(mp,fp2,"o-",color=col,label=mn)
            except: pass
        ax.set(xlabel="Mean Predicted Prob",ylabel="Fraction Positives",
               title=title_c); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle("Fig — Calibration Before and After Isotonic Recalibration",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"Fig_isotonic_calibration.png"),dpi=DPI)
    plt.close(); print("  Saved Fig_isotonic_calibration.png")

    # DeLong heatmap (supplementary / Fig 3 companion)
    fig,ax = plt.subplots(figsize=(8,7))
    annot  = delong_p.applymap(lambda v: f"{v:.3f}" if not np.isnan(v) else "")
    sns.heatmap(delong_p.astype(float),annot=annot,fmt="",
                cmap="RdYlGn_r",vmin=0,vmax=0.05,
                mask=np.eye(len(mnames),dtype=bool),
                ax=ax,linewidths=0.5,cbar_kws={"label":"p-value"})
    ax.set_title("DeLong Pairwise AUC Significance Heatmap\n"
                 "(red=p<0.05 significant; green=n.s.)",fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"Fig_DeLong_heatmap.png"),dpi=DPI)
    plt.close()

    # ── PREDICTION FILES ──────────────────────
    def make_pred_file(y_true,preds_d,id_arr,fname):
        bp  = preds_d[best_mn]
        out = pd.DataFrame()
        out["ID"] = id_arr if id_arr is not None \
                    else [f"v{i}" for i in range(len(y_true))]
        out[f"{best_mn}_score"]         = np.round(bp,4)
        out["Predicted_pathogenicity"]  = np.where(bp>=0.5,"Pathogenic","Benign")
        out["Triage"] = [
            "High-Conf Pathogenic" if p>=0.8
            else "High-Conf Benign" if p<=0.2
            else "Low-Conf Pathogenic" if p>=0.5
            else "Low-Conf Benign" for p in bp]
        out["CLIN_SIG"]         = np.where(y_true==1,"Pathogenic","Benign")
        out["Classification"]   = [
            "True Pathogenic"  if t==1 and p>=0.5
            else "False Pathogenic" if t==0 and p>=0.5
            else "True Benign"      if t==0 and p<0.5
            else "False Benign"
            for t,p in zip(y_true,bp)]
        for mn in preds_d:
            out[f"{mn}_score"] = np.round(preds_d[mn],4)
        out["Ensemble_score"] = np.round(
            np.stack(list(preds_d.values()),axis=1).mean(axis=1),4)
        out.to_excel(os.path.join(OUTPUT_DIR,fname),index=False)
        print(f"  Saved {fname}")
        print(f"    Triage: {out['Triage'].value_counts().to_dict()}")
        print(f"    Class:  {out['Classification'].value_counts().to_dict()}")

    make_pred_file(y_te,  predictions, None,   "predictions_internal.xlsx")
    make_pred_file(y_ext, ext_preds,   id_col, "predictions_external.xlsx")

    # ── FINAL SUMMARY ─────────────────────────
    elapsed = (time.time()-t0)/60
    print("\n"+"="*65)
    print(f"✓ Best model: {best_mn} (score={best_score:.4f})")
    print(f"✓ Internal AUC: {results[best_mn]['AUC']:.4f}")
    print(f"✓ External AUC: {roc_auc_score(y_ext,ext_preds[best_mn]):.4f}")
    print(f"✓ Done in {elapsed:.1f} min")
    print(f"\nOutputs → {OUTPUT_DIR}/")

        all_outputs = [
        ("Fig_ROC_PR_Calibration.png",        "ROC+PR+Cal"),
        ("Fig_statistical_tests.png",         "Statistical tests"),
        ("Fig_correlation_heatmap.png",       "Correlation heatmap"),
        ("Fig_PMI_all_models.png",            "PMI all models"),
        ("Fig_LIME_all_models.png",           "LIME all models"),
        ("Fig_LIME_best_model_TP_TN_FP_FN.png","LIME TP/TN/FP/FN"),
        ("Fig_external_ROC_benchmark.png",    "External ROC benchmark"),
        ("Fig_seed_AUC_boxplot.png",          "Seed AUC boxplot"),
        ("Fig_isotonic_calibration.png",      "Isotonic calibration"),
        ("Fig_DeLong_heatmap.png",            "DeLong heatmap"),
        ("Table_seed_AUC.xlsx",               "Seed AUC"),
        ("Table_internal_metrics_CI.xlsx",    "Internal metrics CI"),
        ("Table_external_metrics_CI.xlsx",    "External metrics CI"),
        ("Table_benchmark_all.xlsx",          "Benchmark all"),
        ("Table_delong_pvalues.xlsx",         "DeLong p-values"),
        ("Table_calibration_ECE_Brier.xlsx",  "Calibration ECE/Brier"),
        ("Table_feature_descriptions.xlsx",   "Feature descriptions"),
        ("Table_leakage_flags.xlsx",          "Leakage flags"),
        ("Table_composite_scores.xlsx",       "Composite scores"),
        ("Table_statistical_tests.xlsx",      "Statistical tests"),
        ("predictions_internal.xlsx",         "Predictions internal"),
        ("predictions_external.xlsx",         "Predictions external"),
    ]
    print("\nChecklist:")
    for fname,desc in all_outputs:
        path   = os.path.join(OUTPUT_DIR,fname)
        status = "✓" if os.path.exists(path) else "✗ MISSING"
        print(f"  {status}  {desc}")


if __name__ == "__main__":
    main()

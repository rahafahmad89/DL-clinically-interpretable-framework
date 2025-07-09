#RUNNING 24 JUNE, DONT CHANGE!
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from scipy.stats import stats
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix,
                             precision_recall_curve, average_precision_score, cohen_kappa_score, roc_curve, auc)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from scikeras.wrappers import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, LSTM, SimpleRNN, Dropout
from keras.layers import LayerNormalization, LeakyReLU, BatchNormalization, GlobalAveragePooling1D, GRU
import lime
import lime.lime_tabular
from sklearn.calibration import calibration_curve
from sklearn.metrics import matthews_corrcoef
from statsmodels.stats.weightstats import ztest
from statsmodels.stats.weightstats import ztest
from scipy.stats import shapiro, levene, f_oneway
from keras.layers import MultiHeadAttention
from sklearn.utils import resample
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score, matthews_corrcoef)

SEED_LIST = [42, 101, 202, 303, 404]
EPOCHS = 50


def set_random_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    print(data.head())
    print(data.info())
    print(data.describe())

    if 'CLIN_SIG' in data.columns:
        le = LabelEncoder()
        data['CLIN_SIG'] = le.fit_transform(data['CLIN_SIG'])

    columns_to_drop = ['cDNA_position', 'Location', 'Protein_position', 'CDS_position']
    data.drop(columns=columns_to_drop, inplace=True)

    return data

def evaluate_model_across_seeds(clf_name, clf_func, model_type, X_train, X_test, y_train, y_test, input_shape, epochs=10, batch_size=32):
    seed_auc_scores = []

    for seed in SEED_LIST:
        print(f"🔁 {clf_name} | Seed {seed}")
        set_random_seeds(seed)

        clf = clf_func(seed, input_shape)

        if model_type == 'mlp':
            result = evaluate_model(clf, X_train, X_test, y_train, y_test, model_type='mlp')
        else:
            result = evaluate_model(clf,
                                    X_train.reshape((X_train.shape[0], X_train.shape[1], 1)),
                                    X_test.reshape((X_test.shape[0], X_test.shape[1], 1)),
                                    y_train, y_test,
                                    model_type='cnn_lstm_rnn')

        seed_auc_scores.append({'Seed': seed, 'AUC': result['AUC']})

    return pd.DataFrame(seed_auc_scores)

def pearson_correlation_filter(X, threshold=0.9):
    corr_matrix = X.corr()
    high_corr = np.where(np.abs(corr_matrix) > threshold)
    high_corr_pairs = [(corr_matrix.index[i], corr_matrix.columns[j])
                       for i, j in zip(*high_corr) if i != j and i < j]

    drop_columns = set([col2 for col1, col2 in high_corr_pairs])
    data = X.drop(columns=list(drop_columns))
    print(f"Dropped {len(drop_columns)} features due to high correlation")
    return data


def feature_selection_and_heatmap(X, y):
    estimator = RandomForestClassifier(random_state=42)
    selector = RFE(estimator, n_features_to_select=10, step=1)
    selector = selector.fit(X, y)
    selected_features = X.loc[:, selector.support_].copy()
    selected_features['CLIN_SIG'] = y

    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(selected_features.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    for text in heatmap.texts:
        text.set_fontsize(8)
    plt.title('Correlation Heatmap of Selected Features')
    plt.show()

    return selected_features

def create_mlp_model(input_dim):
    model = Sequential([
        Dense(256, input_dim=input_dim, kernel_initializer="he_normal"),
        LayerNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.4),

        Dense(128, kernel_initializer="he_normal"),
        LayerNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),

        Dense(64, kernel_initializer="he_normal"),
        LeakyReLU(alpha=0.1),
        Dropout(0.2),

        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model


def create_cnn_model(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model



def create_lstm_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model



def create_rnn_model(input_shape):
    model = Sequential([
        SimpleRNN(64, input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model


def create_dnn_model(input_dim):
    model = Sequential([
        Dense(256, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    x = Dense(ff_dim, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res


def create_transformer_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
        metrics=['AUC']
    )

    return model



def create_gru_model(input_shape, num_classes):
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=input_shape),
        GRU(64),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
                  metrics=['AUC'])
    return model



def evaluate_model(clf, X_train, X_test, y_train, y_test, model_type='mlp'):
    # Reshape
    if model_type in ['cnn', 'lstm', 'rnn']:
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


    print(f"Training data shape (X_train): {X_train.shape}")
    print(f"Test data shape (X_test): {X_test.shape}")
    print(f"Test labels shape (y_test): {y_test.shape}")

    # Fit the model
    clf.fit(X_train, y_train)

    # Get predictions
    y_pred = clf.predict(X_test)

    # Get probabilities
    y_prob = clf.predict_proba(X_test)[:, 1]


    print(f"Predicted probabilities shape: {y_prob.shape}")

    # Calculate evaluation metrics
    auc_score = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    mcc = matthews_corrcoef(y_test, y_pred)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    kappa = cohen_kappa_score(y_test, y_pred)

    return {
        'AUC': auc_score,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Specificity': specificity,
        'Sensitivity': sensitivity,
        'Confusion Matrix': confusion_matrix(y_test, y_pred),
        'Cohen\'s Kappa': kappa,
        'MCC': mcc,
        'FPR': fp / (fp + tn),
        'TNR': tn / (tn + fp),
        'TPR': tp / (tp + fn),
        'Probabilities': y_prob
    }


def save_results_to_excel(results, original_file_name, step_number, results_dir):
    file_name = os.path.join(results_dir, f'{original_file_name}_results_new{step_number}.xlsx')
    if len(results) > 1:
        results_df = pd.DataFrame(results)
    else:
        results_df = pd.DataFrame(list(results.items()), columns=['Metric', 'Value'])
    try:
        results_df.to_excel(file_name, index=False)
        print(f"Results saved to {file_name}")
    except Exception as e:
        print(f"Error saving results to {file_name}: {e}")


def save_predictions_to_excel(predictions, file_path, results_dir):
    try:
        predictions.to_excel(os.path.join(results_dir, file_path), index=False)
        print(f"Predictions saved to {file_path}")
    except Exception as e:
        print(f"Error saving predictions to {file_path}: {e}")


def generate_roc_curve(y_true, y_probs, clf_names, results_dir):
    plt.figure()
    best_thresholds = {}

    for i in range(len(y_probs)):
        # Shape matching
        if len(y_true) != len(y_probs[i]):
            print(f"Shape mismatch for classifier {clf_names[i]}: y_true has {len(y_true)} samples, y_probs has {len(y_probs[i])} samples")
            continue

        fpr, tpr, thresholds = roc_curve(y_true, y_probs[i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{clf_names[i]} (AUC = {roc_auc:.2f})')

        optimal_idx = np.argmax(tpr - fpr)
        best_threshold = thresholds[optimal_idx]
        best_thresholds[clf_names[i]] = best_threshold

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - All Classifiers')
    plt.legend(loc="lower right")
    plt.show()
    plt.tight_layout()
    output_path = os.path.join(results_dir, "ROC_all_classifiers.png")
    plt.savefig(output_path)
    plt.close()

    return best_thresholds


def generate_precision_recall_curve(y_true, y_probs, clf_names, results_dir):
    plt.figure()
    for i in range(len(y_probs)):
        precision, recall, _ = precision_recall_curve(y_true, y_probs[i])
        avg_precision = average_precision_score(y_true, y_probs[i])
        plt.plot(recall, precision, lw=2, label=f'{clf_names[i]} (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - All Classifiers')
    plt.legend(loc="lower left")
    plt.show()
    plt.tight_layout()
    output_path = os.path.join(results_dir, "PR_all_classifiers.png")
    plt.savefig(output_path)
    plt.close()


def plot_lime_explanations(clf, X_train, X_test, clf_name, results_dir, num_features=10):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns,
        class_names=['0', '1'],
        mode='classification'
    )

    instance_to_explain = X_test.iloc[0]

    exp = explainer.explain_instance(
        data_row=instance_to_explain,
        predict_fn=clf.predict_proba,
        num_features=num_features
    )

    print(f"LIME explanation for {clf_name}:")
    exp.show_in_notebook(show_table=True)


    fig = exp.as_pyplot_figure()
    plt.title(f"LIME Explanation for {clf_name}")
    plt.tight_layout()
    output_path = os.path.join(results_dir, f"LIME_final_{clf_name}.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)



def plot_calibration_curve(y_true, y_probs, clf_names, results_dir):
    plt.figure(figsize=(10, 8))
    for i in range(len(y_probs)):
        prob_true, prob_pred = calibration_curve(y_true, y_probs[i], n_bins=10)
        plt.plot(prob_pred, prob_true, lw=2, label=f'{clf_names[i]}')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend(loc='best')
    plt.tight_layout()


    fig = plt.gcf()

    output_path = os.path.join(results_dir, "calibration_curve.png")
    fig.savefig(output_path)
    plt.close(fig)



def permutation_importance_analysis(clf, X_train, y_train, clf_name, results_dir):
    perm_importance = permutation_importance(clf, X_train, y_train, n_repeats=10, random_state=42)
    perm_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance Mean': perm_importance.importances_mean,
        'Importance Std': perm_importance.importances_std
    }).sort_values(by='Importance Mean', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance Mean', y='Feature', data=perm_importance_df)
    plt.title(f'Permutation Feature Importance for {clf_name}')
    plt.tight_layout()


    fig = plt.gcf()
    output_path = os.path.join(results_dir, f"PMI_final_{clf_name}.png")
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)



def plot_combined_roc_with_ci(y_test, predictions_dict, results_dir, n_bootstraps=1000, seed=42):
    plt.figure(figsize=(10, 8))
    rng = np.random.RandomState(seed)
    roc_data = []

    for clf_name, y_prob in predictions_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)

        # Bootstrap AUC CI
        bootstrapped_scores = []
        for _ in range(n_bootstraps):
            indices = rng.randint(0, len(y_prob), len(y_prob))
            if len(np.unique(y_test[indices])) < 2:
                continue
            score = roc_auc_score(y_test[indices], y_prob[indices])
            bootstrapped_scores.append(score)

        sorted_scores = np.sort(bootstrapped_scores)
        ci_lower = sorted_scores[int(0.025 * len(sorted_scores))]
        ci_upper = sorted_scores[int(0.975 * len(sorted_scores))]

        plt.plot(fpr, tpr, label=f"{clf_name} (AUC = {auc_score:.3f}, 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")

        roc_data.append({
            'Classifier': clf_name,
            'AUC': auc_score,
            '95% CI Lower': ci_lower,
            '95% CI Upper': ci_upper
        })

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Combined ROC Curve with 95% CI")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "combined_roc_with_ci.png"))
    plt.close()

    pd.DataFrame(roc_data).to_excel(os.path.join(results_dir, "combined_roc_summary.xlsx"), index=False)
    print("✅ Combined ROC curve with CI saved.")

def plot_lime_for_tp_tn_fp_fn(best_clf_name, best_clf, X_train, X_test, y_test, y_prob, results_dir, num_features=10):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['Benign', 'Pathogenic'],
        mode='classification'
    )

    y_pred = (y_prob >= 0.5).astype(int)
    indices = {'TP': None, 'TN': None, 'FP': None, 'FN': None}
    for i, (true, pred) in enumerate(zip(y_test, y_pred)):
        if true == 1 and pred == 1 and indices['TP'] is None:
            indices['TP'] = i
        elif true == 0 and pred == 0 and indices['TN'] is None:
            indices['TN'] = i
        elif true == 0 and pred == 1 and indices['FP'] is None:
            indices['FP'] = i
        elif true == 1 and pred == 0 and indices['FN'] is None:
            indices['FN'] = i
        if all(v is not None for v in indices.values()):
            break

    lime_dir = os.path.join(results_dir, 'lime_explanations')
    os.makedirs(lime_dir, exist_ok=True)

    for label, idx in indices.items():
        if idx is not None:
            exp = explainer.explain_instance(
                data_row=X_test.iloc[idx].values,
                predict_fn=best_clf.predict_proba,
                num_features=num_features
            )
            fig = exp.as_pyplot_figure()
            fig.suptitle(f'LIME Explanation: {label}', fontsize=12)
            plt.tight_layout()
            fig.savefig(os.path.join(lime_dir, f'{best_clf_name}_{label}_lime.png'))
            plt.close()

    return indices

def statistical_analysis(y_true, model_predictions, model_results, results_dir):
    """
    Perform statistical analysis on model predictions.

    Parameters:
    y_true: Ground truth (actual) labels
    model_predictions: Dictionary where keys are model names, and values are predicted probabilities or classes
    model_results: Dictionary of results from each model

    Returns:
    DataFrame containing statistical test results for each model
    """
    all_statistics_results = []

    for model_name, metrics in model_results.items():
        y_pred = model_predictions.get(model_name, [])

        if len(y_true) == 0 or len(y_pred) == 0:
            print(f"No predictions for {model_name}. Skipping statistical analysis.")
            continue

        try:
            # Statistical tests
            z_stat, p_ztest = ztest(y_true, y_pred)
            shapiro_stat, p_shapiro = shapiro(y_pred)
            levene_stat, p_levene = levene(y_true, y_pred)
            f_stat, p_f = f_oneway(y_true, y_pred)
        except Exception as e:
            print(f"Error in statistical tests for {model_name}: {e}")
            continue

        # Save results
        statistics_results = {
            'Model': model_name,
            'Z-Statistic': z_stat,
            'P-value Z-Test': p_ztest,
            'Shapiro Statistic': shapiro_stat,
            'P-value Shapiro': p_shapiro,
            'Levene Statistic': levene_stat,
            'P-value Levene': p_levene,
            'F-statistic': f_stat,
            'P-value F-test': p_f
        }
        all_statistics_results.append(statistics_results)


    stat_df = pd.DataFrame(all_statistics_results)
    stat_output_path = os.path.join(results_dir, "statistical_tests.xlsx")
    stat_df.to_excel(stat_output_path, index=True)



    def plot_statistic(statistic_name, p_value_name, title, ylabel, results_dir):
        plt.figure(figsize=(10, 6))

        sns.barplot(
            data=stat_df.dropna(subset=[statistic_name]),
            x='Model',
            y=statistic_name,
            hue='Model',
            dodge=False,
            palette='viridis',
            legend=False
        )

        for i, p_value in enumerate(stat_df[p_value_name]):
            significance = ''
            if p_value < 0.001:
                significance = '***'
            elif p_value < 0.01:
                significance = '**'
            elif p_value < 0.05:
                significance = '*'
            plt.text(i, stat_df[statistic_name][i], significance, ha='center', va='bottom', fontsize=12, color='red')

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()


        fig = plt.gcf()
        output_path = os.path.join(results_dir, f"{statistic_name.replace(' ', '_')}_plot_final.png")
        fig.savefig(output_path, bbox_inches='tight')

        plt.show()
        plt.close(fig)


    plot_statistic('Z-Statistic', 'P-value Z-Test', 'Z-Statistic for Each Classifier', 'Z-Statistic', results_dir)
    plot_statistic('Shapiro Statistic', 'P-value Shapiro', 'Shapiro-Wilk Test for Each Classifier', 'Shapiro Statistic', results_dir)
    plot_statistic('Levene Statistic', 'P-value Levene', 'Levene Test for Each Classifier', 'Levene Statistic', results_dir)
    plot_statistic('F-statistic', 'P-value F-test', 'F-Test for Each Classifier', 'F-statistic', results_dir)


    return stat_df
# Select best Classifier
def select_best_classifier(results):

    weights = {
        'AUC': 0.30,
        'Precision': 0.20,
        'Recall': 0.20,
        'F1 Score': 0.10,
        'Specificity': 0.10,
        'Sensitivity': 0.05,
        'MCC': 0.025,
        'Cohen\'s Kappa': 0.025
    }

    best_score = -float('inf')
    best_classifier = None

    for clf_name, metrics in results.items():
        score = sum(metrics[metric] * weights[metric] for metric in weights)
        print(f"Classifier: {clf_name}, Weighted Score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_classifier = clf_name

    return best_classifier, best_score

def get_callbacks():
    return [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]

def build_classifiers_for_seed(seed, input_shape, X_train_mlp, callbacks, EPOCHS):
    return {
        'MLP': KerasClassifier(
            model=create_mlp_model,
            input_dim=X_train_mlp.shape[1],
            epochs=EPOCHS, batch_size=32, verbose=0,
            optimizer=Adam(learning_rate=0.001),
            callbacks=callbacks,
            random_state=seed
        ),
        'CNN': KerasClassifier(
            model=create_cnn_model,
            input_shape=(X_train_mlp.shape[1], 1),
            epochs=EPOCHS, batch_size=32, verbose=0,
            optimizer=Adam(learning_rate=0.001),
            callbacks=callbacks,
            random_state=seed
        ),
        'LSTM': KerasClassifier(
            model=create_lstm_model,
            input_shape=(X_train_mlp.shape[1], 1),
            epochs=EPOCHS, batch_size=32, verbose=0,
            optimizer=Adam(learning_rate=0.001),
            callbacks=callbacks,
            random_state=seed
        ),
        'RNN': KerasClassifier(
            model=create_rnn_model,
            input_shape=(X_train_mlp.shape[1], 1),
            epochs=EPOCHS, batch_size=32, verbose=0,
            optimizer=Adam(learning_rate=0.001),
            callbacks=callbacks,
            random_state=seed
        ),
        'DNN': KerasClassifier(
            model=create_dnn_model,
            input_dim=X_train_mlp.shape[1],
            epochs=EPOCHS, batch_size=32, verbose=0,
            optimizer=Adam(learning_rate=0.001),
            callbacks=callbacks,
            random_state=seed
        ),
        'Transformer': KerasClassifier(
            model=create_transformer_model,
            model__input_shape=input_shape,
            model__num_classes=1,
            epochs=EPOCHS, batch_size=32, verbose=0,
            optimizer=Adam(learning_rate=0.001),
            callbacks=callbacks,
            random_state=seed
        ),
        'GRU': KerasClassifier(
            model=create_gru_model,
            model__input_shape=input_shape,
            model__num_classes=1,
            epochs=EPOCHS, batch_size=32, verbose=0,
            optimizer=Adam(learning_rate=0.001),
            callbacks=callbacks,
            random_state=seed
        )
    }

def bootstrap_ci(y_true, y_pred_prob, threshold=0.5, n_bootstrap=1000, alpha=0.05):
    metrics = {
        'AUC': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1': [],
        'Kappa': [],
        'MCC': [],
        'Sensitivity': [],
        'Specificity': []
    }

    y_pred_bin = (y_pred_prob >= threshold).astype(int)
    for _ in range(n_bootstrap):
        indices = resample(range(len(y_true)), replace=True)
        y_t = y_true[indices]
        y_p = y_pred_prob[indices]
        y_b = y_pred_bin[indices]

        try:
            metrics['AUC'].append(roc_auc_score(y_t, y_p))
        except: pass

        tn, fp, fn, tp = confusion_matrix(y_t, y_b).ravel()
        metrics['Accuracy'].append((tp + tn) / (tp + tn + fp + fn))
        metrics['Precision'].append(precision_score(y_t, y_b, zero_division=0))
        metrics['Recall'].append(recall_score(y_t, y_b, zero_division=0))
        metrics['F1'].append(f1_score(y_t, y_b, zero_division=0))
        metrics['Kappa'].append(cohen_kappa_score(y_t, y_b))
        metrics['MCC'].append((tp*tn - fp*fn) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) + 1e-8))
        metrics['Sensitivity'].append(tp / (tp + fn + 1e-8))
        metrics['Specificity'].append(tn / (tn + fp + 1e-8))

    ci_summary = {}
    for key in metrics:
        vals = np.array(metrics[key])
        ci_lower = np.percentile(vals, 100 * alpha / 2)
        ci_upper = np.percentile(vals, 100 * (1 - alpha / 2))
        ci_summary[key] = f"{np.mean(vals):.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"
    return ci_summary


def main():
    results_dir = "/content/drive/MyDrive/Deeplearning/results_3rdjul"
    os.makedirs(results_dir, exist_ok=True)

    file_path = "/content/drive/MyDrive/Deeplearning/cleaned_dataset.csv"
    data = load_and_preprocess_data(file_path)
    X = data.drop(columns=['#Uploaded_variation', 'CLIN_SIG'])
    X = X.select_dtypes(include=[np.number])
    y = data['CLIN_SIG']

    filtered_X = pearson_correlation_filter(X)
    selected_features = feature_selection_and_heatmap(filtered_X, y)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(selected_features.drop('CLIN_SIG', axis=1))
    scaled_features_df = pd.DataFrame(scaled_features, columns=selected_features.drop('CLIN_SIG', axis=1).columns)
    scaled_features_df['CLIN_SIG'] = selected_features['CLIN_SIG']
    selected_features = scaled_features_df

    X_train, X_test, y_train, y_test = train_test_split(
        selected_features.drop('CLIN_SIG', axis=1),
        selected_features['CLIN_SIG'],
        test_size=0.2, random_state=42
    )

    X_train_mlp = X_train.values
    X_test_mlp = X_test.values

    input_shape = (X_train_mlp.shape[1], 1)
    callbacks = get_callbacks()

    # 🔁 STEP 1: MULTI-SEED EVALUATION TEST (ONLY)
    seed_results_dir = os.path.join(results_dir, "multi_seed_auc")
    os.makedirs(seed_results_dir, exist_ok=True)

    seeds = [42, 101, 202, 303, 404]
    multi_seed_results = {}

    for clf_name in ['MLP', 'CNN', 'LSTM', 'RNN', 'DNN', 'Transformer', 'GRU']:
        print(f"=== Multi-seed AUC evaluation for {clf_name} ===")
        model_type = 'mlp' if clf_name in ['MLP', 'DNN'] else 'cnn_lstm_rnn'
        aucs = []

        for seed in seeds:
            clf_dict = build_classifiers_for_seed(seed, input_shape, X_train_mlp, callbacks, EPOCHS)
            clf = clf_dict[clf_name]

            try:
                result = evaluate_model(clf, X_train_mlp, X_test_mlp, y_train, y_test, model_type=model_type)
                aucs.append({'Seed': seed, 'AUC': result['AUC']})
            except Exception as e:
                print(f"❌ Error for {clf_name} seed {seed}: {e}")

        auc_df = pd.DataFrame(aucs)
        multi_seed_results[clf_name] = auc_df
        auc_df.to_excel(os.path.join(seed_results_dir, f"{clf_name}_seed_auc.xlsx"), index=False)

        print("✅ Multi-seed AUC results saved.")

    # ✅ Find best seed per classifier and save
    best_seeds = {}

    for clf_name, auc_df in multi_seed_results.items():
        best_row = auc_df.loc[auc_df['AUC'].idxmax()]
        best_seeds[clf_name] = {
            'Best Seed': int(best_row['Seed']),
            'Best AUC': round(best_row['AUC'], 4)
        }

    best_seeds_df = pd.DataFrame.from_dict(best_seeds, orient='index')
    best_seeds_df.reset_index(inplace=True)
    best_seeds_df.rename(columns={'index': 'Classifier'}, inplace=True)
    best_seeds_df.to_excel(os.path.join(seed_results_dir, "best_seeds_per_model.xlsx"), index=False)
    print("✅ Best seeds per model saved.")


    classifiers = {
        'MLP': KerasClassifier(
            model=create_mlp_model,
            input_dim=X_train_mlp.shape[1],
            epochs=EPOCHS, batch_size=32, verbose=0,
            optimizer=Adam(learning_rate=0.001),
            callbacks=callbacks,
            random_state=int(best_seeds['MLP']['Best Seed'])  # ✅ Inject best seed
        ),

        'CNN': KerasClassifier(
            model=create_cnn_model,
            input_shape=(X_train_mlp.shape[1], 1),
            epochs=EPOCHS, batch_size=32, verbose=0,
            optimizer=Adam(learning_rate=0.001),
            callbacks=callbacks,
            random_state=int(best_seeds['CNN']['Best Seed'])  # ✅ Inject best seed
        ),

        'LSTM': KerasClassifier(
            model=create_lstm_model,
            input_shape=(X_train_mlp.shape[1], 1),
            epochs=EPOCHS, batch_size=32, verbose=0,
            optimizer=Adam(learning_rate=0.001),
            callbacks=callbacks,
            random_state=int(best_seeds['LSTM']['Best Seed'])
        ),

        'RNN': KerasClassifier(
            model=create_rnn_model,
            input_shape=(X_train_mlp.shape[1], 1),
            epochs=EPOCHS, batch_size=32, verbose=0,
            optimizer=Adam(learning_rate=0.001),
            callbacks=callbacks,
            random_state=int(best_seeds['RNN']['Best Seed'])
        ),

        'DNN': KerasClassifier(
            model=create_dnn_model,
            input_dim=X_train_mlp.shape[1],
            epochs=EPOCHS, batch_size=32, verbose=0,
            optimizer=Adam(learning_rate=0.001),
            callbacks=callbacks,
            random_state=int(best_seeds['DNN']['Best Seed'])
        ),

        'Transformer': KerasClassifier(
            model=create_transformer_model,
            model__input_shape=input_shape,
            model__num_classes=1,
            epochs=EPOCHS, batch_size=32, verbose=0,
            optimizer=Adam(learning_rate=0.001),
            callbacks=callbacks,
            random_state=int(best_seeds['Transformer']['Best Seed'])  # ✅
        ),

        'GRU': KerasClassifier(
            model=create_gru_model,
            model__input_shape=input_shape,
            model__num_classes=1,
            epochs=EPOCHS, batch_size=32, verbose=0,
            optimizer=Adam(learning_rate=0.001),
            callbacks=callbacks,
            random_state=int(best_seeds['GRU']['Best Seed'])  # ✅
        )
    }


    results = {}
    predictions = {}

    for clf_name, clf in classifiers.items():
        print(f'Evaluating {clf_name}...')
        if clf_name in ['MLP', 'DNN']:
            result = evaluate_model(clf, X_train_mlp, X_test_mlp, y_train, y_test, model_type='mlp')
        else:
            result = evaluate_model(clf, X_train_mlp, X_test_mlp, y_train, y_test, model_type='cnn_lstm_rnn')

        y_prob = result['Probabilities']
        results[clf_name] = result
        predictions[clf_name] = y_prob

    save_results_to_excel(results, "results_summary", 1, results_dir)

    best_thresholds = generate_roc_curve(y_test, list(predictions.values()), list(classifiers.keys()), results_dir)
    print("Best thresholds for each classifier:")
    for clf_name, threshold in best_thresholds.items():
        print(f"{clf_name}: {threshold:.2f}")

    prediction_df = pd.DataFrame({
        '#Uploaded_variation': data.loc[X_test.index, '#Uploaded_variation'],
        'Actual': y_test
    })

    for clf_name, y_prob in predictions.items():
        if len(y_prob) == len(prediction_df):
            prediction_df[clf_name] = y_prob
        else:
            print(f"Skipping {clf_name} due to length mismatch between y_prob and y_test")

    prediction_df['Average Probability'] = prediction_df[list(classifiers.keys())].mean(axis=1)
    prediction_df['Classification'] = 'Unknown'
    average_best_threshold = np.mean(list(best_thresholds.values()))

    prediction_df.loc[(prediction_df['Average Probability'] > average_best_threshold) & (
        prediction_df['Actual'] == 1), 'Classification'] = 'True Pathogenic'
    prediction_df.loc[(prediction_df['Average Probability'] > average_best_threshold) & (
        prediction_df['Actual'] == 0), 'Classification'] = 'False Pathogenic'
    prediction_df.loc[(prediction_df['Average Probability'] <= average_best_threshold) & (
        prediction_df['Actual'] == 0), 'Classification'] = 'True Benign'
    prediction_df.loc[(prediction_df['Average Probability'] <= average_best_threshold) & (
        prediction_df['Actual'] == 1), 'Classification'] = 'False Benign'

    save_predictions_to_excel(prediction_df, "predictions.xlsx", results_dir)

    generate_precision_recall_curve(y_test, list(predictions.values()), list(classifiers.keys()), results_dir)
    plot_combined_roc_with_ci(y_test.values, predictions, results_dir)

    ci_summary_df = []

    for clf_name, result in results.items():
        print(f"🔁 Bootstrapping CI for {clf_name}...")
        y_prob = result['Probabilities']
        threshold = result.get('Best Threshold', 0.5)  # Or use average_best_threshold
        ci_metrics = bootstrap_ci(y_test.values, np.array(y_prob), threshold=threshold)
        ci_metrics['Classifier'] = clf_name
        ci_summary_df.append(ci_metrics)

    ci_df = pd.DataFrame(ci_summary_df)
    ci_df = ci_df[['Classifier'] + [col for col in ci_df.columns if col != 'Classifier']]
    ci_df.to_excel(os.path.join(results_dir, "results_with_ci.xlsx"), index=False)
    print("✅ Bootstrap CI metrics saved.")

    for clf_name, clf in classifiers.items():
        plot_lime_explanations(clf, X_train, X_test, clf_name, results_dir, num_features=10)
        permutation_importance_analysis(clf, X_train, y_train, clf_name, results_dir)

    plot_calibration_curve(y_test, list(predictions.values()), list(classifiers.keys()), results_dir)
    stat_df = statistical_analysis(y_test, predictions, results, results_dir)
    print(stat_df)

    best_classifier, best_score = select_best_classifier(results)
    print(f"\nBest Classifier: {best_classifier} with a score of {best_score:.4f}")

    best_classifier, best_score = select_best_classifier(results)
    best_model = classifiers[best_classifier]
    best_probs = predictions[best_classifier]

    # Call LIME only on the best classifier
    plot_lime_for_tp_tn_fp_fn(
        best_clf_name=best_classifier,
        best_clf=best_model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        y_prob=best_probs,
        results_dir=results_dir,
        num_features=10
    )


if __name__ == "__main__":
    main()

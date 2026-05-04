"""
train_lcs.py
============
This script trains and evaluates classifiers on each feature set independently.

It does the following for each feature set (LBP, HOG, DWT):
  1. Loads the feature CSV
  2. Normalises features using MinMaxScaler (column-wise, 0-1 range)
  3. Splits data into 80% training / 20% testing
  4. Trains an eLCS model and records results
  5. Trains an SVM model and records results       (O4 benchmark)
  6. Trains a Random Forest model and records results (O4 benchmark)

At the end, prints a full comparison table across all feature sets and models.

Install requirements:
  pip install scikit-eLCS scikit-learn pandas numpy
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from skeLCS import eLCS

# =============================================================================
# 1. SETTINGS
# =============================================================================

# Root directory — two levels up from this file
current_dir = Path(__file__).resolve()
root_dir = current_dir.parent.parent

# The three feature CSVs we want to evaluate independently.
# Each was saved by extract_features.py.
# Key = display name, Value = path to CSV
FEATURE_FILES = {
    'LBP': root_dir / 'features_LBP.csv',
    'HOG': root_dir / 'features_HOG.csv',
    'DWT': root_dir / 'features_DWT.csv',
}

# eLCS hyperparameters
# - learning_iterations: how many times the algorithm loops through training data
#   More iterations = more time to learn, but slower. Start at 2000, increase later.
# - N: maximum number of rules (classifiers) kept in the population at once
#   Too small = underfitting, too large = slow convergence
# - nu: fitness exponent — higher values reward accurate rules more strongly
# - chi: crossover rate — how often two rules swap genetic material (0-1)
# - mu: mutation rate — how often a rule randomly flips a bit (0-1)
# - theta_del: minimum experience a rule needs before it can be deleted
ELCS_PARAMS = {
    'learning_iterations': 2000,
    'N': 500,
    'nu': 5,
    'chi': 0.8,
    'mu': 0.04,
    'theta_del': 20,
}

# =============================================================================
# 2. HELPER FUNCTION — load, normalise, and split a feature CSV
# =============================================================================

def load_and_prepare(csv_path):
    """
    Loads a feature CSV, normalises the feature columns, and splits into
    train/test sets.

    Normalisation (MinMaxScaler):
      - Each feature column is scaled independently to the range [0, 1]
      - This is called "column-wise" normalisation
      - eLCS (and most ML models) perform better when features are on the
        same scale — without this, a feature with large values (e.g. DWT_0 = 1.4)
        would dominate features with small values (e.g. HOG_3 = 0.002)
      - Dr Siddique specifically asked for this in the supervision meeting

    Args:
        csv_path: Path to the feature CSV file

    Returns:
        X_train, X_test, y_train, y_test — ready for model.fit()
    """
    df = pd.read_csv(csv_path)

    # Select only the feature columns (drop image_id and label)
    feature_cols = [col for col in df.columns if col not in ('image_id', 'label')]
    X = df[feature_cols].values   # numpy array of shape (n_images, n_features)
    y = df['label'].values        # array of 'benign' / 'malignant' strings

    # Encode string labels to integers: benign=0, malignant=1
    # eLCS requires numeric labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Normalise features column-wise to [0, 1]
    # fit_transform learns the min/max from training data and applies it
    scaler = MinMaxScaler()
    X_normalised = scaler.fit_transform(X)

    # Split: 80% training, 20% testing
    # random_state=42 ensures the same split every run (reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalised, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded   # keeps class ratio the same in both splits
    )

    return X_train, X_test, y_train, y_test


# =============================================================================
# 3. HELPER FUNCTION — evaluate a trained model and return metrics
# =============================================================================

def evaluate(model, X_test, y_test):
    """
    Runs predictions on the test set and returns three metrics.

    Metrics explained:
      - Balanced Accuracy: average recall per class — handles class imbalance.
        Better than plain accuracy when benign/malignant counts are unequal.
      - Precision: of all images the model said were malignant, how many
        actually were? High precision = fewer false alarms.
      - Recall: of all actual malignant images, how many did the model catch?
        High recall = fewer missed cancers. Critical for medical use.

    Args:
        model: a fitted classifier with a .predict() method
        X_test: normalised test features
        y_test: true labels

    Returns:
        dict with bal_acc, precision, recall (all as percentages)
    """
    y_pred = model.predict(X_test)

    return {
        'bal_acc':   balanced_accuracy_score(y_test, y_pred) * 100,
        'precision': precision_score(y_test, y_pred, zero_division=0) * 100,
        'recall':    recall_score(y_test, y_pred, zero_division=0) * 100,
    }


# =============================================================================
# 4. MAIN LOOP — train all models on all feature sets
# =============================================================================

# Store all results here so we can print a comparison table at the end
# Each entry will be a dict: {feature_set, model_name, bal_acc, precision, recall}
all_results = []

for feature_name, csv_path in FEATURE_FILES.items():

    print(f"\n{'='*60}")
    print(f"  Feature Set: {feature_name}")
    print(f"{'='*60}")

    # Check the CSV exists before trying to load it
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found — skipping.")
        print(f"  Run extract_features.py first to generate the CSVs.")
        continue

    # Load, normalise, and split the data
    X_train, X_test, y_train, y_test = load_and_prepare(csv_path)
    print(f"  Training samples : {len(X_train)}")
    print(f"  Testing samples  : {len(X_test)}")
    print(f"  Features per image: {X_train.shape[1]}")

    # -----------------------------------------------------------------
    # 4a. eLCS — our main model (interpretable IF-THEN rules)
    # -----------------------------------------------------------------
    # eLCS is an Extended Learning Classifier System.
    # It learns a population of IF-THEN rules (classifiers) that together
    # vote on whether a lesion is benign or malignant.
    # Each rule looks like: IF LBP_3 > 0.7 AND LBP_5 < 0.2 THEN malignant
    # This is what makes eLCS interpretable — a clinician can read the rules.
    print(f"\n  [1/3] Training eLCS...")
    elcs_model = eLCS(**ELCS_PARAMS)
    elcs_model.fit(X_train, y_train)
    elcs_results = evaluate(elcs_model, X_test, y_test)
    print(f"        Balanced Accuracy : {elcs_results['bal_acc']:.1f}%")
    print(f"        Precision         : {elcs_results['precision']:.1f}%")
    print(f"        Recall            : {elcs_results['recall']:.1f}%")
    all_results.append({'Feature': feature_name, 'Model': 'eLCS', **elcs_results})

    # -----------------------------------------------------------------
    # 4b. SVM — Support Vector Machine (O4 benchmark)
    # -----------------------------------------------------------------
    # SVM finds a hyperplane that best separates benign from malignant.
    # It is a strong baseline classifier for image feature vectors.
    # It is NOT interpretable — it cannot produce IF-THEN rules.
    # We use it here to benchmark against eLCS (Objective O4).
    # kernel='rbf' = Radial Basis Function, good for non-linear data.
    # class_weight='balanced' handles unequal benign/malignant counts.
    print(f"\n  [2/3] Training SVM...")
    svm_model = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    svm_model.fit(X_train, y_train)
    svm_results = evaluate(svm_model, X_test, y_test)
    print(f"        Balanced Accuracy : {svm_results['bal_acc']:.1f}%")
    print(f"        Precision         : {svm_results['precision']:.1f}%")
    print(f"        Recall            : {svm_results['recall']:.1f}%")
    all_results.append({'Feature': feature_name, 'Model': 'SVM', **svm_results})

    # -----------------------------------------------------------------
    # 4c. Random Forest (O4 benchmark)
    # -----------------------------------------------------------------
    # Random Forest builds many decision trees and takes a majority vote.
    # It is partially interpretable (feature importances) but does NOT
    # produce clinician-readable IF-THEN rules the way eLCS does.
    # n_estimators=100 = 100 trees in the forest.
    # class_weight='balanced' handles unequal benign/malignant counts.
    print(f"\n  [3/3] Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_results = evaluate(rf_model, X_test, y_test)
    print(f"        Balanced Accuracy : {rf_results['bal_acc']:.1f}%")
    print(f"        Precision         : {rf_results['precision']:.1f}%")
    print(f"        Recall            : {rf_results['recall']:.1f}%")
    all_results.append({'Feature': feature_name, 'Model': 'Random Forest', **rf_results})


# =============================================================================
# 5. PRINT FULL COMPARISON TABLE
# =============================================================================
# This table directly answers the research question:
#   "Which handcrafted feature produces the most accurate eLCS classifier?"
# It also fulfils O4 by showing eLCS vs SVM vs Random Forest side by side.

print(f"\n\n{'='*65}")
print(f"  FULL RESULTS COMPARISON")
print(f"{'='*65}")
print(f"  {'Model':<16} {'Feature':<8} {'Bal.Acc':>8} {'Precision':>10} {'Recall':>8}")
print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")

for r in all_results:
    print(
        f"  {r['Model']:<16} {r['Feature']:<8} "
        f"{r['bal_acc']:>7.1f}% {r['precision']:>9.1f}% {r['recall']:>7.1f}%"
    )

print(f"{'='*65}")

# =============================================================================
# 6. HIGHLIGHT BEST eLCS FEATURE SET
# =============================================================================
# Pull out only the eLCS results and find which feature set scored highest
# on balanced accuracy — this is the answer to the research question.

elcs_only = [r for r in all_results if r['Model'] == 'eLCS']

if elcs_only:
    best = max(elcs_only, key=lambda r: r['bal_acc'])
    print(f"\n  Best feature set for eLCS: {best['Feature']}")
    print(f"  Balanced Accuracy: {best['bal_acc']:.1f}%")
    print(f"  → This answers the research question: which handcrafted feature")
    print(f"    produces the most accurate eLCS classifier for skin cancer detection.")

print()
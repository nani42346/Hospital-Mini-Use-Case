"""Retrain the notebook's final MLP and save it for the Streamlit app.

Run once from this folder:  python train_model.py
Saves artifacts/model.keras, artifacts/preprocess.joblib and artifacts/metrics.json.
"""
import json

import joblib
import numpy as np
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from readmission_pipeline import ARTIFACTS, NUMERICAL_COLUMNS, clean, load_raw, split_features, transform

SEEDS = [42, 7, 21, 123, 2024]
EPOCHS, BATCH = 20, 64  # best setting from the notebook's experiments

np.random.seed(42)

df = clean(load_raw())
X, y, categorical = split_features(df)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

prep = {
    "numerical_columns": NUMERICAL_COLUMNS,
    "categorical_columns": categorical,
    "numeric_imputer": SimpleImputer(strategy="median").fit(X_train[NUMERICAL_COLUMNS]),
    "categorical_imputer": SimpleImputer(strategy="most_frequent").fit(X_train[categorical].astype(object)),
}
prep["scaler"] = StandardScaler().fit(prep["numeric_imputer"].transform(X_train[NUMERICAL_COLUMNS]))
prep["encoder"] = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False).fit(
    prep["categorical_imputer"].transform(X_train[categorical].astype(object)))
# Defaults for fields the app's form does not ask for: most frequent / median training value
prep["defaults"] = {
    **{c: float(X_train[c].median()) for c in NUMERICAL_COLUMNS},
    **{c: X_train[c].mode().iloc[0] for c in categorical},
}
prep["category_options"] = {c: sorted(X_train[c].dropna().astype(str).unique().tolist()) for c in categorical}

Xtr, Xva, Xte = (transform(d, prep) for d in (X_train, X_val, X_test))
ytr, yva, yte = (s.to_numpy() for s in (y_train, y_val, y_test))

weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=ytr)


def build_and_train(seed):
    tf.keras.utils.set_random_seed(seed)
    m = Sequential([
        Input(shape=(Xtr.shape[1],)),
        Dense(64, activation="relu"), Dropout(0.3),
        Dense(32, activation="relu"), Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy")
    m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS, batch_size=BATCH,
          class_weight={0: float(weights[0]), 1: float(weights[1])},
          callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)], verbose=0)
    return m, roc_auc_score(yva, m.predict(Xva, verbose=0).flatten())


# Like the notebook, select the final model on validation ROC-AUC (never on test data)
runs = [build_and_train(seed) for seed in SEEDS]
for seed, (_, auc) in zip(SEEDS, runs):
    print(f"seed {seed}: validation ROC-AUC = {auc:.4f}")
model, best_val_auc = max(runs, key=lambda r: r[1])

# Threshold: highest value in 0.30..0.70 whose validation Recall is >= 0.60 (notebook method)
val_prob = model.predict(Xva, verbose=0).flatten()
candidates = [round(t, 2) for t in np.arange(0.30, 0.75, 0.05) if recall_score(yva, (val_prob >= t).astype(int)) >= 0.60]
THRESHOLD = max(candidates) if candidates else 0.45
prep["threshold"] = THRESHOLD

prob = model.predict(Xte, verbose=0).flatten()
pred = (prob >= THRESHOLD).astype(int)
tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
metrics = {
    "accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred),
    "recall": recall_score(yte, pred), "f1": f1_score(yte, pred), "roc_auc": roc_auc_score(yte, prob),
    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "threshold": THRESHOLD,
    "n_features": int(Xtr.shape[1]), "val_roc_auc": best_val_auc,
}

ARTIFACTS.mkdir(exist_ok=True)
model.save(ARTIFACTS / "model.keras")
joblib.dump(prep, ARTIFACTS / "preprocess.joblib")
(ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
print(json.dumps(metrics, indent=2, default=float))

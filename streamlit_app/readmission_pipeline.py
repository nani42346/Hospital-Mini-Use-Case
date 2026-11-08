"""Data pipeline shared by train_model.py and app.py.

Mirrors the preprocessing in Koteswara_rao_Hospital_Readmission_MLP.ipynb.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
ZIP_PATH = PROJECT_DIR / "diabetes+130-us+hospitals+for+years+1999-2008.zip"
REPORT_CSV = PROJECT_DIR / "patient_readmission_risk_report.csv"
ARTIFACTS = APP_DIR / "artifacts"

EXPIRED_OR_HOSPICE = [11, 13, 14, 19, 20, 21]
SINGLE_VALUE_COLUMNS = ["examide", "citoglipton", "glimepiride-pioglitazone"]
ID_COLUMNS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
NUMERICAL_COLUMNS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]


def load_raw() -> pd.DataFrame:
    with zipfile.ZipFile(ZIP_PATH) as z, z.open("diabetic_data.csv") as f:
        return pd.read_csv(f, na_values="?", keep_default_na=False, low_memory=False)


def load_id_mappings() -> dict[str, dict[str, str]]:
    """Parse IDS_mapping.csv (three tables separated by blank lines)."""
    with zipfile.ZipFile(ZIP_PATH) as z:
        text = z.read("IDS_mapping.csv").decode("utf-8", errors="replace")
    mappings: dict[str, dict[str, str]] = {}
    for block in text.replace("\r", "").split("\n,\n"):
        block = block.strip()
        if not block:
            continue
        table = pd.read_csv(io.StringIO(block), dtype=str, keep_default_na=False)
        key = table.columns[0]
        mappings[key] = dict(zip(table.iloc[:, 0].str.strip(), table.iloc[:, 1].str.strip()))
    return mappings


def group_diagnosis(code):
    if pd.isnull(code):
        return np.nan
    code = str(code)
    if code.startswith("V") or code.startswith("E"):
        return "Other"
    number = float(code)
    if 390 <= number <= 459 or number == 785:
        return "Circulatory"
    if 460 <= number <= 519 or number == 786:
        return "Respiratory"
    if 520 <= number <= 579 or number == 787:
        return "Digestive"
    if int(number) == 250:
        return "Diabetes"
    if 800 <= number <= 999:
        return "Injury"
    if 710 <= number <= 739:
        return "Musculoskeletal"
    if 580 <= number <= 629 or number == 788:
        return "Genitourinary"
    if 140 <= number <= 239:
        return "Neoplasms"
    return "Other"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's data-preparation steps (before splitting)."""
    df = df.copy()
    df["readmitted"] = (df["readmitted"] == "<30").astype(int)
    df = df[~df["discharge_disposition_id"].isin(EXPIRED_OR_HOSPICE)]
    df = df.sort_values("encounter_id").drop_duplicates(subset="patient_nbr", keep="first")
    df = df.drop(columns=["weight"])
    df["medical_specialty"] = df["medical_specialty"].fillna("Unknown")
    df["payer_code"] = df["payer_code"].fillna("Unknown")
    df = df.drop(columns=SINGLE_VALUE_COLUMNS)
    df[ID_COLUMNS] = df[ID_COLUMNS].astype(str)
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col] = df[col].apply(group_diagnosis)
    return df


def split_features(df: pd.DataFrame):
    X = df.drop(columns=["readmitted", "encounter_id", "patient_nbr"])
    y = df["readmitted"]
    categorical = [c for c in X.columns if c not in NUMERICAL_COLUMNS]
    return X, y, categorical


def transform(X: pd.DataFrame, prep: dict) -> np.ndarray:
    """Apply fitted imputers, scaler and encoder (from artifacts) to raw feature rows."""
    num = prep["numeric_imputer"].transform(X[prep["numerical_columns"]])
    cat = prep["categorical_imputer"].transform(X[prep["categorical_columns"]].astype(object))
    num = prep["scaler"].transform(num)
    cat = prep["encoder"].transform(cat)
    return np.concatenate([num, cat], axis=1).astype("float32")


def risk_level(probability: float, threshold: float) -> str:
    if probability >= 0.70:
        return "High"
    if probability >= threshold:
        return "Medium"
    return "Low"

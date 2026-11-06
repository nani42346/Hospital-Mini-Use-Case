# Hospital Readmission Risk Prediction Using an MLP

Predicting whether a diabetic patient will be **readmitted to hospital within 30 days** of discharge, using a Multi-Layer Perceptron (MLP) built with TensorFlow / Keras.

## Business Problem

Hospitals discharge many diabetic patients every day, and staff currently decide from experience who needs extra follow-up care. About **11%** of encounters end in a readmission within 30 days, which costs money, fills beds, and signals gaps in post-discharge care. The goal is a model that flags high-risk patients at discharge so that limited follow-up capacity goes to the patients who need it most.

## Dataset

[Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) (UCI Machine Learning Repository), included here as a zip file.

- 101,766 encounters, 50 columns, 71,518 unique patients
- Target: `readmitted`, converted to binary: `<30` = 1, `>30` / `NO` = 0

## Repository Contents

| File | Description |
|---|---|
| `Koteswara_rao_Hospital_Readmission_MLP.ipynb` | Full notebook: EDA, preprocessing, MLP training, evaluation, error analysis |
| `diabetes+130-us+hospitals+for+years+1999-2008.zip` | Raw dataset (`diabetic_data.csv`, `IDS_mapping.csv`) |
| `patient_readmission_risk_report.csv` | Per-patient test-set output: actual result, predicted result, risk probability, risk level |
| `Error_Analysis_and_Business_Findings_Report.docx` | Written report on error analysis and business findings |
| `MLP_Hospital_Readmission_Assignment.docx` | Assignment brief |

## Approach

1. **Data understanding & EDA:** missing values, duplicates, class imbalance (~89% / 11%), readmission rate by age, prior inpatient visits and length of stay. Prior inpatient visits turned out to be the strongest single signal.
2. **Data preparation**
   - Removed patients who died or were discharged to hospice (they cannot be readmitted)
   - **Prevented leakage** by keeping only the first encounter per patient (69,990 rows)
   - Dropped `weight` (~97% missing); filled `medical_specialty` / `payer_code` with `"Unknown"`
   - Removed constant columns; grouped ICD-9 diagnosis codes into disease categories
   - Stratified 70 / 15 / 15 train / validation / test split
   - Imputation, one-hot encoding and scaling fitted on training data only (155 final features)
3. **Model:** Keras Sequential MLP with Dense 64 → 32 → 16 (ReLU), 30% dropout, and a sigmoid output. Adam (lr = 0.001), binary cross-entropy, balanced class weights, EarlyStopping on validation loss.
4. **Experiments:** compared epoch and batch-size combinations, then selected the best run by validation ROC-AUC (batch size 64).
5. **Threshold tuning:** chose the highest threshold that still reaches **Recall ≥ 0.60** on validation data, which gave **0.45**.
6. **Evaluation & error analysis:** metrics, confusion matrix, ROC curve, a baseline comparison, and a breakdown of false negatives and false positives by patient group.

## Results (Test Set)

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Always predict "No" (baseline) | 0.910 | 0.000 | 0.000 | 0.000 | 0.500 |
| **MLP (threshold 0.45)** | 0.586 | 0.129 | **0.627** | 0.214 | **0.653** |

Confusion matrix: TP = 591, FN = 352, FP = 3,996, TN = 5,560.

- The model catches about **63% of patients who are actually readmitted**, while the baseline catches none despite its higher accuracy.
- Precision (~13%) is above the ~9% base rate, so flagged patients are meaningfully more likely to return.
- ROC-AUC of ~0.65 matches published results on this dataset. Many drivers of readmission (social support, medication adherence) are not in the data.
- The threshold can be raised or lowered to match the hospital's follow-up capacity.

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
jupyter notebook Koteswara_rao_Hospital_Readmission_MLP.ipynb
```

The notebook reads the dataset directly from the zip file in the same folder, so there's no need to extract it.

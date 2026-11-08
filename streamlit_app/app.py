"""Hospital Readmission Risk Prediction — Streamlit app.

Run from this folder:  streamlit run app.py
(First run `python train_model.py` once to create the model artifacts.)
"""
import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from readmission_pipeline import (ARTIFACTS, REPORT_CSV, load_id_mappings, load_raw, risk_level, transform)

st.set_page_config(page_title="Readmission Risk AI", page_icon="🏥", layout="wide")

# ---------------------------------------------------------------- styling
NAVY, CARD, CYAN, TEAL, CORAL, AMBER, INDIGO, TURQ = (
    "#0B1F3A", "#12305A", "#22D3EE", "#14B8A6", "#FF7A59", "#FFB547", "#6366F1", "#2DD4BF")
TIER_COLORS = {"High": CORAL, "Medium": AMBER, "Low": TEAL}

st.markdown(f"""
<style>
.stApp {{ background: radial-gradient(circle at 90% 0%, rgba(0,180,216,.22), transparent 45%),
                     radial-gradient(circle at 0% 100%, rgba(20,184,166,.18), transparent 45%),
                     linear-gradient(135deg, #061227 0%, #0B2447 55%, #0D3B5E 100%); }}
[data-testid="stSidebar"] {{ background: #071630; }}
[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3 {{ font-family: 'Montserrat', 'Segoe UI', sans-serif; letter-spacing: -.3px; }}
.kicker {{ color: {CYAN}; font-size: .8rem; font-weight: 700; letter-spacing: .25em; text-transform: uppercase; margin-bottom: -.4rem; }}
.card {{ background: {CARD}; border-radius: 14px; padding: 18px 20px; height: 100%;
         box-shadow: 0 4px 14px rgba(0,0,0,.25); border: 1px solid rgba(127,231,247,.15); }}
.card h4 {{ margin: 0 0 .35rem 0; font-size: 1.05rem; }}
.card p {{ margin: 0; color: #C9DAEC; font-size: .93rem; line-height: 1.45; }}
.kpi {{ border-radius: 14px; padding: 16px 18px; background: {CARD}; border: 1px solid rgba(127,231,247,.15); }}
.kpi .lab {{ font-size: .72rem; letter-spacing: .2em; font-weight: 700; color: #A9C1DD; text-transform: uppercase; }}
.kpi .val {{ font-size: 2.2rem; font-weight: 800; line-height: 1.15; font-family: 'Montserrat','Segoe UI',sans-serif; }}
.kpi .sub {{ font-size: .82rem; color: #C9DAEC; }}
.pill {{ display: inline-block; padding: 4px 14px; border-radius: 999px; border: 1px solid {CYAN};
         color: #7FE7F7; font-size: .78rem; letter-spacing: .2em; font-weight: 700; }}
.tier {{ border-radius: 14px; padding: 22px; text-align: center; }}
.small-note {{ color: #A9C1DD; font-size: .82rem; }}
</style>
""", unsafe_allow_html=True)


def kicker(text):
    st.markdown(f'<div class="kicker">{text}</div>', unsafe_allow_html=True)


def card(title, body, accent=CYAN, icon=""):
    st.markdown(f'<div class="card" style="border-color:{accent}55"><h4 style="color:{accent}">{icon} {title}</h4><p>{body}</p></div>',
                unsafe_allow_html=True)


def kpi(label, value, sub, color=None, filled=False):
    style = f"background:{color};border:none" if filled else ""
    vcol = "#FFFFFF" if filled or not color else color
    st.markdown(f'<div class="kpi" style="{style}"><div class="lab" style="{"color:#fff" if filled else ""}">{label}</div>'
                f'<div class="val" style="color:{vcol}">{value}</div><div class="sub" style="{"color:#fff" if filled else ""}">{sub}</div></div>',
                unsafe_allow_html=True)


def style_fig(fig, height=330, title=None):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=10, r=10, t=50 if title else 20, b=10),
                      title=dict(text=title, font=dict(size=15)) if title else None,
                      font=dict(color="#DCE9F5"), showlegend=False)
    fig.update_yaxes(gridcolor="rgba(127,231,247,.12)", zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


# ---------------------------------------------------------------- data & model
@st.cache_data(show_spinner="Loading dataset…")
def eda_tables():
    df = load_raw()
    df["y"] = (df["readmitted"] == "<30").astype(int)
    out = {
        "target": df["readmitted"].value_counts().reindex(["NO", ">30", "<30"]),
        "age": df.groupby("age")["y"].mean() * 100,
        "stay": df.groupby("time_in_hospital")["y"].mean() * 100,
        "missing": (df.isnull().mean() * 100).loc[lambda s: s > 0].sort_values(ascending=False),
    }
    df["inp"] = pd.cut(df["number_inpatient"], bins=[-1, 0, 1, 2, 3, 100], labels=["0", "1", "2", "3", "4+"])
    out["inpatient"] = df.groupby("inp", observed=True)["y"].mean() * 100
    num = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
           "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
    out["corr"] = df[num + ["y"]].corr()["y"].drop("y").sort_values()
    return out


@st.cache_resource(show_spinner="Loading MLP model…")
def load_model():
    if not (ARTIFACTS / "model.keras").exists():
        return None, None, None
    import tensorflow as tf
    model = tf.keras.models.load_model(ARTIFACTS / "model.keras")
    prep = joblib.load(ARTIFACTS / "preprocess.joblib")
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    return model, prep, metrics


@st.cache_data
def id_maps():
    return load_id_mappings()


@st.cache_data
def patient_report():
    return pd.read_csv(REPORT_CSV)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🏥 Readmission Risk AI")
    st.caption("30-day readmission prediction with a Multi-Layer Perceptron")
    page = st.radio("Navigate", ["Overview", "Data Insights", "Model Performance", "Risk Predictor", "Patient Report"],
                    label_visibility="collapsed")
    st.divider()
    st.markdown("**Presented by**  \nD. Koteswara Rao")
    st.caption("Educational prototype — not a clinically validated decision system.")


# ================================================================ OVERVIEW
if page == "Overview":
    st.markdown('<span class="pill">AI-POWERED HEALTHCARE ANALYTICS</span>', unsafe_allow_html=True)
    st.title("Hospital Readmission Risk Prediction")
    st.markdown(f"<h3 style='color:{CYAN};margin-top:-.8rem'>Using a Multi-Layer Perceptron (MLP)</h3>", unsafe_allow_html=True)
    st.write("An AI-based approach to identifying diabetic patients at risk of 30-day hospital readmission, "
             "so follow-up support goes to the patients who need it most.")

    c = st.columns(4)
    with c[0]: kpi("Encounters", "101,766", "UCI Diabetes 130-US Hospitals", CYAN)
    with c[1]: kpi("Patients modelled", "69,990", "one encounter per patient", TEAL)
    with c[2]: kpi("Readmitted <30 days", "11.2%", "of all encounters", CORAL)
    with c[3]: kpi("Test Recall", "63%", "of readmitted patients caught", AMBER)

    st.write("")
    kicker("CBIM canvas")
    st.subheader("Framing the business question")
    cbim = [
        ("S · Situation", "The hospital discharges many diabetic patients every day. Staff decide by experience who needs extra care after going home.", CYAN, "🏥"),
        ("C · Complication", "About 11% return within 30 days — costing money and beds, overloading staff, and patients may come back sicker.", CORAL, "⚠️"),
        ("Q · Question", "At discharge, which patients are at high risk of returning within 30 days, so they can get extra care first?", AMBER, "❓"),
        ("SSOT · Single source of truth", "diabetic_data.csv (UCI, 101,766 visits). Decision column <code>readmitted</code>: &lt;30 days → 1, otherwise → 0.", INDIGO, "🗄️"),
        ("Success definition", "Recall ≥ 60% on test data and ROC-AUC clearly above 0.5. Accuracy is not the main metric.", TEAL, "🎯"),
        ("Primary stakeholder", "Discharge planning manager. Also affected: patients &amp; families, doctors &amp; nurses, hospital management.", TURQ, "👤"),
    ]
    for row in (cbim[:3], cbim[3:]):
        cols = st.columns(3)
        for col, (t, b, a, i) in zip(cols, row):
            with col: card(t, b, a, i)
        st.write("")

    kicker("Solution flow")
    cols = st.columns(3)
    with cols[0]: card("Problem", "Readmitted patients are not identified before discharge.", CORAL, "⚠️")
    with cols[1]: card("AI prediction", "The MLP scores each patient's 30-day readmission risk at discharge.", CYAN, "🧠")
    with cols[2]: card("Better follow-up", "Staff prioritise calls, appointments and discharge planning by risk tier.", TEAL, "🤝")


# ================================================================ DATA INSIGHTS
elif page == "Data Insights":
    kicker("Data understanding & EDA")
    st.title("What the data tells us")
    t = eda_tables()
    c = st.columns(3)
    with c[0]: card("Missing values", "weight 96.9%, medical_specialty 49.1%, payer_code 39.6%; all others below 2.3%.", AMBER, "❓")
    with c[1]: card("Duplicates & patients", "0 duplicate rows, but 101,766 visits come from only 71,518 patients — so we keep one visit per patient.", CYAN, "🧬")
    with c[2]: card("Class imbalance", "Only 11.16% are readmitted within 30 days. A model that always says “No” is 89% accurate and useless.", CORAL, "⚖️")
    st.write("")

    col1, col2 = st.columns(2)
    with col1:
        s = t["inpatient"]
        fig = go.Figure(go.Bar(x=s.index.astype(str), y=s.values, marker_color=CORAL, text=[f"{v:.1f}%" for v in s.values], textposition="outside"))
        fig.update_xaxes(title="Inpatient visits in the previous year"); fig.update_yaxes(title="Readmission rate (%)", range=[0, s.max() * 1.2])
        st.plotly_chart(style_fig(fig, title="Readmission rate by previous inpatient visits"), width="stretch")
    with col2:
        s = t["age"]
        fig = go.Figure(go.Bar(x=[a.strip("[)") for a in s.index], y=s.values, marker_color=CYAN, text=[f"{v:.1f}" for v in s.values], textposition="outside"))
        fig.update_xaxes(title="Age group"); fig.update_yaxes(title="Readmission rate (%)", range=[0, s.max() * 1.2])
        st.plotly_chart(style_fig(fig, title="Readmission rate by age group"), width="stretch")
    col1, col2 = st.columns(2)
    with col1:
        s = t["stay"]
        fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines+markers", line=dict(color="#A5B4FC", width=3), marker=dict(size=8)))
        fig.update_xaxes(title="Days in hospital", dtick=1); fig.update_yaxes(title="Readmission rate (%)")
        st.plotly_chart(style_fig(fig, title="Readmission rate by length of stay"), width="stretch")
    with col2:
        s = t["corr"]
        fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=[TEAL if v > 0 else CORAL for v in s.values],
                               text=[f"{v:.3f}" for v in s.values], textposition="outside"))
        fig.update_xaxes(title="Correlation with 30-day readmission", range=[min(s.min() * 1.6, -0.03), s.max() * 1.25])
        st.plotly_chart(style_fig(fig, title="Correlation of numeric features with readmission"), width="stretch")
    col1, col2 = st.columns(2)
    with col1:
        s = t["target"]
        fig = go.Figure(go.Bar(x=["NO", ">30 days", "<30 days"], y=s.values, marker_color=[CYAN, INDIGO, CORAL], text=[f"{v:,}" for v in s.values], textposition="outside"))
        fig.update_yaxes(title="Encounters", range=[0, s.max() * 1.15])
        st.plotly_chart(style_fig(fig, title="Original readmitted values"), width="stretch")
    with col2:
        s = t["missing"]
        fig = go.Figure(go.Bar(x=s.index, y=s.values, marker_color=AMBER, text=[f"{v:.1f}%" for v in s.values], textposition="outside"))
        fig.update_yaxes(title="Missing (%)", range=[0, 110])
        st.plotly_chart(style_fig(fig, title="Missing values by column"), width="stretch")
    st.info("**Takeaway:** previous inpatient visits are the strongest single signal, but every correlation is weak (< 0.2). "
            "No single feature predicts readmission alone, which is why the MLP combines many features.")


# ================================================================ MODEL PERFORMANCE
elif page == "Model Performance":
    kicker("Model evaluation — test data (notebook final run)")
    st.title("Catching most readmissions, honestly measured")
    c = st.columns(5)
    with c[0]: kpi("Recall", "63%", "591 of 943 readmitted caught", CORAL, filled=True)
    with c[1]: kpi("ROC-AUC", "0.65", "vs 0.50 random ranking", TEAL, filled=True)
    with c[2]: kpi("Accuracy", "59%", "lower by design (threshold 0.45)")
    with c[3]: kpi("Precision", "13%", "of flagged patients readmitted")
    with c[4]: kpi("F1-score", "21%", "balance of precision & recall")
    st.write("")

    col1, col2 = st.columns([1, 1])
    with col1:
        z = [[5560, 3996], [352, 591]]
        labels = [["TN<br><b>5,560</b>", "FP<br><b>3,996</b>"], ["FN<br><b>352</b>", "TP<br><b>591</b>"]]
        fig = go.Figure(go.Heatmap(z=[[1, 2], [3, 0]], x=["Predicted: No", "Predicted: Yes"], y=["Actual: No", "Actual: Yes"],
                                   colorscale=[[0, CYAN], [0.34, TEAL], [0.67, AMBER], [1, CORAL]], showscale=False,
                                   text=labels, texttemplate="%{text}", textfont=dict(size=20, color=NAVY), xgap=6, ygap=6))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(fig, 380, "Confusion matrix (10,499 test patients)"), width="stretch")
    with col2:
        cats = ["Accuracy", "Recall", "F1-score", "ROC-AUC ×100"]
        fig = go.Figure([
            go.Bar(name="Always predict 'No'", x=cats, y=[91.0, 0, 0, 50], marker_color="#7D93AD", text=[91, 0, 0, 50], textposition="outside"),
            go.Bar(name="MLP model", x=cats, y=[58.6, 62.7, 21.4, 65.3], marker_color=CYAN, text=[59, 63, 21, 65], textposition="outside"),
        ])
        fig = style_fig(fig, 380, "MLP vs 'always No' baseline (%)")
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.08, x=0), barmode="group")
        fig.update_yaxes(range=[0, 110])
        st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns([1.1, 1])
    with col1:
        feats = ["Previous inpatient visits", "Discharged to home", "Discharged to rehab facility", "Time in hospital",
                 "Discharged to nursing facility (SNF)", "Number of diagnoses", "Age 50–60", "Discharged home w/ home health"]
        vals = [0.0272, 0.0194, 0.0147, 0.0093, 0.0077, 0.0067, 0.0046, 0.0030]
        fig = go.Figure(go.Bar(x=vals[::-1], y=feats[::-1], orientation="h", marker_color=[CYAN] * 7 + [CORAL],
                               text=[f"{v:.3f}" for v in vals[::-1]], textposition="outside"))
        fig.update_xaxes(range=[0, 0.033], title="Drop in validation ROC-AUC when shuffled")
        st.plotly_chart(style_fig(fig, 380, "Permutation feature importance (top 8)"), width="stretch")
    with col2:
        tiers = pd.DataFrame({"Tier": ["High", "Medium", "Low"], "Rate": [28.62, 11.79, 5.97], "Patients": [297, 4273, 5929]})
        fig = go.Figure(go.Bar(x=tiers["Tier"], y=tiers["Rate"], marker_color=[CORAL, AMBER, TEAL],
                               text=[f"{r:.1f}%<br>{p:,} patients" for r, p in zip(tiers["Rate"], tiers["Patients"])], textposition="outside"))
        fig.update_yaxes(range=[0, 38], title="Actual readmission rate (%)")
        st.plotly_chart(style_fig(fig, 380, "Actual readmission rate by predicted risk tier"), width="stretch")

    with st.expander("🧠 Model architecture & training setup"):
        a, b = st.columns(2)
        a.markdown("""
| Layer | Details |
|---|---|
| Input | 155 features (8 scaled numeric + one-hot categorical) |
| Dense | 64 neurons, ReLU → Dropout 30% |
| Dense | 32 neurons, ReLU → Dropout 30% |
| Dense | 16 neurons, ReLU |
| Output | 1 neuron, Sigmoid (probability) |
""")
        b.markdown("""
- **Framework:** TensorFlow / Keras Sequential API (12,609 parameters)
- **Optimizer:** Adam, learning rate 0.001 · **Loss:** binary crossentropy
- **Imbalance:** balanced class weights (readmitted 5.567 vs 0.549)
- **EarlyStopping:** patience 5 on validation loss, best weights restored
- **Selection:** best of 4 epoch/batch settings by validation ROC-AUC
- **Threshold:** 0.45 — highest with validation Recall ≥ 60%
""")

    model, prep, m = load_model()
    if m:
        with st.expander("🔁 Metrics of the retrained model used by the Risk Predictor"):
            st.write("The notebook did not save its model, so `train_model.py` retrains the same pipeline. "
                     "Neural-network training is random, so results differ slightly from the notebook's run above.")
            cc = st.columns(6)
            for col, (k, lab) in zip(cc, [("recall", "Recall"), ("roc_auc", "ROC-AUC"), ("accuracy", "Accuracy"),
                                          ("precision", "Precision"), ("f1", "F1"), ("threshold", "Threshold")]):
                v = m[k]
                col.metric(lab, f"{v:.2f}" if k in ("roc_auc", "threshold") else f"{v:.1%}")


# ================================================================ RISK PREDICTOR
elif page == "Risk Predictor":
    kicker("Live prediction")
    st.title("Patient risk predictor")
    model, prep, m = load_model()
    if model is None:
        st.error("Model artifacts not found. Run `python train_model.py` in the streamlit_app folder first.")
        st.stop()
    maps = id_maps()
    opts, d = prep["category_options"], prep["defaults"]

    def sel(label, col, mapping=None, key=None):
        options = opts[col]
        idx = options.index(str(d[col])) if str(d[col]) in options else 0
        fmt = (lambda v: f"{v} · {mapping.get(v, 'Unknown')}") if mapping else (lambda v: v)
        return st.selectbox(label, options, index=idx, format_func=fmt, key=key or col)

    st.write("Enter information available **at discharge**. Fields not shown use the most common value from the training data.")
    with st.form("patient"):
        st.markdown(f"<h4 style='color:{CYAN}'>👤 Patient</h4>", unsafe_allow_html=True)
        c = st.columns(3)
        with c[0]: age = sel("Age group", "age")
        with c[1]: gender = sel("Gender", "gender")
        with c[2]: race = sel("Race", "race")

        st.markdown(f"<h4 style='color:{CYAN}'>🏥 Admission & discharge</h4>", unsafe_allow_html=True)
        c = st.columns(3)
        with c[0]: adm_type = sel("Admission type", "admission_type_id", maps.get("admission_type_id"))
        with c[1]: adm_src = sel("Admission source", "admission_source_id", maps.get("admission_source_id"))
        with c[2]: disch = sel("Discharge destination", "discharge_disposition_id", maps.get("discharge_disposition_id"))
        c = st.columns(3)
        with c[0]: specialty = sel("Medical specialty", "medical_specialty")
        with c[1]: time_in_hospital = st.slider("Days in hospital", 1, 14, int(d["time_in_hospital"]))
        with c[2]: number_diagnoses = st.slider("Number of diagnoses", 1, 16, int(d["number_diagnoses"]))

        st.markdown(f"<h4 style='color:{CYAN}'>📋 Visit history (previous year)</h4>", unsafe_allow_html=True)
        c = st.columns(3)
        with c[0]: number_inpatient = st.number_input("Inpatient visits", 0, 25, 0)
        with c[1]: number_emergency = st.number_input("Emergency visits", 0, 80, 0)
        with c[2]: number_outpatient = st.number_input("Outpatient visits", 0, 45, 0)

        st.markdown(f"<h4 style='color:{CYAN}'>🔬 Procedures, diagnoses & medication</h4>", unsafe_allow_html=True)
        c = st.columns(3)
        with c[0]: num_lab = st.slider("Lab procedures", 1, 132, int(d["num_lab_procedures"]))
        with c[1]: num_proc = st.slider("Procedures", 0, 6, int(d["num_procedures"]))
        with c[2]: num_med = st.slider("Medications", 1, 81, int(d["num_medications"]))
        c = st.columns(3)
        with c[0]: diag_1 = sel("Primary diagnosis group", "diag_1")
        with c[1]: diag_2 = sel("Secondary diagnosis group", "diag_2")
        with c[2]: diag_3 = sel("Additional diagnosis group", "diag_3")
        c = st.columns(5)
        with c[0]: a1c = sel("A1C result", "A1Cresult")
        with c[1]: glu = sel("Max glucose serum", "max_glu_serum")
        with c[2]: insulin = sel("Insulin", "insulin")
        with c[3]: change = sel("Medication change", "change")
        with c[4]: dmed = sel("Diabetes medication", "diabetesMed")
        submitted = st.form_submit_button("🔮  Predict readmission risk", width="stretch", type="primary")

    if submitted:
        row = dict(d)
        row.update({
            "age": age, "gender": gender, "race": race, "admission_type_id": adm_type, "admission_source_id": adm_src,
            "discharge_disposition_id": disch, "medical_specialty": specialty, "time_in_hospital": time_in_hospital,
            "number_diagnoses": number_diagnoses, "number_inpatient": number_inpatient, "number_emergency": number_emergency,
            "number_outpatient": number_outpatient, "num_lab_procedures": num_lab, "num_procedures": num_proc,
            "num_medications": num_med, "diag_1": diag_1, "diag_2": diag_2, "diag_3": diag_3, "A1Cresult": a1c,
            "max_glu_serum": glu, "insulin": insulin, "change": change, "diabetesMed": dmed,
        })
        X = pd.DataFrame([row])[prep["numerical_columns"] + prep["categorical_columns"]]
        prob = float(model.predict(transform(X, prep), verbose=0).flatten()[0])
        thr = prep["threshold"]
        tier = risk_level(prob, thr)
        color = TIER_COLORS[tier]

        st.divider()
        col1, col2 = st.columns([1, 1.2])
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100, number=dict(suffix="%", font=dict(size=46, color=color)),
                gauge=dict(axis=dict(range=[0, 100], tickcolor="#A9C1DD"), bar=dict(color=color, thickness=0.28),
                           bgcolor="rgba(0,0,0,0)", borderwidth=0,
                           steps=[dict(range=[0, thr * 100], color="rgba(20,184,166,.25)"),
                                  dict(range=[thr * 100, 70], color="rgba(255,181,71,.25)"),
                                  dict(range=[70, 100], color="rgba(255,122,89,.28)")],
                           threshold=dict(line=dict(color="#FFFFFF", width=3), thickness=0.8, value=thr * 100))))
            st.plotly_chart(style_fig(fig, 300, "Predicted readmission risk score"), width="stretch")
        with col2:
            actions = {
                "High": ["Detailed discharge planning", "Medication review with a pharmacist", "Follow-up phone call within 48 hours", "Appointment within 7 days"],
                "Medium": ["Follow-up call within one week", "Scheduled outpatient appointment", "Medication guidance"],
                "Low": ["Standard discharge process", "Routine follow-up"],
            }[tier]
            items = "".join(f"<li>{a}</li>" for a in actions)
            st.markdown(f"""<div class="tier" style="background:{CARD};border:2px solid {color}">
                <div style="font-size:.8rem;letter-spacing:.25em;color:#A9C1DD;font-weight:700">RISK TIER</div>
                <div style="font-size:2.6rem;font-weight:800;color:{color};font-family:Montserrat,Segoe UI">{tier.upper()}</div>
                <div style="color:#DCE9F5">{"Flagged for follow-up" if prob >= thr else "Below the follow-up threshold"} (threshold {thr:.2f})</div>
                <hr style="border-color:rgba(127,231,247,.2)">
                <div style="text-align:left"><b style="color:{color}">Recommended actions</b><ul style="margin-top:.4rem">{items}</ul></div>
                </div>""", unsafe_allow_html=True)
        st.caption("Tiers: High ≥ 0.70 · Medium ≥ threshold · Low below threshold. In the notebook's test set the actual readmission rates were "
                   "28.6% (High), 11.8% (Medium) and 6.0% (Low). The score is a ranking aid, not a diagnosis — "
                   "this educational prototype must not be used for clinical decisions.")


# ================================================================ PATIENT REPORT
elif page == "Patient Report":
    kicker("Patient-level output")
    st.title("Test-set prediction report")
    st.write("Predictions from the notebook for 10,499 test patients. IDs are pseudonymous — original patient numbers are not included.")
    rep = patient_report()
    c = st.columns(3)
    levels = c[0].multiselect("Risk level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    actual = c[1].multiselect("Actual readmission", ["Yes", "No"], default=["Yes", "No"])
    pmin, pmax = c[2].slider("Risk probability range", 0.0, 1.0, (0.0, 1.0), 0.01)
    f = rep[rep["Risk Level"].isin(levels) & rep["Actual Readmission"].isin(actual) & rep["Risk Probability"].between(pmin, pmax)]

    k = st.columns(4)
    with k[0]: kpi("Patients shown", f"{len(f):,}", f"of {len(rep):,}", CYAN)
    with k[1]: kpi("Flagged", f"{(f['Predicted Result'] == 'Yes').sum():,}", "predicted readmission", AMBER)
    with k[2]: kpi("Actually readmitted", f"{(f['Actual Readmission'] == 'Yes').sum():,}", "within 30 days", CORAL)
    rate = (f["Actual Readmission"] == "Yes").mean() * 100 if len(f) else 0
    with k[3]: kpi("Readmission rate", f"{rate:.1f}%", "in this selection", TEAL)
    st.write("")

    col1, col2 = st.columns([1.3, 1])
    with col1:
        fig = go.Figure()
        for lab, colr in (("No", CYAN), ("Yes", CORAL)):
            fig.add_trace(go.Histogram(x=f.loc[f["Actual Readmission"] == lab, "Risk Probability"], name=f"Actual: {lab}",
                                       marker_color=colr, opacity=0.7, nbinsx=40, histnorm="probability density"))
        fig = style_fig(fig, 340, "Risk score distribution by actual outcome")
        fig.update_layout(barmode="overlay", showlegend=True, legend=dict(orientation="h", y=1.1, x=0))
        fig.add_vline(x=0.45, line_dash="dash", line_color="white", annotation_text="threshold 0.45")
        st.plotly_chart(fig, width="stretch")
    with col2:
        counts = f["Risk Level"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
        fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.6, marker=dict(colors=[CORAL, AMBER, TEAL]), sort=False))
        fig = style_fig(fig, 340, "Patients by risk level")
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig, width="stretch")

    st.dataframe(
        f.sort_values("Risk Probability", ascending=False), width="stretch", height=380, hide_index=True,
        column_config={"Risk Probability": st.column_config.ProgressColumn("Risk Probability", min_value=0.0, max_value=1.0, format="%.3f")},
    )
    st.download_button("⬇️ Download filtered report (CSV)", f.to_csv(index=False).encode(), "readmission_risk_filtered.csv", "text/csv")

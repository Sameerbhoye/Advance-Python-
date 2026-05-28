"""
CKD Risk Predictor — Streamlit App
===================================
HOW TO SET UP:
  1. pip install streamlit keras scikit-learn imbalanced-learn pandas numpy

  2. Save your trained artifacts from the notebook:

        model.save("ckd_model.h5")

        import pickle
        with open("scaler.pkl", "wb") as f: pickle.dump(scaler, f)
        with open("pca.pkl",    "wb") as f: pickle.dump(pca, f)

     Place those 3 files in the same folder as this script.

  3. Run:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import os, pickle

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CKD Risk Predictor",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

/* ── global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace !important;
    background-color: #080c12 !important;
    color: #c8ddf0 !important;
}

/* ── grid background ── */
.stApp {
    background-color: #080c12;
    background-image:
        linear-gradient(rgba(0,212,255,.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,.025) 1px, transparent 1px);
    background-size: 40px 40px;
}

/* ── header band ── */
.ckd-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.4rem 1.8rem;
    background: #0e1520;
    border: 1px solid #1e2d42;
    border-radius: 12px;
    margin-bottom: 1.8rem;
}
.ckd-header-logo {
    width: 46px; height: 46px;
    background: linear-gradient(135deg,#00d4ff,#005f73);
    border-radius: 10px;
    display:flex; align-items:center; justify-content:center;
    font-size: 1.5rem; flex-shrink:0;
}
.ckd-header-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.35rem; font-weight: 800;
    color: #e8f4ff; letter-spacing: -.02em;
}
.ckd-header-sub { font-size: .68rem; color: #4a6580; margin-top:2px; }
.ckd-badge {
    margin-left: auto;
    font-size: .62rem; font-weight:600;
    color: #00d4ff; border: 1px solid #00d4ff;
    border-radius: 4px; padding: 3px 10px;
    letter-spacing: .1em; white-space:nowrap;
}

/* ── section labels ── */
.sec-label {
    font-family:'Syne',sans-serif !important;
    font-size:.62rem; font-weight:700;
    letter-spacing:.14em; text-transform:uppercase;
    color:#00d4ff; margin:1.4rem 0 .6rem;
    display:flex; align-items:center; gap:.6rem;
}
.sec-label::after {
    content:''; flex:1; height:1px;
    background: linear-gradient(90deg,#1e2d42,transparent);
}

/* ── input fields ── */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] > div > div {
    background: #0e1520 !important;
    border: 1px solid #1e2d42 !important;
    border-radius: 6px !important;
    color: #e8f4ff !important;
    font-family: 'DM Mono', monospace !important;
    font-size: .82rem !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within,
div[data-testid="stNumberInput"] input:focus {
    border-color: #00d4ff !important;
    box-shadow: 0 0 0 3px rgba(0,212,255,.1) !important;
}
label { color: #4a6580 !important; font-size:.68rem !important; }

/* ── predict button ── */
div[data-testid="stButton"] > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: .9rem !important;
    letter-spacing: .06em !important;
    background: #00d4ff !important;
    color: #080c12 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: .7rem 2.5rem !important;
    width: 100% !important;
    transition: box-shadow .2s !important;
}
div[data-testid="stButton"] > button:hover {
    box-shadow: 0 8px 24px rgba(0,212,255,.3) !important;
}

/* ── result card ── */
.result-card {
    background: #0e1520;
    border: 1px solid #1e2d42;
    border-radius: 12px;
    padding: 1.8rem 2rem;
    margin-top: 1rem;
}
.result-card.ckd    { border-left: 4px solid #ff4d6d; }
.result-card.no-ckd { border-left: 4px solid #06d6a0; }

.result-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.7rem; font-weight: 800;
    letter-spacing: -.03em;
    margin-bottom: .3rem;
}
.result-title.ckd    { color: #ff4d6d; }
.result-title.no-ckd { color: #06d6a0; }

.result-sub { font-size:.7rem; color:#4a6580; margin-bottom:1rem; }

.risk-pill {
    display:inline-block;
    font-size:.65rem; font-weight:700;
    letter-spacing:.1em; text-transform:uppercase;
    border-radius:4px; padding:3px 10px; margin-bottom:1rem;
}
.risk-high   { background:rgba(255,77,109,.15); color:#ff4d6d; border:1px solid #ff4d6d; }
.risk-medium { background:rgba(255,183,3,.15);  color:#ffb703; border:1px solid #ffb703; }
.risk-low    { background:rgba(6,214,160,.15);  color:#06d6a0; border:1px solid #06d6a0; }
.risk-vlow   { background:rgba(6,214,160,.08);  color:#06d6a0; border:1px solid rgba(6,214,160,.35); }

.prob-bar-bg {
    background:#1e2d42; border-radius:99px; height:10px;
    overflow:hidden; margin:0.8rem 0 0.3rem;
}
.prob-bar-fill {
    height:100%; border-radius:99px;
    transition: width 1s ease;
}
.prob-label {
    font-family:'Syne',sans-serif !important;
    font-size:1.1rem; font-weight:700; margin-top:.5rem;
}

.disclaimer {
    margin-top:1.2rem;
    padding:.75rem 1rem;
    background:rgba(255,183,3,.06);
    border-left:3px solid #ffb703;
    border-radius:0 6px 6px 0;
    font-size:.65rem; color:#4a6580; line-height:1.7;
}

/* ── demo mode banner ── */
.demo-banner {
    background:rgba(255,183,3,.07);
    border:1px solid rgba(255,183,3,.3);
    border-radius:8px; padding:.7rem 1rem;
    font-size:.68rem; color:#ffb703; margin-bottom:1.2rem;
}

/* hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stDecoration"] { display:none; }
</style>
""", unsafe_allow_html=True)

# ── Load model artifacts ────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    mdl = scl = pc = None
    try:
        from keras.models import load_model
        if os.path.exists("ckd_model.h5"):
            mdl = load_model("ckd_model.h5")
        if os.path.exists("scaler.pkl"):
            with open("scaler.pkl","rb") as f: scl = pickle.load(f)
        if os.path.exists("pca.pkl"):
            with open("pca.pkl","rb") as f:    pc  = pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load model artifacts: {e}")
    return mdl, scl, pc

model, scaler, pca = load_artifacts()
demo_mode = not all([model, scaler, pca])

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ckd-header">
  <div class="ckd-header-logo">🫘</div>
  <div>
    <div class="ckd-header-title">CKD Risk Predictor</div>
    <div class="ckd-header-sub">Chronic Kidney Disease · Neural Network Classifier · UCI Dataset</div>
  </div>
  <span class="ckd-badge">NN · 99% ACC</span>
</div>
""", unsafe_allow_html=True)

if demo_mode:
    st.markdown("""
    <div class="demo-banner">
      ⚡ <strong>Demo Mode</strong> — model files not found (<code>ckd_model.h5</code>, <code>scaler.pkl</code>, <code>pca.pkl</code>).
      Save them from your notebook and place them alongside this script to enable real predictions.
    </div>
    """, unsafe_allow_html=True)

# ── Form ────────────────────────────────────────────────────────────────────
with st.form("ckd_form"):

    # Section 1: Lab Values
    st.markdown('<div class="sec-label">Laboratory Values</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    age  = c1.number_input("Age (years)",        min_value=1,   max_value=100,  value=48)
    bp   = c2.selectbox("Blood Pressure (mm/Hg)",options=[50,60,70,80,90,100,110,120,140,180], index=3)
    sg   = c3.selectbox("Specific Gravity",      options=[1.005,1.010,1.015,1.020,1.025],       index=3)
    al   = c4.selectbox("Albumin (0–5)",         options=[0,1,2,3,4,5],                         index=0)

    c5, c6, c7, c8 = st.columns(4)
    su   = c5.selectbox("Sugar (0–5)",           options=[0,1,2,3,4,5],  index=0)
    bgr  = c6.number_input("Blood Glucose (mg/dL)",   min_value=0,   max_value=500,  value=120)
    bu   = c7.number_input("Blood Urea (mg/dL)",      min_value=0,   max_value=400,  value=36)
    sc   = c8.number_input("Serum Creatinine (mg/dL)",min_value=0.0, max_value=80.0, value=1.2, step=0.1)

    c9, c10, c11, c12 = st.columns(4)
    sod  = c9.number_input("Sodium (mEq/L)",     min_value=100, max_value=165,  value=137)
    pot  = c10.number_input("Potassium (mEq/L)", min_value=2.0, max_value=50.0, value=4.5, step=0.1)
    hemo = c11.number_input("Hemoglobin (g/dL)", min_value=3.0, max_value=18.0, value=13.5, step=0.1)
    pcv  = c12.number_input("Packed Cell Volume",min_value=9,   max_value=54,   value=41)

    c13, c14 = st.columns(2)
    wc   = c13.number_input("White Cell Count (/µL)", min_value=2000, max_value=30000, value=9800, step=100)
    rc   = c14.number_input("Red Cell Count (M/µL)",  min_value=2.0,  max_value=9.0,  value=5.2, step=0.1)

    # Section 2: Urinalysis
    st.markdown('<div class="sec-label">Urinalysis</div>', unsafe_allow_html=True)
    u1, u2, u3, u4 = st.columns(4)
    rbc  = u1.selectbox("Red Blood Cells",  ["normal","abnormal"])
    pc_v = u2.selectbox("Pus Cell",         ["normal","abnormal"])
    pcc  = u3.selectbox("Pus Cell Clumps",  ["notpresent","present"])
    ba   = u4.selectbox("Bacteria",         ["notpresent","present"])

    # Section 3: Comorbidities
    st.markdown('<div class="sec-label">Comorbidities &amp; Symptoms</div>', unsafe_allow_html=True)
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    htn   = s1.selectbox("Hypertension",          ["no","yes"])
    dm    = s2.selectbox("Diabetes Mellitus",      ["no","yes"])
    cad   = s3.selectbox("Coronary Artery Disease",["no","yes"])
    appet = s4.selectbox("Appetite",               ["good","poor"])
    pe    = s5.selectbox("Pedal Edema",            ["no","yes"])
    ane   = s6.selectbox("Anemia",                 ["no","yes"])

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔬  Run Prediction")

# ── Prediction logic ────────────────────────────────────────────────────────
def encode(val, mapping): return mapping[val]

BINARY = {"yes":1,"no":0}
RBC_M  = {"normal":1,"abnormal":0}
PC_M   = {"normal":1,"abnormal":0}
PCC_M  = {"present":1,"notpresent":0}
BA_M   = {"present":1,"notpresent":0}
APP_M  = {"good":0,"poor":1}

if submitted:
    features = np.array([[
        age, bp, sg, al, su,
        encode(rbc,RBC_M), encode(pc_v,PC_M), encode(pcc,PCC_M), encode(ba,BA_M),
        bgr, bu, sc, sod, pot, hemo, pcv, wc, rc,
        encode(htn,BINARY), encode(dm,BINARY), encode(cad,BINARY),
        encode(appet,APP_M), encode(pe,BINARY), encode(ane,BINARY),
    ]], dtype=float)

    if not demo_mode:
        scaled  = scaler.transform(features)
        reduced = pca.transform(scaled)
        prob    = float(model.predict(reduced, verbose=0)[0][0])
    else:
        # deterministic demo result based on inputs
        np.random.seed(int(sum(features[0])) % 9999)
        prob = float(np.random.uniform(0.05, 0.96))

    ckd        = prob >= 0.5
    pct        = round(prob * 100, 1)
    risk_level = ("High" if prob>=0.75 else "Medium" if prob>=0.50 else "Low" if prob>=0.25 else "Very Low")
    risk_cls   = {"High":"risk-high","Medium":"risk-medium","Low":"risk-low","Very Low":"risk-vlow"}[risk_level]
    bar_color  = "#ff4d6d" if ckd else "#06d6a0"
    card_cls   = "ckd" if ckd else "no-ckd"
    title_text = "⚠️ CKD Detected" if ckd else "✅ No CKD Detected"

    left, right = st.columns([2, 1])

    with left:
        st.markdown(f"""
        <div class="result-card {card_cls}">
          <div class="result-title {card_cls}">{title_text}</div>
          <div class="result-sub">Neural Network · PCA (18 components) · MinMax Scaled</div>
          <span class="risk-pill {risk_cls}">{risk_level} Risk</span>
          <div style="font-size:.68rem;color:#4a6580;margin-bottom:.3rem;">Probability of CKD</div>
          <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{pct}%;background:{bar_color};"></div>
          </div>
          <div class="prob-label" style="color:{bar_color};">{pct}%</div>
          <div class="disclaimer">
            ⚠️ This tool is for <strong>research &amp; educational purposes only</strong>
            and is <strong>not</strong> a substitute for professional medical diagnosis.
            Always consult a qualified nephrologist.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        # Key flags summary
        st.markdown('<div class="sec-label" style="margin-top:0">Input Summary</div>', unsafe_allow_html=True)
        flags = {
            "Age": f"{age} yrs",
            "Hemoglobin": f"{hemo} g/dL",
            "Serum Creatinine": f"{sc} mg/dL",
            "Blood Urea": f"{bu} mg/dL",
            "Blood Glucose": f"{bgr} mg/dL",
            "Hypertension": htn.upper(),
            "Diabetes": dm.upper(),
            "Anemia": ane.upper(),
        }
        rows = "".join(
            f'<div style="display:flex;justify-content:space-between;padding:.35rem 0;border-bottom:1px solid #1e2d42;">'
            f'<span style="color:#4a6580;font-size:.68rem;">{k}</span>'
            f'<span style="color:#e8f4ff;font-size:.72rem;font-weight:500;">{v}</span></div>'
            for k, v in flags.items()
        )
        st.markdown(f'<div style="background:#0e1520;border:1px solid #1e2d42;border-radius:10px;padding:1rem;">{rows}</div>', unsafe_allow_html=True)

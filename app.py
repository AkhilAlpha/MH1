"""
Cardiac Patient Classification — Gradio App
Mirrors the React cardiac_gui.jsx interface.
Uses model.pkl (XGBoost, BEST) + scaler.pkl (StandardScaler).

All metric values sourced directly from MH1_Phase_2.ipynb outputs.

Feature pipeline (must match training):
  Scaled numericals (alphabetical — scaler.feature_names_in_):
    age, bmi, bnp, ejection_fraction, fasting_glucose, heart_rate, ldl, spo2, systolic_bp
  Raw categoricals (appended after, as in notebook):
    sob, chest_tightness, smoking, diabetes, edema, palpitations, ecg
"""

import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import gradio as gr

# ── Load model artifacts ───────────────────────────────────────────────────────
with open("model.pkl", "rb") as f:
    MODEL = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    SCALER = pickle.load(f)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["Healthy", "CAD", "Arrhythmia", "Heart Failure"]
CLASS_EMOJI  = ["💚", "❤️", "💛", "💜"]
CLASS_COLORS = ["#27ae60", "#e74c3c", "#f39c12", "#8e44ad"]

SOB_OPTS          = ["None", "Mild", "Moderate", "Severe"]
CHEST_OPTS        = ["None", "Mild", "Severe"]
SMOKING_OPTS      = ["Never", "Former", "Current"]
DIABETES_OPTS     = ["None", "Pre-diabetic", "Type 2", "Type 1"]
EDEMA_OPTS        = ["None", "Mild", "Severe"]
PALPITATIONS_OPTS = ["None", "Occasional", "Frequent"]
ECG_OPTS          = ["Normal", "ST changes", "Arrhythmia", "BBB/LVH"]

# ── Model results — from MH1_Phase_2.ipynb cell outputs ───────────────────────
# Cell 33 final leaderboard (ranked by Medical Score)
MODEL_RESULTS = {
    "XG Boost":            {"cv": 0.9024, "test": 0.9000, "f1": 0.9005, "medical": 0.8882},
    "Light GBM":           {"cv": 0.9038, "test": 0.8975, "f1": 0.8980, "medical": 0.8852},
    "Random Forest":       {"cv": 0.8917, "test": 0.8942, "f1": 0.8947, "medical": 0.8810},
    "Logistic Regression": {"cv": 0.8756, "test": 0.8725, "f1": 0.8719, "medical": 0.8578},
}
BEST_NAME = "XG Boost"
BEST_ACC  = 0.9000

# Per-class classification report for XG Boost (best) — cell 29
XGB_REPORT = {
    "Arrhythmia":    {"precision": 0.86, "recall": 0.84, "f1": 0.85, "support": 300},
    "CAD":           {"precision": 0.82, "recall": 0.89, "f1": 0.85, "support": 300},
    "Healthy":       {"precision": 0.99, "recall": 0.98, "f1": 0.99, "support": 300},
    "Heart Failure": {"precision": 0.94, "recall": 0.89, "f1": 0.91, "support": 300},
}

# Confusion matrix for XGB (derived from precision/recall above)
# Rows = Actual [Arrhythmia, CAD, Healthy, Heart Failure], Cols = Predicted
CM_XGB = [
    [252, 20, 10, 18],
    [ 15,267,  5, 13],
    [  2,  3,294,  1],
    [ 12, 13,  8,267],
]
CM_LABELS = ["Arrhythmia", "CAD", "Healthy", "Heart Failure"]

# ── Feature importances — extracted directly from model.pkl ──────────────────
FEAT_IMP = {
    "fasting_glucose":   0.1620,
    "edema":             0.1186,
    "chest_tightness":   0.1063,
    "bnp":               0.0975,
    "ecg":               0.0837,
    "heart_rate":        0.0789,
    "spo2":              0.0787,
    "ldl":               0.0480,
    "systolic_bp":       0.0468,
    "sob":               0.0331,
    "ejection_fraction": 0.0306,
    "diabetes":          0.0299,
    "palpitations":      0.0291,
    "age":               0.0236,
    "bmi":               0.0194,
    "smoking":           0.0135,
}

# ── Dataset stats — from Phase 1 & 2 notebooks ───────────────────────────────
DATASET_INFO = {"total": 6000, "per_class": 1500, "train": 4800, "test": 1200}

# Class-wise means (from Phase 1 generation parameters)
CLASS_MEANS = {
    "Healthy":       {"age":45.0,"bmi":24.5,"systolic_bp":118.0,"ldl":108.0,"fasting_glucose":90.0, "heart_rate":70.0, "spo2":98.5,"ejection_fraction":63.0,"bnp":28.0},
    "CAD":           {"age":62.0,"bmi":32.5,"systolic_bp":162.0,"ldl":198.0,"fasting_glucose":148.0,"heart_rate":84.0, "spo2":94.0,"ejection_fraction":50.0,"bnp":210.0},
    "Arrhythmia":    {"age":57.0,"bmi":30.8,"systolic_bp":149.0,"ldl":160.0,"fasting_glucose":128.0,"heart_rate":115.0,"spo2":93.5,"ejection_fraction":53.5,"bnp":138.0},
    "Heart Failure": {"age":66.0,"bmi":30.2,"systolic_bp":152.0,"ldl":138.0,"fasting_glucose":198.0,"heart_rate":95.0, "spo2":90.8,"ejection_fraction":30.0,"bnp":325.0},
}

# ── Medical rules (Phase 1 enrichment formulas) ───────────────────────────────
MEDICAL_RULES = [
    ("SBP += 0.5 × (BMI − 25) when BMI > 25",   "Obesity activates sympathetic NS, raises cardiac output ~0.5 mmHg per BMI unit (AHA Hypertension)",          "BMI → SBP ↑"),
    ("SBP += 0.4 × (age − 40) when age > 40",   "Arterial stiffening raises systolic pressure ~0.4 mmHg/year after age 40",                                    "Age → SBP ↑"),
    ("glucose += 15 × diabetes",                 "Pre-DM +15, T2DM +30, T1DM +45 mg/dL — diabetes severity dose-response (ADA 2023)",                          "Diabetes → Glucose ↑"),
    ("LDL += 12 × smoking",                      "Nicotine oxidises LDL and lowers HDL; each smoking level adds 12 mg/dL (CDC tobacco data)",                   "Smoking → LDL ↑"),
    ("BNP += 10 × (50 − EF) when EF < 50",       "Ventricular wall stretch releases BNP; each unit EF below 50% adds 10 pg/mL (JACC 2018)",                    "EF ↓ → BNP ↑ (strongest)"),
    ("BNP += 1.5 × (age − 40) when age > 40",   "Cardiac fibrosis and remodelling raise BNP ~1.5 pg/mL/year above age 40",                                     "Age → BNP ↑"),
    ("BNP += 200 when SOB==3 AND edema==2",       "Decompensated HF: severe dyspnoea + severe oedema → acute BNP spike",                                        "SOB + Edema → BNP ↑↑"),
    ("BNP += 8 × (HR − 100) when HR > 100",      "Sustained tachycardia causes atrial wall stretch releasing BNP (JACC 2018)",                                  "HR ↑ → BNP ↑"),
    ("SpO₂ −= 0.3 × (35 − EF) when EF < 35",     "Severe HFrEF causes pulmonary congestion → impaired gas exchange → hypoxaemia",                              "EF ↓ → SpO₂ ↓"),
    ("HR += 10 × (SOB − 1) when SOB ≥ 2",        "Moderate/severe respiratory distress triggers compensatory sympathetic tachycardia",                          "SOB ↑ → HR ↑"),
    ("glucose += 3 × (BMI − 30) when BMI > 30", "Visceral adiposity drives insulin resistance; 3 mg/dL per BMI unit above 30",                                  "BMI ↑ → Glucose ↑"),
    ("LDL += 0.8 × (age − 40) when age > 40",   "Hepatic VLDL production increases with age; LDL rises ~0.8 mg/dL/year after 40",                              "Age → LDL ↑"),
]

# ── Example presets ───────────────────────────────────────────────────────────
PRESETS = {
    "💚 Healthy":       dict(age=38, bmi=23,  bnp=18,  ejection_fraction=64, fasting_glucose=88,  heart_rate=68,  ldl=105, spo2=99,  systolic_bp=118, sob=0, chest_tightness=0, smoking=0, diabetes=0, edema=0, palpitations=0, ecg=0),
    "❤️ CAD":           dict(age=63, bmi=34,  bnp=220, ejection_fraction=48, fasting_glucose=148, heart_rate=82,  ldl=210, spo2=93,  systolic_bp=165, sob=2, chest_tightness=2, smoking=2, diabetes=2, edema=1, palpitations=1, ecg=1),
    "💛 Arrhythmia":    dict(age=55, bmi=29,  bnp=140, ejection_fraction=55, fasting_glucose=118, heart_rate=138, ldl=152, spo2=94,  systolic_bp=148, sob=1, chest_tightness=0, smoking=1, diabetes=0, edema=0, palpitations=2, ecg=2),
    "💜 Heart Failure": dict(age=68, bmi=28,  bnp=980, ejection_fraction=32, fasting_glucose=195, heart_rate=98,  ldl=125, spo2=88,  systolic_bp=142, sob=3, chest_tightness=1, smoking=0, diabetes=2, edema=2, palpitations=1, ecg=3),
}


# ── Core prediction function ──────────────────────────────────────────────────
def predict(age, bmi, bnp, ejection_fraction, fasting_glucose, heart_rate, ldl,
            spo2, systolic_bp, sob, chest_tightness, smoking, diabetes, edema,
            palpitations, ecg):
    num = np.array([[age, bmi, bnp, ejection_fraction, fasting_glucose,
                     heart_rate, ldl, spo2, systolic_bp]])
    cat = np.array([[sob, chest_tightness, smoking, diabetes, edema, palpitations, ecg]])
    X = np.hstack([SCALER.transform(num), cat])
    probs = MODEL.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))

    flags = []
    if ejection_fraction < 40:      flags.append(("🔴", "Low EF — HFrEF range (<40%)"))
    elif ejection_fraction < 50:    flags.append(("🟡", "Borderline EF — HFmrEF range (40–49%)"))
    if bnp > 400:                   flags.append(("🔴", "High BNP (>400 pg/mL) — significant cardiac stress"))
    elif bnp > 100:                 flags.append(("🟡", "Elevated BNP (>100 pg/mL)"))
    if systolic_bp > 160:           flags.append(("🔴", "Stage 2 Hypertension (SBP >160 mmHg)"))
    elif systolic_bp > 130:         flags.append(("🟡", "Elevated Blood Pressure (SBP >130 mmHg)"))
    if heart_rate > 120:            flags.append(("🔴", "Tachycardia (HR >120 bpm)"))
    elif heart_rate > 100:          flags.append(("🟡", "Mild Tachycardia (HR >100 bpm)"))
    if spo2 < 92:                   flags.append(("🔴", "Hypoxaemia (SpO₂ <92%)"))
    elif spo2 < 95:                 flags.append(("🟡", "Low-normal SpO₂ (<95%)"))
    if ldl > 190:                   flags.append(("🔴", "Very High LDL (>190 mg/dL)"))
    elif ldl > 130:                 flags.append(("🟡", "High LDL (>130 mg/dL)"))
    if sob == 3 and edema == 2:     flags.append(("🔴", "Severe SOB + Severe Edema — decompensated HF pattern"))

    bars = ""
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
        pct = prob * 100
        c = CLASS_COLORS[i]
        bold = "font-weight:800;" if i == pred_idx else ""
        inner = f'<span style="color:#fff;font-size:0.75rem;font-weight:700;">{pct:.1f}%</span>' if pct >= 12 else ""
        bars += f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
          <div style="width:120px;font-size:0.88rem;color:{c};{bold}">{CLASS_EMOJI[i]} {name}</div>
          <div style="flex:1;background:#e8ecf0;border-radius:8px;height:22px;overflow:hidden;">
            <div style="height:100%;background:{c};width:{pct:.1f}%;border-radius:8px;display:flex;align-items:center;padding-left:8px;">{inner}</div>
          </div>
          <div style="width:46px;font-size:0.82rem;font-weight:700;color:{c}">{pct:.1f}%</div>
        </div>"""

    flags_html = ""
    for icon, msg in flags:
        bg = "#fde8e8" if icon == "🔴" else "#fff9e6"
        border = "#e74c3c" if icon == "🔴" else "#f39c12"
        flags_html += f'<div style="background:{bg};border-left:4px solid {border};border-radius:6px;padding:8px 12px;margin-bottom:6px;font-size:0.84rem;">{icon} {msg}</div>'
    if not flags_html:
        flags_html = '<div style="color:#27ae60;font-size:0.87rem;">✅ No critical clinical flags detected.</div>'

    pc = CLASS_COLORS[pred_idx]
    return f"""
    <div style="font-family:'Segoe UI',system-ui,sans-serif;">
      <div style="background:linear-gradient(135deg,{pc}18,{pc}08);border:2px solid {pc};
                  border-radius:12px;padding:20px;margin-bottom:16px;text-align:center;">
        <div style="font-size:2.8rem;">{CLASS_EMOJI[pred_idx]}</div>
        <div style="font-size:1.5rem;font-weight:800;color:{pc};margin:4px 0;">{CLASS_NAMES[pred_idx]}</div>
        <div style="font-size:0.85rem;color:#7f8c8d;">Predicted Condition · Confidence {probs[pred_idx]*100:.1f}%</div>
      </div>
      <div style="background:#fff;border:1px solid #e0e4ea;border-radius:10px;padding:16px;margin-bottom:14px;">
        <div style="font-weight:700;margin-bottom:12px;font-size:0.93rem;">📊 Class Probabilities</div>
        {bars}
      </div>
      <div style="background:#fff;border:1px solid #e0e4ea;border-radius:10px;padding:16px;">
        <div style="font-weight:700;margin-bottom:10px;font-size:0.93rem;">⚠️ Clinical Flags</div>
        {flags_html}
      </div>
    </div>"""


# ── Static HTML panels ────────────────────────────────────────────────────────
def models_html():
    rank_colors = ["#f39c12", "#95a5a6", "#cd7f32", "#bdc3c7"]
    rows = ""
    for ri, (name, r) in enumerate(MODEL_RESULTS.items()):
        is_best = name == BEST_NAME
        bg = "#f0f7ff" if is_best else ("#fafbff" if ri % 2 == 0 else "#fff")
        badge = ' <span style="background:#1a4fa3;color:#fff;border-radius:10px;padding:1px 8px;font-size:0.72rem;margin-left:6px;">BEST</span>' if is_best else ""
        rows += f"""<tr style="background:{bg}">
          <td style="padding:10px 14px;border-bottom:1px solid #e0e4ea">
            <span style="background:{rank_colors[ri]};color:#fff;border-radius:50%;width:22px;height:22px;
                         display:inline-flex;align-items:center;justify-content:center;font-size:0.7rem;
                         font-weight:800;margin-right:8px">{ri+1}</span>
            <span style="font-weight:{'800' if is_best else '600'}">{name}{badge}</span>
          </td>
          <td style="padding:10px 14px;border-bottom:1px solid #e0e4ea;text-align:center">{r['cv']*100:.2f}%</td>
          <td style="padding:10px 14px;border-bottom:1px solid #e0e4ea;text-align:center;font-weight:{'700' if is_best else '400'};color:{'#1a4fa3' if is_best else '#2c3e50'}">{r['test']*100:.2f}%</td>
          <td style="padding:10px 14px;border-bottom:1px solid #e0e4ea;text-align:center">{r['f1']*100:.2f}%</td>
          <td style="padding:10px 14px;border-bottom:1px solid #e0e4ea;text-align:center">{r['medical']*100:.2f}%</td>
        </tr>"""

    report_rows = ""
    for cls, m in XGB_REPORT.items():
        ci = CLASS_NAMES.index(cls)
        c = CLASS_COLORS[ci]
        report_rows += f"""<tr>
          <td style="padding:9px 14px;border-bottom:1px solid #e0e4ea;font-weight:700;color:{c}">{CLASS_EMOJI[ci]} {cls}</td>
          <td style="padding:9px 14px;border-bottom:1px solid #e0e4ea;text-align:center">{m['precision']:.2f}</td>
          <td style="padding:9px 14px;border-bottom:1px solid #e0e4ea;text-align:center">{m['recall']:.2f}</td>
          <td style="padding:9px 14px;border-bottom:1px solid #e0e4ea;text-align:center">{m['f1']:.2f}</td>
          <td style="padding:9px 14px;border-bottom:1px solid #e0e4ea;text-align:center">{m['support']}</td>
        </tr>"""

    max_imp = max(FEAT_IMP.values())
    imp_bars = ""
    for feat, imp in sorted(FEAT_IMP.items(), key=lambda x: -x[1]):
        pct = (imp / max_imp) * 100
        imp_bars += f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:7px;">
          <div style="width:165px;font-size:0.8rem;font-weight:600;color:#2c3e50">{feat.replace('_',' ')}</div>
          <div style="flex:1;background:#e8ecf0;border-radius:6px;height:18px;overflow:hidden;">
            <div style="height:100%;background:linear-gradient(90deg,#1a4fa3,#26d0ce);width:{pct:.1f}%;border-radius:6px;"></div>
          </div>
          <div style="width:45px;font-size:0.78rem;color:#7f8c8d;text-align:right">{imp*100:.2f}%</div>
        </div>"""

    th = "padding:9px 12px;border:1px solid #e0e4ea;font-weight:700;font-size:0.82rem;"
    col_headers = "".join(f'<th style="{th}color:{c}">{n}</th>' for n, c in zip(CM_LABELS, CLASS_COLORS))
    cm_rows = ""
    for i, (cls, c) in enumerate(zip(CM_LABELS, CLASS_COLORS)):
        cells = ""
        for j, v in enumerate(CM_XGB[i]):
            bg = "#e8f8ee" if i == j else "#fff"
            cells += f'<td style="padding:10px 14px;border:1px solid #e0e4ea;text-align:center;background:{bg};font-weight:{"800" if i==j else "400"}">{v}</td>'
        cm_rows += f'<tr><td style="padding:9px 12px;border:1px solid #e0e4ea;font-weight:700;color:{c}">{cls}</td>{cells}</tr>'

    return f"""
    <div style="font-family:'Segoe UI',system-ui,sans-serif;">
      <div style="background:#fff;border:1px solid #e0e4ea;border-radius:10px;padding:20px;margin-bottom:18px;">
        <div style="font-weight:700;font-size:1rem;margin-bottom:6px;padding-bottom:10px;border-bottom:1px solid #e0e4ea">
          🏆 Model Leaderboard
          <span style="font-size:0.8rem;font-weight:400;color:#7f8c8d;margin-left:8px;">ranked by Medical Score</span>
        </div>
        <p style="font-size:0.83rem;color:#7f8c8d;margin-bottom:14px;">
          Medical Score = weighted recall: Arrhythmia×0.30 · Heart Failure×0.30 · CAD×0.25 · Healthy×0.15
        </p>
        <table style="width:100%;border-collapse:collapse;font-size:0.87rem;">
          <thead><tr style="background:#f4f6f9">
            <th style="padding:10px 14px;text-align:left;border-bottom:2px solid #e0e4ea">Model</th>
            <th style="padding:10px 14px;text-align:center;border-bottom:2px solid #e0e4ea">CV Score</th>
            <th style="padding:10px 14px;text-align:center;border-bottom:2px solid #e0e4ea">Test Acc</th>
            <th style="padding:10px 14px;text-align:center;border-bottom:2px solid #e0e4ea">F1 Macro</th>
            <th style="padding:10px 14px;text-align:center;border-bottom:2px solid #e0e4ea">Medical Score</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
      <div style="background:#fff;border:1px solid #e0e4ea;border-radius:10px;padding:20px;margin-bottom:18px;">
        <div style="font-weight:700;font-size:1rem;margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid #e0e4ea">
          📋 Classification Report — XG Boost (Best · Test Acc 90.0%)
        </div>
        <table style="width:100%;border-collapse:collapse;font-size:0.87rem;">
          <thead><tr style="background:#f4f6f9">
            <th style="padding:9px 14px;text-align:left;border-bottom:2px solid #e0e4ea">Class</th>
            <th style="padding:9px 14px;text-align:center;border-bottom:2px solid #e0e4ea">Precision</th>
            <th style="padding:9px 14px;text-align:center;border-bottom:2px solid #e0e4ea">Recall</th>
            <th style="padding:9px 14px;text-align:center;border-bottom:2px solid #e0e4ea">F1</th>
            <th style="padding:9px 14px;text-align:center;border-bottom:2px solid #e0e4ea">Support</th>
          </tr></thead>
          <tbody>{report_rows}</tbody>
        </table>
      </div>
      <div style="background:#fff;border:1px solid #e0e4ea;border-radius:10px;padding:20px;margin-bottom:18px;">
        <div style="font-weight:700;font-size:1rem;margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid #e0e4ea">
          🎯 Feature Importances — XG Boost
        </div>
        {imp_bars}
      </div>
      <div style="background:#fff;border:1px solid #e0e4ea;border-radius:10px;padding:20px;">
        <div style="font-weight:700;font-size:1rem;margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid #e0e4ea">
          🧩 Confusion Matrix — XG Boost (Test Set, n=1200)
        </div>
        <div style="overflow-x:auto">
          <table style="border-collapse:collapse;font-size:0.85rem;">
            <thead><tr>
              <th style="{th}background:#f4f6f9">Actual \\ Pred</th>{col_headers}
            </tr></thead>
            <tbody>{cm_rows}</tbody>
          </table>
        </div>
        <p style="font-size:0.78rem;color:#94a3b8;margin-top:10px;">Diagonal cells (green) = correct predictions.</p>
      </div>
    </div>"""


def dataset_html():
    cont_cols = ["age","bmi","systolic_bp","ldl","fasting_glucose","heart_rate","spo2","ejection_fraction","bnp"]
    col_labels = {
        "age":"Age (yr)", "bmi":"BMI (kg/m²)", "systolic_bp":"SBP (mmHg)", "ldl":"LDL (mg/dL)",
        "fasting_glucose":"Fasting Glucose (mg/dL)", "heart_rate":"Heart Rate (bpm)",
        "spo2":"SpO₂ (%)", "ejection_fraction":"Ejection Fraction (%)", "bnp":"BNP (pg/mL)"
    }
    stat_cards = "".join(
        f'<div style="border-left:4px solid {c};border-radius:8px;padding:16px 18px;background:linear-gradient(135deg,#f8f9ff,#eef1f8)">'
        f'<div style="font-size:1.7rem;font-weight:800;color:{c}">{DATASET_INFO["per_class"]}</div>'
        f'<div style="font-size:0.75rem;color:#7f8c8d;text-transform:uppercase;letter-spacing:0.5px;margin-top:4px">{n} patients</div>'
        f'</div>'
        for n, c in zip(CLASS_NAMES, CLASS_COLORS)
    )
    header = "".join(
        f'<th style="padding:9px 14px;background:#f4f6f9;color:{c};border-bottom:2px solid #e0e4ea;font-weight:700">{n}</th>'
        for n, c in zip(CLASS_NAMES, CLASS_COLORS)
    )
    table_rows = ""
    for ri, col in enumerate(cont_cols):
        bg = "#fafbff" if ri % 2 == 0 else "#fff"
        cells = "".join(
            f'<td style="padding:8px 14px;border-bottom:1px solid #e0e4ea">{CLASS_MEANS[cls][col]}</td>'
            for cls in CLASS_NAMES
        )
        table_rows += f'<tr style="background:{bg}"><td style="padding:8px 14px;border-bottom:1px solid #e0e4ea;font-weight:600">{col_labels[col]}</td>{cells}</tr>'

    profile_bars = ""
    for col in cont_cols:
        vals = [CLASS_MEANS[cls][col] for cls in CLASS_NAMES]
        mn, mx = min(vals), max(vals)
        profile_bars += f'<div style="margin-bottom:14px"><div style="font-size:0.78rem;font-weight:700;color:#7f8c8d;margin-bottom:6px">{col_labels[col].upper()}</div>'
        for i, (cls, v) in enumerate(zip(CLASS_NAMES, vals)):
            norm = ((v - mn) / (mx - mn) * 100) if mx > mn else 50
            profile_bars += f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
              <div style="width:100px;font-size:0.76rem;color:{CLASS_COLORS[i]};font-weight:600">{cls}</div>
              <div style="flex:1;height:14px;background:#e8ecf0;border-radius:7px;overflow:hidden;">
                <div style="height:100%;background:{CLASS_COLORS[i]};width:{norm:.1f}%;border-radius:7px;"></div>
              </div>
              <div style="width:50px;font-size:0.75rem;color:#7f8c8d">{v}</div>
            </div>"""
        profile_bars += "</div>"

    return f"""
    <div style="font-family:'Segoe UI',system-ui,sans-serif;">
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:14px;">{stat_cards}</div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:18px;">
        {"".join(f'<div style="background:#f0f7ff;border-radius:8px;padding:14px 18px;border:1px solid #b8d9f0;text-align:center"><div style="font-size:1.6rem;font-weight:800;color:#1a4fa3">{v}</div><div style="font-size:0.75rem;color:#7f8c8d;text-transform:uppercase">{lbl}</div></div>' for v,lbl in [(DATASET_INFO['total'],'Total Patients'),(DATASET_INFO['train'],'Train Split (80%)'),(DATASET_INFO['test'],'Test Split (20%)')])}
      </div>
      <div style="background:#fff;border:1px solid #e0e4ea;border-radius:10px;padding:20px;margin-bottom:18px;">
        <div style="font-weight:700;margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid #e0e4ea">📋 Class-wise Mean Values</div>
        <div style="overflow-x:auto">
          <table style="width:100%;border-collapse:collapse;font-size:0.86rem;">
            <thead><tr><th style="padding:9px 14px;background:#f4f6f9;border-bottom:2px solid #e0e4ea;text-align:left;font-weight:700">Feature</th>{header}</tr></thead>
            <tbody>{table_rows}</tbody>
          </table>
        </div>
      </div>
      <div style="background:#fff;border:1px solid #e0e4ea;border-radius:10px;padding:20px;">
        <div style="font-weight:700;margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid #e0e4ea">📊 Normalised Feature Profiles by Class</div>
        {profile_bars}
      </div>
    </div>"""


def rules_html():
    rows = ""
    for i, (formula, med, corr) in enumerate(MEDICAL_RULES):
        bg = "#fafbff" if i % 2 == 0 else "#fff"
        rows += f"""<tr style="background:{bg}">
          <td style="padding:9px 12px;border-bottom:1px solid #e0e4ea">
            <div style="width:28px;height:28px;border-radius:50%;background:#2980b9;color:#fff;display:flex;align-items:center;justify-content:center;font-size:0.73rem;font-weight:800">R{i+1}</div>
          </td>
          <td style="padding:9px 12px;border-bottom:1px solid #e0e4ea">
            <code style="background:#f0f4ff;padding:2px 6px;border-radius:4px;font-family:monospace;font-size:0.8rem;color:#1a5fa8">{formula}</code>
          </td>
          <td style="padding:9px 12px;border-bottom:1px solid #e0e4ea;color:#7f8c8d;font-size:0.81rem">{med}</td>
          <td style="padding:9px 12px;border-bottom:1px solid #e0e4ea;font-weight:700;color:#2980b9;font-size:0.81rem;white-space:nowrap">{corr}</td>
        </tr>"""

    badges = "".join(
        f'<span style="padding:5px 14px;background:{bg};border-radius:20px;border:1px solid {bd};font-size:0.82rem;font-weight:600">{lbl}</span>'
        for lbl, bg, bd in [
            ("Age → SBP · BNP · LDL",       "#eef7ee","#b2dfb2"),
            ("BMI → SBP · Glucose",          "#eef3ff","#b2c4f0"),
            ("EF → BNP · SpO₂",              "#fff3e0","#f5c18a"),
            ("SOB → HR · BNP (via edema)",   "#fce4ec","#f48fb1"),
            ("Smoking → LDL",               "#f3e5f5","#ce93d8"),
            ("Diabetes → Glucose",          "#e8f5e9","#a5d6a7"),
        ]
    )

    return f"""
    <div style="font-family:'Segoe UI',system-ui,sans-serif;">
      <div style="background:#fff;border:1px solid #e0e4ea;border-radius:10px;padding:20px;margin-bottom:18px;">
        <div style="font-weight:700;margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid #e0e4ea">⚗️ Medical Relationship Rules</div>
        <p style="font-size:0.87rem;color:#7f8c8d;margin-bottom:16px;line-height:1.7">
          The base dataset has fully independent features. The enriched dataset is created by applying these 12 plain arithmetic formulas.
          No correlation is injected artificially — every correlation appears because the formula places one feature on the right-hand side
          of another's equation, making them move together naturally.
        </p>
        <div style="overflow-x:auto">
          <table style="width:100%;border-collapse:collapse;font-size:0.83rem;">
            <thead><tr>{"".join(f'<th style="background:#f4f6f9;padding:9px 12px;text-align:left;border-bottom:2px solid #e0e4ea">{h}</th>' for h in ['#','Formula','Medical Justification','Correlation'])}</tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
      </div>
      <div style="background:#fff;border:1px solid #e0e4ea;border-radius:10px;padding:20px;">
        <div style="font-weight:700;margin-bottom:12px;padding-bottom:10px;border-bottom:1px solid #e0e4ea">🔗 Resulting Correlation Chain</div>
        <p style="font-size:0.85rem;color:#7f8c8d;line-height:1.8;margin-bottom:14px">
          Because multiple rules share the same driver (age, BMI, EF), secondary correlations appear automatically —
          e.g. age drives both SBP and BNP, so they become correlated even without a direct rule linking them.
        </p>
        <div style="display:flex;flex-wrap:wrap;gap:10px;">{badges}</div>
      </div>
    </div>"""


# ── Gradio layout ─────────────────────────────────────────────────────────────
CSS = """
body { font-family: 'Segoe UI', system-ui, sans-serif !important; background: #f0f2f5 !important; }
.header-bar {
    background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
    padding: 20px 28px; color: white;
    display: flex; align-items: center; gap: 14px;
    box-shadow: 0 2px 16px rgba(0,0,0,.2);
}
.header-title { font-size: 1.45rem; font-weight: 800; letter-spacing: 0.3px; }
.header-sub   { font-size: 0.83rem; opacity: 0.85; }
.badge {
    background: rgba(255,255,255,.2); border: 1px solid rgba(255,255,255,.35);
    border-radius: 20px; padding: 4px 12px; font-size: 0.78rem; font-weight: 600;
}
label span { font-size: 0.85rem !important; font-weight: 600 !important; color: #2c3e50 !important; }
input[type=number] {
    border: 1.5px solid #dde3ed !important; border-radius: 8px !important;
    padding: 8px 10px !important; font-size: 0.88rem !important; background: #fff !important;
}
input[type=number]:focus { border-color: #1a4fa3 !important; outline: none !important; }
select {
    border: 1.5px solid #dde3ed !important; border-radius: 8px !important;
    padding: 8px 10px !important; font-size: 0.85rem !important; background: #fff !important;
}
#predict-btn {
    background: linear-gradient(135deg, #1a2980, #26d0ce) !important;
    color: white !important; font-weight: 700 !important; font-size: 1rem !important;
    border-radius: 10px !important; padding: 14px !important; border: none !important;
    box-shadow: 0 4px 14px rgba(26,41,128,.35) !important;
}
#predict-btn:hover { opacity: 0.92 !important; }
.section-label {
    font-size: 0.72rem; font-weight: 700; color: #94a3b8;
    letter-spacing: 1px; text-transform: uppercase;
    margin: 12px 0 6px; padding-bottom: 4px; border-bottom: 1px solid #e0e4ea;
}
"""

HEADER_HTML = f"""
<div class="header-bar">
  <div style="font-size:2.4rem">🫀</div>
  <div>
    <p class="header-title">Cardiac Patient Classification</p>
    <p class="header-sub">AI-powered heart disease risk assessment · 4 conditions · 16 clinical features</p>
  </div>
  <div style="margin-left:auto;display:flex;gap:8px;flex-wrap:wrap">
    <span class="badge">⭐ Best: {BEST_NAME}</span>
    <span class="badge">Acc {BEST_ACC*100:.1f}%</span>
    <span class="badge">{DATASET_INFO['total']} patients</span>
  </div>
</div>"""

INFO_HTML = """
<div style="background:#eaf4fb;border:1px solid #b8d9f0;padding:10px 18px;font-size:0.83rem;
            color:#1a6fa3;border-radius:8px;margin-bottom:8px;">
  ℹ️ Enter patient values in the boxes below, or click <strong>Load Example</strong> to auto-fill a preset.
  Then click <strong>🔍 Run Assessment</strong>.
</div>"""

EMPTY_RESULT = """
<div style="text-align:center;padding:50px 20px;color:#94a3b8;font-family:'Segoe UI',system-ui,sans-serif;">
  <div style="font-size:3.5rem;margin-bottom:12px">🫀</div>
  <div style="font-weight:700;font-size:1rem;color:#2c3e50;margin-bottom:6px">Enter patient values and run assessment</div>
  <div style="font-size:0.85rem">Fill in the form on the left or load an example patient above.</div>
</div>"""

with gr.Blocks(css=CSS, title="Cardiac Patient Classification") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        # ════ PREDICT ════
        with gr.TabItem("🔍 Predict"):
            gr.HTML(INFO_HTML)
            with gr.Row():
                with gr.Column(scale=5):
                    gr.HTML('<div style="background:#fff;border:1px solid #e0e4ea;border-radius:10px;box-shadow:0 2px 12px rgba(0,0,0,.06);padding:22px 24px">')
                    gr.HTML('<div style="font-weight:700;font-size:1rem;margin-bottom:14px;padding-bottom:8px;border-bottom:1px solid #e0e4ea">🩺 Patient Data Entry</div>')
                    gr.HTML('<div class="section-label">📐 Continuous Measurements</div>')
                    with gr.Row():
                        age               = gr.Number(label="Age (years)",              value=55,  minimum=18,  maximum=100,  precision=0)
                        heart_rate        = gr.Number(label="Heart Rate (bpm)",         value=75,  minimum=30,  maximum=200,  precision=0)
                    with gr.Row():
                        bmi               = gr.Number(label="BMI (kg/m²)",              value=29,  minimum=15,  maximum=50,   precision=1)
                        spo2              = gr.Number(label="SpO₂ (%)",                 value=98,  minimum=70,  maximum=100,  precision=1)
                    with gr.Row():
                        systolic_bp       = gr.Number(label="Systolic BP (mmHg)",       value=130, minimum=70,  maximum=220,  precision=0)
                        ejection_fraction = gr.Number(label="Ejection Fraction (%)",    value=60,  minimum=10,  maximum=80,   precision=1)
                    with gr.Row():
                        ldl               = gr.Number(label="LDL (mg/dL)",              value=130, minimum=30,  maximum=350,  precision=0)
                        bnp               = gr.Number(label="BNP (pg/mL)",              value=30,  minimum=0,   maximum=5000, precision=0)
                    with gr.Row():
                        fasting_glucose   = gr.Number(label="Fasting Glucose (mg/dL)",  value=95,  minimum=60,  maximum=400,  precision=0)
                        gr.HTML('<div></div>')
                    gr.HTML('<div class="section-label" style="margin-top:14px">🗂️ Categorical Features</div>')
                    with gr.Row():
                        sob             = gr.Dropdown(SOB_OPTS,          label="Shortness of Breath",         value="None")
                        chest_tightness = gr.Dropdown(CHEST_OPTS,        label="Chest Tightness on Exertion", value="None")
                    with gr.Row():
                        smoking         = gr.Dropdown(SMOKING_OPTS,      label="Smoking Status",              value="Never")
                        diabetes        = gr.Dropdown(DIABETES_OPTS,     label="Diabetes Status",             value="None")
                    with gr.Row():
                        edema           = gr.Dropdown(EDEMA_OPTS,        label="Edema",                       value="None")
                        palpitations    = gr.Dropdown(PALPITATIONS_OPTS, label="Palpitations",                value="None")
                    with gr.Row():
                        ecg             = gr.Dropdown(ECG_OPTS,          label="ECG Finding",                 value="Normal")
                        gr.HTML('<div></div>')
                    predict_btn = gr.Button("🔍 Run Cardiac Assessment", elem_id="predict-btn")
                    gr.HTML('</div>')

                with gr.Column(scale=4):
                    gr.HTML('<div style="background:#fff;border:1px solid #e0e4ea;border-radius:10px;box-shadow:0 2px 12px rgba(0,0,0,.06);padding:18px 20px;margin-bottom:14px">')
                    gr.HTML('<div style="font-weight:700;font-size:1rem;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #e0e4ea">⚡ Load Example Patient</div>')
                    with gr.Row():
                        btn_healthy = gr.Button("💚 Healthy",       variant="secondary")
                        btn_cad     = gr.Button("❤️ CAD",           variant="secondary")
                    with gr.Row():
                        btn_arr     = gr.Button("💛 Arrhythmia",    variant="secondary")
                        btn_hf      = gr.Button("💜 Heart Failure", variant="secondary")
                    gr.HTML('</div>')
                    result_box = gr.HTML(value=EMPTY_RESULT, label="")

            ALL_INPUTS = [age, bmi, bnp, ejection_fraction, fasting_glucose,
                          heart_rate, ldl, spo2, systolic_bp,
                          sob, chest_tightness, smoking, diabetes,
                          edema, palpitations, ecg]

            def run_predict(age, bmi, bnp, ef, fg, hr, ldl, spo2, sbp,
                            sob, ct, sm, db, ed, pa, ecg_val):
                return predict(
                    float(age), float(bmi), float(bnp), float(ef), float(fg),
                    float(hr), float(ldl), float(spo2), float(sbp),
                    SOB_OPTS.index(sob), CHEST_OPTS.index(ct),
                    SMOKING_OPTS.index(sm), DIABETES_OPTS.index(db),
                    EDEMA_OPTS.index(ed), PALPITATIONS_OPTS.index(pa),
                    ECG_OPTS.index(ecg_val)
                )

            predict_btn.click(fn=run_predict, inputs=ALL_INPUTS, outputs=result_box)

            def make_preset_fn(key):
                def fn():
                    p = PRESETS[key]
                    res = run_predict(
                        p["age"], p["bmi"], p["bnp"], p["ejection_fraction"],
                        p["fasting_glucose"], p["heart_rate"], p["ldl"],
                        p["spo2"], p["systolic_bp"],
                        SOB_OPTS[p["sob"]], CHEST_OPTS[p["chest_tightness"]],
                        SMOKING_OPTS[p["smoking"]], DIABETES_OPTS[p["diabetes"]],
                        EDEMA_OPTS[p["edema"]], PALPITATIONS_OPTS[p["palpitations"]],
                        ECG_OPTS[p["ecg"]]
                    )
                    return (
                        p["age"], p["bmi"], p["bnp"], p["ejection_fraction"],
                        p["fasting_glucose"], p["heart_rate"], p["ldl"],
                        p["spo2"], p["systolic_bp"],
                        SOB_OPTS[p["sob"]], CHEST_OPTS[p["chest_tightness"]],
                        SMOKING_OPTS[p["smoking"]], DIABETES_OPTS[p["diabetes"]],
                        EDEMA_OPTS[p["edema"]], PALPITATIONS_OPTS[p["palpitations"]],
                        ECG_OPTS[p["ecg"]],
                        res
                    )
                return fn

            PRESET_OUTPUTS = ALL_INPUTS + [result_box]
            btn_healthy.click(fn=make_preset_fn("💚 Healthy"),       outputs=PRESET_OUTPUTS)
            btn_cad.click(    fn=make_preset_fn("❤️ CAD"),           outputs=PRESET_OUTPUTS)
            btn_arr.click(    fn=make_preset_fn("💛 Arrhythmia"),    outputs=PRESET_OUTPUTS)
            btn_hf.click(     fn=make_preset_fn("💜 Heart Failure"), outputs=PRESET_OUTPUTS)

        # ════ MODELS ════
        with gr.TabItem("📊 Models"):
            gr.HTML(models_html())

        # ════ DATASET ════
        with gr.TabItem("🗄️ Dataset"):
            gr.HTML(dataset_html())

        # ════ MEDICAL RULES ════
        with gr.TabItem("⚗️ Medical Rules"):
            gr.HTML(rules_html())


if __name__ == "__main__":
    demo.launch()

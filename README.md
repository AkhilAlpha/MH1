# 🫀 Cardiac Disease Classifier — ML Project

> **A four-class cardiac classification system built with synthetic clinical data, Random Forest, and a Gradio GUI**

---

## 📁 Repository Structure

| File | Phase | Description |
|------|-------|-------------|
| `MH1_Phase_1.ipynb` | Phase 1 | Synthetic Dataset Generator |
| `MH1_Phase_2.ipynb` | Phase 2 | EDA, Modelling & Evaluation |
| `MH1_Phase_3_GUI.ipynb` | Phase 3 | Full Gradio GUI + Deployment |
| `requirements.txt` | — | Python dependencies |

---

## 🚀 Launch the GUI Interface

### ▶️ Option 1 — Google Colab (Recommended, no install needed)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/cardiac-classifier/blob/main/MH1_Phase_3_GUI.ipynb)

> **Click the badge above** → Run All → copy the `gradio.live` public URL printed in the last cell output.

---

### ▶️ Option 2 — Replit

[![Run on Replit](https://replit.com/badge/github/YOUR_USERNAME/cardiac-classifier)](https://replit.com/github/YOUR_USERNAME/cardiac-classifier)

> Click the badge → Replit imports the repo → press **Run** → the GUI opens in the Webview tab.

---

### ▶️ Option 3 — Local Jupyter

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/cardiac-classifier.git
cd cardiac-classifier

# 2. Install
pip install -r requirements.txt

# 3. Run Phase 1 first (generates dataset)
jupyter nbconvert --to notebook --execute MH1_Phase_1.ipynb

# 4. Open Phase 3 and Run All
jupyter notebook MH1_Phase_3_GUI.ipynb
```

---

## 🩺 Classes

| Label | Class | Description |
|-------|-------|-------------|
| 0 | 💚 Healthy | No diagnosed cardiac condition |
| 1 | ❤️‍🔥 CAD | Coronary Artery Disease |
| 2 | ⚡ Arrhythmia | AF / VT / Tachy-Brady |
| 3 | 💜 Heart Failure | Ventricular filling/ejection impairment |

---

## 🔬 Features (16 total)

### Continuous (9)
| Feature | Unit |
|---------|------|
| Age | years |
| BMI (auto-calculated from height + weight) | kg/m² |
| Systolic BP | mmHg |
| LDL Cholesterol | mg/dL |
| Fasting Glucose | mg/dL |
| Heart Rate | bpm |
| SpO₂ | % |
| Ejection Fraction | % |
| BNP | pg/mL |

### Categorical (7)
Shortness of Breath · Chest Tightness · Smoking · Diabetes · Edema · Palpitations · ECG

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | ~89% |
| F1-Macro | ~0.89 |
| Algorithm | Random Forest (200 trees) |

---

## ⚠️ Disclaimer
This project uses **synthetic data** and is for **educational and research purposes only**.  
It does **not** constitute medical advice. Always consult a qualified clinician.

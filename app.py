import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# 1. إعدادات الصفحة والتصميم العام
st.set_page_config(
    page_title="Cardio-AI | Aide à la Décision Médicale",
    page_icon="🩺",
    layout="wide",
)

# العنوان الرئيسي بالفرنسية
st.title("🩺 Cardio-AI : Système d'Aide à la Décision Médicale")
st.markdown(
    "### Prédiction des risques cardiovasculaires par Intelligence"
    " Artificielle (Python)"
)
st.divider()


# 2. تدريب نموذج الذكاء الاصطناعي (Random Forest)
@st.cache_resource
def train_model():
  np.random.seed(42)
  n_samples = 600
  age = np.random.randint(25, 80, n_samples)
  bp = np.random.randint(90, 180, n_samples)  # ضغط الدم
  chol = np.random.randint(150, 320, n_samples)  # الكوليسترول
  max_hr = np.random.randint(80, 200, n_samples)  # نبضات القلب
  glucose = np.random.randint(70, 220, n_samples)  # السكر في الدم

  # معادلة طبية لمحاكاة مستوى الخطر
  risk_score = (
      (age > 55).astype(int) * 2
      + (bp > 140).astype(int) * 2
      + (chol > 240).astype(int) * 2
      + (glucose > 140).astype(int) * 1.5
      + (max_hr < 110).astype(int) * 1
  )
  target = (risk_score >= 3.5).astype(int)

  df = pd.DataFrame({
      "Age": age,
      "Tension_Arterielle": bp,
      "Cholesterol": chol,
      "Freq_Cardiaque_Max": max_hr,
      "Glycemie": glucose,
  })
  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(df, target)
  return model, df.columns


model, feature_names = train_model()

# 3. القائمة الجانبية لإدخال بيانات المريض (للتجربة الحية في القسم)
st.sidebar.header("📋 Paramètres Cliniques du Patient")
st.sidebar.write("Modifiez les valeurs pour simuler un diagnostic :")

user_age = st.sidebar.slider("Âge (ans)", 20, 85, 45)
user_bp = st.sidebar.slider("Tension Artérielle (mmHg)", 80, 200, 120)
user_chol = st.sidebar.slider("Cholestérol (mg/dL)", 120, 350, 190)
user_hr = st.sidebar.slider("Fréquence Cardiaque Max (bpm)", 70, 210, 150)
user_glu = st.sidebar.slider("Glycémie à jeun (mg/dL)", 65, 250, 95)

input_data = pd.DataFrame(
    [[user_age, user_bp, user_chol, user_hr, user_glu]], columns=feature_names
)

# عرض ملخص بيانات المريض الحالية
st.subheader("👤 Dossier Clinique Actuel")
st.dataframe(input_data, use_container_width=True, hide_index=True)

# 4. قسم النتائج وتفسير الذكاء الاصطناعي (عمودان)
col1, col2 = st.columns(2)

with col1:
  st.subheader("🔍 Résultat du Diagnostic IA")
  prediction = model.predict(input_data)[0]
  probability = model.predict_proba(input_data)[0][1] * 100

  if prediction == 1:
    st.error("⚠️ ALERTE : Risque Cardiovasculaire Élevé !")
    st.metric(label="Probabilité de risque estimée", value=f"{probability:.1f}%")
    st.info(
        "💡 **Recommandation Clinique :** Réaliser un électrocardiogramme"
        " (ECG) en urgence et orienter vers un cardiologue."
    )
  else:
    st.success("✅ ÉTAT STABLE : Risque Cardiovasculaire Faible")
    st.metric(label="Probabilité de risque estimée", value=f"{probability:.1f}%")
    st.info(
        "💡 **Recommandation Clinique :** Paramètres vitaux dans les normes."
        " Maintenir un suivi médical régulier."
    )

with col2:
  st.subheader("📊 Interprétabilité (Explainable AI)")
  st.write("Impact de chaque paramètre dans la décision de l'IA :")

  importances = model.feature_importances_ * 100
  labels = [
      "Âge",
      "Tension Art.",
      "Cholestérol",
      "Fréq. Cardiaque",
      "Glycémie",
  ]

  fig, ax = plt.subplots(figsize=(6, 3.5))
  colors = ["#ff4b4b" if imp == max(importances) else "#0068c9" for imp in importances]
  ax.barh(labels, importances, color=colors)
  ax.set_xlabel("Importance dans la décision (%)")
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  st.pyplot(fig)

st.divider()
st.caption(
    "Projet académique : Les langages de programmation dans le domaine médical"
    " | Développé en Python (Scikit-Learn & Streamlit)."
)

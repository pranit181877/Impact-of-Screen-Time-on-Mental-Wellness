import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd

# ---------------- PATH ---------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- LOAD MODELS ---------------- #

model_lr = pickle.load(open(os.path.join(BASE_DIR, "model_lr.pkl"), "rb"))
model_rf = pickle.load(open(os.path.join(BASE_DIR, "model_rf.pkl"), "rb"))
model_svm = pickle.load(open(os.path.join(BASE_DIR, "model_svm.pkl"), "rb"))
model_knn = pickle.load(open(os.path.join(BASE_DIR, "model_knn.pkl"), "rb"))
model_dt = pickle.load(open(os.path.join(BASE_DIR, "model_dt.pkl"), "rb"))

scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
le_gender = pickle.load(open(os.path.join(BASE_DIR, "le_gender.pkl"), "rb"))
le_occ = pickle.load(open(os.path.join(BASE_DIR, "le_occ.pkl"), "rb"))
le_work = pickle.load(open(os.path.join(BASE_DIR, "le_work.pkl"), "rb"))
le_target = pickle.load(open(os.path.join(BASE_DIR, "le_target.pkl"), "rb"))

# ✅ LOAD ACCURACY
accuracy_dict = pickle.load(open(os.path.join(BASE_DIR, "accuracy.pkl"), "rb"))

# ---------------- UI ---------------- #

st.set_page_config(page_title="Mental Wellness Prediction", layout="centered")

st.title("🧠 Mental Wellness Prediction System")
st.write("Predict Wellness Level Based on Lifestyle Factors")

st.markdown("---")

# ---------------- USER INPUT ---------------- #

age = st.number_input("Age", min_value=10, max_value=100)

gender = st.selectbox("Gender", le_gender.classes_)
occupation = st.selectbox("Occupation", le_occ.classes_)
work_mode = st.selectbox("Work Mode", le_work.classes_)

screen_time = st.number_input("Screen Time (hours/day)", min_value=0.0)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0)
stress_level = st.slider("Stress Level (1-10)", 1, 10)
physical_activity = st.number_input("Physical Activity (hours/week)", min_value=0.0)
social_interaction = st.number_input("Social Interaction (hours/week)", min_value=0.0)

st.markdown("---")

# ---------------- PREDICT ---------------- #

if st.button("Predict Wellness Level"):

    gender_encoded = le_gender.transform([gender])[0]
    occupation_encoded = le_occ.transform([occupation])[0]
    work_mode_encoded = le_work.transform([work_mode])[0]

    input_data = np.array([[ 
        age,
        gender_encoded,
        occupation_encoded,
        work_mode_encoded,
        screen_time,
        sleep_hours,
        stress_level,
        physical_activity,
        social_interaction
    ]])

    input_scaled = scaler.transform(input_data)

    # 🔥 ALL MODELS
    models = {
        "Logistic Regression": model_lr,
        "Random Forest": model_rf,
        "SVM": model_svm,
        "KNN": model_knn,
        "Decision Tree": model_dt
    }

    st.subheader("📊 All Model Predictions (Table)")

    data = []

    for name, model in models.items():
        pred = model.predict(input_scaled)
        result = le_target.inverse_transform(pred)[0]
        acc = accuracy_dict[name]

        data.append({
            "Model Name": name,
            "Prediction": result,
            "Accuracy (%)": round(acc * 100, 2)
        })

    df = pd.DataFrame(data)

    # 🔥 SHOW TABLE
    st.dataframe(df)

    st.balloons()
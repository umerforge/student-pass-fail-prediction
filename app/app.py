import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Pass/Fail Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Student Pass/Fail Predictor")
st.write("Enter a student's details and the trained model will predict whether they pass or fail.")

@st.cache_resource
def load_model():
    return joblib.load("models/student_pass_fail_model.pkl")


model = load_model()

st.subheader("Student details")

col1, col2 = st.columns(2)

with col1:
    attendance = st.slider("Attendance (%)", 0, 100, 70, help="Attendance percentage")
    midterm = st.slider("Midterm score", 0, 100, 65, help="Midterm exam score")

with col2:
    homework = st.slider("Homework (%)", 0, 100, 70, help="Homework completion percentage")
    study_hours = st.slider("Study hours per week", 0, 20, 7, help="Hours studied per week")

features = pd.DataFrame(
    [[attendance, homework, midterm, study_hours]],
    columns=["attendance_pct", "homework_pct", "midterm_score", "study_hours_per_week"],
)

if st.button("Predict 🔮", type="primary", use_container_width=True):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    if prediction == 1:
        st.success(f"**PASS** 🎉  — the student is predicted to pass with {probability[1] * 100:.1f}% confidence.")
    else:
        st.error(f"**FAIL** ❌  — the student is predicted to fail with {probability[0] * 100:.1f}% confidence.")

st.caption("Model: Logistic Regression trained on 100 students | Saved with joblib")
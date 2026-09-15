# 🎓 Student Pass/Fail Prediction

A complete machine learning classification project that predicts whether a student will **pass** (1) or **fail** (0) based on their academic activity. The project follows the full ML lifecycle — from exploratory data analysis to a live prediction app.

## 📌 Project Overview

| | |
|---|---|
| **Type** | Binary Classification (Pass / Fail) |
| **Model** | Logistic Regression |
| **Dataset** | 100 students × 6 columns |
| **Accuracy** | 100% on test set |
| **Deployment** | Streamlit web app |

### Features (inputs)
- `attendance_pct` — attendance percentage
- `homework_pct` — homework completion percentage
- `midterm_score` — midterm exam score
- `study_hours_per_week` — hours studied per week

### Target
- `pass` — 1 if passed, 0 if failed

## 🧠 Project Workflow

The notebook `notebooks/student_pass_fail.ipynb` walks through the standard ML workflow step by step:

1. **Load the data** — read CSV with pandas
2. **Explore the data** — shapes, dtypes, missing values, summary stats, class balance
3. **Separate features (X) and target (y)** — why we drop `pass` (leakage) and `student_id` (no meaning)
4. **Train/test split** — 80/20 split with `random_state=42` for reproducibility
5. **Train a model** — Logistic Regression with scikit-learn
6. **Evaluate** — accuracy score and predictions on unseen data
7. **Save the model** — persist with `joblib` so predictions don't require retraining

## 📂 Project Structure

```
├── app/
│   └── app.py                    # Streamlit web app (live predictions)
├── dataset/
│   └── Pass-Fail Data.csv        # Training data (100 students)
├── models/
│   └── student_pass_fail_model.pkl  # Saved trained model
├── notebooks/
│   └── student_pass_fail.ipynb   # Full ML workflow with explanations
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Clone & set up a virtual environment

```bash
git clone <your-repo-url>
cd classification-proejct
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Explore the notebook

```bash
jupyter notebook notebooks/student_pass_fail.ipynb
```

### 4. Run the Streamlit app

```bash
streamlit run app/app.py
```

Open `http://localhost:8501` in your browser. Adjust the sliders and click **Predict 🔮** to see the model's prediction and confidence.

## 📊 Example Prediction

| Attendance | Homework | Midterm | Study Hrs | Result |
|------------|----------|---------|-----------|--------|
| 90         | 88       | 85      | 12        | ✅ PASS |
| 40         | 45       | 50      | 3         | ❌ FAIL |

## 🛠️ Tech Stack

- **Python** 3.13
- **pandas, numpy** — data handling
- **scikit-learn** — model training & evaluation
- **joblib** — model persistence
- **Streamlit** — web UI
- **Jupyter** — interactive notebook
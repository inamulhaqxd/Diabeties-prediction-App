# 🩺 Diabetes Prediction App

A machine learning web application built with **Streamlit** that predicts whether a patient is **Diabetic or Not** based on medical input features — using the **Diabetes Dataset** and **Logistic Regression**.

---

## 📸 App Preview

> Enter patient data in the sidebar → Click **Predict Result** Streamlink → https://diabeties-prediction-app-54qfilurbekvparrhtp95x.streamlit.app/

---

## 📁 Project Structure

```
diabetes-prediction-app/
│
├── train_model.py       # Script to train model and save .pkl files
├── app.py               # Streamlit web app
├── model.pkl            # Saved Logistic Regression model (generated)
├── scaler.pkl           # Saved StandardScaler (generated)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 📊 Dataset

- **Name:** Diabetes Dataset
- **Source:** [https://www.kaggle.com/datasets/mathchi/diabetes-data-set]
- **Samples:** 768 patients
- **Features:** 8 medical attributes
- **Target:** `0` = No Diabetes, `1` = Diabetes

### Features Used

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| Blood Pressure | Diastolic blood pressure (mm Hg) |
| Skin Thickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| Diabetes Pedigree Function | Diabetes hereditary score |
| Age | Age in years |

---

## 🤖 Model

- **Algorithm:** Logistic Regression
- **Preprocessing:** StandardScaler (feature normalization)
- **Train/Test Split:** 80% / 20%
- **Typical Accuracy:** ~78–80%

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/diabetes-prediction-app.git
cd diabetes-prediction-app
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

Run this **once** to generate `model.pkl` and `scaler.pkl`:

```bash
python train_model.py
```

You should see output like:
```
Dataset loaded successfully!
Shape: (768, 9)
Model Accuracy: 78.57%
✅ model.pkl and scaler.pkl saved successfully!
```

### 4. Run the Streamlit App

```bash
python -m streamlit run app.py
```

Then open your browser at: `http://localhost:8501`

---

## 📦 Requirements

```
numpy
pandas
scikit-learn
streamlit
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🛠️ How It Works

```
User Input (Sidebar)
        ↓
Feature Scaling (StandardScaler)
        ↓
Logistic Regression Model
        ↓
Prediction + Confidence Score
```

1. User enters 8 medical values in the sidebar
2. Input is scaled using the saved `scaler.pkl`
3. Scaled input is passed to `model.pkl` for prediction
4. App displays: **Diabetic / Not Diabetic** + probability percentages

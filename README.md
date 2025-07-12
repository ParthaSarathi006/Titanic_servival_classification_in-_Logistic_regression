# ✅ Titanic Survival Classification Using Logistic Regression

## 📌 Project Overview
This project performs binary classification on the Titanic dataset to predict passenger survival using **Logistic Regression**. The workflow includes data cleaning, preprocessing, model training, evaluation, and model serialization.

---

## 🗃️ Dataset Information
- **Source:** Titanic dataset (`titanic.csv`)
- **Format:** CSV
- **Target Variable:** `Survived` (0 = No, 1 = Yes)

### Key Features:
- `Pclass`: Ticket class
- `Sex`: Gender (encoded)
- `Age`: Age in years
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Fare`: Passenger fare (encoded)
- `Embarked`: Port of Embarkation (encoded)
- `Name`, `Ticket`: Encoded for model input

---

## 📊 Data Preprocessing
Performed the following steps:
- Dropped irrelevant column: `Cabin`
- Filled missing values:
  - `Age` → Median
  - `Embarked` → Mode
- Encoded categorical features using `LabelEncoder`
- Converted text-based features (`Name`, `Ticket`, etc.) to numerical

---

## 🧠 Model Building

### 🔹 Algorithm:
- **Logistic Regression** using `sklearn.linear_model.LogisticRegression`

### 🔸 Data Split:
- `train_test_split()` with `test_size=0.2` and `random_state=120`

---

## ✅ Evaluation
- Used `accuracy_score()` from `sklearn.metrics`
- Compared predicted vs actual labels
- Output is a single accuracy metric

---

## 💾 Model Saving
Saved trained logistic regression model using:

### Option 1: `joblib` (commented in code)
```python
import joblib
joblib.dump(lr, 'model.pkl')

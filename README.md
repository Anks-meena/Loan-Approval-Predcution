# Loan Approval Prediction 🏦💳

A machine learning project that predicts whether a loan application will be approved based on applicant details such as income, education, credit history, and loan amount. This model uses **Logistic Regression** for binary classification and achieves **78.86% accuracy**.

---

## 📌 Features
- Data preprocessing:
  - Missing value imputation (mode for categorical, median for numerical)
  - Label encoding and one-hot encoding for categorical variables
- Logistic Regression model for binary classification
- Train-test split for unbiased evaluation
- Accuracy calculation and prediction output
- Clean, modular, and well-documented Python code

---

## 🛠 Tech Stack
- **Python**  
- **Pandas** – Data handling  
- **Scikit-learn** – Machine learning model & preprocessing  
- **NumPy** – Numerical computations  
- **Matplotlib / Seaborn**

---

## 📂 Dataset
The dataset `loan.csv` contains details of loan applicants such as:
- ApplicantIncome & CoapplicantIncome
- LoanAmount & Loan_Amount_Term
- Credit_History
- Gender, Education, Marital Status
- Loan_Status (Target variable)

---

**##📊Model Performance**

- Algorithm: Logistic Regression
- Accuracy: 78.86%
- Possible Improvements:
  * Feature engineering (e.g., TotalIncome)
  * Hyperparameter tuning
  * Trying other models like Random Forest, XGBoost

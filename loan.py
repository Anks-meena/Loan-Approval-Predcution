"""
Loan Approval Prediction
Custom Implementation by [Your Name]
"""

# Import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import os
import pandas as pd

# Change working directory to script's folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Load dataset
data = pd.read_csv("loan.csv")

# Encode target variable (Y/N to 1/0)
label_encoder = LabelEncoder()
data['Loan_Status'] = label_encoder.fit_transform(data['Loan_Status'])

# Fill missing values instead of dropping
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']:
    data[col] = data[col].fillna(data[col].mode()[0])


for col in ['LoanAmount', 'Loan_Amount_Term']:
    data[col] = data[col].fillna(data[col].median())


# Train-test split
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Separate features and target
X_train = train.drop(columns=['Loan_ID', 'Loan_Status'])
y_train = train['Loan_Status']

X_test = test.drop(columns=['Loan_ID', 'Loan_Status'])
y_test = test['Loan_Status']










# One-hot encode categorical variables
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Align columns so train and test match
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Create and train model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Output results
print("Predicted Values on Test Data:\n", y_pred)
print("\nAccuracy Score on Test Data: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

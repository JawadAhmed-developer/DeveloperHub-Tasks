# ==============================
# Heart Disease Prediction Project
# ==============================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# 2. Load Dataset
df = pd.read_csv("heart.csv")

# 3. Data Cleaning
print("Missing Values:\n", df.isnull().sum())
df.fillna(df.mean(), inplace=True)
df.drop_duplicates(inplace=True)

# 4. Exploratory Data Analysis (EDA)

# Target Distribution
sns.countplot(x='target', data=df)
plt.title("Heart Disease Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Feature Analysis
sns.boxplot(x='target', y='age', data=df)
plt.show()

sns.countplot(x='cp', hue='target', data=df)
plt.show()

# 5. Feature & Target Split
X = df.drop("target", axis=1)
y = df["target"]

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# 8. Model Training
# ==============================

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# ==============================
# 9. Evaluation
# ==============================

# Accuracy
print("\n--- Accuracy ---")
print("Logistic Regression:", accuracy_score(y_test, y_pred_lr))
print("Decision Tree:", accuracy_score(y_test, y_pred_dt))

# Confusion Matrix (Logistic Regression)
cm = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve (Logistic Regression)
y_prob = lr.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label="AUC = " + str(roc_auc))
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

print("ROC-AUC Score:", roc_auc)

# ==============================
# 10. Feature Importance
# ==============================

# Logistic Regression Importance
importance_lr = pd.Series(lr.coef_[0], index=X.columns)
importance_lr.sort_values().plot(kind='barh')
plt.title("Feature Importance (Logistic Regression)")
plt.show()

# Decision Tree Importance
importance_dt = pd.Series(dt.feature_importances_, index=X.columns)
importance_dt.sort_values().plot(kind='barh')
plt.title("Feature Importance (Decision Tree)")
plt.show()

# ==============================
# 11. Conclusion
# ==============================

print("\nProject Completed Successfully!")
print("Model predicts heart disease using patient health data.")
print("Important features include cp, thalach, oldpeak, and chol.")
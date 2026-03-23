# DeveloperHub-Tasks

# 📊 Machine Learning Projects Collection

This repository contains multiple machine learning projects focused on data analysis, visualization, classification, and regression.

---

# 🔷 1. Iris Dataset Exploration & Visualization

## 📌 Objective

To explore and visualize the Iris dataset using different data analysis and visualization techniques.

## 🔧 Tasks Performed

* Loaded dataset using Pandas
* Explored dataset structure (shape, columns, info, describe)
* Created visualizations:

  * Scatter Plot (Sepal Length vs Sepal Width)
  * Histogram with KDE
  * Box Plot

## 📊 Key Insights

* Different species show clear separation in feature space
* Sepal measurements vary across species
* Data is clean and well-structured

---

# 🔷 2. House Price Prediction

## 📌 Objective

To predict house prices based on features such as size, bedrooms, and location.

## 🔧 Steps Performed

* Data preprocessing:

  * Handled missing values
  * Converted categorical variables using one-hot encoding
  * Removed duplicates
* Performed Exploratory Data Analysis (EDA)
* Applied feature scaling
* Trained **Linear Regression model**

## 📈 Evaluation Metrics

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

## 📊 Visualization

* Actual vs Predicted Prices scatter plot
* Feature importance graph

## 📊 Key Insights

* Features like size and location strongly influence price
* Model provides a reasonable estimation of house prices

---

# 🔷 3. Heart Disease Prediction

## 📌 Objective

To predict whether a person is at risk of heart disease based on medical data.

## 🔧 Steps Performed

* Data cleaning:

  * Handled missing values
  * Removed duplicates
* Performed EDA:

  * Target distribution
  * Correlation heatmap
  * Feature analysis
* Trained models:

  * Logistic Regression
  * Decision Tree

## 📈 Evaluation Metrics

* Accuracy
* Confusion Matrix
* ROC Curve & AUC Score

## 📊 Feature Importance

* Identified key features influencing prediction such as:

  * Chest pain (cp)
  * Maximum heart rate (thalach)
  * Oldpeak
  * Cholesterol (chol)

## 📊 Key Insights

* Logistic Regression performs well for classification
* Decision Tree provides interpretability
* Model effectively distinguishes between disease and non-disease cases

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

# 🚀 How to Run

1. Clone the repository
2. Install required libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Place datasets in the project directory:

   * `iris.csv`
   * `house.csv`
   * `heart.csv`
4. Run the Python file:

   ```bash
   python filename.py
   ```

---

# 📌 Conclusion

This repository demonstrates:

* Data exploration and visualization techniques
* Regression modeling for price prediction
* Classification modeling for medical diagnosis
* Model evaluation using standard metrics

---

# 👨‍💻 Author

**Muhammad Jawad Ahmed**

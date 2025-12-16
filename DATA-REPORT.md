# Data Report – Titanic Dataset

## 1. Dataset Overview

- **Dataset Name:** Titanic – Machine Learning from Disaster
- **Source:** Kaggle
- **Task Type:** Binary Classification
- **Target Variable:** `Survived`

The dataset contains passenger information such as age, gender, ticket class, fare, and embarkation port.  
The objective is to predict whether a passenger survived the Titanic disaster.

---

## 2. Data Loading

- Raw data was loaded using a Python data pipeline located at:
src/pipelines/data_pipeline.py

---

## 3. Data Cleaning

### 3.1 Missing Values Handling

| Column | Strategy | Reason |
|------|---------|-------|
| Age | Median | Robust to outliers |
| Fare | Median | Skewed distribution |
| Embarked | Mode | Categorical feature |
| Cabin | Dropped | High percentage of missing values |

All missing values were handled using training-data statistics to avoid data leakage.

---

### 3.2 Duplicate Removal

- Duplicate rows were identified and removed.
- This ensured data consistency and prevented biased learning.

---

### 3.3 Outlier Handling

- Outliers in numerical features (`Age`, `Fare`) were handled using the **IQR method**.
- Extreme values were capped instead of removed to avoid unnecessary data loss.

---

## 4. Processed Dataset

- The cleaned dataset was saved as:
src/data/processed/final.csv

---

## 5. Exploratory Data Analysis (EDA)

EDA was performed in:
src/notebooks/EDA.ipynb

### 5.1 Missing Values Heatmap
- Confirms no missing values remain after cleaning.

### 5.2 Correlation Matrix
- Shows relationships between numerical features.
- Identified moderate correlation between `Fare` and `Pclass`.

### 5.3 Feature Distributions
- Age and Fare distributions show right skewness.
- Confirms effectiveness of outlier capping.

### 5.4 Target Variable Distribution
- Target (`Survived`) shows class imbalance.
- Indicates the need for imbalance-aware metrics during modeling.

### 5.5 Key Insights
- Survival rate is higher for:
  - Female passengers
  - Higher passenger classes
- Fare and class have a noticeable impact on survival probability.

---
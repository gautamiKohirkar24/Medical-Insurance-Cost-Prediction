# Medical-Insurance-Cost-Prediction
# Medical Cost Price Prediction

This project focuses on predicting medical insurance costs using patient data such as age, gender, BMI, number of children, smoking status, and region. The goal is to build a regression model to estimate insurance charges and analyze the impact of various factors on the cost.

---

## Table of Contents
- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Dataset Overview](#dataset-overview)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Predictive System](#predictive-system)
- [Results](#results)

---

## Introduction

Health insurance costs can vary significantly based on multiple factors. This project leverages machine learning techniques, specifically Linear Regression, to predict medical insurance costs based on patient demographics and lifestyle indicators.

The project demonstrates:
- Data collection and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering and encoding
- Model training and evaluation
- Building a simple predictive system

---

## Technologies Used

The following libraries and tools were used in this project:

- Python 3.9+
- Numpy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## Dataset Overview

The dataset is sourced from a CSV file and contains the following columns:
- **age**: Age of the patient
- **sex**: Gender (male/female)
- **bmi**: Body Mass Index
- **children**: Number of children/dependents covered
- **smoker**: Smoking status (yes/no)
- **region**: Region in the US (southeast, southwest, northwest, northeast)
- **charges**: Medical insurance costs (target variable)

### Dataset Properties:
- **Shape**: 1338 rows and 7 columns
- **No missing values**: The dataset is clean and ready for analysis.
- **Categorical Columns**: `sex`, `smoker`, `region`

---

## Exploratory Data Analysis (EDA)

### Key Insights:
1. **Age Distribution**: Most patients are between 20–60 years old.
2. **BMI Analysis**: Observed an average BMI of ~30, with a few outliers.
3. **Smoking Impact**: Smokers incur significantly higher charges than non-smokers.
4. **Children Impact**: Having more children has a slight impact on insurance charges.
5. **Regional Analysis**: Charges vary across regions, with no significant outliers.

Plots and visualizations include:
- Age distribution
- BMI distribution
- Charges distribution
- Count plots for categorical features

---

## Data Preprocessing

Before feeding the data into the model:
1. **Encoding Categorical Variables**:
    - `sex` converted to 0 (male) and 1 (female)
    - `smoker` converted to 0 (yes) and 1 (no)
    - `region` converted to numerical values (0–3)
2. **Splitting Data**: 
    - Features (`X`) and target (`Y`) were separated.
    - Dataset split into 80% training and 20% testing subsets.

---

## Model Training

A **Linear Regression** model was trained using the `scikit-learn` library.

### Model Training Steps:
1. Load and preprocess the dataset.
2. Train the Linear Regression model on training data (`X_train`, `Y_train`).
3. Evaluate the model using R-squared scores on both training and testing datasets.

---

## Model Evaluation

### Results:
- **Training R² Score**: 0.7515
- **Testing R² Score**: 0.7447

The model performs well in predicting insurance charges with an acceptable level of accuracy.

---

## Predictive System

A predictive system was built to estimate insurance costs based on user inputs.

### Example Prediction:
Input Data: 
- `age`: 31
- `sex`: female
- `bmi`: 25.74
- `children`: 0
- `smoker`: no
- `region`: southeast

Predicted Insurance Cost: **$3760.08**

---

## Results

- Smoking status and BMI significantly influence medical insurance costs.
- The model provides an effective framework for predicting charges using linear regression.
- This project showcases how machine learning can assist in cost prediction and trend analysis.


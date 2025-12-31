# 🏯 House Prices – Advanced Regression Techniques

A beginner-friendly Machine Learning project using the Ames Housing Dataset.

---

## Project Overview

This project builds a **regression model** to predict house prices using real housing data.  
It follows a clean and professional Machine Learning workflow used in industry.

You will learn:

- How to structure ML projects
- How to clean and prepare datasets
- How to train and evaluate ML models
- How to save and reuse trained models
- How to make predictions

---

## 💿 Dataset Information

- Dataset: Ames Housing Dataset
- Kaggle Competition: House Prices – Advanced Regression Techniques
- Files used:
  - `train.csv`
  - `test.csv`

---

## 🛠 Libraries Used

| Purpose               | Library                |
| --------------------- | ---------------------- |
| Numerical Computing   | numpy                  |
| Data Handling         | pandas                 |
| Visualization         | matplotlib, seaborn    |
| Machine Learning      | scikit-learn, lightgbm |
| Model Saving          | joblib                 |
| Environment Variables | python-dotenv          |

Install all dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn lightgbm joblib python-dotenv
```

### What is a Virtual Environment?

A virtual environment (venv) is a private workspace for your project.
It keeps your libraries isolated so that different projects do not break each other.

📁 Project Structure

ames-housing-ml/
│

├ data/

- train.csv

- test.csv

├ src/

- data_preprocessing.py

- feature_engineering.py

- train_model.py

- predict.py

├ models/

- house_price_model.pkl

└─ README.md

## 🪜 Step-by-Step Workflow

### Step 1: Create Project Folder

mkdir ames-housing-ml
cd ames-housing-ml

### Step 2: Create Virtual Environment

python -m venv venv
source venv/bin/activate # On Linux/Mac
venv\Scripts\activate # On Windows

### Step 3: Install Libraries

pip install numpy pandas scikit-learn matplotlib seaborn lightgbm joblib python-dotenv

### Step 4: Download Dataset

Download train.csv and test.csv from Kaggle and place them inside the data/ folder.

### Step 5: Data Preprocessing

Handle missing values
Separate numerical and categorical columns
Prepare data for training

### Step 6: Feature Engineering

Convert categorical values into numeric values
Prepare clean features for the model

### Step 7: Train Model

python src/train_model.py
This will train the model and save it into the models/ folder.

### Step 8: Make Predictions

python src/predict.py
Predictions will be generated for test data.
📈 Evaluation Metric
We use Root Mean Squared Error (RMSE) to evaluate the model.
Lower RMSE means better predictions.

📦 **Final Output**

Trained ML Model saved in models/house_price_model.pkl
Prediction file for submission

🎯 **Goal**

Build a real world ML regression pipeline and understand how professional ML projects are structured.

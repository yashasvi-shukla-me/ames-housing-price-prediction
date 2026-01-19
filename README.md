# 🏯 Ames House Price Prediction - ML Web Application

An end to end machine learning application that predicts house prices using the Ames Housing dataset.  
The project goes beyond a Kaggle notebook and demonstrates **production-ready ML inference**, exposed via a FastAPI backend and consumed by a browser-based frontend.

🔗 **Live Demo:** https://ames-house-price-predict.netlify.app  
🔗 **Backend API:** https://ames-house-price-api.onrender.com  
🔗 **API Docs (Swagger):** https://ames-house-price-api.onrender.com/docs

---

## 🚀 Project Overview

Most Kaggle projects stop at model training.  
This project converts a trained ML model into a **real-world application** with:

- Robust preprocessing at inference time
- A public REST API
- A live frontend UI
- Safe handling of partial user input
- Stable and reproducible predictions

Users can change house attributes in the browser and instantly see updated price predictions.

---

## 🧠 Key Features

- End-to-end ML pipeline (training → inference → UI)
- FastAPI backend with schema validation
- Robust preprocessing:
  - Persistent median imputer (no data leakage)
  - Safe handling of missing numeric & categorical features
  - Stable feature ordering between training and inference
- One-Hot Encoding with unseen category handling
- Browser-based frontend (HTML + JavaScript)
- Fully deployed (Netlify + Render)

---

## 🏗️ System Architecture

Frontend (Browser)

↓

FastAPI Backend (/predict)

↓

Inference Pipeline (Imputer → Encoder → Model)

↓

Predicted House Price

---

## 🛠️ Tech Stack

### Machine Learning

- Python
- Scikit-learn
- Pandas, NumPy

### Backend

- FastAPI
- Uvicorn
- Pydantic (request validation)

### Frontend

- HTML
- CSS
- JavaScript (Fetch API)

### Deployment

- Backend: Render
- Frontend: Netlify

---

## 📊 Model Details

- Dataset: Ames Housing Dataset
- Model: Gradient Boosting Regressor
- Target transformation: `log1p(SalePrice)`
- Inference output converted back using `expm1`

### Important ML Engineering Decisions

- Saved and reused preprocessing artifacts (imputer, encoder)
- Prevented training–inference skew
- Expanded partial user input to full training schema
- Enforced exact feature order during inference

These steps ensure **correct and stable predictions** in production.

---

### ▶️ Running Locally

1. Install dependencies
   pip install -r requirements.txt
2. Train the model
   python -m src.train
3. Start the backend
   uvicorn src.app:app --reload
4. Open the frontend
   Open frontend/index.html in a browser.

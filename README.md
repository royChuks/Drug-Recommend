# 🩺 Clinical Drug Recommendation System

A modern, machine learning-powered clinical decision support system that recommends drugs based on patient symptoms and profiles. The system employs multiple classification algorithms and provides deep analytics into model performance.

## 🚀 Features

- **Multi-Algorithm Support**: Compare recommendations and performance across:
  - Logistic Regression (LR)
  - Naive Bayes (NB)
  - Support Vector Machine (SVM)
  - Random Forest (RF)
  - XGBoost (XGB)
- **Age-Based Confidence Scaling**: Automatically adjusts recommendation confidence based on patient age groups (Pediatrics, Adult, Geriatric).
- **Comprehensive Analytics**:
  - Algorithm comparison charts.
  - Detailed model metrics (Precision, Recall, F1-Score, Accuracy).
  - Confusion Matrix analysis.
  - Feature correlation insights.
- **Modern UI**: A responsive, glassmorphism-inspired interface built with Materialize CSS.

## 🛠️ Technology Stack

- **Backend**: Python, FastAPI, Scikit-learn, Pandas, XGBoost.
- **Frontend**: HTML5, Vanilla JavaScript, Materialize CSS, Chart.js.
- **Deployment**: Uvicorn (ASGI server).

## 📁 Project Structure

```text
├── backend/
│   ├── main.py               # FastAPI application & API endpoints
│   ├── model.py              # ML logic, training, and prediction
│   ├── disease_drug_dataset.csv # Training data
│   ├── requirements.txt      # Python dependencies
│   └── model_cache/          # Cached trained models
├── frontend/
│   └── index.html            # Main UI (Single Page Application)
└── README.md
```

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.8+
- Modern Web Browser

### 2. Backend Setup
Navigate to the `backend` directory and install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

### 3. Run the API
Start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload
```
The backend will be available at `http://localhost:8000`.

### 4. Frontend Setup
Simply open `frontend/index.html` in your browser. 
> **Note**: Ensure the backend is running first so the frontend can fetch recommendations and metrics.

## 📊 API Endpoints

- `POST /predict`: Get drug recommendations for a specific patient profile.
- `GET /metrics/{algo}`: Retrieve performance metrics for a specific algorithm.
- `POST /compare-algorithms`: Compare results across all supported algorithms.
- `POST /analytics/charts`: Get formatted data for visualization charts.

## 💡 How it Works

1. **Input**: User enters a disease/symptom, patient age, gender, and weight.
2. **Processing**: The backend vectorizes the input and runs it through the selected ML model.
3. **Adjustment**: Confidence scores are adjusted based on the patient's age group relative to the training data distribution.
4. **Visualization**: The UI displays the results and provides options to view detailed model analytics.

---
*Disclaimer: This system is for educational/informational purposes and should not replace professional medical advice.*

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from model import predict_drugs, get_model_metrics, compare_algorithms_for_disease, get_algorithm_comparison_chart_data, load_data, get_model
import os
import traceback

app = FastAPI(title="Drug Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Preload data and models on startup"""
    try:
        print("Preloading data...")
        load_data()
        print("Data loaded successfully!")
        # Preload one model to verify everything works
        get_model('lr')
        print("Models ready!")
    except Exception as e:
        print(f"Error during startup: {e}")
        traceback.print_exc()

@app.get("/status")
def get_status():
    """Check if data and models are loaded"""
    from model import df, grouped, models_cache
    return {
        "data_loaded": df is not None,
        "num_diseases": len(grouped) if grouped is not None else 0,
        "models_loaded": list(models_cache.keys()),
        "dataset_file_exists": os.path.exists("disease_drug_dataset.csv"),
        "model_cache_exists": os.path.exists("model_cache")
    }

class PatientProfile(BaseModel):
    disease: str
    age: int
    gender: str
    weight: float = None
    medical_history: str = None
    algo: str = 'lr'  # Algorithm: lr, nb, svm, rf, xgb

@app.post("/predict")
def predict(input: PatientProfile):
    recommendations = predict_drugs(input.disease, algo=input.algo, age=input.age)
    print("Recommendations:", recommendations)
    return {
        "disease": input.disease,
        "recommendations": recommendations,
        "age": input.age,
        "gender": input.gender,
        "weight": input.weight,
        "medical_history": input.medical_history,
        "algo": input.algo
    }

@app.get("/metrics/{disease}/{algo}")
def metrics(disease: str, algo: str):
    return get_model_metrics(algo=algo, disease=disease)

@app.post("/compare-algorithms")
def compare_algorithms(input: PatientProfile):
    """Compare all algorithms for a given disease/patient profile"""
    comparison = compare_algorithms_for_disease(input.disease, age=input.age)
    return comparison

@app.post("/analytics/charts")
def get_analytics_charts(input: PatientProfile):
    """Get chart data for algorithm comparison analytics - specific to patient's disease"""
    chart_data = get_algorithm_comparison_chart_data(input.disease, age=input.age)
    return chart_data

@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML file"""
    frontend_paths = ["../frontend/index.html", "static/index.html", "index.html"]
    for path in frontend_paths:
        if os.path.exists(path):
            return FileResponse(path)
    return {"error": "Frontend not found", "paths_checked": frontend_paths}

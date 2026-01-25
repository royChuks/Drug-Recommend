from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import predict_drugs, get_model_metrics, compare_algorithms_for_disease, get_algorithm_comparison_chart_data

app = FastAPI(title="Drug Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/metrics/{algo}")
def metrics(algo: str):
    return get_model_metrics(algo=algo)

@app.post("/compare-algorithms")
def compare_algorithms(input: PatientProfile):
    """Compare all algorithms for a given disease/patient profile"""
    comparison = compare_algorithms_for_disease(input.disease, age=input.age)
    return comparison

@app.post("/analytics/charts")
def get_analytics_charts(input: PatientProfile):
    """Get chart data for algorithm comparison analytics"""
    chart_data = get_algorithm_comparison_chart_data(input.disease, age=input.age)
    return chart_data

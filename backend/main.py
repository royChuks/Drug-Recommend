import logging
import sys
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exception_handlers import http_exception_handler
from starlette.exceptions import HTTPException
from pydantic import BaseModel
from model import predict_drugs, get_model_metrics, compare_algorithms_for_disease, get_algorithm_comparison_chart_data, load_data, get_model
import os
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Drug Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom 405 handler to return JSON instead of HTML
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == 405:
        logger.warning(f"405 Method Not Allowed: {request.method} {request.url.path}")
        return JSONResponse(
            status_code=405,
            content={
                "error": "Method Not Allowed",
                "detail": f"Endpoint '{request.url.path}' does not support {request.method}.",
                "available_methods": "Check the API documentation."
            },
        )
    # For 404, return JSON as well
    if exc.status_code == 404:
        return JSONResponse(
            status_code=404,
            content={
                "error": "Not Found",
                "detail": f"Endpoint '{request.url.path}' does not exist.",
                "available_endpoints": [
                    "GET  /",
                    "GET  /status",
                    "POST /predict",
                    "GET  /metrics/{disease}/{algo}",
                    "POST /compare-algorithms",
                    "POST /analytics/charts",
                ]
            },
        )
    # For all other HTTP exceptions, use default handler
    return await http_exception_handler(request, exc)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"→ {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"← {request.method} {request.url.path} => {response.status_code}")
    return response

@app.on_event("startup")
async def startup_event():
    """Preload data and models on startup"""
    try:
        logger.info("Preloading data...")
        load_data()
        logger.info("Data loaded successfully!")
        # Preload one model to verify everything works
        get_model('lr')
        logger.info("Models ready!")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        traceback.print_exc()

@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML file"""
    # Single source of truth: backend/static/index.html
    static_path = os.path.join("static", "index.html")
    if os.path.exists(static_path):
        return FileResponse(static_path, media_type="text/html")
    # Fallback for backward compatibility
    fallback_path = os.path.join("..", "frontend", "index.html")
    if os.path.exists(fallback_path):
        return FileResponse(fallback_path, media_type="text/html")
    return JSONResponse(
        status_code=404,
        content={"error": "Frontend not found", "paths_checked": [static_path, fallback_path]}
    )

@app.get("/favicon.ico")
async def favicon():
    """Prevent 404 errors for browser favicon requests"""
    return JSONResponse(content={"status": "no favicon"})

@app.get("/status")
def get_status():
    """Check if data and models are loaded"""
    try:
        from model import df, grouped, models_cache
        return {
            "data_loaded": df is not None,
            "num_diseases": len(grouped) if grouped is not None else 0,
            "models_loaded": list(models_cache.keys()),
            "dataset_file_exists": os.path.exists("disease_drug_dataset.csv"),
            "model_cache_exists": os.path.exists("model_cache")
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

class PatientProfile(BaseModel):
    disease: str
    age: int
    gender: str
    weight: float = None
    medical_history: str = None
    algo: str = 'lr'  # Algorithm: lr, nb, svm, rf, xgb

@app.post("/predict")
def predict(input: PatientProfile):
    """Get drug recommendations for a patient profile"""
    try:
        logger.info(f"Predict request: disease={input.disease}, age={input.age}, algo={input.algo}")
        recommendations = predict_drugs(input.disease, algo=input.algo, age=input.age)
        logger.info(f"Predict success: {len(recommendations)} recommendations")
        return {
            "disease": input.disease,
            "recommendations": recommendations,
            "age": input.age,
            "gender": input.gender,
            "weight": input.weight,
            "medical_history": input.medical_history,
            "algo": input.algo
        }
    except Exception as e:
        logger.error(f"Predict failed: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Prediction failed: {str(e)}"}
        )

@app.get("/metrics/{disease}/{algo}")
def metrics(disease: str, algo: str):
    """Get model evaluation metrics for a specific disease and algorithm"""
    try:
        logger.info(f"Metrics request: disease={disease}, algo={algo}")
        result = get_model_metrics(algo=algo, disease=disease)
        return result
    except Exception as e:
        logger.error(f"Metrics failed: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Metrics computation failed: {str(e)}"}
        )

@app.post("/compare-algorithms")
def compare_algorithms(input: PatientProfile):
    """Compare all algorithms for a given disease/patient profile"""
    try:
        logger.info(f"Compare algorithms request: disease={input.disease}, age={input.age}")
        comparison = compare_algorithms_for_disease(input.disease, age=input.age)
        return comparison
    except Exception as e:
        logger.error(f"Compare algorithms failed: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Algorithm comparison failed: {str(e)}"}
        )

@app.post("/analytics/charts")
def get_analytics_charts(input: PatientProfile):
    """Get chart data for algorithm comparison analytics - specific to patient's disease"""
    try:
        logger.info(f"Analytics charts request: disease={input.disease}, age={input.age}")
        chart_data = get_algorithm_comparison_chart_data(input.disease, age=input.age)
        return chart_data
    except Exception as e:
        logger.error(f"Analytics charts failed: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Chart data generation failed: {str(e)}"}
        )

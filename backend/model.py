import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    XGBClassifier = None
    HAS_XGB = False
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss, multilabel_confusion_matrix, accuracy_score, confusion_matrix
import os
import joblib
import threading
from functools import lru_cache
import time

# --- Lazy Loading Setup ---
MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_cache")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Global variables for lazy loading
df = None
grouped = None
vectorizer = None
mlb = None
drug_age_map = None
X = None
y = None
drug_classes = None

def load_data():
    """Lazy load data and preprocessing"""
    global df, grouped, vectorizer, mlb, drug_age_map, X, y, drug_classes

    if df is not None:
        return  # Already loaded

    print("Loading data and preprocessing...")
    csv_path = os.path.join(os.path.dirname(__file__), "disease_drug_dataset.csv")
    df = pd.read_csv(csv_path)
    grouped = df.groupby("disease_name")['drug_generic_name'].apply(list).reset_index()
    grouped["disease_name"] = grouped["disease_name"].str.lower().str.strip()
    X_text = grouped["disease_name"]
    y_labels = grouped["drug_generic_name"]

    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    X = vectorizer.fit_transform(X_text)
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y_labels)

    # Age group mapping
    drug_age_map = dict(zip(df['drug_generic_name'], df.get('age_group', 'adult')))
    drug_classes = mlb.classes_

    print("Data loaded successfully!")

def get_patient_age_group(age: int) -> str:
    """Determine patient age group from age - optimized"""
    if age <= 15:
        return 'pediatrics'
    elif age >= 50:
        return 'geriatric'
    return 'adult'

# --- Lazy Model Loading ---
models_cache = {}
model_lock = threading.Lock()

def get_model(algo: str):
    """Lazy load and cache individual models"""
    load_data()  # Ensure data is loaded

    if algo in models_cache:
        return models_cache[algo]

    with model_lock:
        model_path = os.path.join(MODEL_CACHE_DIR, f"{algo}_model.pkl")

        if os.path.exists(model_path):
            # Load from cache
            try:
                model = joblib.load(model_path)
                models_cache[algo] = model
                return model
            except Exception as e:
                print(f"Error loading cached model {algo}: {e}")

        # Create and train model
        print(f"Training {algo} model...")
        if algo == "lr":
            model = OneVsRestClassifier(LogisticRegression(max_iter=300, solver="liblinear"))
        elif algo == "nb":
            model = OneVsRestClassifier(MultinomialNB())
        elif algo == "svm":
            model = OneVsRestClassifier(LinearSVC(max_iter=300, dual=False))
        elif algo == "rf":
            model = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        elif algo == "xgb" and HAS_XGB:
            model = OneVsRestClassifier(XGBClassifier(eval_metric="logloss", random_state=42))
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

        model.fit(X, y)

        # Cache the model
        try:
            joblib.dump(model, model_path)
            models_cache[algo] = model
            print(f"{algo} model trained and cached successfully!")
        except Exception as e:
            print(f"Error caching model {algo}: {e}")

        return model

def apply_age_adjustment(prob: float, drug_age_group: str, patient_age_group: str) -> tuple:
    """Optimized age adjustment logic"""
    if drug_age_group == patient_age_group:
        return min(1.0, prob * 1.2), "perfect_match"
    elif (patient_age_group == 'adult' and drug_age_group in ['pediatrics', 'geriatric']) or \
         (patient_age_group == 'pediatrics' and drug_age_group == 'geriatric') or \
         (patient_age_group == 'geriatric' and drug_age_group == 'pediatrics'):
        return max(0.0, prob * 0.7), "mismatch"
    else:
        return prob * 0.9, "partial_match"

def predict_drugs(disease: str, threshold=0.02, algo='lr', top_k_fallback=3, age: int = None):
    """
    Optimized drug prediction with age-based confidence adjustment.
    """
    disease = disease.lower().strip()
    model = get_model(algo)
    vec = vectorizer.transform([disease])
    
    # Fast probability prediction
    try:
        probs = model.predict_proba(vec)[0]
    except Exception:
        preds = model.predict(vec)[0]
        probs = preds if hasattr(preds, '__iter__') else [preds]
    
    # Convert to numpy for faster operations
    probs_array = np.array(probs, dtype=np.float32)
    
    # Get patient age group
    patient_age_group = get_patient_age_group(age) if age is not None else None
    
    # Vectorized age adjustment (much faster)
    if patient_age_group:
        adjusted_probs = probs_array.copy()
        age_matches = []
        
        for i, drug in enumerate(drug_classes):
            drug_age_group = drug_age_map.get(drug, 'adult')
            adjusted_prob, match_type = apply_age_adjustment(probs_array[i], drug_age_group, patient_age_group)
            adjusted_probs[i] = adjusted_prob
            age_matches.append((match_type, drug_age_group))
    else:
        adjusted_probs = probs_array
        age_matches = [None] * len(drug_classes)
    
    # Fast threshold filtering using numpy
    above_threshold = adjusted_probs >= threshold
    
    if np.any(above_threshold):
        # Get indices where threshold is met
        indices = np.where(above_threshold)[0]
        results = []
        
        for idx in indices:
            result = {
                "drug": drug_classes[idx],
                "confidence": float(adjusted_probs[idx]),
                "base_confidence": float(probs_array[idx]),
                "age_group": age_matches[idx][1] if age_matches[idx] else drug_age_map.get(drug_classes[idx], 'adult')
            }
            if age_matches[idx]:
                result["age_match"] = age_matches[idx][0]
            results.append(result)
    else:
        # Fallback: get top_k using numpy argsort (faster than sorted)
        top_indices = np.argsort(adjusted_probs)[::-1][:top_k_fallback]
        results = []
        
        for idx in top_indices:
            result = {
                "drug": drug_classes[idx],
                "confidence": float(adjusted_probs[idx]),
                "base_confidence": float(probs_array[idx]),
                "age_group": drug_age_map.get(drug_classes[idx], 'adult')
            }
            if age_matches[idx]:
                result["age_match"] = age_matches[idx][0]
            results.append(result)
    
    # Sort by confidence (already sorted if fallback, but ensure it)
    return sorted(results, key=lambda x: x["confidence"], reverse=True)

def get_disease_specific_indices(disease: str):
    """Get training data indices for a specific disease"""
    load_data()
    disease_lower = disease.lower().strip()
    disease_mask = df['disease_name'].str.lower().str.strip() == disease_lower
    indices = np.where(disease_mask.values)[0]
    
    # Map grouped data indices to original X,y indices
    grouped_disease_idx = grouped[grouped['disease_name'] == disease_lower].index
    return grouped_disease_idx.tolist() if len(grouped_disease_idx) > 0 else None

def get_model_metrics(algo='lr', disease: str = None):
    """Get model evaluation metrics. If disease specified, metrics are for that disease only."""
    model = get_model(algo)
    
    # Use disease-specific data if provided
    if disease:
        disease_idx = get_disease_specific_indices(disease)
        if disease_idx is None or len(disease_idx) == 0:
            return {"error": f"Disease '{disease}' not found in training data", "algorithm": algo}
        
        # Get disease-specific training data
        X_disease = X[disease_idx]
        y_disease = y[disease_idx]
        scope = f"disease: {disease}"
    else:
        X_disease = X
        y_disease = y
        scope = "all diseases"
    
    try:
        y_pred = model.predict(X_disease)
    except Exception as e:
        print("Metric prediction error", e)
        return {"error": str(e)}
    
    precision = precision_score(y_disease, y_pred, average='micro', zero_division=0)
    recall = recall_score(y_disease, y_pred, average='micro', zero_division=0)
    f1 = f1_score(y_disease, y_pred, average='micro', zero_division=0)
    hamming = hamming_loss(y_disease, y_pred)
    conf_matrix = multilabel_confusion_matrix(y_disease, y_pred).tolist()
    
    # Confusion matrix analysis
    conf_matrix_array = multilabel_confusion_matrix(y_disease, y_pred)
    conf_matrices_detailed = []

    # Per-label analysis from confusion matrices with comprehensive metrics
    for i, cm in enumerate(conf_matrix_array):
        tn, fp, fn, tp = cm.ravel()
        label_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        label_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        label_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        label_f1 = 2 * (label_precision * label_recall) / (label_precision + label_recall) if (label_precision + label_recall) > 0 else 0
        
        # Additional metrics
        label_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Diagnostic odds ratio (DOR) - higher is better
        dor = (tp * tn) / (fp * fn) if (fp * fn) > 0 else 0
        dor = 0.0 if np.isinf(dor) else dor

        conf_matrices_detailed.append({
            "label": mlb.classes_[i],
            "matrix": cm.tolist(),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "precision": float(label_precision),
            "recall": float(label_recall),
            "f1_score": float(label_f1),
            "specificity": float(label_specificity),
            "accuracy": float(label_accuracy),
            "false_positive_rate": float(false_positive_rate),
            "false_negative_rate": float(false_negative_rate),
            "diagnostic_odds_ratio": float(dor),
            "support": int(tp + fn)  # Total actual positives for this label
        })

    # Accuracy calculation for multilabel
    accuracy = np.mean(np.any(y_disease == y_pred, axis=1))

    # Correlation analysis - Enhanced
    correlations = {}

    # Feature-target correlation (how well the text features predict each drug)
    try:
        y_pred_proba = model.predict_proba(X_disease)
        feature_importance = {}
        X_dense = X_disease.toarray()
        
        for i, drug in enumerate(mlb.classes_):
            # Calculate correlation between feature vector and drug predictions
            drug_predictions = y_pred_proba[:, i]
            
            # Correlation with individual features
            feature_correlations = []
            feature_correlation_details = []
            
            for j in range(X_dense.shape[1]):
                try:
                    corr = np.corrcoef(X_dense[:, j], drug_predictions)[0, 1]
                    # Only include valid correlations (not NaN or Inf)
                    if not np.isnan(corr) and not np.isinf(corr):
                        feature_correlations.append(float(corr))
                        feature_correlation_details.append({
                            "feature_index": int(j),
                            "correlation": float(corr),
                            "abs_correlation": float(abs(corr))
                        })
                except:
                    pass
            
            # Sort by absolute correlation value
            feature_correlation_details.sort(key=lambda x: x["abs_correlation"], reverse=True)
            top_features = feature_correlation_details[:5] if len(feature_correlation_details) >= 5 else feature_correlation_details

            # Calculate safe statistics (handle empty list)
            mean_corr = float(np.mean(feature_correlations)) if feature_correlations else 0.0
            max_corr = float(np.max(feature_correlations)) if feature_correlations else 0.0
            min_corr = float(np.min(feature_correlations)) if feature_correlations else 0.0
            std_corr = float(np.std(feature_correlations)) if feature_correlations else 0.0
            
            # Ensure values are JSON-safe
            feature_importance[drug] = {
                "mean_correlation": 0.0 if np.isnan(mean_corr) or np.isinf(mean_corr) else mean_corr,
                "max_correlation": 0.0 if np.isnan(max_corr) or np.isinf(max_corr) else max_corr,
                "min_correlation": 0.0 if np.isnan(min_corr) or np.isinf(min_corr) else min_corr,
                "correlation_std": 0.0 if np.isnan(std_corr) or np.isinf(std_corr) else std_corr,
                "num_features_analyzed": len(feature_correlations),
                "top_correlated_features": top_features,
                "correlation_strength": "Strong" if mean_corr > 0.5 else ("Moderate" if mean_corr > 0.3 else "Weak")
            }
    except Exception as e:
        # Fallback if predict_proba is not available
        print(f"Correlation analysis error: {e}")
        feature_importance = {drug: {
            "mean_correlation": 0.0,
            "max_correlation": 0.0,
            "min_correlation": 0.0,
            "correlation_std": 0.0,
            "num_features_analyzed": 0,
            "top_correlated_features": [],
            "correlation_strength": "Weak"
        } for drug in mlb.classes_}

    # Drug-drug correlation and similarity analysis
    try:
        drug_cooccurrence = np.dot(y_disease.T, y_disease) / len(y_disease)
        # Replace any NaN or Inf values in cooccurrence
        drug_cooccurrence = np.nan_to_num(drug_cooccurrence, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize cooccurrence to similarity scores (0-1)
        drug_cooccurrence_normalized = drug_cooccurrence / (np.max(drug_cooccurrence) + 1e-10)
        
        # Calculate correlation matrix safely
        try:
            drug_correlation_matrix = np.corrcoef(drug_cooccurrence)
            # Replace NaN/Inf in correlation matrix
            drug_correlation_matrix = np.nan_to_num(drug_correlation_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        except:
            # If corrcoef fails, use identity matrix as fallback
            drug_correlation_matrix = np.eye(len(drug_cooccurrence))
        
        # Find most similar drug pairs
        drug_similarity_pairs = []
        for i in range(len(mlb.classes_)):
            for j in range(i+1, len(mlb.classes_)):
                similarity = float(drug_cooccurrence_normalized[i, j])
                if similarity > 0.1:  # Only include significant similarities
                    drug_similarity_pairs.append({
                        "drug_1": mlb.classes_[i],
                        "drug_2": mlb.classes_[j],
                        "similarity": similarity,
                        "cooccurrence_rate": float(drug_cooccurrence[i, j])
                    })
        
        # Sort by similarity
        drug_similarity_pairs.sort(key=lambda x: x["similarity"], reverse=True)
        top_similar_pairs = drug_similarity_pairs[:10]
        
    except Exception as e:
        print(f"Drug cooccurrence error: {e}")
        drug_cooccurrence = np.zeros((len(mlb.classes_), len(mlb.classes_)))
        drug_correlation_matrix = np.eye(len(mlb.classes_))
        top_similar_pairs = []

    correlations["feature_importance"] = feature_importance
    correlations["drug_cooccurrence"] = drug_cooccurrence.tolist()
    correlations["drug_correlation_matrix"] = drug_correlation_matrix.tolist()
    correlations["top_similar_drug_pairs"] = top_similar_pairs
    correlations["total_features_analyzed"] = X_dense.shape[1]

    # Model characteristics
    supervised_algorithms = ["lr", "nb", "svm", "rf", "xgb"]
    model_characteristics = {
        "is_supervised": algo in supervised_algorithms,
        "algorithm_type": "Supervised Learning" if algo in supervised_algorithms else "Unsupervised Learning",
        "learning_paradigm": {
            "lr": "Linear Classification",
            "nb": "Probabilistic Classification",
            "svm": "Maximum Margin Classification",
            "rf": "Ensemble Decision Trees",
            "xgb": "Gradient Boosting"
        }.get(algo, "Unknown"),
        "predictability_assessment": {
            "high_precision": precision > 0.8,
            "high_recall": recall > 0.8,
            "balanced_performance": abs(precision - recall) < 0.1,
            "overall_accuracy": accuracy
        }
    }

    # Confusion matrix interpretability - Enhanced analysis
    avg_precision = float(np.mean([cm["precision"] for cm in conf_matrices_detailed])) if conf_matrices_detailed else 0.0
    avg_recall = float(np.mean([cm["recall"] for cm in conf_matrices_detailed])) if conf_matrices_detailed else 0.0
    avg_specificity = float(np.mean([cm["specificity"] for cm in conf_matrices_detailed])) if conf_matrices_detailed else 0.0
    avg_f1 = float(np.mean([cm["f1_score"] for cm in conf_matrices_detailed])) if conf_matrices_detailed else 0.0
    
    # Ensure values are JSON-safe
    avg_precision = 0.0 if np.isnan(avg_precision) or np.isinf(avg_precision) else avg_precision
    avg_recall = 0.0 if np.isnan(avg_recall) or np.isinf(avg_recall) else avg_recall
    avg_specificity = 0.0 if np.isnan(avg_specificity) or np.isinf(avg_specificity) else avg_specificity
    avg_f1 = 0.0 if np.isnan(avg_f1) or np.isinf(avg_f1) else avg_f1
    
    # Calculate overall diagnostic metrics
    total_tp = int(np.sum(conf_matrix_array[:, 1, 1]))
    total_fp = int(np.sum(conf_matrix_array[:, 0, 1]))
    total_fn = int(np.sum(conf_matrix_array[:, 1, 0]))
    total_tn = int(np.sum(conf_matrix_array[:, 0, 0]))
    
    # Overall diagnostic metrics
    overall_sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0
    
    # Class balance analysis
    label_supports = [cm["support"] for cm in conf_matrices_detailed]
    class_imbalance_ratio = max(label_supports) / (min(label_supports) + 1e-10) if label_supports else 0
    
    confusion_summary = {
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn,
        "total_true_negatives": total_tn,
        "average_precision_per_label": float(avg_precision),
        "average_recall_per_label": float(avg_recall),
        "average_specificity_per_label": float(avg_specificity),
        "average_f1_per_label": float(avg_f1),
        "overall_sensitivity": float(overall_sensitivity),
        "overall_specificity": float(overall_specificity),
        "class_imbalance_ratio": float(class_imbalance_ratio),
        "num_labels_analyzed": len(conf_matrices_detailed),
        "model_diagnostic_assessment": {
            "sensitivity_level": "High" if overall_sensitivity > 0.8 else ("Medium" if overall_sensitivity > 0.6 else "Low"),
            "specificity_level": "High" if overall_specificity > 0.8 else ("Medium" if overall_specificity > 0.6 else "Low"),
            "balance_assessment": "Balanced" if class_imbalance_ratio < 2.0 else ("Moderately Imbalanced" if class_imbalance_ratio < 5.0 else "Highly Imbalanced"),
            "false_positive_concern": "Low" if total_fp < total_fn else "High",
            "false_negative_concern": "Low" if total_fn < total_fp else "High"
        }
    }

    return {
        "algorithm": algo,
        "data_scope": scope,
        "model_characteristics": model_characteristics,
        "basic_metrics": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "hamming_loss": hamming
        },
        "confusion_matrix_analysis": {
            "summary": confusion_summary,
            "per_label_details": conf_matrices_detailed,
            "legacy_confusion_matrices": conf_matrix  # Keep for backward compatibility
        },
        "correlation_analysis": correlations,
        "predictability_insights": {
            "model_reliability": "High" if f1 > 0.8 else ("Medium" if f1 > 0.6 else "Low"),
            "prediction_confidence": "High" if accuracy > 0.8 else ("Medium" if accuracy > 0.6 else "Low"),
            "correlation_strength": "Strong" if len(feature_importance) > 0 and np.mean([fi["mean_correlation"] for fi in feature_importance.values()]) > 0.5 else "Moderate"
        },
        "labels": mlb.classes_.tolist()  # Keep for backward compatibility
    }

def compare_algorithms_for_disease(disease: str, age: int = None):
    """
    Compare all algorithms for a specific disease.
    Returns performance metrics and recommendations for each algorithm.
    """
    disease = disease.lower().strip()

    comparison = {
        "disease": disease,
        "age": age,
        "algorithms": {},
        "metrics": {},
        "timing": {}
    }

    patient_age_group = get_patient_age_group(age) if age is not None else None

    # List of available algorithms
    available_algos = ["lr", "nb", "svm", "rf"]
    if HAS_XGB:
        available_algos.append("xgb")

    for algo_name in available_algos:
        start_time = time.time()

        model = get_model(algo_name)
        vec = vectorizer.transform([disease])

        try:
            probs = model.predict_proba(vec)[0]
        except Exception:
            preds = model.predict(vec)[0]
            probs = preds if hasattr(preds, '__iter__') else [preds]

        probs_array = np.array(probs, dtype=np.float32)

        # Get top 5 recommendations
        top_indices = np.argsort(probs_array)[::-1][:5]
        recommendations = []

        for idx in top_indices:
            drug = drug_classes[idx]
            base_conf = float(probs_array[idx])
            drug_age_group = drug_age_map.get(drug, 'adult')

            if patient_age_group:
                adj_conf, match_type = apply_age_adjustment(base_conf, drug_age_group, patient_age_group)
            else:
                adj_conf, match_type = base_conf, None

            recommendations.append({
                "drug": drug,
                "confidence": adj_conf,
                "base_confidence": base_conf,
                "age_group": drug_age_group,
                "age_match": match_type
            })

        # Get disease-specific model metrics
        disease_idx = get_disease_specific_indices(disease)
        if disease_idx and len(disease_idx) > 0:
            X_disease = X[disease_idx]
            y_disease = y[disease_idx]
            y_pred = model.predict(X_disease)
            precision = precision_score(y_disease, y_pred, average='micro', zero_division=0)
            recall = recall_score(y_disease, y_pred, average='micro', zero_division=0)
            f1 = f1_score(y_disease, y_pred, average='micro', zero_division=0)
        else:
            # Fallback to overall metrics if disease not found
            y_pred = model.predict(X)
            precision = precision_score(y, y_pred, average='micro', zero_division=0)
            recall = recall_score(y, y_pred, average='micro', zero_division=0)
            f1 = f1_score(y, y_pred, average='micro', zero_division=0)

        elapsed = time.time() - start_time

        comparison["algorithms"][algo_name] = {
            "recommendations": recommendations,
            "top_confidence": recommendations[0]["confidence"] if recommendations else 0
        }

        comparison["metrics"][algo_name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        comparison["timing"][algo_name] = round(elapsed * 1000, 2)  # ms

    return comparison

def get_algorithm_comparison_chart_data(disease: str, age: int = None):
    """
    Get formatted data for chart visualization comparing algorithms.
    """
    comparison = compare_algorithms_for_disease(disease, age)
    
    # Format for chart.js or similar
    algorithms = list(comparison["algorithms"].keys())
    
    # Chart data for metrics comparison
    metrics_chart = {
        "labels": algorithms,
        "datasets": [
            {
                "label": "Precision",
                "data": [comparison["metrics"][a]["precision"] * 100 for a in algorithms],
                "backgroundColor": "rgba(46, 139, 87, 0.6)"
            },
            {
                "label": "Recall",
                "data": [comparison["metrics"][a]["recall"] * 100 for a in algorithms],
                "backgroundColor": "rgba(70, 130, 180, 0.6)"
            },
            {
                "label": "F1 Score",
                "data": [comparison["metrics"][a]["f1_score"] * 100 for a in algorithms],
                "backgroundColor": "rgba(255, 165, 0, 0.6)"
            }
        ]
    }
    
    # Timing chart
    timing_chart = {
        "labels": algorithms,
        "datasets": [{
            "label": "Prediction Time (ms)",
            "data": [comparison["timing"][a] for a in algorithms],
            "backgroundColor": "rgba(220, 20, 60, 0.6)"
        }]
    }
    
    # Top confidence comparison
    confidence_chart = {
        "labels": algorithms,
        "datasets": [{
            "label": "Top Confidence Score",
            "data": [comparison["algorithms"][a]["top_confidence"] * 100 for a in algorithms],
            "backgroundColor": "rgba(138, 43, 226, 0.6)"
        }]
    }
    
    return {
        "comparison": comparison,
        "charts": {
            "metrics": metrics_chart,
            "timing": timing_chart,
            "confidence": confidence_chart
        }
    }

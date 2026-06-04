"""Retrain all models from the CSV dataset and save them to model_cache/"""
import os
import sys
import shutil

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")

def main():
    print("=" * 60)
    print("Retraining all models from CSV data...")
    print("=" * 60)

    # Clear the model cache to force fresh training
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Cleared model cache at {CACHE_DIR}")

    # Import AFTER clearing cache so model.py creates fresh cache dir
    from model import load_data, get_model, HAS_XGB

    ALGOS = ["lr", "nb", "svm", "rf"]
    if HAS_XGB:
        ALGOS.append("xgb")

    # Load the data
    load_data()

    # Train and cache each model
    for algo in ALGOS:
        print(f"\n--- Training {algo.upper()} ---")
        try:
            model = get_model(algo)
            print(f"{algo.upper()} model trained and saved successfully!")
        except Exception as e:
            print(f"ERROR training {algo}: {e}")

    print("\n" + "=" * 60)
    print("All models retrained successfully!")
    print("=" * 60)

    # Verify saved files
    print(f"\nFiles in {CACHE_DIR}:")
    for f in os.listdir(CACHE_DIR):
        fpath = os.path.join(CACHE_DIR, f)
        size = os.path.getsize(fpath)
        print(f"  {f}  ({size / 1024:.1f} KB)")

if __name__ == "__main__":
    main()

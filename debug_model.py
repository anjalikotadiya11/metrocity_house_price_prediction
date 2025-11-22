import joblib
import pandas as pd

print("--- Inspecting Model Features ---")

try:
    # Load the feature columns file
    with open('feature_columns.txt', 'r') as f:
        features = f.read().strip().split(',')
    
    print(f"Total Features Expected: {len(features)}")
    print("\n--- First 20 Features ---")
    print(features[:20])
    
    print("\n--- City-related Features Found ---")
    city_cols = [c for c in features if 'city' in c.lower() or 'mumbai' in c.lower() or 'bangalore' in c.lower()]
    if city_cols:
        for col in city_cols:
            print(f"  - {col}")
    else:
        print("  WARNING: No city columns found! Did you train without city encoding?")

except FileNotFoundError:
    print("Error: feature_columns.txt not found. Run model_training.ipynb first.")
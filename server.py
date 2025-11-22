import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import os

# --- 1. APP CONFIGURATION ---
app = Flask(__name__, template_folder='.', static_folder='.')

# --- 2. LOAD TRAINED ARTIFACTS ---
MODEL_FILE = 'house_price_model.pkl'
SCALER_FILE = 'scaler.pkl'
FEATURES_FILE = 'feature_columns.txt'

print("--- Server Starting ---")
model = None
scaler = None
model_columns = []

try:
    # Load Model
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
        print(f"✅ Loaded Model: {MODEL_FILE}")
    else:
        print(f"❌ Error: {MODEL_FILE} not found. Run training notebook first.")

    # Load Scaler
    if os.path.exists(SCALER_FILE):
        scaler = joblib.load(SCALER_FILE)
        print(f"✅ Loaded Scaler: {SCALER_FILE}")
    else:
        print(f"❌ Error: {SCALER_FILE} not found.")

    # Load Feature List
    if os.path.exists(FEATURES_FILE):
        with open(FEATURES_FILE, 'r') as f:
            model_columns = f.read().strip().split(',')
        print(f"✅ Loaded Feature List ({len(model_columns)} features)")
        
        # DEBUG: Print available city columns to help verify training
        city_cols = [c for c in model_columns if 'city' in c.lower()]
        print(f"ℹ️  City Columns in Model: {city_cols}")
        
    else:
        print(f"❌ Error: {FEATURES_FILE} not found.")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")

# --- 3. ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

# Explicitly serve static files (fixes potential CSS loading issues)
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

@app.route('/predict', methods=['POST'])
def predict():
    # Safety Check
    if not model or not scaler or not model_columns:
        return jsonify({'error': 'Server not ready. Model files missing.'})

    try:
        # A. Get JSON Data from Frontend
        data = request.json
        print(f"\n--- Incoming Prediction Request ---")
        print(f"Input Data: {data}")

        # B. Extract & Parse Inputs
        try:
            # Text Inputs
            city_input = data['city'].strip()
            status = data['status']
            
            # Numeric Inputs (Handle conversion errors)
            area = float(data['area_sqft'])
            baths = float(data['baths'])
            beds = float(data['beds'])
            
            # Optional Inputs (Default to 0 if missing or empty)
            balcony = float(data.get('balcony', 0))
            parking = float(data.get('parking', 0))
            floor = float(data.get('floor', 0))
            total_floors = float(data.get('total_floors', 0))
            
            # Pre-process Status (Ready to Move = 1, Under Construction = 0)
            status_ready = 1 if 'ready' in str(status).lower() else 0

        except (ValueError, KeyError) as e:
            return jsonify({'error': f'Invalid input data: {str(e)}'})

        # C. Initialize Input DataFrame (All zeros)
        # This creates a row with all possible columns from training
        input_df = pd.DataFrame(columns=model_columns)
        input_df.loc[0] = 0
        
        # D. Set Raw Numerical Values
        input_df.loc[0, 'area_sqft'] = area
        input_df.loc[0, 'baths'] = baths
        input_df.loc[0, 'beds'] = beds
        input_df.loc[0, 'balcony'] = balcony
        input_df.loc[0, 'parking'] = parking
        input_df.loc[0, 'floor'] = floor
        input_df.loc[0, 'total_floors'] = total_floors
        input_df.loc[0, 'status_ready'] = status_ready
        
        # E. Apply Scaling (Normalization)
        # We must use the same scaler from training on the same columns
        # IMPORTANT: Only scale columns that exist in the input AND were scaled in training
        scale_cols = ['area_sqft', 'baths', 'beds', 'balcony', 'parking', 'floor', 'total_floors']
        
        # Only select cols that are actually in our input_df to prevent errors
        cols_to_scale_present = [c for c in scale_cols if c in input_df.columns]
        
        if cols_to_scale_present:
            # Transform only the relevant columns
            input_df[cols_to_scale_present] = scaler.transform(input_df[cols_to_scale_present])
        else:
             print("Warning: No scaler columns found in input dataframe.")

        # F. Handle City Encoding (Smart Fuzzy Match)
        # Try exact match first: 'city_Mumbai'
        exact_col = f"city_{city_input}"
        
        found_col = None
        if exact_col in input_df.columns:
            found_col = exact_col
        else:
            # Fallback: Case-insensitive search
            # e.g., matches 'City_Mumbai' if input is 'Mumbai'
            for col in input_df.columns:
                if f"city_{city_input}".lower() == col.lower():
                    found_col = col
                    break
        
        if found_col:
            input_df.loc[0, found_col] = 1
            print(f"✅ Set City Column: '{found_col}' to 1")
        else:
            print(f"⚠️ Warning: City '{city_input}' NOT found in training data. Prediction might be inaccurate.")

        # G. Make Prediction
        # Ensure data types are float for the model
        X_input = input_df[model_columns].astype(float)
        prediction = model.predict(X_input)[0]
        
        # Prevent negative prices
        final_price = max(0, prediction)
        
        print(f"💰 Predicted Price: {final_price}")
        
        # H. Return Result
        return jsonify({'predicted_price': round(final_price, 2)})

    except Exception as e:
        print(f"Prediction Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("--- Starting Flask Server ---")  
    print("Go to: http://127.0.0.1:5000")
    app.run(debug=True)
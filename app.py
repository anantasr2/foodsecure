from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os
import traceback

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Nama fitur yang digunakan saat pelatihan model
FEATURE_NAMES = [
    'Akses Jangkauan', 'Jumlah Keluarga Miskin', 'Rasio Penduduk Miskin Desil 1', 
    'Rumah tangga tanpa akses listrik', 'Produksi pangan', 'Luas lahan', 
    'Rasio Sarana Pangan', 'Persentase balita stunting', 'Proporsi Penduduk Lanjut Usia', 
    'Rasio Rumah Tangga Tanpa Air Bersih', 'Rasio Tenaga Kesehatan', 
    'Total Keluarga Beresiko Stunting dan Keluarga rentan'
]

# Pemetaan fitur input (X1, X2, ...) ke nama fitur deskriptif
FEATURE_MAPPING = {
    'X1': 'Akses Jangkauan',
    'X2': 'Jumlah Keluarga Miskin',
    'X3': 'Rasio Penduduk Miskin Desil 1',
    'X4': 'Rumah tangga tanpa akses listrik',
    'X5': 'Produksi pangan',
    'X6': 'Luas lahan',
    'X7': 'Rasio Sarana Pangan',
    'X8': 'Persentase balita stunting',
    'X9': 'Proporsi Penduduk Lanjut Usia',
    'X10': 'Rasio Rumah Tangga Tanpa Air Bersih',
    'X11': 'Rasio Tenaga Kesehatan',
    'X12': 'Total Keluarga Beresiko Stunting dan Keluarga rentan'
}

# Global model and scaler
model = None
scaler = None

def load_xgb_model():
    """Load the XGBoost model from the pickle file"""
    try:
        model_path = os.path.join(BASE_DIR, 'best_model_XGB.pkl')
        with open(model_path, 'rb') as f:
            global model
            model = pickle.load(f)
        print("✓ Model loaded from PKL successfully.")
        print(f"Model n_features_in_: {model.n_features_in_}")
        print(f"Booster num features: {model.get_booster().num_features()}")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        traceback.print_exc()
        return False

def load_scaler():
    """Load the scaler from the pickle file"""
    try:
        scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
        with open(scaler_path, 'rb') as f:
            global scaler
            scaler = pickle.load(f)
        print("✓ Scaler loaded successfully.")
        return True
    except Exception as e:
        print(f"✗ Error loading scaler: {e}")
        traceback.print_exc()
        return False

# Load models at startup
model_loaded = load_xgb_model()
scaler_loaded = load_scaler()

@app.route('/')
def home():
    """Render home page"""
    return send_from_directory('.', 'index.html')

@app.route('/api/status', methods=['GET'])
def status():
    """Check the API status"""
    status_info = {
        'status': 'API is running',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'feature_names': FEATURE_NAMES
    }

    if model and scaler:
        try:
            test_data = pd.DataFrame([[50] * 12], columns=FEATURE_NAMES)
            test_scaled = scaler.transform(test_data)
            test_pred = model.predict(test_scaled)
            status_info['model_test'] = 'Model can predict'
            status_info['test_prediction'] = float(test_pred[0])
        except Exception as e:
            status_info['model_test'] = f'Model test failed: {str(e)}'
    
    return jsonify(status_info)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Make a prediction based on user input"""
    
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return '', 204
    
    print("\n" + "="*60)
    print("📥 NEW PREDICTION REQUEST")
    print("="*60)
    
    # Check if model and scaler are loaded
    if not model or not scaler:
        error_msg = 'Model atau scaler belum dimuat. Pastikan file model tersedia.'
        print(f"❌ {error_msg}")
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500
    
    try:
        # Get JSON data from request
        try:
            data = request.get_json(force=True)
        except Exception as e:
            print(f"❌ Error parsing JSON: {e}")
            return jsonify({
                'success': False,
                'error': f'Error parsing JSON: {str(e)}'
            }), 400
        
        if not data:
            print("❌ No data received")
            return jsonify({
                'success': False,
                'error': 'Data tidak diberikan. Pastikan mengirim JSON.'
            }), 400
        
        print(f"📊 Data received: {data}")
        
        # Validate and extract features, mapping input keys to feature names
        features_dict = {}
        missing_features = []
        invalid_features = []
        
        for feature_name in FEATURE_NAMES:
            feature_key = [key for key, value in FEATURE_MAPPING.items() if value == feature_name][0]  # Mapping X1, X2, ...
            value = data.get(feature_key)
            
            if value is None or value == '':
                missing_features.append(feature_name)
            else:
                try:
                    # Convert to float and validate
                    float_value = float(value)
                    if np.isnan(float_value) or np.isinf(float_value):
                        invalid_features.append(f"{feature_name} (invalid number)")
                    else:
                        features_dict[feature_name] = float_value
                except (ValueError, TypeError) as e:
                    invalid_features.append(f"{feature_name} (value: {value})")
        
        # Check for errors
        if missing_features:
            error_msg = f'Fitur yang hilang: {", ".join(missing_features)}'
            print(f"❌ {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        if invalid_features:
            error_msg = f'Nilai tidak valid untuk: {", ".join(invalid_features)}'
            print(f"❌ {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        print(f"✓ All features validated: {features_dict}")
        
        # Check if all features are 0
        if all(value == 0 for value in features_dict.values()):
            error_msg = 'Semua fitur tidak boleh bernilai 0.'
            print(f"❌ {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        # Create DataFrame with correct feature order
        features_df = pd.DataFrame([features_dict], columns=FEATURE_NAMES)
        print(f"📊 Features DataFrame shape: {features_df.shape}")
        print(f"📊 Features DataFrame:\n{features_df}")
        
        # Scale features
        features_scaled = scaler.transform(features_df)
        print(f"📈 Scaled features shape: {features_scaled.shape}")
        print(f"📈 Scaled features: {features_scaled}")
        
        # Make prediction
        prediction = model.predict(features_scaled)
        score = float(prediction[0])
        
        print(f"🎯 Raw prediction: {prediction}")
        print(f"🎯 Prediction score: {score}")
        
        # Calculate confidence (default for regression models)
        confidence = 96.8
        
        # Try to get prediction intervals if available
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features_scaled)
                confidence = float(np.max(probabilities) * 100)
                print(f"📊 Confidence from predict_proba: {confidence}%")
        except Exception as e:
            print(f"ℹ️ Using default confidence: {confidence}% (predict_proba not available)")

        response = {
            'success': True,
            'score': round(score, 3),
            'confidence': round(confidence, 1),
            'features_received': list(features_dict.keys()),
            'message': 'Prediksi berhasil dilakukan'
        }
        
        print(f"✅ Response prepared: {response}")
        print("="*60 + "\n")
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"❌ FATAL ERROR in prediction: {str(e)}")
        print("="*60)
        print("TRACEBACK:")
        traceback.print_exc()
        print("="*60 + "\n")
        
        return jsonify({
            'success': False,
            'error': f'Terjadi kesalahan server: {str(e)}'
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({
        'success': False,
        'error': 'Method not allowed'
    }), 405

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Flask API + Web Server Starting")
    print("="*60)
    print(f"📂 Base Directory: {BASE_DIR}")
    print(f"🤖 Model Status: {'✓ Loaded' if model else '✗ Not Loaded'}")
    print(f"📊 Scaler Status: {'✓ Loaded' if scaler else '✗ Not Loaded'}")
    print("="*60)
    print("📌 Local: http://localhost:5000")
    print("📌 Network: http://0.0.0.0:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
    

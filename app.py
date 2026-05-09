"""
app.py – Smart Dairy Livestock Monitoring and Decision Support System
Flask backend: loads ML model, handles prediction requests, generates recommendations.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────
app = Flask(__name__)
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

# ─────────────────────────────────────────────
# Load model at startup
# ─────────────────────────────────────────────
model_data = joblib.load(MODEL_PATH)
clf        = model_data['model']
FEATURES   = model_data['features']
MODEL_NAME = model_data['best_name']
ALL_RESULTS = model_data.get('all_results', {})

print(f"✅  Model loaded: {MODEL_NAME}  (accuracy={model_data['accuracy']*100:.1f}%)")

# ─────────────────────────────────────────────
# Thresholds (physiological reference values)
# ─────────────────────────────────────────────
THRESHOLDS = {
    'temperature_high': 39.2,   # °C
    'temperature_low':  37.8,
    'humidity_high':    75,     # %
    'humidity_low':     40,
    'milk_yield_low':   15,     # L/day
    'weight_low':       420,    # kg
    'heart_rate_high':  80,     # bpm
    'heart_rate_low':   55,
    'activity_low':     50,     # %
}

# ─────────────────────────────────────────────
# Recommendation Engine
# ─────────────────────────────────────────────
def generate_recommendations(data: dict, prediction: int) -> list:
    """
    Rule-based recommendation engine.
    Returns a list of recommendation dicts with keys:
        icon, title, message, severity  (info | warning | danger)
    """
    recs = []
    temp     = data['temperature']
    humidity = data['humidity']
    milk     = data['milk_yield']
    weight   = data['weight']
    hr       = data['heart_rate']
    activity = data['activity_level']
    T = THRESHOLDS

    # ── Health Status ────────────────────────
    if prediction == 1:
        recs.append({
            'icon': 'bi-exclamation-triangle-fill',
            'title': 'Immediate Veterinary Consultation Required',
            'message': (
                'The animal shows signs of illness. Isolate it from the herd immediately '
                'to prevent disease spread. Contact a licensed veterinarian as soon as possible.'
            ),
            'severity': 'danger'
        })
        recs.append({
            'icon': 'bi-shield-exclamation',
            'title': 'Herd Isolation Protocol',
            'message': (
                'Separate the affected animal into a quarantine pen. '
                'Monitor other herd members closely for similar symptoms.'
            ),
            'severity': 'danger'
        })

    # ── Body Temperature ─────────────────────
    if temp > T['temperature_high']:
        recs.append({
            'icon': 'bi-thermometer-high',
            'title': 'High Body Temperature Detected',
            'message': (
                f'Temperature is {temp}°C, above the safe threshold of {T["temperature_high"]}°C. '
                'Increase water availability (at least 100 L/day), provide shade, '
                'improve air circulation, and reduce physical activity.'
            ),
            'severity': 'warning'
        })
    elif temp < T['temperature_low']:
        recs.append({
            'icon': 'bi-thermometer-low',
            'title': 'Low Body Temperature Detected',
            'message': (
                f'Temperature is {temp}°C, below the safe threshold of {T["temperature_low"]}°C. '
                'Ensure the animal is housed in a warm, sheltered environment. '
                'Check for signs of infection or metabolic disorders.'
            ),
            'severity': 'warning'
        })

    # ── Humidity ─────────────────────────────
    if humidity > T['humidity_high']:
        recs.append({
            'icon': 'bi-wind',
            'title': 'High Humidity – Improve Ventilation',
            'message': (
                f'Relative humidity is {humidity}%, above the recommended {T["humidity_high"]}%. '
                'Install or run ventilation fans, ensure proper drainage, '
                'and reduce stocking density to lower moisture levels.'
            ),
            'severity': 'warning'
        })
    elif humidity < T['humidity_low']:
        recs.append({
            'icon': 'bi-droplet-half',
            'title': 'Low Humidity – Risk of Respiratory Issues',
            'message': (
                f'Relative humidity is {humidity}%, which is too dry. '
                'Use misters or increase water troughs nearby to raise humidity. '
                'Monitor for dry cough or nasal discharge.'
            ),
            'severity': 'info'
        })

    # ── Milk Yield ────────────────────────────
    if milk < T['milk_yield_low']:
        recs.append({
            'icon': 'bi-cup-hot-fill',
            'title': 'Low Milk Yield – Review Nutrition',
            'message': (
                f'Milk yield is {milk} L/day, below the target of {T["milk_yield_low"]} L/day. '
                'Increase high-energy feed rations (TMR), ensure adequate water intake, '
                'and consult a nutritionist to balance the diet.'
            ),
            'severity': 'warning'
        })

    # ── Weight ────────────────────────────────
    if weight < T['weight_low']:
        recs.append({
            'icon': 'bi-bar-chart-fill',
            'title': 'Underweight – Supplement Nutrition',
            'message': (
                f'Body weight is {weight} kg, below the healthy benchmark of {T["weight_low"]} kg. '
                'Add energy-dense supplements (molasses, bypass fat, protein meals) to the diet. '
                'Schedule a body condition scoring (BCS) evaluation weekly.'
            ),
            'severity': 'warning'
        })

    # ── Heart Rate ────────────────────────────
    if hr > T['heart_rate_high']:
        recs.append({
            'icon': 'bi-heart-pulse-fill',
            'title': 'Elevated Heart Rate',
            'message': (
                f'Heart rate is {hr} bpm, above the normal range of '
                f'{T["heart_rate_low"]}–{T["heart_rate_high"]} bpm. '
                'This may indicate fever, stress, or cardiac issues. '
                'Provide a calm environment and seek veterinary assessment.'
            ),
            'severity': 'warning'
        })
    elif hr < T['heart_rate_low']:
        recs.append({
            'icon': 'bi-heart-fill',
            'title': 'Low Heart Rate – Possible Bradycardia',
            'message': (
                f'Heart rate is {hr} bpm, below {T["heart_rate_low"]} bpm. '
                'Could indicate bloat, vagal tone increase, or hypothermia. '
                'Conduct a thorough clinical examination immediately.'
            ),
            'severity': 'warning'
        })

    # ── Activity Level ────────────────────────
    if activity < T['activity_low']:
        recs.append({
            'icon': 'bi-person-walking',
            'title': 'Low Activity Level',
            'message': (
                f'Activity level is {activity}%, below the expected {T["activity_low"]}%. '
                'Reduced activity is an early indicator of lameness, pain, or illness. '
                'Examine hooves for laminitis and assess overall well-being.'
            ),
            'severity': 'info'
        })

    # ── Positive Feedback ─────────────────────
    if prediction == 0 and len(recs) == 0:
        recs.append({
            'icon': 'bi-check-circle-fill',
            'title': 'All Parameters Within Normal Range',
            'message': (
                'The animal appears to be in excellent health. '
                'Continue routine monitoring, balanced nutrition, and regular vaccinations '
                'to maintain optimal productivity.'
            ),
            'severity': 'success'
        })

    if prediction == 0 and len(recs) > 0:
        recs.insert(0, {
            'icon': 'bi-check-circle-fill',
            'title': 'Animal is Healthy – Monitor Minor Alerts',
            'message': (
                'The ML model predicts the animal is currently Healthy. '
                'However, some parameters are slightly outside optimal ranges. '
                'Follow the recommendations below for preventive care.'
            ),
            'severity': 'success'
        })

    return recs


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route('/')
def home():
    """Home / landing page."""
    model_stats = {
        'model_name':  MODEL_NAME,
        'accuracy':    round(model_data['accuracy'] * 100, 2),
        'precision':   round(model_data['precision'] * 100, 2),
        'recall':      round(model_data['recall'] * 100, 2),
        'f1':          round(model_data['f1'] * 100, 2),
        'cv_accuracy': round(model_data['cv_accuracy'] * 100, 2),
        'all_results': {
            k: {
                'accuracy':  round(v['accuracy'] * 100, 2),
                'precision': round(v['precision'] * 100, 2),
                'recall':    round(v['recall'] * 100, 2),
                'f1':        round(v['f1'] * 100, 2),
            }
            for k, v in ALL_RESULTS.items()
        }
    }
    return render_template('home.html', stats=model_stats)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction form and result page."""
    if request.method == 'GET':
        return render_template('predict.html')

    # ── Parse form input ─────────────────────
    try:
        data = {
            'temperature':   float(request.form['temperature']),
            'humidity':      float(request.form['humidity']),
            'milk_yield':    float(request.form['milk_yield']),
            'weight':        float(request.form['weight']),
            'heart_rate':    float(request.form['heart_rate']),
            'activity_level': float(request.form['activity_level']),
        }
    except (KeyError, ValueError) as e:
        return render_template('predict.html', error=f"Invalid input: {e}")

    # ── ML Prediction ─────────────────────────
    X_input    = pd.DataFrame([data], columns=FEATURES)
    prediction = int(clf.predict(X_input)[0])
    proba      = clf.predict_proba(X_input)[0]
    confidence = round(float(proba[prediction]) * 100, 1)

    # ── Risk Level ────────────────────────────
    if prediction == 1:
        risk_level = 'High' if confidence > 80 else 'Medium'
    else:
        risk_level = 'Low' if confidence > 80 else 'Medium'

    # ── Recommendations ───────────────────────
    recommendations = generate_recommendations(data, prediction)

    result = {
        'prediction':       prediction,
        'prediction_label': 'Sick / At Risk' if prediction == 1 else 'Healthy',
        'confidence':       confidence,
        'risk_level':       risk_level,
        'recommendations':  recommendations,
        'input_data':       data,
        'model_name':       MODEL_NAME,
    }

    return render_template('result.html', result=result)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for prediction (for AJAX calls)."""
    try:
        payload = request.get_json(force=True)
        data = {
            'temperature':    float(payload['temperature']),
            'humidity':       float(payload['humidity']),
            'milk_yield':     float(payload['milk_yield']),
            'weight':         float(payload['weight']),
            'heart_rate':     float(payload['heart_rate']),
            'activity_level': float(payload['activity_level']),
        }
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    X_input    = pd.DataFrame([data], columns=FEATURES)
    prediction = int(clf.predict(X_input)[0])
    proba      = clf.predict_proba(X_input)[0]
    confidence = round(float(proba[prediction]) * 100, 1)
    risk_level = ('High' if confidence > 80 else 'Medium') if prediction == 1 else \
                 ('Low'  if confidence > 80 else 'Medium')

    return jsonify({
        'prediction':       prediction,
        'prediction_label': 'Sick / At Risk' if prediction == 1 else 'Healthy',
        'confidence':       confidence,
        'risk_level':       risk_level,
        'recommendations':  generate_recommendations(data, prediction),
    })


@app.route('/about')
def about():
    return render_template('about.html')


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

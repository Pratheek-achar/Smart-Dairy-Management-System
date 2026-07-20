"""
app.py – Smart Dairy Livestock Monitoring and Decision Support System
Flask backend: loads ML model, handles prediction requests, generates recommendations.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, redirect, url_for, flash
import database as db

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

# Initialise SQLite database
db.init_db()

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


# ═════════════════════════════════════════════════════════════════
# MODULE 3 — ANIMAL BEHAVIOUR TRACKING
# ═════════════════════════════════════════════════════════════════

def _get_stream():
    """Lazy-load VideoStream so the app boots even without a camera."""
    try:
        from behaviour_tracking.video_stream import VideoStream
        vs = VideoStream.get_instance()
        if not vs.running:
            import threading
            t = threading.Thread(target=vs.start, daemon=True)
            t.start()
        return vs
    except Exception as e:
        print(f"[Tracking] VideoStream unavailable: {e}")
        return None


@app.route('/tracking')
def tracking():
    """Render the live animal behaviour tracking dashboard."""
    return render_template('tracking.html')


@app.route('/video_feed')
def video_feed():
    """MJPEG stream of annotated camera frames."""
    vs = _get_stream()
    if vs is None:
        return jsonify({'error': 'Camera module unavailable'}), 503

    return Response(
        stream_with_context(vs.mjpeg_generator()),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/tracking_stats')
def tracking_stats():
    """JSON snapshot of current tracking state for dashboard polling."""
    vs = _get_stream()
    if vs is None:
        return jsonify({'error': 'Camera module unavailable'}), 503
    stats = vs.get_stats()
    # Make alerts JSON-serialisable
    for a in stats.get('alerts', []):
        a['timestamp'] = round(a.get('timestamp', 0), 1)
    return jsonify(stats)


@app.route('/tracking_alerts')
def tracking_alerts():
    """Server-Sent Events stream for real-time alert notifications."""
    import json as _json
    import time as _time

    def generate():
        vs = _get_stream()
        last_ts = 0.0
        while True:
            if vs:
                stats = vs.get_stats()
                for alert in stats.get('alerts', []):
                    ts = alert.get('timestamp', 0)
                    if ts > last_ts:
                        last_ts = ts
                        data = _json.dumps(alert)
                        yield f"data: {data}\n\n"
            _time.sleep(0.5)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


@app.route('/tracking/upload', methods=['POST'])
def tracking_upload():
    """Accept an uploaded video file and switch the stream source to it."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    f    = request.files['video']
    path = os.path.join(BASE_DIR, 'static', 'uploads', f.filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f.save(path)
    vs = _get_stream()
    if vs:
        vs.switch_source(path)
    return jsonify({'status': 'ok', 'path': path})


@app.route('/tracking/use_webcam', methods=['POST'])
def tracking_use_webcam():
    """Switch back to webcam (source index 0)."""
    vs = _get_stream()
    if vs:
        vs.switch_source(0)
    return jsonify({'status': 'ok', 'source': 'webcam'})


# ═════════════════════════════════════════════════════════════════
# MODULE 4 — ANIMAL PROFILE MANAGEMENT
# ═════════════════════════════════════════════════════════════════

@app.route('/animals')
def animals_list():
    """List all animal profiles."""
    animals = db.get_all_animals()
    return render_template('animals.html', animals=animals)


@app.route('/animals/new', methods=['GET', 'POST'])
def animal_new():
    """Register a new animal."""
    if request.method == 'POST':
        uid = db.create_animal(request.form.to_dict())
        return redirect(url_for('animal_profile', uid=uid))
    return render_template('animal_form.html', animal=None)


@app.route('/animals/<uid>')
def animal_profile(uid):
    """View an animal's full profile."""
    animal = db.get_animal(uid)
    if not animal:
        return render_template('404.html'), 404
    stats          = db.get_animal_stats(uid)
    health_records = db.get_health_records(uid, limit=20)
    behaviour_logs = db.get_behaviour_logs(uid, limit=20)
    return render_template('animal_profile.html',
                           animal=animal,
                           stats=stats,
                           health_records=health_records,
                           behaviour_logs=behaviour_logs)


@app.route('/animals/<uid>/edit', methods=['GET', 'POST'])
def animal_edit(uid):
    """Edit an animal's profile."""
    animal = db.get_animal(uid)
    if not animal:
        return redirect(url_for('animals_list'))
    if request.method == 'POST':
        db.update_animal(uid, request.form.to_dict())
        return redirect(url_for('animal_profile', uid=uid))
    return render_template('animal_form.html', animal=animal)


@app.route('/animals/<uid>/delete', methods=['POST'])
def animal_delete(uid):
    """Delete an animal and all associated records."""
    db.delete_animal(uid)
    return redirect(url_for('animals_list'))


@app.route('/animals/<uid>/save_health', methods=['POST'])
def animal_save_health(uid):
    """Save a health prediction result to an animal's record."""
    animal = db.get_animal(uid)
    if not animal:
        return jsonify({'error': 'Animal not found'}), 404
    data = request.get_json() or request.form.to_dict()
    record_id = db.add_health_record(uid, {
        'temperature':    data.get('temperature'),
        'humidity':       data.get('humidity'),
        'milk_yield':     data.get('milk_yield'),
        'weight_kg':      data.get('weight'),
        'heart_rate':     data.get('heart_rate'),
        'activity_level': data.get('activity_level'),
        'prediction':     data.get('prediction'),
        'confidence':     data.get('confidence'),
        'risk_level':     data.get('risk_level'),
        'recommendations': data.get('recommendations', []),
    })
    if request.is_json:
        return jsonify({'status': 'saved', 'record_id': record_id})
    return redirect(url_for('animal_profile', uid=uid))


@app.route('/animals/<uid>/save_behaviour', methods=['POST'])
def animal_save_behaviour(uid):
    """Log a behaviour event to an animal's record."""
    animal = db.get_animal(uid)
    if not animal:
        return jsonify({'error': 'Animal not found'}), 404
    data = request.get_json() or {}
    record_id = db.add_behaviour_log(uid, {
        'behaviour':   data.get('behaviour'),
        'duration_sec': data.get('duration_sec', 0),
        'velocity':    data.get('velocity', 0),
        'alert_msg':   data.get('alert_msg', ''),
    })
    return jsonify({'status': 'saved', 'record_id': record_id})


@app.route('/api/animals')
def api_animals():
    """JSON list of all animals (for dropdowns in predict/tracking forms)."""
    animals = db.get_all_animals(status_filter='Active')
    return jsonify([{'uid': a['uid'], 'name': a['name'], 'breed': a['breed']} for a in animals])


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)

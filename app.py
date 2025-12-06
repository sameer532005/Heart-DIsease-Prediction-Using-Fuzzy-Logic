from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime

app = Flask(__name__)

# ---- Simple login (demo only) ----
USERNAME = "admin"
PASSWORD = "1234"

# ---- Dataset for default medians ----
DATA_PATH = "D:/SOFT COMPUTING LAB/disease_prediction_fuzzy_heart/archive/cleaned_merged_heart_dataset.csv"
_df = pd.read_csv(DATA_PATH)

# ---- Fuzzy logic helpers ----
def trimf_single(x, a, b, c):
    x = float(x)
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a + 1e-9)
    if b < x < c:
        return (c - x) / (c - b + 1e-9)
    return 0.0

REP = {'low': 20.0, 'medium': 50.0, 'high': 80.0}


def patient_to_features_from_form(form):
    """Convert form fields (except name) into numeric features for fuzzy model."""
    return dict(
        age=float(form.get('age', 0)),
        trestbps=float(form.get('trestbps', 0)),
        chol=float(form.get('chol', 0)),
        thalach=float(form.get('thalach', 0)),
        oldpeak=float(form.get('oldpeak', 0)),
        sex=int(form.get('sex', 0)),
        cp=int(form.get('cp', 0)),
        fbs=int(form.get('fbs', 0)),
        exang=int(form.get('exang', 0)),
        ca=int(form.get('ca', 0)),
        thal=int(form.get('thal', 1)),
        restecg=int(form.get('restecg', 0)),
        slope=int(form.get('slope', 1)),
    )


def fuzzy_risk(features):
    age = features['age']
    trestbps = features['trestbps']
    chol = features['chol']
    thalach = features['thalach']
    oldpeak = features['oldpeak']
    exang = features['exang']
    cp = features['cp']
    fbs = features['fbs']
    ca = features['ca']
    thal = features['thal']

    age_y = trimf_single(age, 20, 30, 40)
    age_m = trimf_single(age, 35, 47.5, 60)
    age_o = trimf_single(age, 55, 72.5, 90)

    bp_l = trimf_single(trestbps, 80, 95, 110)
    bp_n = trimf_single(trestbps, 100, 115, 130)
    bp_h = trimf_single(trestbps, 120, 150, 200)

    ch_l = trimf_single(chol, 100, 140, 180)
    ch_m = trimf_single(chol, 170, 205, 240)
    ch_h = trimf_single(chol, 220, 320, 600)

    th_l = trimf_single(thalach, 60, 85, 110)
    th_m = trimf_single(thalach, 100, 120, 140)
    th_h = trimf_single(thalach, 130, 160, 210)

    op_l = trimf_single(oldpeak, 0, 0.3, 1)
    op_m = trimf_single(oldpeak, 0.5, 1.25, 2)
    op_h = trimf_single(oldpeak, 1.5, 3, 6)

    activ_low = 0.0
    activ_med = 0.0
    activ_high = 0.0

    # Example fuzzy rules
    activ_high = max(activ_high, min(age_o, ch_h))
    activ_high = max(activ_high, min(th_l, 1.0 if exang == 1 else 0.0))
    activ_high = max(activ_high, op_h)

    activ_med = max(activ_med, min(age_m, ch_m, bp_h))
    activ_med = max(activ_med, min(th_m, op_m))

    activ_low = max(activ_low, min(age_y, th_h, ch_l))

    if cp in [2, 3]:
        activ_low = max(activ_low, min(th_h, 0.6))
    if fbs == 1 and ca > 0:
        activ_med = max(activ_med, 0.6 * 0.6)
    if thal in [2, 3]:
        activ_high = max(activ_high, 0.7)

    numerator = (activ_low * REP['low'] +
                 activ_med * REP['medium'] +
                 activ_high * REP['high'])
    denom = (activ_low + activ_med + activ_high) + 1e-9
    score = numerator / denom
    label = 'low' if score < 35 else ('medium' if score < 65 else 'high')

    return {
        'score': round(float(score), 2),
        'label': label,
        'activations': {
            'low': activ_low,
            'medium': activ_med,
            'high': activ_high
        }
    }

# ---- SQLite storage ----
DB_PATH = "predictions.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            name TEXT,
            age REAL,
            sex INTEGER,
            cp INTEGER,
            trestbps REAL,
            chol REAL,
            fbs INTEGER,
            restecg INTEGER,
            thalach REAL,
            exang INTEGER,
            oldpeak REAL,
            slope INTEGER,
            ca INTEGER,
            thal INTEGER,
            score REAL,
            label TEXT,
            low REAL,
            med REAL,
            high REAL
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(patient_name, features, res):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO predictions (
            created_at, name,
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal,
            score, label, low, med, high
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now().isoformat(timespec="seconds"),
        patient_name,
        features['age'], features['sex'], features['cp'], features['trestbps'],
        features['chol'], features['fbs'], features['restecg'],
        features['thalach'], features['exang'], features['oldpeak'],
        features['slope'], features['ca'], features['thal'],
        res['score'], res['label'],
        res['activations']['low'], res['activations']['medium'], res['activations']['high']
    ))
    conn.commit()
    conn.close()


init_db()  # create table on startup

# ---- Routes ----

@app.route('/')
def index():
    return render_template('index.html', error=None)


@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username', '')
    password = request.form.get('password', '')
    if username == USERNAME and password == PASSWORD:
        return redirect(url_for('dashboard'))
    return render_template('index.html', error="Invalid credentials")


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/input')
def input_form():
    # default medians
    med = _df.median(numeric_only=True).to_dict()
    return render_template(
        'input.html',
        age=int(med.get('age', 50)),
        sex=int(med.get('sex', 1)),
        cp=int(med.get('cp', 0)),
        trestbps=int(med.get('trestbps', 120)),
        chol=int(med.get('chol', 240)),
        fbs=int(med.get('fbs', 0)),
        restecg=int(med.get('restecg', 0)),
        thalach=int(med.get('thalach', 140)),
        exang=int(med.get('exang', 0)),
        oldpeak=float(med.get('oldpeak', 1.0)),
        slope=int(med.get('slope', 1)),
        ca=int(med.get('ca', 0)),
        thal=int(med.get('thal', 2))
    )


@app.route('/predict', methods=['POST'])
def predict():
    # name is only stored/displayed, not used in fuzzy model
    patient_name = request.form.get('patient_name', '').strip() or "Unknown"
    features = patient_to_features_from_form(request.form)
    res = fuzzy_risk(features)

    save_prediction(patient_name, features, res)

    return render_template(
        'result.html',
        score=res['score'],
        label=res['label'],
        low=res['activations']['low'],
        med=res['activations']['medium'],
        high=res['activations']['high'],
        features=features,
        patient_name=patient_name
    )


@app.route('/patients')
def list_patients():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT id, created_at, name, age, sex, score, label
        FROM predictions
        ORDER BY id DESC
    """)
    rows = c.fetchall()
    conn.close()
    return render_template('patients.html', patients=rows)


@app.route('/patients/<int:pid>')
def view_patient(pid):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM predictions WHERE id = ?", (pid,))
    row = c.fetchone()
    conn.close()
    if row is None:
        return "Prediction not found", 404

    features = {
        'age': row['age'],
        'sex': row['sex'],
        'cp': row['cp'],
        'trestbps': row['trestbps'],
        'chol': row['chol'],
        'fbs': row['fbs'],
        'restecg': row['restecg'],
        'thalach': row['thalach'],
        'exang': row['exang'],
        'oldpeak': row['oldpeak'],
        'slope': row['slope'],
        'ca': row['ca'],
        'thal': row['thal'],
    }

    return render_template(
        'result.html',
        score=row['score'],
        label=row['label'],
        low=row['low'],
        med=row['med'],
        high=row['high'],
        features=features,
        patient_name=row['name'] or "Unknown"
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

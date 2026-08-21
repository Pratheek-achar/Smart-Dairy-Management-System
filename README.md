# 🐄 Smart Dairy Livestock Monitoring & Decision Support System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B6B?style=for-the-badge)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white)

**An AI-powered full-stack web application for intelligent dairy farm management.**

</div>

---

## 📌 Overview

Smart Dairy is a complete livestock decision support system built for dairy farm management. It combines **Machine Learning**, **Computer Vision**, and a **full-stack web interface** to provide real-time health monitoring, behaviour surveillance, animal profile management, and vaccination scheduling — all accessible from a browser.

---

## 🚀 Features & Modules

### 🧠 Module 1 — Livestock Health Prediction
- Predict **Healthy** or **Sick/At Risk** using ML classification
- Input: Body Temperature, Humidity, Milk Yield, Weight, Heart Rate, Activity Level
- Compares **Decision Tree**, **Random Forest**, and **Logistic Regression**
- Displays Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Feature Importance
- Personalised veterinary recommendations based on prediction results
- Save prediction results directly to any animal's profile

### 📹 Module 2 — Animal Behaviour Tracking (Live Camera)
- **YOLOv8** real-time animal detection from webcam or uploaded video
- **DeepSORT** multi-object tracking with persistent IDs across frames
- Behaviour classification: **Resting · Feeding · Walking · Stress · Aggressive · Isolated**
- Live MJPEG video stream with bounding boxes and behaviour labels
- Real-time **Chart.js** behaviour distribution & velocity charts
- Smart alert engine with **WARNING / CRITICAL** levels and suppression logic
- Manual **Start / Stop toggle** — tracking only runs when explicitly started
- Assign any tracked animal directly to an animal profile

### 🐄 Module 3 — Animal Profile Management
- Register animals with auto-generated unique IDs (`SDMS-YYYY-NNNN`)
- Store: Name, Breed, Age, Gender, Weight, Ear Tag, Color, Health Status, Notes
- Full **CRUD** — Create, Read, Update, Delete
- **Health Check History** — every prediction linked to the animal profile
- **Weight Trend Chart** — visualise weight changes over time via Chart.js
- **Behaviour Log** — all live tracking events assignable to specific animals
- **Vaccination & Treatment panels** embedded on every animal profile page
- Search & filter animals by name, breed, status, gender

### 📅 Module 4 — Vaccination & Treatment Scheduler
- Schedule vaccinations with dose number, due date, next due date, and vet name
- Log treatments: medicine name, dosage, treatment type, cost, vet details
- **FullCalendar** interactive calendar — colour-coded by status
- **Auto-overdue detection** — status changes to Overdue when date passes
- **Mark Done** workflow — record administered date and administering vet
- Status tracking: `Pending` · `Completed` · `Overdue` · `Skipped` · `Active` · `Cancelled`
- Full CRUD for both vaccinations and treatments
- Events visible on both the scheduler calendar and individual animal profiles

### 📊 Live Home Dashboard
- **Live KPI cards** — active animals, total registered, upcoming & overdue vaccines, ML accuracy
- **4 Module cards** with real counts and direct action buttons
- **Animated ML performance bars** — Accuracy, Precision, Recall, F1
- **Algorithm comparison table** with Best badge
- **Recent Health Checks** activity feed (last 5 predictions)
- **Upcoming Vaccinations** sidebar panel (next 5 due, overdue highlighted)
- **Quick Action buttons** — Register Animal, Schedule Vaccine, Start Tracking, Run Prediction

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, Flask 2.x |
| Machine Learning | Scikit-learn (Random Forest, Decision Tree, Logistic Regression) |
| Computer Vision | OpenCV, YOLOv8 (Ultralytics), DeepSORT |
| Database | SQLite (built-in `sqlite3`) |
| Frontend | HTML5, CSS3, Bootstrap 5, Vanilla JS |
| Charts | Chart.js, FullCalendar 6 |
| Fonts & Icons | Google Fonts (Inter), Bootstrap Icons |

---

## 📁 Project Structure

```
Smart-Dairy-Management-System/
│
├── app.py                          # Flask application — all routes
├── database.py                     # SQLite CRUD layer
├── train_model.py                  # ML model training script
├── generate_dataset.py             # Synthetic dataset generator
├── model.pkl                       # Trained ML model
├── livestock_health_dataset.csv    # Training dataset
│
├── behaviour_tracking/             # Module 2 — CV pipeline
│   ├── __init__.py
│   ├── detection.py                # YOLOv8 wrapper
│   ├── tracker.py                  # DeepSORT tracker
│   ├── behaviour_analyzer.py       # Kinematic behaviour classifier
│   ├── anomaly_detector.py         # Alert engine
│   └── video_stream.py             # MJPEG stream orchestrator
│
├── templates/
│   ├── base.html                   # Base layout & navbar
│   ├── home.html                   # Live home dashboard
│   ├── predict.html                # Health prediction form
│   ├── result.html                 # Prediction results + save to profile
│   ├── tracking.html               # Live behaviour tracking dashboard
│   ├── animals.html                # Animal profiles list
│   ├── animal_profile.html         # Individual animal profile + scheduler panels
│   ├── animal_form.html            # Add / Edit animal form
│   ├── scheduler.html              # Vaccination & treatment scheduler
│   └── about.html                  # About page
│
└── static/
    ├── css/style.css               # All custom styles (dark theme)
    ├── js/main.js                  # Frontend scripts
    └── images/                     # Generated ML charts (confusion matrix, feature importance)
```

---

## 🗄️ Database Schema

```
animals               — Animal profiles (uid, name, breed, age, gender, weight, status…)
health_records        — ML prediction history per animal
behaviour_logs        — Live tracking events per animal
vaccination_schedules — Vaccination schedule per animal (dose, dates, status)
treatments            — Treatment logs (medicine, dosage, vet, cost, dates)
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Pratheek-achar/Smart-Dairy-Management-System.git
cd Smart-Dairy-Management-System
```

### 2. Install dependencies
```bash
pip install flask scikit-learn pandas numpy joblib matplotlib seaborn
pip install ultralytics opencv-python deep-sort-realtime
```

### 3. Train the ML model
```bash
python train_model.py
```

### 4. Run the application
```bash
python app.py
```

### 5. Open in browser
```
http://127.0.0.1:5000
```

> **Note:** YOLOv8 weights (`yolov8n.pt`, ~6 MB) auto-download on first use of the Live Tracking module. The SQLite database (`smartdairy.db`) is created automatically on first run.

---

## 🔗 Available Pages

| URL | Page |
|---|---|
| `/` | Home Dashboard |
| `/predict` | Health Prediction Form |
| `/tracking` | Live Behaviour Tracking |
| `/animals` | Animal Profile List |
| `/animals/<uid>` | Individual Animal Profile |
| `/animals/new` | Register New Animal |
| `/scheduler` | Vaccination & Treatment Scheduler |
| `/about` | About Page |

---

## 📸 Key UI Highlights

- 🌑 **Full dark theme** throughout — `#060e1c` background, `#22d3ee` accent
- 🎨 Glassmorphism cards with hover lift animations
- 📊 Live animated metric bars on dashboard
- 📅 FullCalendar with colour-coded vaccination events
- 📹 Real-time MJPEG video feed with overlaid detection boxes
- 📱 Fully responsive layout (Bootstrap 5 grid)

---

## 👨‍💻 Author

**Pratheek Achar**
Final Year Project — Smart Dairy Management System

---

## 📄 License

MIT License

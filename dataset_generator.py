"""
Dataset Generator for Smart Dairy Livestock Monitoring System
Generates a realistic synthetic dataset for model training
"""

import numpy as np
import pandas as pd
import random

np.random.seed(42)
random.seed(42)

n_samples = 2000

# ─────────────────────────────────────────────
# Generate HEALTHY animals (label = 0)
# ─────────────────────────────────────────────
n_healthy = 1200

healthy_temperature  = np.random.normal(38.5, 0.4, n_healthy)   # °C  (normal 38.0–39.0)
healthy_humidity     = np.random.normal(60,   8,   n_healthy)   # %
healthy_milk_yield   = np.random.normal(22,   3,   n_healthy)   # litres/day
healthy_weight       = np.random.normal(480,  40,  n_healthy)   # kg
healthy_heart_rate   = np.random.normal(68,   5,   n_healthy)   # bpm
healthy_activity     = np.random.normal(75,   8,   n_healthy)   # %

healthy_df = pd.DataFrame({
    'temperature':   healthy_temperature,
    'humidity':      healthy_humidity,
    'milk_yield':    healthy_milk_yield,
    'weight':        healthy_weight,
    'heart_rate':    healthy_heart_rate,
    'activity_level': healthy_activity,
    'disease_label': 0
})

# ─────────────────────────────────────────────
# Generate SICK/AT-RISK animals (label = 1)
# ─────────────────────────────────────────────
n_sick = 800

sick_temperature = np.random.normal(39.8, 0.6, n_sick)    # elevated
sick_humidity    = np.random.normal(78,   10,  n_sick)    # high humidity
sick_milk_yield  = np.random.normal(10,   4,   n_sick)    # reduced yield
sick_weight      = np.random.normal(420,  45,  n_sick)    # lower weight
sick_heart_rate  = np.random.normal(85,   10,  n_sick)    # elevated bpm
sick_activity    = np.random.normal(40,   12,  n_sick)    # reduced activity

sick_df = pd.DataFrame({
    'temperature':   sick_temperature,
    'humidity':      sick_humidity,
    'milk_yield':    sick_milk_yield,
    'weight':        sick_weight,
    'heart_rate':    sick_heart_rate,
    'activity_level': sick_activity,
    'disease_label': 1
})

# ─────────────────────────────────────────────
# Combine and shuffle
# ─────────────────────────────────────────────
df = pd.concat([healthy_df, sick_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Clip to realistic physiological bounds
df['temperature']   = df['temperature'].clip(36.5, 42.0)
df['humidity']      = df['humidity'].clip(20, 100)
df['milk_yield']    = df['milk_yield'].clip(0, 40)
df['weight']        = df['weight'].clip(300, 700)
df['heart_rate']    = df['heart_rate'].clip(40, 120)
df['activity_level']= df['activity_level'].clip(5, 100)

# Round to 2 decimal places for realism
df = df.round(2)

# Save
df.to_csv('dataset.csv', index=False)
print(f"Dataset generated: {len(df)} records  |  Healthy={n_healthy}  Sick={n_sick}")
print(df.head())
print(df['disease_label'].value_counts())

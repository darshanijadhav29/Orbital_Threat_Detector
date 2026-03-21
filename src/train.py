import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

merged = pd.read_parquet("/home/darshani/lightkurve-env/space-debris-detector/data/merged/ML_merged.parquet")

# collision risk model
print("training collision risk model...")

merged['BSTAR'] = pd.to_numeric(merged['BSTAR'], errors='coerce')
merged['PERIGEE'] = pd.to_numeric(merged['PERIGEE'], errors='coerce')

def get_risk_label(row):
    if row['PERIGEE'] < 300 and abs(row['BSTAR']) > 0.0001:
        return 'HIGH'
    elif row['PERIGEE'] < 600 and abs(row['BSTAR']) > 0.00001:
        return 'MEDIUM'
    else:
        return 'LOW'

merged['risk_label'] = merged.apply(get_risk_label, axis=1)

risk_features = [
    'ECCENTRICITY', 'INCLINATION', 'MEAN_MOTION', 'MEAN_MOTION_DOT',
    'SEMIMAJOR_AXIS', 'PERIOD', 'APOGEE', 'RCSVALUE',
    'altitude_km', 'speed_km_s', 'attr_mass', 'attr_xSectAvg', 'tle_age_days'
]

risk_features = [f for f in risk_features if f in merged.columns]
X = merged[risk_features].apply(pd.to_numeric, errors='coerce').fillna(0)
y = merged['risk_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
risk_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
risk_model.fit(X_train, y_train)

print(classification_report(y_test, risk_model.predict(X_test), zero_division=0))
joblib.dump(risk_model, "/home/darshani/lightkurve-env/space-debris-detector/models/risk_model.pkl")
joblib.dump(risk_features, "/home/darshani/lightkurve-env/space-debris-detector/models/risk_features.pkl")
print("collision risk model saved!!")

# decay risk model
print("\ntraining decay risk model...")

def decay_risk(row):
    if row['PERIGEE'] < 300 and abs(row['BSTAR']) > 0.0001:
        return 'IMMINENT'
    elif row['PERIGEE'] < 500 and abs(row['BSTAR']) > 0.00001:
        return 'ELEVATED'
    elif row['PERIGEE'] < 800:
        return 'MODERATE'
    else:
        return 'STABLE'

merged['decay_risk'] = merged.apply(decay_risk, axis=1)

decay_features = [
    'ECCENTRICITY', 'INCLINATION', 'MEAN_MOTION', 'MEAN_MOTION_DOT',
    'SEMIMAJOR_AXIS', 'PERIOD', 'APOGEE', 'RCSVALUE',
    'altitude_km', 'speed_km_s', 'attr_mass', 'attr_xSectAvg', 'tle_age_days'
]

decay_features = [f for f in decay_features if f in merged.columns]
X2 = merged[decay_features].apply(pd.to_numeric, errors='coerce').fillna(0)
y2 = merged['decay_risk']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)
decay_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
decay_model.fit(X_train2, y_train2)

print(classification_report(y_test2, decay_model.predict(X_test2), zero_division=0))
joblib.dump(decay_model, "/home/darshani/lightkurve-env/space-debris-detector/models/decay_model.pkl")
joblib.dump(decay_features, "/home/darshani/lightkurve-env/space-debris-detector/models/decay_features.pkl")
print("decay risk model saved!!")
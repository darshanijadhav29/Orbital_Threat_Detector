import pandas as pd
import numpy as np
import joblib

risk_model = joblib.load("/home/darshani/lightkurve-env/space-debris-detector/models/risk_model.pkl")
decay_model = joblib.load("/home/darshani/lightkurve-env/space-debris-detector/models/decay_model.pkl")
risk_features = joblib.load("/home/darshani/lightkurve-env/space-debris-detector/models/risk_features.pkl")
decay_features = joblib.load("/home/darshani/lightkurve-env/space-debris-detector/models/decay_features.pkl")

merged = pd.read_parquet("/home/darshani/lightkurve-env/space-debris-detector/data/merged/ML_merged.parquet")

def predict(norad_id):
    obj = merged[merged['NORAD_CAT_ID'] == norad_id]
    
    if len(obj) == 0:
        print("object not found!!")
        return
    
    obj = obj.iloc[0]
    
    # risk predictions
    X_risk = pd.DataFrame([obj[risk_features]]).apply(pd.to_numeric, errors='coerce').fillna(0)
    X_decay = pd.DataFrame([obj[decay_features]]).apply(pd.to_numeric, errors='coerce').fillna(0)
    
    risk = risk_model.predict(X_risk)[0]
    decay = decay_model.predict(X_decay)[0]
    
    # propulsion calculations
    deltav = obj.get('deltav_km_s', 'not calculated')
    expellant = obj.get('expellant_mass_kg', 'not calculated')
    deorbit = obj.get('deorbit_days', 'not calculated')
    
    print(f"\nobject: {obj['OBJECT_NAME']}")
    print(f"norad id: {norad_id}")
    print(f"type: {obj['OBJECT_TYPE']}")
    print(f"altitude: {round(obj['altitude_km'], 1)} km")
    print(f"\ncollision risk: {risk}")
    print(f"decay risk: {decay}")
    print(f"\ndelta-v to deorbit: {deltav} km/s")
    print(f"xenon expellant needed: {expellant} kg")
    print(f"deorbit time: {deorbit} days")

# test it on ISS
predict(25544)
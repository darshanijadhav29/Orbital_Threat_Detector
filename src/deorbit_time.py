import numpy as np
import pandas as pd

THRUST_N = 0.2
THRUST_KMS2 = THRUST_N / 1000

def deorbit_time(mass_kg, dv_km_s):
    if pd.isna(mass_kg) or pd.isna(dv_km_s):
        return np.nan
    if mass_kg <= 0 or dv_km_s <= 0:
        return np.nan
    time_seconds = (mass_kg * dv_km_s) / THRUST_KMS2
    return round(time_seconds / 86400, 1)

def deorbit_time_for_dataset(df):
    df['attr_mass'] = pd.to_numeric(df['attr_mass'], errors='coerce')
    df['deorbit_days'] = df.apply(
        lambda row: deorbit_time(row['attr_mass'], row['deltav_km_s']), axis=1
    )
    print("deorbit time calculated for", df['deorbit_days'].notna().sum(), "objects")
    return df
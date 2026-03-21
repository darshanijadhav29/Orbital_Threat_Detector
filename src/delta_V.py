import numpy as np
import pandas as pd

MU = 398600
R_EARTH = 6371

def calc_deltav(altitude_km):
    if pd.isna(altitude_km) or altitude_km < 200:
        return np.nan
    r1 = R_EARTH + altitude_km
    r_perigee = R_EARTH + 200
    v_circular = np.sqrt(MU / r1)
    v_transfer = np.sqrt(MU * (2/r1 - 2/(r1 + r_perigee)))
    return round(abs(v_circular - v_transfer), 4)

def deltav_for_dataset(df):
    df['deltav_km_s'] = df['altitude_km'].apply(calc_deltav)
    print("delta-v calculated for", df['deltav_km_s'].notna().sum(), "objects")
    return df
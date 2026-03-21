import numpy as np
import pandas as pd

ISP = 2000
G0 = 0.00981

def expellant_needed(dry_mass_kg, dv_km_s):
    if pd.isna(dry_mass_kg) or pd.isna(dv_km_s):
        return np.nan
    if dry_mass_kg <= 0 or dv_km_s <= 0:
        return np.nan
    mass_ratio = np.exp(dv_km_s / (ISP * G0))
    return round(dry_mass_kg * (mass_ratio - 1), 3)

def expellant_for_dataset(df):
    df['attr_mass'] = pd.to_numeric(df['attr_mass'], errors='coerce')
    df['expellant_mass_kg'] = df.apply(
        lambda row: expellant_needed(row['attr_mass'], row['deltav_km_s']), axis=1
    )
    df['expellant_percent'] = (df['expellant_mass_kg'] / df['attr_mass']) * 100
    print("expellant calculated for", df['expellant_mass_kg'].notna().sum(), "objects")
    return df
import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday
from datetime import datetime, timezone

def propagate(tle_df):
    now = datetime.now(timezone.utc)
    jd, fr = jday(now.year, now.month, now.day, now.hour, now.minute, now.second)
    
    tle_df['EPOCH'] = pd.to_datetime(tle_df['EPOCH'])
    tle_df['tle_age_days'] = (now - tle_df['EPOCH'].dt.tz_localize('UTC')).dt.days
    tle_df['PERIAPSIS'] = pd.to_numeric(tle_df['PERIAPSIS'], errors='coerce')
    
    usable = tle_df[~((tle_df['tle_age_days'] > 30) & (tle_df['PERIAPSIS'] < 200))].copy()
    print("propagating", len(usable), "objects")
    
    rows = []
    for i, row in usable.iterrows():
        try:
            sat = Satrec.twoline2rv(row['TLE_LINE1'], row['TLE_LINE2'])
            err, r, v = sat.sgp4(jd, fr)
            rows.append({
                'NORAD_CAT_ID': row['NORAD_CAT_ID'],
                'error': err,
                'rx_km': r[0], 'ry_km': r[1], 'rz_km': r[2],
                'vx_km_s': v[0], 'vy_km_s': v[1], 'vz_km_s': v[2],
                'tle_age_days': row['tle_age_days']
            })
        except:
            rows.append({
                'NORAD_CAT_ID': row['NORAD_CAT_ID'],
                'error': 999,
                'rx_km': None, 'ry_km': None, 'rz_km': None,
                'vx_km_s': None, 'vy_km_s': None, 'vz_km_s': None,
                'tle_age_days': row['tle_age_days']
            })
    
    sgp4_df = pd.DataFrame(rows)
    sgp4_df = sgp4_df[sgp4_df['error'] == 0].copy()
    sgp4_df['altitude_km'] = np.sqrt(sgp4_df['rx_km']**2 + sgp4_df['ry_km']**2 + sgp4_df['rz_km']**2) - 6371
    sgp4_df['speed_km_s'] = np.sqrt(sgp4_df['vx_km_s']**2 + sgp4_df['vy_km_s']**2 + sgp4_df['vz_km_s']**2)
    sgp4_df['regime'] = pd.cut(sgp4_df['altitude_km'],
        bins=[0, 450, 2000, 35786, 1000000],
        labels=['VLEO', 'LEO', 'MEO', 'GEO'])
    
    return sgp4_df
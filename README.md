# # space debris risk assessment and deorbit planning system

built a system that identifies high risk space debris and calculates how to deorbit them using plasma propulsion.

## what this does

- fetches real orbital data from space-track.org and ESA DISCOS APIs
- propagates 66,000+ objects using SGP4 to get current positions
- predicts collision risk (high/medium/low) using random forest — 95% accuracy
- predicts decay risk — 94% accuracy
- calculates delta-V needed to deorbit each object
- estimates xenon expellant needed for hall thruster deorbit
- calculates deorbit time using plasma propulsion (0.2N hall thruster, Isp 2000s)
- powered by solar panels — zero combustion, no harmful emissions

## example

```
object: ISS (ZARYA)
norad id: 25544
type: PAYLOAD
altitude: 423.3 km

collision risk: MEDIUM
decay risk: ELEVATED

delta-v to deorbit: 0.0642 km/s
xenon expellant needed: 1474.889 kg
deorbit time: 1671.9 days
```

## data sources

- space-track.org → TLE, SATCAT, conjunction, decay, boxscore
- ESA DISCOS → physical properties (mass, shape, cross section)
- 66,666 objects in final merged dataset

## project structure

```
data/
├── cleaned/
├── merged/
└── raw/

notebooks/
├── raw_to_cleaned.ipynb
├── merging.ipynb
├── sgp4_calculation.ipynb
├── tle_orbital_elements.ipynb
├── orbit_simulation.ipynb
├── discos_visuals.ipynb
├── satcat_classification_visuals.ipynb
├── risk_prediction.ipynb
├── decay_prediction.ipynb
├── delta_v_calculator.ipynb
├── expellant_mass.ipynb
└── deorbit_time_calculator.ipynb

src/
├── sgp4_propagate.py
├── deltav.py
├── expellant.py
├── deorbit_time.py
├── train.py
└── predict.py
models/
├── risk_model.pkl
├── decay_model.pkl
├── risk_features.pkl
└── decay_features.pkl
```

## ml models

**collision risk classifier**

- features: eccentricity, inclination, mean motion, period, apogee, altitude, speed, mass
- model: random forest (100 trees, class balanced)
- accuracy: 95%

**decay risk classifier**

- categories: imminent, elevated, moderate, stable
- model: random forest (100 trees, class balanced)
- accuracy: 95%

## propulsion analysis

using hall thruster specs:

- isp: 2000s
- thrust: 0.2N
- expellant: xenon (inert gas, zero combustion)
- power source: solar panels

for each object calculates:

1. delta-V needed (hohmann transfer to 200km perigee)
2. xenon expellant mass (tsiolkovsky rocket equation)
3. time to deorbit (F = ma)

## quick start

```bash
git clone https://github.com/darshanijadhav29/Orbital_Threat_Detector 
cd Orbital_Threat_Detector
pip install pandas numpy matplotlib seaborn scikit-learn sgp4 joblib

# train models
python src/train.py

# predict risk for any object by NORAD ID
python src/predict.py
```

you need a space-track.org account to get data.

## findings

- 66,666 objects tracked across VLEO, LEO, MEO and GEO
- 19,517 objects classified as IMMINENT decay risk
- ISS needs only 0.0642 km/s delta-V to deorbit
- GEO objects need up to 1.48 km/s delta-V
- hall thruster can deorbit small LEO debris in under 100 days

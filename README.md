# # space debris detection and deorbit planning system

built a system that identifies high risk space debris and calculates how to deorbit them using plasma propulsion.

## what this does

it fetches TLE data and satellite catalog info from space-track.org, then pulls physical properties like mass, cross section and shape from ESA DISCOS. one annoying thing i ran into early, space-track uses NORAD IDs and DISCOS uses COSPAR IDs, so i had to corelate the two. there were total 66K+ TLE objects and 88K+ DISCOS objects after merging i ended up with 66,727 objects across VLEO, LEO, MEO and GEO. all of those get propagated to current epoch using SGP4. then two random forest classifiers run on the final dataset called ML_merged one for collision risk (high/medium/low) and one for decay risk (imminent/elevated/moderate/stable). both hit 95% accuracy. for anything that comes out high risk, the system calculates the delta-V needed for a hohmann transfer down to 200km perigee, the xenon propellant mass using tsiolkovsky, and total deorbit time assuming a 0.2N hall thruster with Isp of 2000s. i kept it to electric propulsion the whole way through and no chemical thrusters. chemical propulsion produces combustion byproducts and adds to the debris problem which kind of defeats the point.

## ISS(example)

```
object: ISS (ZARYA)
norad id: 25544
type: PAYLOAD
altitude: 427.3 km

collision risk: MEDIUM
decay risk: ELEVATED

delta-v to deorbit: 0.0654 km/s
xenon expellant needed: 1502.503 kg
deorbit time: 1703.1 days
```

## data sources

space-track.org gave me the TLE data, SATCAT, conjunction records and decay predictions. i also pulled boxscore initially but ended up not using it in the ML, it was more just for getting a sense of how many objects are up there by type. ESA DISCOS handled the physical properties like mass, dimensions, shape, cross section area. total final merged i got was of 66,727 objects in ML(final) merged dataset

## how the projectt looks

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
├── plots
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

## ml models(for calculating risk and decay for satellites)

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
you need a space-track.org account to fetch data free but takes
about a day to get approved. also don't remove the time.sleep()
calls in the fetch notebooks, space-track rate limits pretty
aggressively and you'll get blocked.

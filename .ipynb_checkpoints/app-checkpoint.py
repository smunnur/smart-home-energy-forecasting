import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import numpy as np

# Load saved objects
model = joblib.load('smart_home_energy_forecaster.pkl')
le_appliance = joblib.load('label_encoder_appliance.pkl')
le_season = joblib.load('label_encoder_season.pkl')

st.title("⚡ Smart Home Energy Forecaster")
st.write("Predict hourly energy consumption (kWh) for your household")

# Inputs
appliance = st.selectbox("Select Appliance", le_appliance.classes_)
season = st.selectbox("Select Season", le_season.classes_)
temp_c = st.slider("Outdoor Temperature (°C)", -10.0, 40.0, 20.0)
household = st.slider("Household Size", 1, 5, 3)
time_str = st.time_input("Time of Day", datetime.strptime("12:00","%H:%M").time())
date_str = st.date_input("Date", datetime.today())

# Derived features
hour = time_str.hour
month = date_str.month
day_of_week = date_str.weekday()
is_weekend = int(day_of_week in [5,6])

X_new = pd.DataFrame([[
    le_appliance.transform([appliance])[0],
    temp_c,
    le_season.transform([season])[0],
    household,
    hour,
    month,
    day_of_week,
    is_weekend
]], columns=[
    'Appliance Encoded','Outdoor Temperature (°C)','Season Encoded',
    'Household Size','Hour','Month','DayOfWeek','IsWeekend'
])

if st.button("Predict Energy Usage"):
    pred = model.predict(X_new)[0]
    st.success(f"Predicted Consumption: {pred:.3f} kWh")

    # Hourly forecast visualization
    hours = list(range(24))
    preds = []
    for h in hours:
        X_new.iloc[0,4] = h  # replace hour
        preds.append(model.predict(X_new)[0])
    st.line_chart(pd.DataFrame({'Hour':hours,'Predicted kWh':preds}).set_index('Hour'))

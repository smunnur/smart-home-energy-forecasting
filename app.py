import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# PAGE SETTINGS
# ----------------------------------------------------------
st.set_page_config(
    page_title="Smart Home Energy Forecaster",
    page_icon="⚡",
    layout="wide",
)

# ----------------------------------------------------------
# THEME FUNCTION (DARK MODE VIA CSS VARIABLES)
# ----------------------------------------------------------
def set_theme(dark=False):
    if dark:
        primary_bg = "#0E1117"
        primary_text = "#F5F5F5"
        card_bg = "#161B22"
        border_color = "#30363d"
    else:
        primary_bg = "#FFFFFF"
        primary_text = "#000000"
        card_bg = "#F5F7FA"
        border_color = "#DDDDDD"

    st.markdown(f"""
    <style>
    :root {{
        --primary-bg: {primary_bg};
        --primary-text: {primary_text};
        --card-bg: {card_bg};
        --border-color: {border_color};
    }}

    html, body, [class*="main"] {{
        background-color: var(--primary-bg) !important;
        color: var(--primary-text) !important;
    }}

    .metric-card {{
        background-color: var(--card-bg);
        color: var(--primary-text);
        padding: 18px;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0px 4px 10px rgba(0,0,0,0.12);
        text-align: center;
        margin-bottom: 10px;
    }}

    .section-title {{
        font-size: 22px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 5px;
    }}

    .footer {{
        text-align: center;
        color: gray;
        margin-top: 30px;
        font-size: 14px;
    }}
    </style>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------
# LOAD MODEL & ENCODERS
# ----------------------------------------------------------
model = joblib.load('smart_home_energy_forecaster.pkl')
le_appliance = joblib.load('label_encoder_appliance.pkl')
le_season = joblib.load('label_encoder_season.pkl')

APPLIANCE_EMOJIS = {
    "Fridge": "🧊 Fridge",
    "Oven": "🔥 Oven",
    "Heater": "♨️ Heater",
    "Microwave": "📡 Microwave",
    "Dishwasher": "🧺 Dishwasher",
    "Washer": "🌀 Washer",
    "Dryer": "🌬 Dryer",
}

# ----------------------------------------------------------
# SIDEBAR CONTROLS
# ----------------------------------------------------------
st.sidebar.title("⚙️ Controls")

dark_mode = st.sidebar.toggle("🌙 Dark Mode")
set_theme(dark_mode)

# Cost per kWh for cost estimation
cost_per_kwh = st.sidebar.slider("Cost per kWh (USD)", 0.05, 0.50, 0.15, 0.01)

# Appliance with emojis where possible
appliance_options = []
for a in le_appliance.classes_:
    label = APPLIANCE_EMOJIS.get(a, a)
    appliance_options.append((label, a))

appliance_label = st.sidebar.selectbox(
    "Appliance",
    [opt[0] for opt in appliance_options]
)
# Map label back to true appliance name
appliance = [a for (label, a) in appliance_options if label == appliance_label][0]

season = st.sidebar.selectbox("Season", le_season.classes_)
temperature = st.sidebar.slider("Outdoor Temperature (°C)", -10.0, 40.0, 20.0)
household = st.sidebar.slider("Household Size", 1, 5, 3)
time_val = st.sidebar.time_input("Time of Day", datetime.strptime("12:00", "%H:%M").time())
date_val = st.sidebar.date_input("Date", datetime.today())

st.sidebar.caption("Tip: Try extreme temperatures or different appliances to see how the prediction changes.")

# ----------------------------------------------------------
# TITLE + TABS
# ----------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center; color:#4A90E2;'>⚡ Smart Home Energy Forecasting Dashboard</h1>",
    unsafe_allow_html=True
)
st.write("")

tab_overview, tab_forecast, tab_analysis, tab_about = st.tabs(
    ["📌 Overview", "📊 Forecast", "📈 Analysis", "ℹ️ About"]
)

# ----------------------------------------------------------
# FEATURE ENGINEERING
# ----------------------------------------------------------
hour = time_val.hour
month = date_val.month
day_of_week = date_val.weekday()
is_weekend = int(day_of_week in [5, 6])

X_input = pd.DataFrame([[
    le_appliance.transform([appliance])[0],
    temperature,
    le_season.transform([season])[0],
    household,
    hour,
    month,
    day_of_week,
    is_weekend
]], columns=[
    "Appliance Encoded", "Outdoor Temperature (°C)", "Season Encoded",
    "Household Size", "Hour", "Month", "DayOfWeek", "IsWeekend"
])

# Init session state for history
if "history" not in st.session_state:
    st.session_state["history"] = []

# ----------------------------------------------------------
# PREDICTION
# ----------------------------------------------------------
predict_btn = st.sidebar.button("🔮 Predict Energy Usage")

if predict_btn:
    prediction = float(model.predict(X_input)[0])
    est_cost = prediction * cost_per_kwh

    # Save to history
    st.session_state["history"].append({
        "Appliance": appliance,
        "Season": season,
        "Temp (°C)": temperature,
        "Household": household,
        "Hour": hour,
        "Pred kWh": prediction,
        "Cost ($)": est_cost
    })

    # ---------------- OVERVIEW TAB -----------------
    with tab_overview:
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Predicted Consumption", f"{prediction:.3f} kWh")
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Estimated Cost", f"${est_cost:.2f}")
            st.caption(f"Assuming {cost_per_kwh:.2f} USD per kWh")
            st.markdown("</div>", unsafe_allow_html=True)

        with c3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Household Size", f"{household} person(s)")
            st.caption(f"{'Weekend' if is_weekend else 'Weekday'} · {season}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 🧠 Explanation (High-Level)")
        st.write(
            f"This prediction is based on **{appliance}** usage in a **{household}-person** household, "
            f"at **{hour}:00** during **{season}**, with an outdoor temperature of **{temperature}°C**. "
            f"The model has learned from past data that both **appliance type** and **temperature** "
            f"play an important role in determining how much energy is typically consumed."
        )

        if st.session_state["history"]:
            st.markdown("### 📜 Recent Predictions")
            hist_df = pd.DataFrame(st.session_state["history"])
            st.dataframe(hist_df.tail(10), use_container_width=True)

    # ---------------- FORECAST TAB -----------------
    with tab_forecast:
        st.markdown("<div class='section-title'>Hourly Forecast for the Day</div>", unsafe_allow_html=True)

        hours = list(range(24))
        hourly_preds = []
        for h in hours:
            X_hour = X_input.copy()
            X_hour.iloc[0, 4] = h
            hourly_preds.append(model.predict(X_hour)[0])

        forecast_df = pd.DataFrame({"Hour": hours, "Predicted kWh": hourly_preds}).set_index("Hour")

        if dark_mode:
            plt.style.use("dark_background")
        else:
            plt.style.use("default")

        st.line_chart(forecast_df)

        st.caption("This chart assumes all other conditions stay the same while only the hour changes.")

    # ---------------- ANALYSIS TAB -----------------
    with tab_analysis:
        st.markdown("<div class='section-title'>Compare Appliances Under Same Conditions</div>", unsafe_allow_html=True)

        compare_vals = []
        compare_labels = []
        for a in le_appliance.classes_:
            temp_row = X_input.copy()
            temp_row.iloc[0, 0] = le_appliance.transform([a])[0]
            compare_vals.append(model.predict(temp_row)[0])
            compare_labels.append(APPLIANCE_EMOJIS.get(a, a))

        compare_df = pd.DataFrame({
            "Appliance": compare_labels,
            "Predicted kWh": compare_vals
        }).set_index("Appliance")

        st.bar_chart(compare_df)

        st.caption(
            "This comparison shows how different appliances would behave under the **same** household, time, "
            "temperature, and seasonal conditions."
        )

        if st.session_state["history"]:
            st.markdown("<div class='section-title'>History Summary</div>", unsafe_allow_html=True)
            hist_df = pd.DataFrame(st.session_state["history"])
            st.write("Average predicted kWh from this session:", round(hist_df["Pred kWh"].mean(), 3))

    # ---------------- ABOUT TAB -----------------
    with tab_about:
        st.markdown("### ℹ️ About This App")
        st.write("""
        This dashboard is part of a semester project on **AI for Smart Home Energy Prediction**.
        It uses a machine learning model trained on smart home appliance usage data
        to estimate energy consumption in kilowatt-hours (kWh).

        **Key Features:**
        - Predicts energy usage based on appliance, season, temperature, time, and household size.
        - Provides cost estimation based on configurable electricity price.
        - Visualizes hourly energy forecasts for the selected day.
        - Compares different appliances under the same conditions.
        - Maintains a local prediction history for the current session.
        """)

        st.write("""
        **Technical Stack:**
        - Python, scikit-learn, joblib
        - Streamlit for the web interface
        - Trained model: Regressor over engineered features (Appliance, Season, Time, etc.)
        """)

    # ---------------- DOWNLOAD (BOTTOM OF OVERVIEW TAB) -----------------
    with tab_overview:
        st.markdown("### ⬇️ Download Current Prediction")
        out_df = pd.DataFrame([{
            "Appliance": appliance,
            "Season": season,
            "Temp (°C)": temperature,
            "Household": household,
            "Hour": hour,
            "Predicted kWh": prediction,
            "Cost ($)": est_cost
        }])
        st.download_button("Download as CSV", out_df.to_csv(index=False), "prediction_record.csv")

    # ---------------- FOOTER -----------------
    st.markdown("<div class='footer'>Built with ❤️ using Streamlit & Machine Learning</div>", unsafe_allow_html=True)

else:
    with tab_overview:
        st.info("Use the controls in the left sidebar and click **🔮 Predict Energy Usage** to get started.")
    with tab_forecast:
        st.info("A forecast will appear here after you generate a prediction.")
    with tab_analysis:
        st.info("Analysis and comparisons will appear here after you generate a prediction.")
    with tab_about:
        st.markdown("This tab explains the project once a prediction has been run at least once.")

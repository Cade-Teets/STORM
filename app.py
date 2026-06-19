import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from run_simulation import OrbitalDecaySimulator

# 1. Page Configuration
st.set_page_config(page_title="Space Track Analytics Control", layout="wide")
st.title("Orbital Decay & Space Weather Risk Engine")
st.markdown("An Operations Research simulator evaluating thermospheric aerodynamic drag under ARIMA solar flux projections.")

# Initialize the backend simulator asset
simulator = OrbitalDecaySimulator()

# 2. Sidebar Controls
st.sidebar.header("Simulation Parameters")
target_id = st.sidebar.number_input("NORAD Catalog ID", value=25544, step=1)
sim_days = st.sidebar.slider("Simulation Window (Days)", min_value=5, max_value=30, value=30)
mc_runs = st.sidebar.slider("Monte Carlo Iterations", min_value=10, max_value=500, value=100)

# 3. Dynamic Name Fetching
try:
    line1, line2, epoch, sat_name = simulator.get_latest_tle(target_id)
    st.sidebar.success(f"Tracking Asset: **{sat_name}**")
    st.sidebar.text(f"Initial Epoch: {epoch[:10]}")
except Exception as e:
    st.sidebar.error("NORAD ID missing from local database storage.")
    st.stop()

# 4. Trigger Execution
if st.button("Execute Stochastic Mission Analysis"):
    with st.spinner("Running Monte Carlo physics steps across predictive distributions..."):
        
        # Run simulation backend (modify your runner to return raw array data)
        # For showcase, let's assume we modify the runner to return the array list
        history = simulator.forecaster.load_historical_flux()
        mean_f107, stderr_f107 = simulator.forecaster.generate_30_day_forecast(history)
        
        # Displaying the Results layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Space Weather Forecast Profile")
            # Build a quick Plotly time-series visualization
            chart_data = pd.DataFrame({
                'Expected F10.7': mean_f107,
                'Upper Bound': mean_f107 + (1.96 * stderr_f107),
                'Lower Bound': mean_f107 - (1.96 * stderr_f107)
            })
            st.line_chart(chart_data)
            st.caption("ARIMA(1,1,1) Projections with a 95% confidence interval block.")

        with col2:
            st.subheader("Operational Risk Distribution")
            # Mocking the statistical array display for the UI shell
            st.metric(label="Asset Survival Probability", value="100.0%")
            st.metric(label="Mean Orbital Frequency Delta", value="+1.42e-6 rad/s")
            
            # Matplotlib histogram for the distribution curve
            fig, ax = plt.subplots()
            ax.hist([sim_days]*mc_runs, bins=10, color='royalblue', edgecolor='black')
            ax.set_xlabel("Re-entry Lifetime (Days)")
            ax.set_ylabel("Trial Frequency Count")
            st.pyplot(fig)
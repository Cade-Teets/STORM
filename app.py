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
            st.line_chart(
                chart_data, 
                x_label="Forecast Date", 
                y_label="F10.7 Solar Flux (sfu)"
            )
            st.caption("ARIMA(1,1,1) Projections with a 95% confidence interval block.")

        with col2:
            st.subheader("Operational Risk Distribution")
            
            # Run the actual Monte Carlo engine to get real statistical data
            results = simulator.run_monte_carlo(norad_id=target_id, num_runs=mc_runs, sim_days=sim_days)
            
            # Calculate real summary metrics
            survived_count = results.count(sim_days)
            survival_rate = (survived_count / len(results)) * 100
            
            crashed_runs = [r for r in results if r < sim_days]
            if crashed_runs:
                avg_lifetime = sum(crashed_runs) / len(crashed_runs)
                avg_lifetime_str = f"Day {avg_lifetime:.1f}"
            else:
                avg_lifetime_str = f"Survived >{sim_days} Days"
                
            st.metric(label="Asset Survival Probability", value=f"{survival_rate:.1f}%")
            st.metric(label="Average Lifetime (Crashed Paths)", value=avg_lifetime_str)
            
            # Matplotlib histogram for the distribution curve
            fig, ax = plt.subplots()
            ax.hist(results, bins=15, range=(0, sim_days), color='royalblue', edgecolor='black')
            ax.set_xlabel("Re-entry Lifetime (Days)")
            ax.set_ylabel("Trial Frequency Count")
            st.pyplot(fig)
# Satellite Tracking and Orbital Risk Model (STORM) 🛰️⛈️
### *An AeroDecay-MC Stochastic Simulation Engine*

An end-to-end data pipeline and numerical physics simulation suite designed to predict Low Earth Orbit (LEO) satellite decay trajectories. By coupling time-series statistical modeling with standard orbital propagation algorithms, STORM evaluates satellite re-entry profiles against dynamic atmospheric drag variations driven by unpredictable space weather anomalies.

---

## 📊 Core Architecture & Framework

STORM bypasses generic static vacuum constraints by pulling live environmental telemetry from external tracking networks and running localized numerical integrations.

1. **Automated Telemetry Ingestion Engine:** Connects directly via the Space-Track API to pull the latest satellite General Perturbations (GP) datasets (TLEs). Simultaneously, it ingests decades of historical daily solar flux index ($F_{10.7}$) records from CelesTrak's archives, ensuring statistical models are trained on chronologically accurate solar cycle data.
2. **Predictive Time-Series Forecaster:** Trains an Auto Regressive Integrated Moving Average (ARIMA) time-series forcasting model using historic solar flux histories to project a 30-day space weather baseline, complete with a dynamically expanding standard error band representing atmospheric uncertainty.
3. **Dynamic Deceleration Core ($a_D$):** Bypasses the static constraints of SGP4's BSTAR parameter by calculating orbital decay using a true ballistic coefficient. Atmospheric density is evaluated dynamically via a **13-layer US Standard Atmosphere (1976)** piecewise exponential model. This properly scales scale-height ($H$) and air resistance as satellites plunge deep into the thermosphere (below 200km). Daily orbital kinetic energy drainage is then computed via the first-principles aerodynamic drag equation:

$$a_D = -\frac{1}{2}\rho \left(\frac{C_D A}{m}\right) v^2$$

4. **Stochastic Monte Carlo Analyzer:** Executes parallel simulation iterations ($N=100$) where daily solar weather variables are randomly sampled from the widening ARIMA forecasting variance distribution to map structural asset survival probabilities.

---

## 💻 Operations Control Dashboard

The engine is wrapped in an interactive, web-based **Streamlit UI**, serving as a predictive command room for operations research analytics.

* **Asset Verification:** Dynamically queries the localized database to display human-readable tracking names (e.g., *ISS (ZARYA)*) and current epoch metadata based on user-inputted NORAD IDs.
* **Risk Mapping:** Plots live 30-day ARIMA confidence bounds and outputs a trial frequency histogram depicting exactly when re-entry risk spikes occur.

---

## 🚀 Installation & Local Deployment

### 1. Prerequisites & Environment Setup
Clone the repository and initialize a localized virtual environment environment to segregate dependency versions:

```bash
git clone [https://github.com/yourusername/satellite-tracking-orbital-risk-model.git](https://github.com/yourusername/satellite-tracking-orbital-risk-model.git)
cd satellite-tracking-orbital-risk-model
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Running the CLI Simulation Engine
STORM can be run directly from the terminal. The engine relies on an internal `ASSET_DB` dictionary inside `run_simulation.py` to securely look up specific spacecraft physical properties (Mass, Area, $C_D$) based on their NORAD ID.

To run a clean deterministic baseline (using the mean solar forecast):
```bash
python run_simulation.py --id 25544 --mode deterministic --days 30
```

To run a full stochastic Monte Carlo decay analysis:
```bash
python run_simulation.py --id 37820 --mode stochastic --days 60
```
*(Note: To simulate a new asset, add its physical parameters to the `ASSET_DB` dictionary first).*

### 3. Launching the Web Dashboard (Streamlit)
To visualize the decay predictions and Monte Carlo confidence intervals via the interactive web UI, run:
```bash
streamlit run app.py
```
This will automatically open the Operations Control Dashboard in your default browser at `http://localhost:8501`.

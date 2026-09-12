import sqlite3
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

class SolarForecaster:
    def __init__(self, db_path: str = "satellite_data.db"):
        self.db_path = db_path

    def load_historical_flux(self, cutoff_date: str = None) -> pd.Series:
        """Pulls historical F10.7 data from the DB and sets up a Pandas time-series.
            If cutoff_date is given, only data on or before that date is used.
        """
        conn = sqlite3.connect(self.db_path)

        # Query data sorted chronologically
        # Add cutoff date to query if provided
        if cutoff_date:
            query = "SELECT date, radio_flux FROM weather_history WHERE date >= ? ORDER BY date ASC"
            df = pd.read_sql_query(query, conn, params=(cutoff_date,)) 
        else:
            query = "SELECT date, radio_flux FROM weather_history ORDER BY date ASC"
            df = pd.read_sql_query(query, conn)
        
        conn.close()

        # Convert date column to datetime objects and set it as the index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Ensure regular daily frequency
        df = df.asfreq('D')

        # Clean any gaps using a forward-fill imputation
        df['radio_flux'] = df['radio_flux'].ffill()
        
        return df['radio_flux']

    def generate_30_day_forecast(self, series: pd.Series, steps: int = 30):
        """Fits an ARIMA model and returns the forecast with prediction intervals."""
        # Fit the ARIMA model (p=1, d=1, q=1 is a solid baseline for daily solar flux trends)
        model = ARIMA(series, order=(1, 1, 1))
        model_fit = model.fit()

        # Get forecast steps ahead
        forecast_res = model_fit.get_forecast(steps=steps)

        # Extract expected mean values and the variance bounds
        forecast_mean = forecast_res.summary_frame()['mean']
        forecast_stderr = forecast_res.summary_frame()['mean_se']

        return forecast_mean, forecast_stderr

if __name__ == "__main__":
    forecaster = SolarForecaster()
    history = forecaster.load_historical_flux()
    
    if not history.empty:
        mean, stderr = forecaster.generate_30_day_forecast(history)
        
        print("\n--- 30-DAY SOLAR FLUX FORECAST ---")
        for i in range(5):  # Print just the first 5 days to verify it works
            target_date = mean.index[i].strftime('%Y-%m-%d')
            print(f"Date: {target_date} | Expected F10.7: {mean.iloc[i]:.2f} sfu | Uncertainty (StdErr): +/- {stderr.iloc[i]:.2f}")
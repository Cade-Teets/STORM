import numpy as np
import pandas as pd
import logging
import sqlite3
from sgp4.api import Satrec
from sat_physics import altitude_from_tle
from run_simulation import OrbitalDecaySimulator

logging.basicConfig(level=logging.INFO, format='%(message)s')

class StormValidator:
    def __init__(self):
        self.simulator = OrbitalDecaySimulator()
        
    def get_backtest_tle(self, norad_id: int, days_before_decay: int):
        """Pulls a historical TLE from exactly N days before the asset's final recorded epoch."""
        conn = sqlite3.connect(self.simulator.db_path)
        cursor = conn.cursor()
        
        # 1. Find the final date the satellite existed
        cursor.execute("SELECT MAX(epoch) FROM gp_history WHERE norad_cat_id = ?", (norad_id,))
        final_epoch = cursor.fetchone()[0]
        
        # 2. Query the TLE from N days prior
        query = """
            SELECT tle_line1, tle_line2, epoch, object_name 
            FROM gp_history 
            WHERE norad_cat_id = ? 
            AND date(epoch) <= date(?, ?) 
            ORDER BY epoch DESC LIMIT 1;
        """
        offset_string = f"-{days_before_decay} days"
        cursor.execute(query, (norad_id, final_epoch, offset_string))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise ValueError(f"Could not find a TLE {days_before_decay} days prior to decay.")
        return row[0], row[1], row[2], row[3], final_epoch

    def backtest_historical_decay(self, norad_id: int, known_reentry_day: int):
        logging.info(f"Initiating STORM Backtest for NORAD ID {norad_id}")
        
        # Pull the TLE from exactly 25 days out
        line1, line2, epoch_str, sat_name, final_epoch = self.get_backtest_tle(norad_id, known_reentry_day)
        actual_days_before_decay = (pd.to_datetime(final_epoch)- pd.to_datetime(epoch_str)).days
        
        logging.info(f"Backtest Starting Line: {sat_name} | Epoch: {epoch_str}")
        
        # OVERRIDE the simulator's TLE fetcher just for this run so it uses our custom lines
        self.simulator.get_latest_tle = lambda norad_id=25544: (line1, line2, epoch_str, sat_name)
        
        # Run the engine
        predicted_reentry_days = self.simulator.run_monte_carlo(
            norad_id=norad_id, 
            num_runs=100, 
            sim_days=40 
        )
        
        # Calculate the statistical summary for 95% confidence interval
        mean_prediction = np.mean(predicted_reentry_days)
        lower_bound = np.percentile(predicted_reentry_days, 2.5)
        upper_bound = np.percentile(predicted_reentry_days, 97.5)
        error_days = mean_prediction - actual_days_before_decay
        captured_in_interval = lower_bound <= known_reentry_day <= upper_bound
        
        print("\n" + "="*45)
        print("STORM MODEL VALIDATION REPORT")
        print("="*45)
        print(f"Target Known Re-entry: Day {known_reentry_day}")
        print(f"STORM Mean Prediction: Day {mean_prediction:.1f}")
        print(f"STORM 95% Interval:    [Day {lower_bound:.1f} - Day {upper_bound:.1f}]")
        print("-" * 45)
        print(f"Absolute Error:        {abs(error_days):.1f} Days")
        print(f"Interval Capture:      {'SUCCESS' if captured_in_interval else 'FAILED'}")
        print("="*45 + "\n")

    def check_real_decay_rate(self, norad_id: int, min_gap_days: int = 5):
        """Compares altitude derived from two real, well-separated historical TLEs
        to get a ground-truth decay rate, independent of any simulation."""
        conn = sqlite3.connect(self.simulator.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT tle_line1, tle_line2, epoch FROM gp_history
            WHERE norad_cat_id = ? ORDER BY epoch ASC
        """, (norad_id,))
        rows = cursor.fetchall()
        conn.close()

        if len(rows) < 2:
            raise ValueError("Not enough TLEs to compute a real decay rate.")

        first = rows[0]
        # find a later TLE at least min_gap_days after the first
        first_epoch = pd.to_datetime(first[2])
        target = None
        for row in rows[1:]:
            gap = (pd.to_datetime(row[2]) - first_epoch).total_seconds() / 86400
            if gap >= min_gap_days:
                target = row
                break

        if target is None:
            raise ValueError(f"No TLE found at least {min_gap_days} days after the first.")

        sat1 = Satrec.twoline2rv(first[0], first[1])
        sat2 = Satrec.twoline2rv(target[0], target[1])

        alt1 = altitude_from_tle(sat1)
        alt2 = altitude_from_tle(sat2)
        elapsed_days = (pd.to_datetime(target[2]) - first_epoch).total_seconds() / 86400
        real_drop_per_day = (alt1 - alt2) / elapsed_days

        print(f"TLE 1: {first[2]} | Alt: {alt1:.2f} km")
        print(f"TLE 2: {target[2]} | Alt: {alt2:.2f} km")
        print(f"Elapsed: {elapsed_days:.1f} days")
        print(f"Real observed decay rate: {real_drop_per_day*1000:.1f} m/day")
        return real_drop_per_day


if __name__ == "__main__":
    validator = StormValidator()
    # validator.check_real_decay_rate(norad_id=25544)
    
    # Run the Tiangong-1 Backtest
    validator.backtest_historical_decay(norad_id=37820, known_reentry_day=25)
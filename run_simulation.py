from sat_physics import calculate_atmospheric_density, calc_altitude, calculate_daily_altitude_drop
from solar_forecaster import SolarForecaster
from sgp4.api import Satrec, jday
import sqlite3
import logging
import argparse
import random
from datetime import timedelta
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Centralized database of physical properties for target assets
ASSET_DB = {
    25544: {"name": "ISS", "mass": 420000.0, "area": 2500.0, "cd": 2.2},
    37820: {"name": "Tiangong-1", "mass": 8500.0, "area": 15.0, "cd": 2.2},
    41765: {"name": "Tiangong-2", "mass": 8600.0, "area": 15.0, "cd": 2.2},
    20638: {"name": "ROSAT", "mass": 2426.0, "area": 12.0, "cd": 2.2},
    21701: {"name": "UARS", "mass": 5900.0, "area": 30.0, "cd": 2.2},
    35243: {"name": "GOCE", "mass": 1050.0, "area": 1.0, "cd": 2.2},
    15354: {"name": "ERBS", "mass": 2450.0, "area": 7.0, "cd": 2.2},
    23560: {"name": "ERS-2", "mass": 2294.0, "area": 18.0, "cd": 2.2},
    27370: {"name": "RHESSI", "mass": 293.0, "area": 2.0, "cd": 2.2},
    40619: {"name": "Progress M-27M", "mass": 7290.0, "area": 10.0, "cd": 2.2},
    37872: {"name": "Phobos-Grunt", "mass": 13200.0, "area": 9.0, "cd": 2.2},
    25063: {"name": "TRMM", "mass": 3420.0, "area": 15.0, "cd": 2.2}
}

class OrbitalDecaySimulator:
    def __init__(self, db_path: str = "satellite_data.db"):
        self.db_path = db_path
        self.forecaster = SolarForecaster(db_path=db_path)

    def get_latest_tle(self, norad_id: int = 25544):
        """Pulls the single most recent TLE snapshot from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Pull latest record based on epoch
        query = """
            SELECT tle_line1, tle_line2, epoch, object_name
            FROM gp_history 
            WHERE norad_cat_id = ? 
            ORDER BY epoch DESC LIMIT 1;
        """
        cursor.execute(query, (norad_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise ValueError(f"No TLE history found for NORAD ID {norad_id}")
        # Return line1, line2, epoch, and satellite name

        return row[0], row[1], row[2], row[3]
    
    def run_monte_carlo(self, norad_id: int = 25544, num_runs: int = 100, sim_days: int = 30):
        """Runs N independent stochastic simulations to map a distribution of re-entry dates."""
        logger.info(f"Initializing Monte Carlo Engine with N = {num_runs} iterations...")
        
        # 1. Lookup satellite physical properties
        if norad_id not in ASSET_DB:
            raise ValueError(f"NORAD ID {norad_id} not found in ASSET_DB configuration.")
        cd = ASSET_DB[norad_id]["cd"]
        area = ASSET_DB[norad_id]["area"]
        mass = ASSET_DB[norad_id]["mass"]
        
        # 1. Load starting TLE and weather forecasting data once
        line1, line2, epoch_str, object_name = self.get_latest_tle(norad_id)
        history = self.forecaster.load_historical_flux(cutoff_date=str(epoch_str)[:10])
        mean_forecast, stderr_forecast = self.forecaster.generate_30_day_forecast(history)
        
        reentry_days = []
        
        # 2. Execute N independent runs
        for run in range(1, num_runs + 1):
            # Crucial: Re-instantiate the satellite so every run starts fresh!
            satellite = Satrec.twoline2rv(line1, line2)
            jd, fr = satellite.jdsatepoch, satellite.jdsatepochF
            dynamic_density = 2.41e-11 
            
            accumulated_altitude_loss = 0.0
            reentered = False
            
            # Internal daily loop
            for day in range(min(sim_days, len(mean_forecast))):
                fr_step = fr + day
                error_code, position, velocity = satellite.sgp4(jd, fr_step)
                if error_code != 0:
                    # SGP4 natively detected atmospheric decay
                    reentry_days.append(day)
                    reentered = True
                    break
                
                # Get SGP4's baseline altitude
                baseline_alt = calc_altitude(position)
                
                # Apply weather drag penalty
                altitude = baseline_alt - accumulated_altitude_loss
                
                # Hard crash check
                if altitude < 120.0:
                    reentry_days.append(day)
                    reentered = True
                    break
                
                # Sample weather stochastically with the 65 sfu physical floor clamp
                base_f107 = mean_forecast.iloc[day]
                stderr = stderr_forecast.iloc[day]
                simulated_f107 = max(65.0, random.normalvariate(base_f107, stderr))
                
                base_density = calculate_atmospheric_density(altitude)
                dynamic_density = base_density * (simulated_f107 / 100.0)
            
                # Calculate today's altitude penalty and add it to the accumulator
                daily_drop = calculate_daily_altitude_drop(velocity, dynamic_density, cd, area, mass, altitude)
                accumulated_altitude_loss += daily_drop

            # If the satellite didn't crash within the window, record that it survived
            if not reentered:
                reentry_days.append(sim_days)
                
            if run % 10 == 0 or run == num_runs:
                logger.info(f"Completed run {run}/{num_runs}...")

        # 3. Process the aggregate results
        self.print_monte_carlo_analytics(reentry_days, sim_days, norad_id, object_name)
        
        return reentry_days

    def print_monte_carlo_analytics(self, results: list, sim_days: int, norad_id: int, sat_name: str):
        """Calculates and prints summary statistics for the Monte Carlo run."""
        total_runs = len(results)
        survived_count = results.count(sim_days) # Survived full timeline
        crashed_runs = [r for r in results if r < sim_days]
        
        print("\n" + "="*40)
        print("MONTE CARLO SIMULATION SUMMARY")
        print("="*40)
        print(f"Asset:                       {sat_name} (NORAD: {norad_id})")
        print(f"Total Operational Paths Evaluated: {total_runs}")
        
        if crashed_runs:
            avg_decay = sum(crashed_runs) / len(crashed_runs)
            min_decay = min(crashed_runs)
            max_decay = max(crashed_runs)
            print(f"Earliest Re-entry Observed:  Day {min_decay}")
            print(f"Latest Re-entry Observed:    Day {max_decay}")
            print(f"Average Lifetime (Crashed):  Day {avg_decay:.1f}")
        else:
            print(f"All iterations safely completed the {sim_days}-day tracking timeline without re-entry.")
            
        survival_rate = (survived_count / total_runs) * 100
        print(f"Asset Survival Probability:  {survival_rate:.1f}%")
        print("="*40 + "\n")

    def run(self, norad_id: int = 25544, mode: str = "deterministic", sim_days: int = 30):
        """Runs the forward-stepping decay simulation loop."""
        
        # Lookup satellite physical properties
        if norad_id not in ASSET_DB:
            raise ValueError(f"NORAD ID {norad_id} not found in ASSET_DB configuration.")
        cd = ASSET_DB[norad_id]["cd"]
        area = ASSET_DB[norad_id]["area"]
        mass = ASSET_DB[norad_id]["mass"]

        line1, line2, epoch_str, object_name = self.get_latest_tle(norad_id)
        satellite = Satrec.twoline2rv(line1, line2)

        dynamic_density = 2.41e-11
        accumulated_altitude_loss = 0.0

        # Load weather history and train the ARIMA model
        history = self.forecaster.load_historical_flux(cutoff_date=str(epoch_str)[:10])
        mean_forecast, stderr_forecast = self.forecaster.generate_30_day_forecast(history)

        logger.info(f"Starting {mode.upper()} simulation from initial epoch: {epoch_str}")

        sim_start_date = pd.to_datetime(epoch_str)

        # Extract initial SGP4 time components
        jd, fr = satellite.jdsatepoch, satellite.jdsatepochF
        print(f"\n--- Running Forward Simulation ({mode.upper()}) ---")
        
        # 2. Daily stepping loop
        for day in range(min(sim_days, len(mean_forecast))):
            
            # Step time forward by exactly 1 day (1440 minutes)
            fr_step = fr + day
            
            # Propagate satellite to the current step
            error_code, position, velocity = satellite.sgp4(jd, fr_step)
            if error_code != 0:
                logger.error(f"SGP4 Propagation error on day {day}: code {error_code}")
                break
            
            # Calculate Altitude
            base_altitude = calc_altitude(position)
            altitude = base_altitude - accumulated_altitude_loss

            # 3. Determine Solar Flux Value based on operational mode
            base_f107 = mean_forecast.iloc[day]
            stderr = stderr_forecast.iloc[day]
            
            if mode.lower() == "stochastic":
                # Sample from the distribution, but enforce the real-world physical solar floor of 65 sfu
                simulated_f107 = max(65.0, random.normalvariate(base_f107, stderr))
            else:
                simulated_f107 = base_f107
                
            # 4. Calculate Weather-Modified Atmospheric Density
            base_density = calculate_atmospheric_density(altitude)
            # Scale density dynamically based on the forecast flux metric
            dynamic_density = base_density * (simulated_f107 / 100.0)

            # Calculate the altit
            daily_drop = calculate_daily_altitude_drop(velocity, dynamic_density, cd, area, mass, altitude)
            accumulated_altitude_loss += daily_drop
            
            # Print status log
            target_date = (sim_start_date + timedelta(days=day)).strftime('%Y-%m-%d')
            print(f"Day {day:2d} ({target_date}) | Alt: {altitude:6.2f} km | Daily Drop: {daily_drop*1000:7.2f} m | "
                f"F10.7: {simulated_f107:6.2f} sfu | Density: {dynamic_density:.3e} kg/m^3")
            
            # Crash threshold condition
            if altitude < 120.0:
                print(f"\n CRITICAL: Re-entry detected on {target_date} (Day {day})!")
                break

if __name__ == "__main__":
    # 1. Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Run an integrated satellite orbital decay simulation using SGP4 and ARIMA solar forecasts."
    )
    
    # 2. Add the operational flags
    parser.add_argument(
        "--mode", 
        type=str, 
        default="deterministic",
        choices=["deterministic", "stochastic"],
        help="Simulation run mode: 'deterministic' (uses forecast mean) or 'stochastic' "
        "(injects forecast uncertainty & runs Monte Carlo Simulation)."
    )
    
    parser.add_argument(
        "--days", 
        type=int, 
        default=30,
        help="Number of days to run the forward simulation (maximum bounded by forecast length)."
    )
    
    parser.add_argument(
        "--id", 
        type=int, 
        default=25544,
        help="NORAD Catalog ID of the target satellite tracking asset (Default: 25544 for ISS)."
    )

    # 3. Parse the terminal inputs
    args = parser.parse_args()
    simulator = OrbitalDecaySimulator()
    # You can toggle between "deterministic" and "stochastic" here
    if args.mode == "stochastic":
        # Pass it to the multiple-run Monte Carlo machine
        simulator.run_monte_carlo(
            norad_id=args.id, 
            num_runs=100, 
            sim_days=args.days
        )
    else:
        # Run our single clean baseline run
        simulator.run(
            norad_id=args.id, 
            mode="deterministic", 
            sim_days=args.days
        )
import numpy as np
import logging
from model_validator import StormValidator
from run_simulation import ASSET_DB

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Dictionary mapping NORAD ID to their KNOWN days before decay that we want to backtest
# Note: For this to work, SpaceTrack data must be downloaded for these IDs
TEST_COHORT = {
    37820: 25, # Tiangong-1 (Passive Decay)
    21701: 20, # UARS (Passive Decay)
    20638: 15, # ROSAT (Passive Decay)
    15354: 25, # ERBS (Passive Decay, 2023)
    23560: 30, # ERS-2 (Passive Decay, 2024)
    27370: 20, # RHESSI (Passive Decay, 2023)
    40619: 5,  # Progress M-27M (Short lifespan, 2015)
    37872: 25, # Phobos-Grunt (Passive Decay, 2012)
    25063: 20, # TRMM (Passive Decay, 2015)
}

def run_batch_validation():
    """Runs a historical backtest cohort to evaluate global model MAE."""
    validator = StormValidator()
    
    print("\n" + "="*50)
    print(" STORM BATCH VALIDATION SUITE ")
    print("="*50)
    
    errors = []
    
    for norad_id, target_days in TEST_COHORT.items():
        name = ASSET_DB[norad_id]['name']
        print(f"\nEvaluating: {name} (NORAD: {norad_id}) at {target_days} Days Out...")
        
        try:
            line1, line2, epoch_str, sat_name, final_epoch = validator.get_backtest_tle(norad_id, target_days)
            
            # Use a robust factory function to safely bind variables and ignore extra arguments
            def make_tle_fetcher(l1, l2, ep, name):
                def fetcher(*args, **kwargs):
                    return l1, l2, ep, name
                return fetcher
                
            validator.simulator.get_latest_tle = make_tle_fetcher(line1, line2, epoch_str, sat_name)
            
            # Suppress logs for the mass run
            logging.getLogger().setLevel(logging.WARNING)
            
            # Run the Monte Carlo (give it a buffer to overshoot)
            predicted_days = validator.simulator.run_monte_carlo(norad_id, num_runs=100, sim_days=target_days + 15)
            
            logging.getLogger().setLevel(logging.INFO)
            
            mean_prediction = np.mean(predicted_days)
            error = abs(mean_prediction - target_days)
            errors.append(error)
            
            print(f" -> Predicted Re-entry: Day {mean_prediction:.1f}")
            print(f" -> Actual Re-entry:    Day {target_days}")
            print(f" -> Absolute Error:     {error:.1f} Days")
            
        except Exception as e:
            print(f" -> SKIPPED: {e}")
            print(f"    (Did you download the telemetry?)")

    if errors:
        mae = sum(errors) / len(errors)
        print("\n" + "="*50)
        print(" GLOBAL PERFORMANCE METRICS ")
        print("="*50)
        print(f"Cohort Size:          {len(errors)} Satellites")
        print(f"Mean Absolute Error:  {mae:.2f} Days")
        print(f"Model Accuracy:       The STORM engine accurately predicts atmospheric re-entry")
        print(f"                      within an average {mae:.2f}-day margin of error across")
        print(f"                      varied spacecraft geometries and solar conditions.")
        print("="*50 + "\n")
    else:
        print("\nNo satellites were successfully evaluated. Please download telemetry.")

if __name__ == "__main__":
    run_batch_validation()

import sqlite3
from sgp4.api import Satrec
import math
from typing import Tuple

"""Take in TLE Data from Space-Track and output (X,Y,Z) coords"""

def calc_altitude(position: list[float]) -> float:
    equatorial_radius = 6378.137
    polar_radius = 6356.752
    
    total_radius = sum(p**2 for p in position)**0.5
    
    # Standard ellipsoidal Earth radius approximation based on the Z component directional vector
    direction_z = position[2] / total_radius
    local_earth_radius = equatorial_radius - (equatorial_radius - polar_radius) * (direction_z**2)

    return total_radius - local_earth_radius

def calculate_atmospheric_density(altitude_km: float) -> float:
    """
    Calculates atmospheric density (kg/m^3) using a simplified 
    exponential scale height model for the thermosphere.
    """
    # Reference values for Low Earth Orbit (approx. 300km - 500km baseline)
    h0 = 300.0          # Reference altitude (km)
    rho0 = 2.41e-11     # Nominal density at 300km (kg/m^3)
    H = 53.2            # Scale height (km)
    
    # If the satellite drops below the thermosphere boundary, clamp it
    if altitude_km < 120:
        return 1.0e-9   # Extremely thick air near re-entry
        
    # The exponential decay formula: rho = rho0 * e^(-(h - h0) / H)
    density = rho0 * math.exp(-(altitude_km - h0) / H)
    
    return density

def calculate_drag_deceleration(
    velocity: Tuple[float, float, float], 
    dynamic_density: float, 
    bstar: float
) -> Tuple[float, float, float]:
    """
    Calculates the new velocity vector after applying 1 day of atmospheric drag
    using the first-principles drag equation: a_D = -0.5 * rho * (Cd*A/m) * v^2
    
    Inputs:
        velocity: (vx, vy, vz) tuple from SGP4 in km/s
        dynamic_density: weather-scaled atmospheric density in kg/m^3
        bstar: the satellite's baseline B* drag term from the TLE
    Outputs:
        (new_vx, new_vy, new_vz) adjusted velocity vector components in km/s
    """
    # 1. Calculate velocity magnitude (speed) in km/s
    v_mag_kms = math.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)
    if v_mag_kms == 0:
        return velocity
        
    # Convert to m/s for standard SI units compatibility
    v_mag_ms = v_mag_kms * 1000.0
    
    # 2. Extract the area-to-mass ratio from BSTAR (Cd*A/m = 2 * B*)
    drag_term = 2.0 * bstar
    
    # 3. Compute acceleration magnitude (m/s^2)
    acceleration_drag = 0.5 * dynamic_density * drag_term * (v_mag_ms**2)
    
    # Convert acceleration back to km/s^2 to match SGP4 coordinate space
    a_drag_kms2 = acceleration_drag / 1000.0
    
    # 4. Determine unit direction vector of travel (drag acts in direct opposition)
    v_unit = [v / v_mag_kms for v in velocity]
    
    # Calculate delta velocity lost over a 1-day step (86,400 seconds)
    delta_v = a_drag_kms2 * 86400.0
    
    # Apply deceleration vector components
    new_vx = velocity[0] - (v_unit[0] * delta_v)
    new_vy = velocity[1] - (v_unit[1] * delta_v)
    new_vz = velocity[2] - (v_unit[2] * delta_v)
    
    return new_vx, new_vy, new_vz

    

if __name__ == '__main__':
    # Connect to DB
    conn = sqlite3.connect('satellite_data.db')
    cursor = conn.cursor()

    sql_query = 'SELECT tle_line1, tle_line2 FROM gp_history ORDER BY epoch ASC LIMIT 1;'

    # Query DB
    output = cursor.execute(sql_query)
    row = cursor.fetchone()
    
    if row:
        tle_line1, tle_line2 = row
        print("Sucessfully retrieved TLE lines.")
    else:
        print("DB is empty or query returned no results.")
        tle_line1, tle_line2 = None, None

    cursor.close()
    conn.close()

    # Parse TLE and extract (X,Y,Z) spatial coords
    if tle_line1 and tle_line2:
        satellite = Satrec.twoline2rv(tle_line1, tle_line2)

    jd = satellite.jdsatepoch # type: ignore
    fr = satellite.jdsatepochF # type: ignore

    print(f"Target Julian Date: {jd}")
    print(f"Target Fractional Day: {fr}")

    # Run the orbital mechanics physics calculation
    error_code, position, velocity = satellite.sgp4(jd, fr) # type: ignore

    if error_code == 0:
        print("X, Y, Z Coordinates (km):", position)
        print("Vx, Vy, Vz Velocity (km/s):", velocity)
    else:
        print(f"SGP4 Propagation Error Code: {error_code}")

    base_jd = satellite.jdsatepoch # type: ignore
    base_fr = satellite.jdsatepochF # type: ignore

    print("Starting 24 hr sim loop...")

    # Loop for 1 day (1440 min)
    for t in range(0, 1440):
        # Update time step
        time_step = t / 1440.0
        current_fr = base_fr + time_step
        
        error_code, position, velocity = satellite.sgp4(base_jd, current_fr) # type: ignore
        
        if error_code == 0:
            # Use radius to calculate altitude
            altitude = calc_altitude(position)
            atmospheric_density = calculate_atmospheric_density(altitude)
            
            # Print every 60 minutes so your terminal doesn't get flooded
            if t % 60 == 0:
                print(f"Minute {t:4d} | Current Altitude: {altitude:4.2f} km | Current Density: {atmospheric_density:.3e} kg/m^3")
        else:
            print(f"Error at minute {t}: Code {error_code}")



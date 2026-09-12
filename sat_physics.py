import sqlite3
from sgp4.api import Satrec
import math

MU = 398600.4418          # Earth's standard gravitational parameter km^3/s^2
EARTH_RADIUS = 6371.0      # km

"""Take in TLE Data from Space-Track and output (X,Y,Z) coords"""
def calc_altitude(position: list[float]) -> float:
    equatorial_radius = 6378.137
    polar_radius = 6356.752
    
    total_radius = sum(p**2 for p in position)**0.5
    
    # Standard ellipsoidal Earth radius approximation based on the Z component directional vector
    direction_z = position[2] / total_radius
    local_earth_radius = equatorial_radius - (equatorial_radius - polar_radius) * (direction_z**2)

    return total_radius - local_earth_radius

def altitude_from_tle(satellite) -> float:
    n_rad_per_sec = satellite.no_kozai / 60.0 # sgp4 stores mean motion in rad/min
    a = (MU / (n_rad_per_sec ** 2)) ** (1/3) # semi-major axis, km

    return a - EARTH_RADIUS 

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

import math

def calculate_daily_altitude_drop(
    velocity: tuple[float, float, float], 
    dynamic_density: float, 
    cd: float,
    area: float,
    mass: float,
    current_alt: float
) -> float:
    """
    Calculates the orbital decay (altitude drop in km) over 1 day 
    caused by atmospheric drag using true physical properties.
    """
    # Calculate velocity magnitude (speed) in km/s
    v_mag_kms = math.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)
    if v_mag_kms == 0:
        return 0.0
    
    # Convert velocity to m/s for drag force calculation
    v_mag_ms = v_mag_kms * 1000.0
    
    # Calculate true ballistic coefficient (B = Cd * A / m)
    # Area in m^2, Mass in kg, Cd is dimensionless
    ballistic_factor = (cd * area) / mass
    
    # Calculate drag acceleration in m/s^2
    # a_drag = 0.5 * rho * B * v^2
    a_drag_ms2 = 0.5 * dynamic_density * ballistic_factor * (v_mag_ms**2)
    
    # Convert acceleration back to km/s^2 for orbital mechanics compatibility
    a_drag_kms2 = a_drag_ms2 / 1000.0
    
    # Total velocity lost in one day (86,400 seconds)
    delta_v_kms = a_drag_kms2 * 86400.0
    
    # Translate velocity loss directly to altitude loss
    orbital_radius = current_alt + EARTH_RADIUS
    
    alt_drop_km = (2.0 * (orbital_radius**2) * v_mag_kms * delta_v_kms) / MU
    
    return alt_drop_km

    

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



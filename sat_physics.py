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
    Calculates atmospheric density (kg/m^3) using a multi-layered 
    exponential scale height model for the thermosphere.
    
    Data Source: U.S. Standard Atmosphere, 1976 (NASA-TM-X-74335)
    Tabular approximation derived from Vallado, "Fundamentals of Astrodynamics 
    and Applications" (Table 8-4).
    Reference: https://ntrs.nasa.gov/citations/19770009539
    """
    # If the satellite drops below 100km, the atmosphere is extremely thick.
    if altitude_km < 100:
        return 5.297e-7 * math.exp(-(altitude_km - 100.0) / 5.877)
        
    # Layer definitions: (h_base [km], rho_base [kg/m^3], scale_height [km])
    layers = [
        (100.0, 5.297e-7,  5.877),
        (120.0, 2.438e-8,  9.473),
        (150.0, 2.070e-9,  22.22),
        (200.0, 2.541e-10, 37.10),
        (250.0, 6.073e-11, 45.54),
        (300.0, 1.916e-11, 53.62),
        (400.0, 2.803e-12, 65.51),
        (500.0, 5.215e-13, 75.83),
        (600.0, 1.137e-13, 91.56),
        (700.0, 2.928e-14, 114.7),
        (800.0, 9.387e-15, 137.6),
        (900.0, 3.480e-15, 157.0),
        (1000.0, 1.455e-15, 172.9)
    ]
    
    # Find the appropriate layer starting from the top down
    h0, rho0, H = layers[0]
    for i in range(len(layers) - 1, -1, -1):
        if altitude_km >= layers[i][0]:
            h0, rho0, H = layers[i]
            break
            
    # The exponential decay formula for the localized layer
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

    





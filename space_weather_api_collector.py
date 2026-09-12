import os
import sys
import time
import argparse
import sqlite3
import logging
from typing import Dict, Any, List, Optional
import requests
import pandas as pd
import datetime

# Configure structured logging to look like a professional military operational pipeline
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ingest_engine.log")
    ]
)
logger = logging.getLogger("NOAAIngest")


class NOAASpaceWeatherCollector:
    BASE_URL = 'https://services.swpc.noaa.gov/text/daily-solar-indices.txt'

    def __init__(self, db_path: str = "satellite_data.db"):
        self.db_path = db_path
        self.session = requests.Session()
        
        # Initialize local storage database
        self._init_db()
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weather_history (
                date TEXT,
                radio_flux INTEGER,
                sunspot_num INTEGER,
                sunspot_area INTEGER,
                new_regions INTEGER,
                solar_mean_field INTEGER,
                xray_bkgd_flux TEXT,
                c_flares INTEGER,
                m_flares INTEGER,
                x_flares INTEGER,
                optical_s_flares INTEGER,
                optical_1_flares INTEGER,
                optical_2_flares INTEGER,
                optical_3_flares INTEGER
            )
        """)
        conn.commit()
        conn.close()
        logger.debug("Database initialized successfully.")
    def fetch_weather_history(self) -> str:
        """Hits CelesTrak's server and brings back the complete historical CSV data block."""
        full_url = "https://celestrak.org/SpaceData/SW-All.csv"

        time.sleep(1.0)

        try: 
            response = self.session.get(full_url, timeout=30)
            response.raise_for_status()
            return response.text

        except Exception as e:
            logger.error(f"Failed to return CelesTrak Space Weather Data: {e}")
            return ""
    
    def parse_weather_text(self, raw_txt: str) -> List[Dict[str, Any]]:
        """Takes the raw CSV data string block and processes it into a clean list of dictionaries."""
        if not raw_txt:
            return []
        
        records = []
        raw_lines = raw_txt.splitlines()

        for line in raw_lines[1:]:  # Skip the header row
            if not line.strip() or line.startswith('DATE'):
                continue
            
            parts = line.split(',')

            if len(parts) < 25:
                continue

            try:
                date_str = parts[0]
                f107_str = parts[24]  # F10.7_OBS is column 25
                
                if not f107_str:
                    continue
                    
                solar_data = {
                    "date": date_str,
                    "radio_flux": float(f107_str),
                    "sunspot_num": int(parts[23]) if parts[23] else 0,
                    "sunspot_area": 0,
                    "new_regions": 0,
                    "solar_mean_field": 0,
                    "xray_bkgd_flux": "0", 
                    "c_flares": 0,
                    "m_flares": 0,
                    "x_flares": 0,
                    "optical_s_flares": 0,
                    "optical_1_flares": 0,
                    "optical_2_flares": 0,
                    "optical_3_flares": 0,
                }
                records.append(solar_data)
            except (IndexError, ValueError) as parse_err:
                logger.debug(f"Skipping line due to layout variance: {parse_err}")
                continue
        
        return records
    def save_records_to_db(self, records: List[Dict[str, Any]]) -> int:
        """Saves weather records into local SQLite storage, filtering out duplicates."""
        if not records:
            return 0
        
        inserted_count = 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for r in records:
                try:
                    cursor.execute("""
                    INSERT OR IGNORE INTO weather_history (
                        date, radio_flux, sunspot_num,
                        sunspot_area, new_regions, solar_mean_field, 
                        xray_bkgd_flux, c_flares, m_flares, x_flares,
                        optical_s_flares, optical_1_flares,
                        optical_2_flares, optical_3_flares        
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                    r.get("date"),
                    r.get("radio_flux"),
                    r.get("sunspot_num"),
                    r.get("sunspot_area"),
                    r.get("new_regions"),
                    r.get("solar_mean_field"),
                    r.get("xray_bkgd_flux"),
                    r.get("c_flares"),
                    r.get("m_flares"),
                    r.get("x_flares"),
                    r.get("optical_s_flares"),
                    r.get("optical_1_flares"),
                    r.get("optical_2_flares"),
                    r.get("optical_3_flares"),
                    ))
                    
                    # Increment insertion count if row insertion is sucessful
                    if cursor.rowcount > 0:
                        inserted_count += 1

                except Exception as e:
                    logger.debug(f"Row insertion error (ignored due to format edge cases): {e}")
            
            conn.commit()
            
        logger.info(f"Database sync finished. Added {inserted_count} new historical records to {self.db_path}.")
        return inserted_count
    
def check_local_cache(db_path: str) -> bool:
    """
    Checks if the local database already has up-to-date space weather records.
    Returns True if we have data for today, skipping the network call.
    """
    if not os.path.exists(db_path):
        return False

    # Get today's date in YYYY-MM-DD format
    today_str = datetime.date.today().isoformat()

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        try:
            # Check if today's date exists in the table
            cursor.execute(
                "SELECT COUNT(*) FROM weather_history WHERE date = ?", 
                (today_str,)
            )
            count = cursor.fetchone()[0]
            
            if count > 0:
                logger.info(f"Cache Hit: Space weather data for {today_str} is already cached. Skipping download.")
                return True
        except sqlite3.OperationalError:
            # Table doesn't exist yet
            return False

    return False
    
def main():
    
    collector = NOAASpaceWeatherCollector()
    # 1. Check local cache first
    if check_local_cache(collector.db_path):
        logger.info("Pipeline ending cleanly using cached data.")
        return
    
    # 2. Otherwise, fetch, parse, and save
    logger.info("Cache miss or out-of-date. Querying NOAA server...")
    
    # Fetch the raw text chunk
    raw_string_data = collector.fetch_weather_history()

    # Turn that chunk into clean dictionary records
    clean_records = collector.parse_weather_text(raw_string_data)

    # Pass those records to your database insertion method
    collector.save_records_to_db(clean_records)


if __name__ == "__main__":
    main()
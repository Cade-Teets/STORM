#!/usr/bin/env python3
"""
Space-Track.org Historical TLE / GP Data Ingestion & Caching Engine

This script handles secure authentication, API rate limiting, and persistent 
local storage (SQLite) of historical satellite orbits. It acts as the core 
Data Engineering and ETL foundation for the STORM predictive model, ensuring 
high-fidelity time-series data for downstream Data Science tasks.

Prerequisites (MacOS/Linux):
    python3 -m pip install -r requirements.txt
Usage:
    python space_track_api_collector.py --norad 25544 --db satellite_data.db
"""

import os
import sys
import time
import argparse
import sqlite3
import logging
from typing import Dict, Any, List
import requests
from dotenv import load_dotenv
from pathlib import Path
import json

load_dotenv() # This automatically pulls the variables into the script

# Configure structured logging for the Data Engineering pipeline
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ingest_engine.log")
    ]
)
logger = logging.getLogger("SpaceTrackIngest")

class RateLimiter:
    """Enforces Space-Track's general throttle (30/min, 300/hour)
    AND tracks per-object gp_history bulk-pull history so it's never repeated."""

    def __init__(self, state_file: str = "api_call_log.json"):
        self.state_file = Path(state_file)
        self.call_timestamps = []
        self.bulk_history_pulled = set()  # norad_ids already fully backfilled
        self._load_state()

    def _load_state(self):
        if self.state_file.exists():
            data = json.loads(self.state_file.read_text())
            self.call_timestamps = data.get("call_timestamps", [])
            self.bulk_history_pulled = set(data.get("bulk_history_pulled", []))

    def _save_state(self):
        self.state_file.write_text(json.dumps({
            "call_timestamps": self.call_timestamps,
            "bulk_history_pulled": list(self.bulk_history_pulled),
        }))

    def wait_if_needed(self):
        now = time.time()
        # Drop timestamps older than 1 hour
        self.call_timestamps = [t for t in self.call_timestamps if now - t < 3600]

        last_minute = [t for t in self.call_timestamps if now - t < 60]
        if len(last_minute) >= 25:  # buffer under the 30/min cap
            time.sleep(60 - (now - last_minute[0]))

        if len(self.call_timestamps) >= 250:  # buffer under the 300/hour cap
            sleep_time = 3600 - (now - self.call_timestamps[0])
            if sleep_time > 0:
                time.sleep(sleep_time)

    def record_call(self):
        self.call_timestamps.append(time.time())
        self._save_state()

    def already_bulk_pulled(self, norad_id: int) -> bool:
        return norad_id in self.bulk_history_pulled

    def mark_bulk_pulled(self, norad_id: int):
        self.bulk_history_pulled.add(norad_id)
        self._save_state()

class SpaceTrackCollector:
    """Handles secure session-based connections to Space-Track.org with built-in caching."""
    
    BASE_URL = "https://www.space-track.org"
    LOGIN_URL = f"{BASE_URL}/ajaxauth/login"
    QUERY_URL = f"{BASE_URL}/basicspacedata/query"
    
    def __init__(self, identity: str, password: str, db_path: str = "satellite_data.db", rate_limiter: RateLimiter = None):
        self.identity = identity
        self.password = password
        self.db_path = db_path
        self.session = requests.Session()
        self.is_authenticated = False
        self.rate_limiter = rate_limiter or RateLimiter()
        self._init_db() # Initialize local storage database

    def _init_db(self):
        """Creates the local database schema if it does not exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Table to store historical TLE element data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS gp_history (
                    gp_id INTEGER PRIMARY KEY,
                    norad_cat_id INTEGER NOT NULL,
                    object_name TEXT,
                    object_id TEXT,
                    epoch TEXT NOT NULL,
                    mean_motion REAL,
                    eccentricity REAL,
                    inclination REAL,
                    ra_of_asc_node REAL,
                    arg_of_pericenter REAL,
                    mean_anomaly REAL,
                    bstar REAL,
                    mean_motion_dot REAL,
                    mean_motion_ddot REAL,
                    tle_line0 TEXT,
                    tle_line1 TEXT,
                    tle_line2 TEXT,
                    UNIQUE(norad_cat_id, epoch)
                )
            """)
            conn.commit()
            logger.debug("Database initialized successfully.")

    def authenticate(self) -> bool:
        """Logs into Space-Track.org to acquire the session cookies."""
        logger.info("Initiating secure login handshake with Space-Track.org...")
        payload = {
            "identity": self.identity,
            "password": self.password
        }
        
        try:
            response = self.session.post(self.LOGIN_URL, data=payload, timeout=15)
            response.raise_for_status()
            
            # Simple check to see if we didn't get kicked back to login
            if "Login Failed" in response.text or response.status_code == 401:
                logger.error("Authentication failed: Invalid credentials provided.")
                self.is_authenticated = False
                return False
                
            self.is_authenticated = True
            logger.info("Secure handshake complete. Session initialized.")
            return True
        except Exception as e:
            logger.critical(f"Connection failure during authentication: {e}")
            self.is_authenticated = False
            return False

    def fetch_gp_history(self, norad_id: int, limit: int = 2000, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """
        Queries the gp_history API class for an object's full historical orbit records.
        Using JSON format to ensure long-term stability and support for Catalog IDs > 99,999.
        """
        if not self.is_authenticated:
            if not self.authenticate():
                raise ConnectionError("Cannot execute query: Client is unauthenticated.")
                
        logger.info(f"Querying historical orbit trajectory for NORAD ID: {norad_id}...")
        
        # Build REST-compliant query path
        # Sorting by epoch descending to get the time-series back from its last position
        query_path = f"/class/gp_history/norad_cat_id/{norad_id}"
        if start_date and end_date:
            query_path += f"/EPOCH/{start_date}--{end_date}"
            logger.info(f"Restricting query to EPOCH range: {start_date} to {end_date}")
        query_path += f"/orderby/epoch%20desc/limit/{limit}/format/json"
        full_url = f"{self.QUERY_URL}{query_path}"
        
        # Polite delay to honor API throttling guidelines
        time.sleep(1.0)
        
        try:
            self.rate_limiter.wait_if_needed() # sleeps if near rate limit
            
            response = self.session.get(full_url, timeout=30)
            self.rate_limiter.record_call()
            
            if response.status_code == 204:
                logger.warning(f"No records found for NORAD ID {norad_id}.")
                return []
                
            response.raise_for_status()
            records = response.json()
            logger.info(f"Retrieved {len(records)} orbital records from Space-Track API.")
            return records
            
        except Exception as e:
            logger.error(f"Failed to fetch data for NORAD ID {norad_id}: {e}")
            return []

    def save_records_to_db(self, records: List[Dict[str, Any]]) -> int:
        """Saves orbital records into local SQLite storage, filtering out duplicates."""
        if not records:
            return 0
            
        inserted_count = 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for r in records:
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO gp_history (
                            gp_id, norad_cat_id, object_name, object_id, epoch,
                            mean_motion, eccentricity, inclination, ra_of_asc_node,
                            arg_of_pericenter, mean_anomaly, bstar, mean_motion_dot,
                            mean_motion_ddot, tle_line0, tle_line1, tle_line2
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        r.get("GP_ID"),
                        int(r.get("NORAD_CAT_ID") or 0),
                        r.get("OBJECT_NAME"),
                        r.get("OBJECT_ID"),
                        r.get("EPOCH"),
                        float(r.get("MEAN_MOTION") or 0.0),
                        float(r.get("ECCENTRICITY") or 0.0),
                        float(r.get("INCLINATION") or 0.0),
                        float(r.get("RA_OF_ASC_NODE") or 0.0),
                        float(r.get("ARG_OF_PERICENTER") or 0.0),
                        float(r.get("MEAN_ANOMALY") or 0.0),
                        float(r.get("BSTAR") or 0.0),
                        float(r.get("MEAN_MOTION_DOT") or 0.0),
                        float(r.get("MEAN_MOTION_DDOT") or 0.0),
                        r.get("TLE_LINE0"),
                        r.get("TLE_LINE1"),
                        r.get("TLE_LINE2")
                    ))
                    if cursor.rowcount > 0:
                        inserted_count += 1
                except Exception as e:
                    logger.debug(f"Row insertion error (ignored due to format edge cases): {e}")
            
            conn.commit()
            
        logger.info(f"Database sync finished. Added {inserted_count} new historical records to {self.db_path}.")
        return inserted_count


def main():
    parser = argparse.ArgumentParser(description="STORM Data Pipeline: Space-Track ETL")
    parser.add_argument("--norad", type=int, default=25544, help="NORAD Catalog ID (default: 25544 - ISS)")
    parser.add_argument("--db", type=str, default="satellite_data.db", help="Path to local SQLite database")
    parser.add_argument("--limit", type=int, default=1000, help="Max records to pull from Space-Track API")

    args = parser.parse_args()

    # Extract credentials
    username = os.getenv("SPACETRACK_USER")
    password = os.getenv("SPACETRACK_PASS")
    
    if not username or not password:
        logger.warning("Space-Track environment variables not detected.")
        print("Please enter your Space-Track.org login credentials below:")
        username = input("Email / Username: ").strip()
        import getpass
        password = getpass.getpass("Password: ").strip()
        
    if not username or not password:
        logger.critical("Missing credentials. Ingestion terminated.")
        sys.exit(1)
        
    # Initialize Ingestion Engine
    rate_limiter = RateLimiter()
    collector = SpaceTrackCollector(identity=username, password=password, db_path=args.db, rate_limiter=rate_limiter)
    
    # Fetch and store TLE's
    try:
        raw_records = collector.fetch_gp_history(norad_id=args.norad, limit=args.limit)
        new_insertions = collector.save_records_to_db(raw_records)
        
        # Verify success by loading a subset into Pandas
        if new_insertions > 0:
            with sqlite3.connect(args.db) as conn:
                df = pd.read_sql_query(
                    """SELECT epoch, mean_motion, bstar
                    FROM gp_history
                    WHERE norad_cat_id = ?
                    ORDER BY epoch DESC LIMIT 5""",
                    conn,
                    params=(args.norad,)
                )
            print("\n" + "="*50)
            print(" SUCCESSFUL BASELINE INGESTION SAMPLE ")
            print("="*50)
            print(df)
            print("="*50 + "\n")
            
    except Exception as e:
        logger.critical(f"ETL pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
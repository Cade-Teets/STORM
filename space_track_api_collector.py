#!/usr/bin/env python3
"""
Space-Track.org Historical TLE / GP Data Ingestion & Caching Engine

This script handles secure authentication, API rate limiting, and persistent 
local storage (SQLite) of historical satellite orbits. It is designed as 
the data-engineering foundation (Phase 1) for a Space Force / Air Force 
15A Operations Research portfolio project.

Prerequisites (MacOS/Linux):
    python3 -m pip install -r requirements.txt
Usage:
    python space_track_collector.py --norad 25544 --output-db satellite_data.db
"""

import os
import sys
import time
import argparse
import sqlite3
import logging
from typing import Dict, Any, List
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv() # This automatically pulls the variables into the script

# Configure structured logging to look like a professional military operational pipeline
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ingest_engine.log")
    ]
)
logger = logging.getLogger("SpaceTrackIngest")


class SpaceTrackCollector:
    """Handles secure session-based connections to Space-Track.org with built-in caching."""
    
    BASE_URL = "https://www.space-track.org"
    LOGIN_URL = f"{BASE_URL}/ajaxauth/login"
    QUERY_URL = f"{BASE_URL}/basicspacedata/query"
    
    def __init__(self, identity: str, password: str, db_path: str = "satellite_data.db"):
        self.identity = identity
        self.password = password
        self.db_path = db_path
        self.session = requests.Session()
        self.is_authenticated = False
        
        # Initialize local storage database
        self._init_db()

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
            conn.close()
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

    def fetch_gp_history(self, norad_id: int, limit: int = 2000) -> List[Dict[str, Any]]:
        """
        Queries the gp_history API class for an object's full historical orbit records.
        Using JSON format to ensure long-term stability and support for Catalog IDs > 99,999.
        """
        if not self.is_authenticated:
            if not self.authenticate():
                raise ConnectionError("Cannot execute query: Client is unauthenticated.")
                
        logger.info(f"Querying historical orbit trajectory for NORAD ID: {norad_id}...")
        
        # Build REST-compliant query path
        # Sorting by epoch ascending allows us to construct a proper forward time-series
        query_path = (
            f"/class/gp_history"
            f"/norad_cat_id/{norad_id}"
            f"/orderby/epoch%20asc"
            f"/limit/{limit}"
            f"/format/json"
        )
        full_url = f"{self.QUERY_URL}{query_path}"
        
        # Polite delay to honor API throttling guidelines
        time.sleep(1.0)
        
        try:
            response = self.session.get(full_url, timeout=30)
            
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


def check_local_cache(norad_id: int, db_path: str) -> bool:
    """Verifies if we already have sufficient historical data cached locally."""
    if not os.path.exists(db_path):
        return False
        
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM gp_history WHERE norad_cat_id = ?", (norad_id,))
        count = cursor.fetchone()[0]
        
    if count > 0:
        logger.info(f"Cache Hit: Found {count} records locally in SQLite database. Skipping active API fetch.")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Space Force Portolio Ingestion: Space-Track ETL")
    parser.add_argument("--norad", type=int, default=25544, help="NORAD Catalog ID (default: 25544 - ISS)")
    parser.add_argument("--db", type=str, default="satellite_data.db", help="Path to local SQLite database")
    parser.add_argument("--limit", type=int, default=1000, help="Max records to pull from Space-Track API")
    
    args = parser.parse_args()
    
    # 1. Check local cache first to protect API limits
    if check_local_cache(args.norad, args.db):
        logger.info("Local caching operational. Pipeline ending cleanly.")
        sys.exit(0)
        
    # 2. Extract Credentials
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
        
    # 3. Initialize Ingestion Engine
    collector = SpaceTrackCollector(identity=username, password=password, db_path=args.db)
    
    # 4. Fetch and Store
    try:
        raw_records = collector.fetch_gp_history(norad_id=args.norad, limit=args.limit)
        new_insertions = collector.save_records_to_db(raw_records)
        
        # Verify success by loading a subset into Pandas
        if new_insertions > 0 or check_local_cache(args.norad, args.db):
            with sqlite3.connect(args.db) as conn:
                df = pd.read_sql_query(
                    "SELECT epoch, mean_motion, bstar FROM gp_history WHERE norad_cat_id = ? ORDER BY epoch ASC LIMIT 5",
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
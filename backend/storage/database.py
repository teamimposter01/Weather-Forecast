"""
Database storage layer using DuckDB for efficient tabular weather data handling.
Includes duplicate prevention, upsert capabilities, and data integrity audits.
"""
import os
import duckdb
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from config.settings import DB_PATH, PROCESSED_DATA_DIR

class StorageEngine:
    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        self._init_db()

    def get_connection(self, read_only: bool = False):
        """Get DuckDB database connection with read-only and in-memory fallback for serverless."""
        if not os.path.exists(self.db_path) and read_only:
            return duckdb.connect(":memory:")
        try:
            return duckdb.connect(self.db_path, read_only=read_only)
        except Exception:
            try:
                return duckdb.connect(self.db_path, read_only=True)
            except Exception:
                return duckdb.connect(":memory:")

    def _init_db(self):
        """Initialize database tables if they do not exist."""
        try:
            with self.get_connection(read_only=False) as conn:
                # Observations / Reanalysis Table
                conn.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    timestamp_utc TIMESTAMP,
                    location_id VARCHAR,
                    latitude DOUBLE,
                    longitude DOUBLE,
                    temperature_2m DOUBLE,
                    precipitation_mm DOUBLE,
                    wind_speed_ms DOUBLE,
                    u_wind_ms DOUBLE,
                    v_wind_ms DOUBLE,
                    relative_humidity_pct DOUBLE,
                    surface_pressure_hpa DOUBLE,
                    data_source VARCHAR,
                    PRIMARY KEY (timestamp_utc, location_id, data_source)
                );
            """)

            # Model Forecasts Table (ECMWF, GFS, AI)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS forecasts (
                    issue_time_utc TIMESTAMP,
                    valid_time_utc TIMESTAMP,
                    lead_time_hours INTEGER,
                    location_id VARCHAR,
                    latitude DOUBLE,
                    longitude DOUBLE,
                    model_name VARCHAR,
                    temperature_2m DOUBLE,
                    precipitation_mm DOUBLE,
                    wind_speed_ms DOUBLE,
                    u_wind_ms DOUBLE,
                    v_wind_ms DOUBLE,
                    relative_humidity_pct DOUBLE,
                    surface_pressure_hpa DOUBLE,
                    PRIMARY KEY (issue_time_utc, lead_time_hours, location_id, model_name)
                );
            """)

            # Historical Skill Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS historical_skill (
                    location_id VARCHAR,
                    variable VARCHAR,
                    model_name VARCHAR,
                    lead_time_hours INTEGER,
                    season VARCHAR,
                    mae DOUBLE,
                    rmse DOUBLE,
                    bias DOUBLE,
                    sample_count INTEGER,
                    updated_at_utc TIMESTAMP,
                    PRIMARY KEY (location_id, variable, model_name, lead_time_hours, season)
                );
            """)

            # Blended Forecasts Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS blended_forecasts (
                    issue_time_utc TIMESTAMP,
                    valid_time_utc TIMESTAMP,
                    lead_time_hours INTEGER,
                    location_id VARCHAR,
                    latitude DOUBLE,
                    longitude DOUBLE,
                    variable VARCHAR,
                    ecmwf_value DOUBLE,
                    gfs_value DOUBLE,
                    ai_value DOUBLE,
                    blended_value DOUBLE,
                    ecmwf_weight DOUBLE,
                    gfs_weight DOUBLE,
                    ai_weight DOUBLE,
                    uncertainty_std DOUBLE,
                    weather_regime VARCHAR,
                    PRIMARY KEY (issue_time_utc, lead_time_hours, location_id, variable)
                );
            """)

            # Extreme Weather Probabilities Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS extreme_probabilities (
                    issue_time_utc TIMESTAMP,
                    valid_time_utc TIMESTAMP,
                    lead_time_hours INTEGER,
                    location_id VARCHAR,
                    heatwave_prob DOUBLE,
                    heavy_rainfall_prob DOUBLE,
                    high_wind_prob DOUBLE,
                    PRIMARY KEY (issue_time_utc, lead_time_hours, location_id)
                );
            """)

            # Model Metadata Registry
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_registry (
                    model_name VARCHAR,
                    variable VARCHAR,
                    version VARCHAR,
                    training_start TIMESTAMP,
                    training_end TIMESTAMP,
                    feature_schema JSON,
                    metrics JSON,
                    created_at_utc TIMESTAMP,
                    PRIMARY KEY (model_name, variable, version)
                );
            """)
        except Exception:
            pass

    def upsert_dataframe(self, df: pd.DataFrame, table_name: str, unique_cols: List[str]):
        """
        Deduplicate and upsert pandas DataFrame into DuckDB table based on unique key columns.
        Prevents duplicate records if API calls are repeated.
        """
        if df.empty:
            return

        df_clean = df.drop_duplicates(subset=unique_cols, keep="last")

        with self.get_connection() as conn:
            # Create temporary staging table
            conn.execute("CREATE TEMPORARY TABLE staging AS SELECT * FROM df_clean")
            
            # Delete existing matching records in target table to avoid constraint error
            where_clause = " AND ".join([f"{table_name}.{col} = staging.{col}" for col in unique_cols])
            conn.execute(f"DELETE FROM {table_name} WHERE EXISTS (SELECT 1 FROM staging WHERE {where_clause})")
            
            # Insert deduplicated staging records
            conn.execute(f"INSERT INTO {table_name} SELECT * FROM staging")
            conn.execute("DROP TABLE staging")

    def save_dataframe(self, df: pd.DataFrame, table_name: str, mode: str = "append"):
        """Save a pandas DataFrame into a DuckDB table."""
        if df.empty:
            return

        unique_key_map = {
            "observations": ["timestamp_utc", "location_id", "data_source"],
            "forecasts": ["issue_time_utc", "lead_time_hours", "location_id", "model_name"],
            "blended_forecasts": ["issue_time_utc", "lead_time_hours", "location_id", "variable"],
            "extreme_probabilities": ["issue_time_utc", "lead_time_hours", "location_id"],
            "historical_skill": ["location_id", "variable", "model_name", "lead_time_hours", "season"],
        }

        if mode == "overwrite":
            with self.get_connection() as conn:
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                df_clean = df.drop_duplicates(subset=unique_key_map.get(table_name, None), keep="last") if table_name in unique_key_map else df
                conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df_clean")
        else:
            if table_name in unique_key_map:
                self.upsert_dataframe(df, table_name, unique_key_map[table_name])
            else:
                with self.get_connection() as conn:
                    conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df WHERE 1=0")
                    conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")

    def query(self, sql_query: str, params: Optional[list] = None) -> pd.DataFrame:
        """Run a SQL query and return a pandas DataFrame."""
        with self.get_connection() as conn:
            if params:
                return conn.execute(sql_query, params).df()
            return conn.execute(sql_query).df()

    def execute(self, sql_query: str, params: Optional[list] = None):
        """Execute a SQL statement (INSERT, UPDATE, DELETE)."""
        with self.get_connection() as conn:
            if params:
                conn.execute(sql_query, params)
            else:
                conn.execute(sql_query)

    def check_data_integrity(self) -> Dict[str, Any]:
        """
        Perform a comprehensive data integrity audit across all tables.
        Reports total records, duplicate counts, missing timestamps, missing coordinates, and data freshness.
        """
        report = {}
        tables = ["observations", "forecasts", "blended_forecasts", "extreme_probabilities", "historical_skill"]
        
        for table in tables:
            try:
                df = self.query(f"SELECT * FROM {table}")
                total = len(df)
                if total == 0:
                    report[table] = {"total_records": 0, "duplicates": 0, "missing_values": 0, "status": "EMPTY"}
                    continue

                # Check duplicates based on key
                unique_keys = {
                    "observations": ["timestamp_utc", "location_id", "data_source"],
                    "forecasts": ["issue_time_utc", "lead_time_hours", "location_id", "model_name"],
                    "blended_forecasts": ["issue_time_utc", "lead_time_hours", "location_id", "variable"],
                    "extreme_probabilities": ["issue_time_utc", "lead_time_hours", "location_id"],
                    "historical_skill": ["location_id", "variable", "model_name", "lead_time_hours", "season"],
                }.get(table, [])

                duplicates = int(total - len(df.drop_duplicates(subset=unique_keys))) if unique_keys else 0
                nulls = int(df.isnull().sum().sum())

                # Freshness check
                last_ts = None
                for col in ["valid_time_utc", "issue_time_utc", "timestamp_utc", "updated_at_utc"]:
                    if col in df.columns and not df[col].isnull().all():
                        last_ts = str(df[col].max())
                        break

                report[table] = {
                    "total_records": total,
                    "duplicates": duplicates,
                    "null_count": nulls,
                    "last_timestamp_utc": last_ts,
                    "status": "HEALTHY" if duplicates == 0 and nulls == 0 else "DEGRADED"
                }
            except Exception as e:
                report[table] = {"error": str(e), "status": "ERROR"}

        return report

# Global storage instance
storage = StorageEngine()

from src.utils.exception import CustomException
from src.utils.logger import logging
from dataclasses import dataclass
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pandas as pd
import shutil
import os


@dataclass
class DataIngestionConfig:
    data_dir: str = os.path.join(os.getcwd(), "data")
    artifacts_dir: str = os.path.join(os.getcwd(), "artifacts")
    raw_data: str = os.path.join("artifacts", "raw_data.csv")
    oil: str = os.path.join("artifacts", "oil.csv")
    stores: str = os.path.join("artifacts", "stores.csv")
    holidays: str = os.path.join("artifacts", "holidays.csv")
    env_file_path: str = os.path.join("secrets.env")


class DataIngestion:
    def __init__(self):
        self.dataingestionconfig = DataIngestionConfig()
        logging.info(">>> DATA INGESTION STARTED <<<")

    def _find_file(self, candidates):
        """Helper to find the first existing candidate file."""
        for path in candidates:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return path
        return None

    def _load_from_local_files(self) -> bool:
        """
        Attempts to ingest from local CSV files located in `data/` or `artifacts/`.
        Accepts both original Kaggle names (`train.csv`, `holidays_events.csv`)
        and pipeline names (`raw_data.csv`, `holidays.csv`).
        """
        train_path = self._find_file([
            os.path.join(self.dataingestionconfig.data_dir, "train.csv"),
            os.path.join(self.dataingestionconfig.data_dir, "raw_data.csv"),
            self.dataingestionconfig.raw_data,
        ])
        oil_path = self._find_file([
            os.path.join(self.dataingestionconfig.data_dir, "oil.csv"),
            self.dataingestionconfig.oil,
        ])
        stores_path = self._find_file([
            os.path.join(self.dataingestionconfig.data_dir, "stores.csv"),
            self.dataingestionconfig.stores,
        ])
        holidays_path = self._find_file([
            os.path.join(self.dataingestionconfig.data_dir, "holidays_events.csv"),
            os.path.join(self.dataingestionconfig.data_dir, "holidays.csv"),
            self.dataingestionconfig.holidays,
        ])

        if train_path and oil_path and stores_path and holidays_path:
            logging.info("Found local CSV files. Ingesting into artifacts/...")
            os.makedirs(self.dataingestionconfig.artifacts_dir, exist_ok=True)

            if os.path.abspath(train_path) != os.path.abspath(self.dataingestionconfig.raw_data):
                shutil.copyfile(train_path, self.dataingestionconfig.raw_data)
            if os.path.abspath(oil_path) != os.path.abspath(self.dataingestionconfig.oil):
                shutil.copyfile(oil_path, self.dataingestionconfig.oil)
            if os.path.abspath(stores_path) != os.path.abspath(self.dataingestionconfig.stores):
                shutil.copyfile(stores_path, self.dataingestionconfig.stores)
            if os.path.abspath(holidays_path) != os.path.abspath(self.dataingestionconfig.holidays):
                shutil.copyfile(holidays_path, self.dataingestionconfig.holidays)

            logging.info("Local datasets successfully ingested into artifacts/")
            print(f"Data ingestion successful: Loaded local CSV files from '{os.path.dirname(train_path)}'")
            return True

        return False

    def _get_engine(self):
        """Creates and returns a SQLAlchemy engine connected to PostgreSQL."""
        load_dotenv(self.dataingestionconfig.env_file_path)

        host = os.getenv("PG_HOST", "localhost")
        port = os.getenv("PG_PORT", "5432")
        database = os.getenv("PG_DATABASE", "store_sales")
        user = os.getenv("PG_USER")
        password = os.getenv("PG_PASSWORD")

        if not user or not password:
            return None

        connection_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        engine = create_engine(connection_url, connect_args={"sslmode": "require"})
        logging.info(f"PostgreSQL engine created for host={host} db={database}")
        return engine

    def _load_from_database(self, engine) -> bool:
        """Extracts data from PostgreSQL tables into CSV artifacts."""
        logging.info("Connecting to PostgreSQL to extract datasets...")
        with engine.connect() as conn:
            data = pd.read_sql(text("SELECT * FROM store_sales ORDER BY date, store_nbr, family;"), conn)
            oil = pd.read_sql(text("SELECT * FROM oil_prices ORDER BY date;"), conn)
            stores = pd.read_sql(text("SELECT * FROM stores ORDER BY store_nbr;"), conn)
            holidays = pd.read_sql(text("SELECT * FROM holidays ORDER BY date;"), conn)

        logging.info(
            f"Extracted — sales: {len(data)} rows | oil: {len(oil)} rows | "
            f"stores: {len(stores)} rows | holidays: {len(holidays)} rows"
        )

        os.makedirs(self.dataingestionconfig.artifacts_dir, exist_ok=True)
        data.to_csv(self.dataingestionconfig.raw_data, index=False)
        oil.to_csv(self.dataingestionconfig.oil, index=False)
        stores.to_csv(self.dataingestionconfig.stores, index=False)
        holidays.to_csv(self.dataingestionconfig.holidays, index=False)
        print("Data ingestion successful: Extracted datasets from PostgreSQL database.")
        return True

    def load_dataset(self):
        """
        Orchestrates dataset loading with automatic fallback:
        1. If local CSV files exist in `data/` or `artifacts/`, ingests them directly.
        2. If PostgreSQL credentials are configured, pulls from PostgreSQL.
        3. If neither, provides clear instructions on where to place the data.
        """
        logging.info("Executing load_dataset function")
        try:
            # 1. Check local files first (e.g. data/ folder)
            if self._load_from_local_files():
                logging.info(">>> DATA INGESTION COMPLETE <<<")
                return

            # 2. Check PostgreSQL database
            engine = self._get_engine()
            if engine is not None:
                if self._load_from_database(engine):
                    logging.info(">>> DATA INGESTION COMPLETE <<<")
                    return

            # 3. Neither found — raise actionable guidance
            header_line = "=" * 60
            msg = (
                f"\n{header_line}\n"
                "NO DATA SOURCE FOUND!\n"
                "Please choose one of the two options:\n"
                "  Option A (Quickest - Local):\n"
                "    Place your 4 Kaggle CSV files inside the 'data/' folder:\n"
                "      - data/train.csv\n"
                "      - data/oil.csv\n"
                "      - data/stores.csv\n"
                "      - data/holidays_events.csv\n\n"
                "  Option B (Database - PostgreSQL):\n"
                "    Configure PG_USER and PG_PASSWORD in 'secrets.env' and seed\n"
                "    the database using 'db/seed_from_csv.sql'.\n"
                f"{header_line}"
            )
            raise FileNotFoundError(msg)

        except Exception as e:
            logging.info(CustomException(e))
            print(CustomException(e))

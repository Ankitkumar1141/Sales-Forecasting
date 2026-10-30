-- =============================================================
-- db/seed_from_csv.sql
-- Helper script: load CSV exports into PostgreSQL tables
-- =============================================================
-- Usage (psql):
--   \copy store_sales   FROM 'path/to/train.csv'      CSV HEADER
--   \copy oil_prices    FROM 'path/to/oil.csv'         CSV HEADER
--   \copy stores        FROM 'path/to/stores.csv'      CSV HEADER
--   \copy holidays      FROM 'path/to/holidays_events.csv' CSV HEADER
--
-- For Supabase / Neon use the Table Editor "Import from CSV" feature
-- or the platform's psql connection string with the \copy commands above.
-- =============================================================

-- Truncate first (safe for a fresh load)
TRUNCATE TABLE store_sales, oil_prices, stores, holidays RESTART IDENTITY CASCADE;

-- Then run the \copy commands shown in the header above.

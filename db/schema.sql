-- =============================================================
-- db/schema.sql
-- PostgreSQL schema for the Store Sales Forecasting project
-- =============================================================
-- Run once against your database to create all required tables:
--
--   psql -h <HOST> -U <USER> -d <DATABASE> -f db/schema.sql
--
-- Or paste into the Supabase / Neon / Railway SQL editor.
-- =============================================================

-- -------------------------
-- Table: store_sales
-- -------------------------
-- One row per (date, store_nbr, family) combination.
-- Mirrors the Kaggle "Store Sales - Time Series Forecasting" train.csv.
CREATE TABLE IF NOT EXISTS store_sales (
    id          BIGINT        PRIMARY KEY,       -- original row id
    date        DATE          NOT NULL,
    store_nbr   SMALLINT      NOT NULL,
    family      VARCHAR(50)   NOT NULL,
    sales       NUMERIC(14,4) NOT NULL DEFAULT 0,
    onpromotion INTEGER       NOT NULL DEFAULT 0
);

-- Index for time-range and store/family queries
CREATE INDEX IF NOT EXISTS idx_sales_date       ON store_sales (date);
CREATE INDEX IF NOT EXISTS idx_sales_store_fam  ON store_sales (store_nbr, family);

-- -------------------------
-- Table: oil_prices
-- -------------------------
-- Daily WTI crude oil prices (dcoilwtico).
CREATE TABLE IF NOT EXISTS oil_prices (
    date        DATE          PRIMARY KEY,
    dcoilwtico  NUMERIC(8,4)              -- nullable: some trading days missing
);

-- -------------------------
-- Table: stores
-- -------------------------
-- Static metadata about each store.
CREATE TABLE IF NOT EXISTS stores (
    store_nbr   SMALLINT      PRIMARY KEY,
    city        VARCHAR(60)   NOT NULL,
    state       VARCHAR(60)   NOT NULL,
    type        CHAR(1)       NOT NULL,   -- A, B, C, D or E
    cluster     SMALLINT      NOT NULL
);

-- -------------------------
-- Table: holidays
-- -------------------------
-- National, regional and local holiday/event calendar.
CREATE TABLE IF NOT EXISTS holidays (
    id          SERIAL        PRIMARY KEY,
    date        DATE          NOT NULL,
    type        VARCHAR(20)   NOT NULL,   -- Holiday, Event, Transfer, Work Day, Bridge
    locale      VARCHAR(20)   NOT NULL,   -- National, Regional, Local
    locale_name VARCHAR(60)   NOT NULL,
    description VARCHAR(200),
    transferred BOOLEAN       NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_holidays_date ON holidays (date);

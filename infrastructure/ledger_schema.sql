-- RAGShield — PostgreSQL Schema
-- Ledger server (10.0.0.110)
-- Run as: psql -U ragshield_app -d ragshield -f ledger_schema.sql

-- ── Create database (run as postgres superuser) ───────────────────────
-- CREATE DATABASE ragshield;
-- CREATE USER ragshield_app WITH PASSWORD 'ragshield2025';
-- GRANT ALL PRIVILEGES ON DATABASE ragshield TO ragshield_app;

-- ── Latency log ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS latency_log (
    id             BIGSERIAL PRIMARY KEY,
    component      VARCHAR(50)   NOT NULL,  -- retrieval | classification | generation
    latency_ms     NUMERIC(12,3) NOT NULL,
    run_id         VARCHAR(100),
    doc_id         VARCHAR(200),
    label          SMALLINT,                -- 0=benign, 1=injection (ground truth)
    recorded_at    TIMESTAMP     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_latency_component   ON latency_log (component);
CREATE INDEX IF NOT EXISTS idx_latency_run_id      ON latency_log (run_id);
CREATE INDEX IF NOT EXISTS idx_latency_recorded_at ON latency_log (recorded_at);

-- ── Evaluation runs ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS evaluation_run (
    id              BIGSERIAL PRIMARY KEY,
    run_id          VARCHAR(100)  NOT NULL UNIQUE,
    model           VARCHAR(50)   NOT NULL,
    injection_pct   NUMERIC(5,3)  NOT NULL,
    n_docs          INTEGER,
    tp              INTEGER,
    fp              INTEGER,
    tn              INTEGER,
    fn              INTEGER,
    f1              NUMERIC(8,4),
    mcc             NUMERIC(8,4),
    precision_score NUMERIC(8,4),
    recall_score    NUMERIC(8,4),
    far             NUMERIC(8,4),
    run_time_s      NUMERIC(8,1),
    completed_at    TIMESTAMP,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eval_run_model ON evaluation_run (model);
CREATE INDEX IF NOT EXISTS idx_eval_run_pct   ON evaluation_run (injection_pct);

-- ── Classifier decisions ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS classifier_decision (
    id              BIGSERIAL PRIMARY KEY,
    run_id          VARCHAR(100)  NOT NULL,
    doc_id          VARCHAR(200),
    model           VARCHAR(50)   NOT NULL,
    true_label      SMALLINT,
    pred_label      SMALLINT,
    confidence      NUMERIC(6,4),
    classification_ms NUMERIC(12,3),
    decided_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_decision_run_id ON classifier_decision (run_id);
CREATE INDEX IF NOT EXISTS idx_decision_model  ON classifier_decision (model);

-- ── Useful views ─────────────────────────────────────────────────────

-- Per-model latency percentiles
CREATE OR REPLACE VIEW v_latency_by_model AS
SELECT
    run_id,
    component,
    COUNT(*)                                              AS n,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP
          (ORDER BY latency_ms)::NUMERIC, 1)             AS p50_ms,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP
          (ORDER BY latency_ms)::NUMERIC, 1)             AS p95_ms,
    ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP
          (ORDER BY latency_ms)::NUMERIC, 1)             AS p99_ms,
    ROUND(AVG(latency_ms)::NUMERIC, 1)                  AS avg_ms
FROM latency_log
GROUP BY run_id, component
ORDER BY run_id, component;

-- Overall latency by component (all runs)
CREATE OR REPLACE VIEW v_latency_summary AS
SELECT
    component,
    COUNT(*)                                              AS n,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP
          (ORDER BY latency_ms)::NUMERIC, 1)             AS p50_ms,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP
          (ORDER BY latency_ms)::NUMERIC, 1)             AS p95_ms,
    ROUND(AVG(latency_ms)::NUMERIC, 1)                  AS avg_ms
FROM latency_log
GROUP BY component;

-- Evaluation results ranked by MCC
CREATE OR REPLACE VIEW v_results_ranked AS
SELECT
    model,
    injection_pct,
    mcc,
    f1,
    far,
    tp, fp, tn, fn,
    run_id
FROM evaluation_run
ORDER BY mcc DESC;

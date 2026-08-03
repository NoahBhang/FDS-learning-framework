"""SQLite schema and connection contract for persisted FDS analyses."""

from __future__ import annotations

import sqlite3


SCHEMA_VERSION = 1
BUSY_TIMEOUT_MS = 5_000


class UnsupportedSchemaVersionError(RuntimeError):
    """Raised when a database was created by a newer schema version."""


class SchemaValidationError(RuntimeError):
    """Raised when a versioned database does not match the required schema."""


_TABLE_STATEMENTS = (
    """
    CREATE TABLE analysis_runs (
        analysis_run_id TEXT PRIMARY KEY CHECK (trim(analysis_run_id) <> ''),
        source_name TEXT NOT NULL CHECK (trim(source_name) <> ''),
        ruleset_version TEXT NOT NULL CHECK (trim(ruleset_version) <> ''),
        created_at TEXT NOT NULL CHECK (trim(created_at) <> ''),
        input_row_count INTEGER NOT NULL CHECK (input_row_count >= 0),
        fraud_score REAL NOT NULL CHECK (fraud_score >= 0.0 AND fraud_score <= 1.0),
        transaction_count INTEGER NOT NULL CHECK (transaction_count >= 0),
        finding_count INTEGER NOT NULL CHECK (finding_count >= 0),
        error_count INTEGER NOT NULL CHECK (error_count >= 0)
    )
    """,
    """
    CREATE TABLE transaction_snapshots (
        snapshot_id INTEGER PRIMARY KEY,
        analysis_run_id TEXT NOT NULL,
        canonical_transaction_id TEXT NOT NULL
            CHECK (trim(canonical_transaction_id) <> ''),
        source_position INTEGER NOT NULL CHECK (source_position >= 0),
        source_row_id_type TEXT NOT NULL
            CHECK (source_row_id_type IN ('none', 'int', 'str')),
        source_row_id_text TEXT,
        step INTEGER,
        transaction_datetime TEXT,
        action TEXT NOT NULL CHECK (trim(action) <> ''),
        amount REAL NOT NULL,
        actor_account TEXT NOT NULL CHECK (trim(actor_account) <> ''),
        target_account TEXT,
        counterparty_account TEXT,
        old_balance_actor REAL,
        new_balance_actor REAL,
        old_balance_target REAL,
        new_balance_target REAL,
        description TEXT,
        bank_code TEXT,
        source_format TEXT,
        CHECK (
            (source_row_id_type = 'none' AND source_row_id_text IS NULL)
            OR (source_row_id_type IN ('int', 'str') AND source_row_id_text IS NOT NULL)
        ),
        FOREIGN KEY (analysis_run_id) REFERENCES analysis_runs(analysis_run_id)
            ON DELETE CASCADE,
        UNIQUE (analysis_run_id, source_position),
        UNIQUE (analysis_run_id, snapshot_id)
    )
    """,
    """
    CREATE TABLE alerts (
        alert_id TEXT PRIMARY KEY CHECK (trim(alert_id) <> ''),
        analysis_run_id TEXT NOT NULL UNIQUE,
        fraud_score REAL NOT NULL CHECK (fraud_score >= 0.0 AND fraud_score <= 1.0),
        risk_level TEXT CHECK (
            risk_level IS NULL OR risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
        ),
        status TEXT NOT NULL DEFAULT 'OPEN'
            CHECK (status IN ('OPEN', 'REVIEWING', 'CONFIRMED', 'DISMISSED')),
        created_at TEXT NOT NULL CHECK (trim(created_at) <> ''),
        FOREIGN KEY (analysis_run_id) REFERENCES analysis_runs(analysis_run_id)
            ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE rule_findings (
        finding_id INTEGER PRIMARY KEY,
        analysis_run_id TEXT NOT NULL,
        rule_id TEXT NOT NULL CHECK (trim(rule_id) <> ''),
        rule_name TEXT NOT NULL CHECK (trim(rule_name) <> ''),
        risk_type TEXT NOT NULL CHECK (trim(risk_type) <> ''),
        triggered INTEGER NOT NULL CHECK (triggered IN (0, 1)),
        rule_score INTEGER NOT NULL CHECK (
            rule_score BETWEEN 0 AND 100
            AND ((triggered = 0 AND rule_score = 0)
                 OR (triggered = 1 AND rule_score > 0))
        ),
        reason TEXT NOT NULL CHECK (trim(reason) <> ''),
        execution_order INTEGER NOT NULL CHECK (execution_order >= 0),
        FOREIGN KEY (analysis_run_id) REFERENCES analysis_runs(analysis_run_id)
            ON DELETE CASCADE,
        UNIQUE (analysis_run_id, rule_id),
        UNIQUE (analysis_run_id, execution_order),
        UNIQUE (analysis_run_id, finding_id)
    )
    """,
    """
    CREATE TABLE finding_evidence (
        finding_evidence_id INTEGER PRIMARY KEY,
        analysis_run_id TEXT NOT NULL,
        finding_id INTEGER NOT NULL,
        snapshot_id INTEGER NOT NULL,
        evidence_order INTEGER NOT NULL CHECK (evidence_order >= 0),
        FOREIGN KEY (analysis_run_id, finding_id)
            REFERENCES rule_findings(analysis_run_id, finding_id) ON DELETE CASCADE,
        FOREIGN KEY (analysis_run_id, snapshot_id)
            REFERENCES transaction_snapshots(analysis_run_id, snapshot_id)
            ON DELETE CASCADE,
        UNIQUE (finding_id, evidence_order),
        UNIQUE (finding_id, snapshot_id)
    )
    """,
    """
    CREATE TABLE rule_execution_errors (
        error_id INTEGER PRIMARY KEY,
        analysis_run_id TEXT NOT NULL,
        rule_id TEXT NOT NULL CHECK (trim(rule_id) <> ''),
        rule_name TEXT NOT NULL CHECK (trim(rule_name) <> ''),
        error_type TEXT NOT NULL CHECK (trim(error_type) <> ''),
        message TEXT NOT NULL CHECK (trim(message) <> '' AND length(message) <= 500),
        execution_order INTEGER CHECK (execution_order IS NULL OR execution_order >= 0),
        FOREIGN KEY (analysis_run_id) REFERENCES analysis_runs(analysis_run_id)
            ON DELETE CASCADE,
        UNIQUE (analysis_run_id, rule_id)
    )
    """,
)

_INDEX_STATEMENTS = (
    "CREATE INDEX idx_alerts_created_at ON alerts(created_at DESC)",
    "CREATE INDEX idx_rule_findings_run_order ON rule_findings(analysis_run_id, execution_order)",
    "CREATE INDEX idx_rule_findings_rule_id ON rule_findings(rule_id)",
    "CREATE INDEX idx_finding_evidence_order ON finding_evidence(finding_id, evidence_order)",
    # This named UNIQUE index supplies both the contract and uniqueness without a duplicate index.
    "CREATE UNIQUE INDEX idx_transaction_snapshots_run_transaction "
    "ON transaction_snapshots(analysis_run_id, canonical_transaction_id)",
)

_REQUIRED_COLUMNS = {
    "analysis_runs": frozenset({
        "analysis_run_id", "source_name", "ruleset_version", "created_at",
        "input_row_count", "fraud_score", "transaction_count", "finding_count",
        "error_count",
    }),
    "transaction_snapshots": frozenset({
        "snapshot_id", "analysis_run_id", "canonical_transaction_id",
        "source_position", "source_row_id_type", "source_row_id_text", "action",
        "amount", "actor_account",
    }),
    "alerts": frozenset({
        "alert_id", "analysis_run_id", "fraud_score", "risk_level", "status",
        "created_at",
    }),
    "rule_findings": frozenset({
        "finding_id", "analysis_run_id", "rule_id", "rule_name", "risk_type",
        "triggered", "rule_score", "reason", "execution_order",
    }),
    "finding_evidence": frozenset({
        "finding_evidence_id", "analysis_run_id", "finding_id", "snapshot_id",
        "evidence_order",
    }),
    "rule_execution_errors": frozenset({
        "error_id", "analysis_run_id", "rule_id", "rule_name", "error_type",
        "message", "execution_order",
    }),
}

_REQUIRED_INDEXES = frozenset({
    "idx_alerts_created_at",
    "idx_rule_findings_run_order",
    "idx_rule_findings_rule_id",
    "idx_finding_evidence_order",
    "idx_transaction_snapshots_run_transaction",
})


def configure_connection(connection: sqlite3.Connection) -> None:
    """Apply safe per-connection SQLite settings without taking ownership of it."""
    connection.execute("PRAGMA foreign_keys = ON")
    if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        raise SchemaValidationError("SQLite foreign key enforcement could not be enabled.")
    connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    # In-memory databases legitimately retain journal_mode=memory. File databases normally
    # return wal, while constrained/read-only environments may retain another safe mode.
    try:
        connection.execute("PRAGMA journal_mode = WAL").fetchone()
    except sqlite3.OperationalError:
        # WAL can be unavailable for read-only, locked, or otherwise constrained files.
        pass


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create or validate schema v1 atomically, leaving the connection open."""
    configure_connection(connection)
    version = _schema_version(connection)
    if version > SCHEMA_VERSION:
        raise UnsupportedSchemaVersionError(
            f"Database schema version {version} is newer than supported version {SCHEMA_VERSION}."
        )
    if version == SCHEMA_VERSION:
        validate_schema(connection)
        return
    if version != 0:
        raise SchemaValidationError(f"Invalid SQLite schema version: {version}.")

    connection.execute("SAVEPOINT fds_schema_v1")
    try:
        for statement in (*_TABLE_STATEMENTS, *_INDEX_STATEMENTS):
            connection.execute(statement)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        validate_schema(connection)
    except Exception:
        connection.execute("ROLLBACK TO SAVEPOINT fds_schema_v1")
        connection.execute("RELEASE SAVEPOINT fds_schema_v1")
        raise
    connection.execute("RELEASE SAVEPOINT fds_schema_v1")


def validate_schema(connection: sqlite3.Connection) -> None:
    """Validate the version, core objects, columns, and FK enforcement."""
    if _schema_version(connection) != SCHEMA_VERSION:
        raise SchemaValidationError(
            f"Expected SQLite schema version {SCHEMA_VERSION}."
        )
    if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        raise SchemaValidationError("SQLite foreign key enforcement is disabled.")

    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    missing_tables = set(_REQUIRED_COLUMNS) - tables
    if missing_tables:
        raise SchemaValidationError(
            f"Required SQLite tables are missing: {', '.join(sorted(missing_tables))}."
        )
    indexes = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name NOT LIKE 'sqlite_%'"
        )
    }
    missing_indexes = _REQUIRED_INDEXES - indexes
    if missing_indexes:
        raise SchemaValidationError(
            f"Required SQLite indexes are missing: {', '.join(sorted(missing_indexes))}."
        )
    for table, required in _REQUIRED_COLUMNS.items():
        actual = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        missing = required - actual
        if missing:
            raise SchemaValidationError(
                f"Required columns are missing from {table}: {', '.join(sorted(missing))}."
            )


def _schema_version(connection: sqlite3.Connection) -> int:
    row = connection.execute("PRAGMA user_version").fetchone()
    if row is None or type(row[0]) is not int or row[0] < 0:
        raise SchemaValidationError("SQLite returned an invalid schema version.")
    return row[0]

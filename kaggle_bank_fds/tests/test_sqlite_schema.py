"""Contract tests for SQLite persistence schema v1."""

from __future__ import annotations

import sqlite3

import pytest

from kaggle_bank_fds.src.persistence.sqlite_schema import (
    BUSY_TIMEOUT_MS,
    SCHEMA_VERSION,
    SchemaValidationError,
    UnsupportedSchemaVersionError,
    configure_connection,
    initialize_schema,
    validate_schema,
)


TABLES = {
    "analysis_runs", "transaction_snapshots", "alerts", "rule_findings",
    "finding_evidence", "rule_execution_errors",
}
INDEXES = {
    "idx_alerts_created_at", "idx_rule_findings_run_order",
    "idx_rule_findings_rule_id", "idx_finding_evidence_order",
    "idx_transaction_snapshots_run_transaction",
}


@pytest.fixture
def connection():
    value = sqlite3.connect(":memory:")
    initialize_schema(value)
    yield value
    value.close()


def _run(connection, run_id="run-1"):
    connection.execute(
        "INSERT INTO analysis_runs VALUES (?, 'source', 'v1', '2026-08-04T00:00:00Z', 1, 0.5, 1, 1, 0)",
        (run_id,),
    )


def _snapshot(connection, run_id="run-1", snapshot_id=1, transaction_id="tx-1", position=0,
              row_type="int", row_text="10"):
    connection.execute(
        """INSERT INTO transaction_snapshots
           (snapshot_id, analysis_run_id, canonical_transaction_id, source_position,
            source_row_id_type, source_row_id_text, action, amount, actor_account)
           VALUES (?, ?, ?, ?, ?, ?, 'TRANSFER', 10.0, 'actor')""",
        (snapshot_id, run_id, transaction_id, position, row_type, row_text),
    )


def _finding(connection, run_id="run-1", finding_id=1, rule_id="rule-1", order=0,
             triggered=1, score=30):
    connection.execute(
        """INSERT INTO rule_findings
           (finding_id, analysis_run_id, rule_id, rule_name, risk_type, triggered,
            rule_score, reason, execution_order)
           VALUES (?, ?, ?, 'Rule', 'risk', ?, ?, 'reason', ?)""",
        (finding_id, run_id, rule_id, triggered, score, order),
    )


def test_configure_connection_contract_and_idempotence():
    connection = sqlite3.connect(":memory:")
    configure_connection(connection)
    configure_connection(connection)
    assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == BUSY_TIMEOUT_MS
    assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "memory"
    assert connection.execute("SELECT 1").fetchone()[0] == 1
    connection.close()


def test_file_connection_uses_wal(tmp_path):
    path = tmp_path / "schema.sqlite3"
    connection = sqlite3.connect(path)
    configure_connection(connection)
    assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    connection.close()


def test_closed_connection_reports_sqlite_error():
    connection = sqlite3.connect(":memory:")
    connection.close()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        configure_connection(connection)


def test_schema_creation_version_objects_and_idempotence():
    connection = sqlite3.connect(":memory:")
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
    initialize_schema(connection)
    initialize_schema(connection)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    tables = {r[0] for r in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )}
    indexes = {r[0] for r in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
    )}
    assert tables == TABLES
    assert INDEXES <= indexes
    validate_schema(connection)


def test_future_version_is_rejected():
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA user_version = 2")
    with pytest.raises(UnsupportedSchemaVersionError, match="newer"):
        initialize_schema(connection)


def test_version_one_with_missing_schema_is_rejected():
    connection = sqlite3.connect(":memory:")
    configure_connection(connection)
    connection.execute("PRAGMA user_version = 1")
    with pytest.raises(SchemaValidationError, match="missing"):
        initialize_schema(connection)


def test_validation_rejects_missing_required_index(connection):
    connection.execute("DROP INDEX idx_alerts_created_at")
    with pytest.raises(SchemaValidationError, match="indexes are missing"):
        validate_schema(connection)


def test_validation_rejects_missing_required_column():
    connection = sqlite3.connect(":memory:")
    configure_connection(connection)
    initialize_schema(connection)
    connection.execute("ALTER TABLE alerts RENAME TO alerts_valid")
    connection.execute(
        "CREATE TABLE alerts (alert_id TEXT PRIMARY KEY, analysis_run_id TEXT, fraud_score REAL)"
    )
    with pytest.raises(SchemaValidationError, match="columns are missing"):
        validate_schema(connection)


def test_validation_rejects_disabled_foreign_keys(connection):
    connection.commit()
    connection.execute("PRAGMA foreign_keys = OFF")
    with pytest.raises(SchemaValidationError, match="disabled"):
        validate_schema(connection)


def test_migration_failure_rolls_back_all_new_objects_and_version():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE alerts (wrong INTEGER)")
    with pytest.raises(sqlite3.OperationalError, match="already exists"):
        initialize_schema(connection)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
    tables = {r[0] for r in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )}
    assert tables == {"alerts"}


def test_foreign_keys_are_enforced_after_initialize(connection):
    with pytest.raises(sqlite3.IntegrityError):
        _snapshot(connection, run_id="missing")


@pytest.mark.parametrize("values", [
    ("", 0.5, 0, 0, 0, 0),
    ("run", -0.1, 0, 0, 0, 0),
    ("run", 1.1, 0, 0, 0, 0),
    ("run", 0.5, -1, 0, 0, 0),
    ("run", 0.5, 0, -1, 0, 0),
    ("run", 0.5, 0, 0, -1, 0),
    ("run", 0.5, 0, 0, 0, -1),
])
def test_analysis_run_constraints(connection, values):
    run_id, score, inputs, transactions, findings, errors = values
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO analysis_runs VALUES (?, 's', 'v', 'now', ?, ?, ?, ?, ?)",
            (run_id, inputs, score, transactions, findings, errors),
        )


def test_transaction_uniqueness_scope_and_source_type(connection):
    _run(connection)
    _run(connection, "run-2")
    _snapshot(connection)
    with pytest.raises(sqlite3.IntegrityError):
        _snapshot(connection, snapshot_id=2, transaction_id="tx-1", position=1)
    with pytest.raises(sqlite3.IntegrityError):
        _snapshot(connection, snapshot_id=3, transaction_id="tx-3", position=0)
    _snapshot(connection, "run-2", 4, "tx-1", 0)
    with pytest.raises(sqlite3.IntegrityError):
        _snapshot(connection, "run-2", 5, "tx-5", 1, "float", "10")
    with pytest.raises(sqlite3.IntegrityError):
        _snapshot(connection, "run-2", 6, "tx-6", 2, "none", "10")
    with pytest.raises(sqlite3.IntegrityError):
        _snapshot(connection, "run-2", 7, "tx-7", 3, "int", None)


def test_alert_constraints(connection):
    _run(connection)
    connection.execute("INSERT INTO alerts VALUES ('a1', 'run-1', .5, NULL, 'OPEN', 'now')")
    for sql in (
        "INSERT INTO alerts VALUES ('a2', 'run-1', .5, NULL, 'OPEN', 'now')",
        "INSERT INTO alerts VALUES ('a3', 'missing', .5, NULL, 'OPEN', 'now')",
        "INSERT INTO alerts VALUES ('a4', 'run-1', .5, NULL, 'BAD', 'now')",
        "INSERT INTO alerts VALUES ('a5', 'run-1', .5, 'EXTREME', 'OPEN', 'now')",
        "INSERT INTO alerts VALUES ('a6', 'run-1', 1.1, NULL, 'OPEN', 'now')",
    ):
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(sql)


@pytest.mark.parametrize("triggered,score", [(2, 10), (0, 10), (1, 0)])
def test_finding_trigger_score_constraints(connection, triggered, score):
    _run(connection)
    with pytest.raises(sqlite3.IntegrityError):
        _finding(connection, triggered=triggered, score=score)


def test_finding_uniqueness_and_parent_constraints(connection):
    _run(connection)
    _finding(connection)
    with pytest.raises(sqlite3.IntegrityError):
        _finding(connection, finding_id=2, order=1)
    with pytest.raises(sqlite3.IntegrityError):
        _finding(connection, finding_id=3, rule_id="rule-3", order=0)
    with pytest.raises(sqlite3.IntegrityError):
        _finding(connection, "missing", 4, "rule-4", 4)


def test_evidence_constraints_and_cross_run_integrity(connection):
    _run(connection)
    _run(connection, "run-2")
    _snapshot(connection)
    _snapshot(connection, "run-2", 2, "tx-2", 0)
    _finding(connection)
    _finding(connection, "run-2", 2, "rule-2", 0)
    connection.execute("INSERT INTO finding_evidence VALUES (1, 'run-1', 1, 1, 0)")
    for values in (
        (2, "run-1", 1, 1, 1),       # duplicate snapshot link
        (3, "run-1", 1, 2, 1),       # cross-run snapshot
        (4, "run-2", 1, 2, 1),       # cross-run finding
        (5, "run-1", 999, 1, 1),     # missing finding
        (6, "run-1", 1, 999, 1),     # missing snapshot
        (7, "run-1", 1, 1, -1),      # invalid order
    ):
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("INSERT INTO finding_evidence VALUES (?, ?, ?, ?, ?)", values)


def test_error_constraints(connection):
    _run(connection)
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO rule_execution_errors VALUES (1, 'missing', 'r', 'R', 'E', 'm', 0)"
        )
    for message, order in (("x" * 501, 0), ("message", -1)):
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO rule_execution_errors VALUES (2, 'run-1', 'r', 'R', 'E', ?, ?)",
                (message, order),
            )


def test_analysis_run_delete_cascades_only_its_graph(connection):
    for run_id, offset in (("run-1", 0), ("run-2", 10)):
        _run(connection, run_id)
        _snapshot(connection, run_id, offset + 1, f"tx-{offset}", 0)
        _finding(connection, run_id, offset + 1, f"rule-{offset}", 0)
        connection.execute(
            "INSERT INTO finding_evidence VALUES (?, ?, ?, ?, 0)",
            (offset + 1, run_id, offset + 1, offset + 1),
        )
        connection.execute(
            "INSERT INTO alerts VALUES (?, ?, .5, NULL, 'OPEN', 'now')",
            (f"alert-{offset}", run_id),
        )
        connection.execute(
            "INSERT INTO rule_execution_errors VALUES (?, ?, ?, 'R', 'E', 'message', 1)",
            (offset + 1, run_id, f"error-{offset}"),
        )
    connection.execute("DELETE FROM analysis_runs WHERE analysis_run_id='run-1'")
    for table in TABLES:
        assert connection.execute(
            f"SELECT count(*) FROM {table} WHERE analysis_run_id='run-1'"
            if table != "analysis_runs" else
            "SELECT count(*) FROM analysis_runs WHERE analysis_run_id='run-1'"
        ).fetchone()[0] == 0
    assert connection.execute("SELECT count(*) FROM analysis_runs").fetchone()[0] == 1


def test_finding_delete_cascades_link_not_snapshot(connection):
    _run(connection)
    _snapshot(connection)
    _finding(connection)
    connection.execute("INSERT INTO finding_evidence VALUES (1, 'run-1', 1, 1, 0)")
    connection.execute("DELETE FROM rule_findings WHERE finding_id=1")
    assert connection.execute("SELECT count(*) FROM finding_evidence").fetchone()[0] == 0
    assert connection.execute("SELECT count(*) FROM transaction_snapshots").fetchone()[0] == 1


def test_source_row_id_type_and_text_preserve_int_str_and_none(connection):
    _run(connection)
    for snapshot_id, row_type, row_text in ((1, "int", "10"), (2, "str", "10"),
                                             (3, "none", None)):
        _snapshot(connection, snapshot_id=snapshot_id,
                  transaction_id=f"tx-{snapshot_id}", position=snapshot_id - 1,
                  row_type=row_type, row_text=row_text)
    assert connection.execute(
        "SELECT source_row_id_type, source_row_id_text FROM transaction_snapshots ORDER BY snapshot_id"
    ).fetchall() == [("int", "10"), ("str", "10"), ("none", None)]

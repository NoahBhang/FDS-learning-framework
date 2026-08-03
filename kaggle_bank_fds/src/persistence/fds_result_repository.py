"""Atomic SQLite writer for validated bank FDS analysis artifacts."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from types import TracebackType
from uuid import uuid4

from .persistence_models import AnalysisPersistenceArtifact, TransactionSnapshot
from .persistence_read_models import (
    AlertDetail,
    AlertSummary,
    AnalysisRunRecord,
    EvidenceRecord,
    RuleExecutionErrorRecord,
    RuleFindingRecord,
)
from .sqlite_schema import (
    SchemaValidationError,
    configure_connection,
    initialize_schema,
    validate_schema,
)


class RepositoryClosedError(RuntimeError):
    """Raised when repository work is requested after its lifecycle ended."""


class FdsResultRepository:
    """Persist one immutable analysis artifact as a single SQLite transaction."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        alert_id_factory: Callable[[], str] | None = None,
        _owns_connection: bool = False,
    ) -> None:
        if not isinstance(connection, sqlite3.Connection):
            raise TypeError("connection must be sqlite3.Connection.")
        self._connection = connection
        self._owns_connection = _owns_connection
        self._closed = False
        self._alert_id_factory = alert_id_factory or (lambda: str(uuid4()))
        self._ensure_connection_open()
        configure_connection(connection)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        alert_id_factory: Callable[[], str] | None = None,
    ) -> FdsResultRepository:
        """Open an owned file connection; the parent directory must already exist."""
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be str or pathlib.Path.")
        connection = sqlite3.connect(Path(path))
        try:
            return cls(
                connection,
                alert_id_factory=alert_id_factory,
                _owns_connection=True,
            )
        except Exception:
            connection.close()
            raise

    def initialize_schema(self) -> None:
        self._ensure_open()
        initialize_schema(self._connection)

    def save_analysis(self, artifact: AnalysisPersistenceArtifact) -> str:
        self._ensure_open()
        validate_schema(self._connection)
        prepared = self._prepare_artifact(artifact)

        self._connection.execute("SAVEPOINT fds_save_analysis")
        try:
            self._insert_analysis_run(artifact, prepared.created_at)
            snapshot_ids = self._insert_transactions(
                artifact.analysis_run_id, prepared.transactions
            )
            if prepared.create_alert:
                self._insert_alert(artifact, prepared.alert_id, prepared.created_at)
            finding_ids = self._insert_findings(
                artifact.analysis_run_id, prepared.findings
            )
            self._insert_evidence(
                artifact.analysis_run_id, prepared.findings, finding_ids, snapshot_ids
            )
            self._insert_errors(artifact.analysis_run_id, prepared.errors)
        except Exception:
            self._connection.execute("ROLLBACK TO SAVEPOINT fds_save_analysis")
            self._connection.execute("RELEASE SAVEPOINT fds_save_analysis")
            raise
        self._connection.execute("RELEASE SAVEPOINT fds_save_analysis")
        return artifact.analysis_run_id

    def get_analysis_run(self, analysis_run_id: str) -> AnalysisRunRecord | None:
        self._prepare_read()
        run_id = _required_identifier(analysis_run_id, "analysis_run_id")
        row = self._connection.execute(
            """SELECT analysis_run_id, source_name, ruleset_version, created_at,
                      input_row_count, fraud_score, transaction_count, finding_count,
                      error_count
               FROM analysis_runs WHERE analysis_run_id = ?""",
            (run_id,),
        ).fetchone()
        return None if row is None else _analysis_run_record(row)

    def list_alerts(self, *, limit: int = 100) -> tuple[AlertSummary, ...]:
        self._prepare_read()
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer.")
        if not 1 <= limit <= 1_000:
            raise ValueError("limit must be between 1 and 1000.")
        rows = self._connection.execute(
            """SELECT a.alert_id, a.analysis_run_id, a.fraud_score, a.risk_level,
                      a.status, a.created_at,
                      COALESCE(SUM(CASE WHEN f.triggered = 1 THEN 1 ELSE 0 END), 0)
               FROM alerts AS a
               LEFT JOIN rule_findings AS f ON f.analysis_run_id = a.analysis_run_id
               GROUP BY a.alert_id, a.analysis_run_id, a.fraud_score, a.risk_level,
                        a.status, a.created_at
               ORDER BY a.created_at DESC, a.alert_id DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return tuple(_alert_summary(row) for row in rows)

    def get_alert_by_run_id(self, analysis_run_id: str) -> AlertSummary | None:
        self._prepare_read()
        run_id = _required_identifier(analysis_run_id, "analysis_run_id")
        row = self._connection.execute(
            """SELECT a.alert_id, a.analysis_run_id, a.fraud_score, a.risk_level,
                      a.status, a.created_at,
                      COALESCE(SUM(CASE WHEN f.triggered = 1 THEN 1 ELSE 0 END), 0)
               FROM alerts AS a
               LEFT JOIN rule_findings AS f ON f.analysis_run_id = a.analysis_run_id
               WHERE a.analysis_run_id = ?
               GROUP BY a.alert_id, a.analysis_run_id, a.fraud_score, a.risk_level,
                        a.status, a.created_at""",
            (run_id,),
        ).fetchone()
        return None if row is None else _alert_summary(row)

    def get_alert_detail(self, alert_id: str) -> AlertDetail | None:
        self._prepare_read()
        normalized_id = _required_identifier(alert_id, "alert_id")
        self._connection.execute("SAVEPOINT fds_read_alert_detail")
        try:
            header = self._connection.execute(
                """SELECT a.alert_id, a.analysis_run_id, a.fraud_score, a.risk_level,
                          a.status, a.created_at,
                          r.source_name, r.ruleset_version, r.created_at,
                          r.input_row_count, r.fraud_score, r.transaction_count,
                          r.finding_count, r.error_count,
                          (SELECT COUNT(*) FROM rule_findings AS tf
                           WHERE tf.analysis_run_id = a.analysis_run_id AND tf.triggered = 1)
                   FROM alerts AS a
                   JOIN analysis_runs AS r ON r.analysis_run_id = a.analysis_run_id
                   WHERE a.alert_id = ?""",
                (normalized_id,),
            ).fetchone()
            if header is None:
                self._connection.execute("RELEASE SAVEPOINT fds_read_alert_detail")
                return None
            finding_rows = self._connection.execute(
                """SELECT finding_id, rule_id, rule_name, risk_type, triggered,
                          rule_score, reason, execution_order
                   FROM rule_findings WHERE analysis_run_id = ?
                   ORDER BY execution_order ASC, finding_id ASC""",
                (header[1],),
            ).fetchall()
            evidence_rows = self._connection.execute(
                """SELECT e.finding_id, t.canonical_transaction_id, t.source_position,
                          t.source_row_id_type, t.source_row_id_text, t.step,
                          t.transaction_datetime, t.action, t.amount, t.actor_account,
                          t.target_account, t.counterparty_account, t.old_balance_actor,
                          t.new_balance_actor, t.old_balance_target, t.new_balance_target,
                          t.description, t.bank_code, t.source_format, e.evidence_order
                   FROM finding_evidence AS e
                   JOIN transaction_snapshots AS t
                     ON t.analysis_run_id = e.analysis_run_id
                    AND t.snapshot_id = e.snapshot_id
                   WHERE e.analysis_run_id = ?
                   ORDER BY e.finding_id ASC, e.evidence_order ASC""",
                (header[1],),
            ).fetchall()
            error_rows = self._connection.execute(
                """SELECT rule_id, rule_name, error_type, message, execution_order
                   FROM rule_execution_errors WHERE analysis_run_id = ?
                   ORDER BY execution_order IS NULL ASC, execution_order ASC,
                            error_id ASC""",
                (header[1],),
            ).fetchall()
            detail = _alert_detail(header, finding_rows, evidence_rows, error_rows)
        except Exception:
            self._connection.execute("ROLLBACK TO SAVEPOINT fds_read_alert_detail")
            self._connection.execute("RELEASE SAVEPOINT fds_read_alert_detail")
            raise
        self._connection.execute("RELEASE SAVEPOINT fds_read_alert_detail")
        return detail

    def _prepare_read(self) -> None:
        self._ensure_open()
        validate_schema(self._connection)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_connection:
            self._connection.close()

    def __enter__(self) -> FdsResultRepository:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RepositoryClosedError("FDS result repository is closed.")
        self._ensure_connection_open()

    def _ensure_connection_open(self) -> None:
        try:
            self._connection.execute("SELECT 1")
        except sqlite3.ProgrammingError as exc:
            raise RepositoryClosedError("SQLite connection is closed.") from exc

    def _prepare_artifact(self, artifact: AnalysisPersistenceArtifact) -> _PreparedArtifact:
        if type(artifact) is not AnalysisPersistenceArtifact:
            raise TypeError("artifact must be AnalysisPersistenceArtifact.")
        finding_ids = {value.rule_id for value in artifact.rule_findings}
        error_ids = {value.rule_id for value in artifact.rule_errors}
        if finding_ids & error_ids:
            raise ValueError("A rule cannot be both a finding and an execution error.")

        transactions = tuple(sorted(artifact.transactions, key=lambda value: value.source_position))
        canonical_ids = {value.canonical_transaction_id for value in transactions}
        for finding in artifact.rule_findings:
            unknown_ids = set(finding.evidence_transaction_ids) - canonical_ids
            if unknown_ids:
                raise ValueError("Finding evidence references an unknown transaction.")
        for transaction in transactions:
            _serialize_optional_datetime(transaction.transaction_datetime)
            _source_row_id_columns(transaction.source_row_id)

        findings = tuple(sorted(artifact.rule_findings, key=lambda value: value.execution_order))
        errors = tuple(sorted(artifact.rule_errors, key=lambda value: value.execution_order))
        create_alert = artifact.fraud_score > 0.0 or any(value.triggered for value in findings)
        alert_id = None
        if create_alert:
            alert_id = self._alert_id_factory()
            if not isinstance(alert_id, str) or not alert_id.strip():
                raise ValueError("alert_id_factory must return a non-empty string.")
        return _PreparedArtifact(
            created_at=_serialize_datetime(artifact.created_at),
            transactions=transactions,
            findings=findings,
            errors=errors,
            create_alert=create_alert,
            alert_id=alert_id,
        )

    def _insert_analysis_run(self, artifact: AnalysisPersistenceArtifact, created_at: str) -> None:
        self._connection.execute(
            """INSERT INTO analysis_runs
               (analysis_run_id, source_name, ruleset_version, created_at,
                input_row_count, fraud_score, transaction_count, finding_count, error_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                artifact.analysis_run_id, artifact.source_name, artifact.ruleset_version,
                created_at, artifact.input_row_count, artifact.fraud_score,
                len(artifact.transactions), len(artifact.rule_findings), len(artifact.rule_errors),
            ),
        )

    def _insert_transactions(
        self, analysis_run_id: str, transactions: tuple[TransactionSnapshot, ...]
    ) -> dict[str, int]:
        snapshot_ids: dict[str, int] = {}
        for value in transactions:
            source_type, source_text = _source_row_id_columns(value.source_row_id)
            cursor = self._connection.execute(
                """INSERT INTO transaction_snapshots
                   (analysis_run_id, canonical_transaction_id, source_position,
                    source_row_id_type, source_row_id_text, step, transaction_datetime,
                    action, amount, actor_account, target_account, counterparty_account,
                    old_balance_actor, new_balance_actor, old_balance_target,
                    new_balance_target, description, bank_code, source_format)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    analysis_run_id, value.canonical_transaction_id,
                    value.source_position, source_type, source_text, value.step,
                    _serialize_optional_datetime(value.transaction_datetime), value.action_type,
                    value.amount, value.actor_account, value.target_account,
                    value.counterparty_name, value.balance_before, value.balance_after,
                    value.target_balance_before, value.target_balance_after, value.description,
                    value.bank_name, value.source_format,
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a transaction snapshot ID.")
            snapshot_ids[value.canonical_transaction_id] = cursor.lastrowid
        return snapshot_ids

    def _insert_alert(
        self, artifact: AnalysisPersistenceArtifact, alert_id: str | None, created_at: str
    ) -> None:
        self._connection.execute(
            """INSERT INTO alerts
               (alert_id, analysis_run_id, fraud_score, risk_level, status, created_at)
               VALUES (?, ?, ?, NULL, 'OPEN', ?)""",
            (alert_id, artifact.analysis_run_id, artifact.fraud_score, created_at),
        )

    def _insert_findings(self, analysis_run_id: str, findings: tuple) -> dict[str, int]:
        finding_ids: dict[str, int] = {}
        for value in findings:
            cursor = self._connection.execute(
                """INSERT INTO rule_findings
                   (analysis_run_id, rule_id, rule_name, risk_type, triggered,
                    rule_score, reason, execution_order)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    analysis_run_id, value.rule_id, value.rule_name, value.risk_type,
                    int(value.triggered), value.rule_score, value.reason, value.execution_order,
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a rule finding ID.")
            finding_ids[value.rule_id] = cursor.lastrowid
        return finding_ids

    def _insert_evidence(
        self, analysis_run_id: str, findings: tuple, finding_ids: dict, snapshot_ids: dict
    ) -> None:
        for finding in findings:
            for evidence_order, transaction_id in enumerate(finding.evidence_transaction_ids):
                snapshot_id = snapshot_ids.get(transaction_id)
                if snapshot_id is None:
                    raise ValueError("Finding evidence references an unknown transaction.")
                self._connection.execute(
                    """INSERT INTO finding_evidence
                       (analysis_run_id, finding_id, snapshot_id, evidence_order)
                       VALUES (?, ?, ?, ?)""",
                    (
                        analysis_run_id, finding_ids[finding.rule_id], snapshot_id,
                        evidence_order,
                    ),
                )

    def _insert_errors(self, analysis_run_id: str, errors: tuple) -> None:
        for value in errors:
            self._connection.execute(
                """INSERT INTO rule_execution_errors
                   (analysis_run_id, rule_id, rule_name, error_type, message, execution_order)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    analysis_run_id, value.rule_id, value.rule_name, value.error_type,
                    value.message, value.execution_order,
                ),
            )


class _PreparedArtifact:
    def __init__(self, *, created_at, transactions, findings, errors, create_alert, alert_id):
        self.created_at = created_at
        self.transactions = transactions
        self.findings = findings
        self.errors = errors
        self.create_alert = create_alert
        self.alert_id = alert_id


def _source_row_id_columns(value: str | int | None) -> tuple[str, str | None]:
    if value is None:
        return "none", None
    if isinstance(value, bool):
        raise TypeError("source_row_id must not be bool.")
    if isinstance(value, int):
        return "int", str(value)
    if isinstance(value, str):
        return "str", value
    raise TypeError("source_row_id must be str, int, or None.")


def _serialize_datetime(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime values must be timezone-aware.")
    return value.astimezone(timezone.utc).isoformat()


def _serialize_optional_datetime(value: datetime | None) -> str | None:
    return None if value is None else _serialize_datetime(value)


def _required_identifier(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")
    return value


def _parse_datetime(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise SchemaValidationError(f"Stored {name} must be ISO-8601 text.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise SchemaValidationError(f"Stored {name} is not valid ISO-8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SchemaValidationError(f"Stored {name} must be timezone-aware.")
    return parsed.astimezone(timezone.utc)


def _parse_optional_datetime(value: object, name: str) -> datetime | None:
    return None if value is None else _parse_datetime(value, name)


def _restore_source_row_id(source_type: object, source_text: object) -> str | int | None:
    if source_type == "none" and source_text is None:
        return None
    if source_type == "str" and isinstance(source_text, str):
        return source_text
    if source_type == "int" and isinstance(source_text, str):
        try:
            return int(source_text)
        except ValueError as exc:
            raise SchemaValidationError("Stored integer source row ID is invalid.") from exc
    raise SchemaValidationError("Stored source row ID type/text combination is invalid.")


def _analysis_run_record(row: tuple) -> AnalysisRunRecord:
    return AnalysisRunRecord(
        analysis_run_id=row[0], source_name=row[1], ruleset_version=row[2],
        created_at=_parse_datetime(row[3], "analysis run created_at"),
        input_row_count=row[4], fraud_score=row[5], transaction_count=row[6],
        finding_count=row[7], error_count=row[8],
    )


def _alert_summary(row: tuple) -> AlertSummary:
    return AlertSummary(
        alert_id=row[0], analysis_run_id=row[1], fraud_score=row[2], risk_level=row[3],
        status=row[4], created_at=_parse_datetime(row[5], "alert created_at"),
        triggered_rule_count=row[6],
    )


def _evidence_record(row: tuple) -> EvidenceRecord:
    return EvidenceRecord(
        canonical_transaction_id=row[1], source_position=row[2],
        source_row_id=_restore_source_row_id(row[3], row[4]), step=row[5],
        transaction_datetime=_parse_optional_datetime(row[6], "transaction datetime"),
        action=row[7], amount=row[8], actor_account=row[9], target_account=row[10],
        counterparty_account=row[11], old_balance_actor=row[12], new_balance_actor=row[13],
        old_balance_target=row[14], new_balance_target=row[15], description=row[16],
        bank_code=row[17], source_format=row[18], evidence_order=row[19],
    )


def _alert_detail(header: tuple, finding_rows: list, evidence_rows: list, error_rows: list) -> AlertDetail:
    summary = _alert_summary((*header[:6], header[14]))
    run = _analysis_run_record((header[1], header[6], header[7], *header[8:14]))
    grouped: dict[int, list[EvidenceRecord]] = {row[0]: [] for row in finding_rows}
    for row in evidence_rows:
        if row[0] not in grouped:
            raise SchemaValidationError("Evidence references an unknown finding.")
        grouped[row[0]].append(_evidence_record(row))
    findings = []
    for row in finding_rows:
        if row[4] not in (0, 1):
            raise SchemaValidationError("Stored finding triggered value is invalid.")
        evidence = tuple(grouped[row[0]])
        if row[4] == 1 and not evidence:
            raise SchemaValidationError("Triggered finding has no evidence.")
        if row[4] == 0 and evidence:
            raise SchemaValidationError("Clean finding unexpectedly has evidence.")
        findings.append(RuleFindingRecord(
            rule_id=row[1], rule_name=row[2], risk_type=row[3], triggered=bool(row[4]),
            rule_score=row[5], reason=row[6], execution_order=row[7], evidence=evidence,
        ))
    errors = tuple(RuleExecutionErrorRecord(*row) for row in error_rows)
    if {value.rule_id for value in findings} & {value.rule_id for value in errors}:
        raise SchemaValidationError("A rule is stored as both finding and execution error.")
    return AlertDetail(summary=summary, analysis_run=run, findings=tuple(findings), errors=errors)

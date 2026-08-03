"""Typed persistence boundary for bank FDS analysis results."""

from .analysis_artifact_builder import (
    analyze_with_plugins_for_persistence,
    build_analysis_artifact,
)
from .persistence_models import (
    AnalysisPersistenceArtifact,
    RuleExecutionErrorSnapshot,
    RuleFindingSnapshot,
    TransactionSnapshot,
)
from .fds_result_repository import FdsResultRepository, RepositoryClosedError
from .persistence_read_models import (
    AlertDetail,
    AlertSummary,
    AnalysisRunRecord,
    EvidenceRecord,
    RuleExecutionErrorRecord,
    RuleFindingRecord,
)
from .sqlite_schema import (
    BUSY_TIMEOUT_MS,
    SCHEMA_VERSION,
    SchemaValidationError,
    UnsupportedSchemaVersionError,
    configure_connection,
    initialize_schema,
    validate_schema,
)

__all__ = [
    "AnalysisPersistenceArtifact",
    "RuleExecutionErrorSnapshot",
    "RuleFindingSnapshot",
    "TransactionSnapshot",
    "FdsResultRepository",
    "RepositoryClosedError",
    "AlertDetail",
    "AlertSummary",
    "AnalysisRunRecord",
    "EvidenceRecord",
    "RuleExecutionErrorRecord",
    "RuleFindingRecord",
    "BUSY_TIMEOUT_MS",
    "SCHEMA_VERSION",
    "SchemaValidationError",
    "UnsupportedSchemaVersionError",
    "analyze_with_plugins_for_persistence",
    "build_analysis_artifact",
    "configure_connection",
    "initialize_schema",
    "validate_schema",
]

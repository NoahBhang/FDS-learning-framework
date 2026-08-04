"""Streamlit operations shell for the explainable Bank FDS."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import logging
import sqlite3
from typing import Iterator

import streamlit as st

from kaggle_bank_fds.src.persistence import (
    AlertDetail,
    FdsResultRepository,
    RepositoryClosedError,
    SchemaValidationError,
    UnsupportedSchemaVersionError,
)
from kaggle_bank_fds.src.services.fds_analysis_service import (
    AnalysisServiceIntegrityError,
    AnalysisServiceResult,
    FdsAnalysisService,
)
from kaggle_bank_fds.src.ui.app_config import (
    AppConfig,
    prepare_db_parent_directory,
    resolve_bank_fds_db_path,
)
from kaggle_bank_fds.src.ui.csv_preflight import (
    CsvPreflightError,
    ParsedCsvUpload,
    parse_and_validate_paysim_csv,
)
from kaggle_bank_fds.src.ui.presenters import (
    AnalysisSummaryView,
    RISK_SCORE_CAPTION,
    to_alert_detail,
    to_alert_summary,
    to_analysis_summary,
    to_error_views,
    to_evidence_table,
    to_finding_views,
)


_SESSION_DEFAULTS = {
    "bank_fds_upload_fingerprint": None,
    "bank_fds_last_run_id": None,
    "bank_fds_last_alert_id": None,
    "bank_fds_last_summary": None,
    "bank_fds_selected_alert_id": None,
    "bank_fds_completed_action_token": None,
}
_REPOSITORY_ERRORS = (
    SchemaValidationError,
    UnsupportedSchemaVersionError,
    RepositoryClosedError,
    sqlite3.Error,
    OSError,
)
_LOGGER = logging.getLogger(__name__)


@contextmanager
def _open_repository(config: AppConfig) -> Iterator[FdsResultRepository]:
    """Open, initialize, and reliably close one operation-scoped repository."""
    with FdsResultRepository.from_path(config.db_path) as repository:
        repository.initialize_schema()
        yield repository


def _initialize_session_state() -> None:
    for key, value in _SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _upload_fingerprint(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _reset_last_analysis() -> None:
    st.session_state.bank_fds_last_run_id = None
    st.session_state.bank_fds_last_alert_id = None
    st.session_state.bank_fds_last_summary = None
    st.session_state.bank_fds_completed_action_token = None


def _run_analysis(
    parsed: ParsedCsvUpload,
    config: AppConfig,
) -> AnalysisServiceResult:
    """Analyze and persist once using one operation-scoped repository."""
    with _open_repository(config) as repository:
        service = FdsAnalysisService(
            repository,
            source_name=config.source_name,
            ruleset_version=config.ruleset_version,
        )
        return service.analyze_and_persist(parsed.dataframe)


def _load_alert_detail(config: AppConfig, alert_id: str) -> AlertDetail | None:
    with _open_repository(config) as repository:
        return repository.get_alert_detail(alert_id)


def _render_sidebar(config: AppConfig, *, database_ready: bool) -> None:
    with st.sidebar:
        st.header("시스템 정보")
        st.write("시스템 유형: Explainable Rule-Based FDS")
        st.write("데이터: PaySim synthetic")
        st.write(f"Ruleset: {config.ruleset_version}")
        st.write(f"최대 업로드 행 수: {config.max_upload_rows:,}")
        st.write(f"Evidence 기본 표시 수: {config.max_evidence_rows:,}")
        st.write("SQLite schema v1")
        if database_ready:
            st.success("DB 준비 완료")
        else:
            st.error("DB 초기화 실패")


def _render_analysis_tab(config: AppConfig) -> None:
    st.subheader("새 분석")
    st.caption(
        f"PaySim CSV 파일을 업로드해 주세요. 최대 {config.max_upload_rows:,}행, "
        f"{config.max_upload_bytes // (1024 * 1024)}MiB까지 분석할 수 있습니다."
    )
    uploaded_file = st.file_uploader(
        "PaySim 거래 CSV 업로드",
        type=["csv"],
        key="bank_fds_csv_uploader",
    )
    if uploaded_file is None:
        st.info("분석할 PaySim CSV 파일을 업로드해 주세요.")
        st.button("분석 실행", disabled=True, key="bank_fds_analyze_button")
        return

    file_bytes = uploaded_file.getvalue()
    file_size = getattr(uploaded_file, "size", len(file_bytes))
    fingerprint = _upload_fingerprint(file_bytes)
    if st.session_state.bank_fds_upload_fingerprint != fingerprint:
        st.session_state.bank_fds_upload_fingerprint = fingerprint
        _reset_last_analysis()
    st.write(f"파일명: {uploaded_file.name}")
    st.write(f"파일 크기: {file_size:,} bytes")
    try:
        parsed = parse_and_validate_paysim_csv(
            file_bytes,
            filename=uploaded_file.name,
            file_size_bytes=file_size,
            max_rows=config.max_upload_rows,
            max_bytes=config.max_upload_bytes,
            preview_rows=config.preview_rows,
        )
    except CsvPreflightError as exc:
        st.error(str(exc))
        st.button("분석 실행", disabled=True, key="bank_fds_analyze_button")
        return

    row_metric, column_metric = st.columns(2)
    row_metric.metric("거래 행 수", f"{parsed.row_count:,}")
    column_metric.metric("컬럼 수", f"{parsed.column_count:,}")
    st.dataframe(parsed.preview, width="stretch", hide_index=True)
    st.success("CSV 사전 검증이 완료되었습니다.")
    completed = st.session_state.bank_fds_completed_action_token is not None
    completed_for_upload = (
        completed
        and st.session_state.bank_fds_completed_action_token.startswith(f"{fingerprint}:")
    )
    if completed_for_upload:
        run_requested = st.button(
            "같은 파일 다시 분석",
            key="bank_fds_reanalyze_button",
        )
    else:
        run_requested = st.button(
            "분석 실행",
            key="bank_fds_analyze_button",
        )

    current_result = None
    if run_requested:
        try:
            with st.spinner("거래를 분석하고 결과를 저장하는 중입니다."):
                current_result = _run_analysis(parsed, config)
        except AnalysisServiceIntegrityError as exc:
            _LOGGER.warning("Analysis integrity failure: %s", type(exc).__name__)
            st.error("분석 결과 검증에 실패했습니다. 다시 실행해 주세요.")
        except (ValueError, TypeError) as exc:
            _LOGGER.warning("Analysis input failure: %s", type(exc).__name__)
            st.error("거래 입력 형식을 확인한 뒤 다시 실행해 주세요.")
        except _REPOSITORY_ERRORS as exc:
            _LOGGER.warning("Analysis persistence failure: %s", type(exc).__name__)
            st.error("분석 결과를 DB에 저장하지 못했습니다. DB 상태를 확인해 주세요.")
        except Exception as exc:  # defensive UI boundary
            _LOGGER.warning("Unexpected analysis failure: %s", type(exc).__name__)
            st.error("분석을 완료하지 못했습니다. 잠시 후 다시 시도해 주세요.")
        else:
            summary = to_analysis_summary(current_result)
            st.session_state.bank_fds_last_run_id = summary.analysis_run_id
            st.session_state.bank_fds_last_alert_id = summary.alert_id
            st.session_state.bank_fds_last_summary = summary
            st.session_state.bank_fds_completed_action_token = (
                f"{fingerprint}:{summary.analysis_run_id}"
            )
            st.success("분석과 결과 저장이 완료되었습니다.")

    _render_current_analysis(config, current_result=current_result)


def _render_current_analysis(
    config: AppConfig,
    *,
    current_result: AnalysisServiceResult | None,
) -> None:
    summary = (
        to_analysis_summary(current_result)
        if current_result is not None
        else st.session_state.bank_fds_last_summary
    )
    if not isinstance(summary, AnalysisSummaryView):
        return

    _render_analysis_summary(summary)
    if current_result is not None:
        if current_result.alert is not None:
            _render_alert_detail(current_result.alert, config=config)
        else:
            _render_snapshot_findings_and_errors(current_result, config=config)
        return

    alert_id = st.session_state.bank_fds_last_alert_id
    if alert_id is None:
        st.info(
            "정상 분석 결과가 저장되었습니다. 상세 clean finding 재조회는 "
            "현재 v1 범위에서 제공하지 않습니다."
        )
        return
    try:
        detail = _load_alert_detail(config, alert_id)
    except _REPOSITORY_ERRORS:
        st.error("저장된 분석 상세를 불러오지 못했습니다.")
        return
    if detail is None:
        st.error("저장된 분석 상세를 찾을 수 없습니다.")
        return
    _render_alert_detail(detail, config=config)


def _render_analysis_summary(summary: AnalysisSummaryView) -> None:
    st.divider()
    st.subheader("분석 결과")
    score, transactions, triggered, errors = st.columns(4)
    score.metric("규칙 기반 위험점수", summary.score_text)
    transactions.metric("거래 수", f"{summary.transaction_count:,}")
    triggered.metric("탐지 Rule 수", summary.triggered_rule_count)
    errors.metric("오류 Rule 수", summary.error_count)
    st.write(summary.status_text)
    st.caption(RISK_SCORE_CAPTION)
    st.write(f"분석 시각: {summary.created_at_text}")
    with st.expander("분석 기술 정보", expanded=False):
        st.write(f"분석 Run ID: {summary.analysis_run_id}")
        st.write(f"Alert ID: {summary.alert_id or '미생성'}")
        st.write(f"데이터 Source: {summary.source_name}")
        st.write(f"Ruleset: {summary.ruleset_version}")


def _render_alert_detail(detail: AlertDetail, *, config: AppConfig) -> None:
    view = to_alert_detail(detail)
    finding_by_id = {finding.rule_id: finding for finding in detail.findings}
    _render_findings(
        view.triggered_findings,
        finding_by_id=finding_by_id,
        config=config,
        triggered=True,
    )
    _render_findings(
        view.clean_findings,
        finding_by_id=finding_by_id,
        config=config,
        triggered=False,
    )
    _render_errors(view.errors)


def _render_snapshot_findings_and_errors(
    result: AnalysisServiceResult,
    *,
    config: AppConfig,
) -> None:
    del config  # clean snapshots have no Evidence records to materialize
    findings = to_finding_views(result.artifact.rule_findings)
    clean = tuple(finding for finding in findings if not finding.triggered)
    triggered = tuple(finding for finding in findings if finding.triggered)
    _render_findings(triggered, finding_by_id={}, config=None, triggered=True)
    _render_findings(clean, finding_by_id={}, config=None, triggered=False)
    _render_errors(to_error_views(result.artifact.rule_errors))


def _render_findings(
    findings,
    *,
    finding_by_id: dict,
    config: AppConfig | None,
    triggered: bool,
) -> None:
    if not findings:
        return
    st.subheader("탐지된 Rule" if triggered else "정상 판정 Rule")
    if triggered:
        st.caption("개별 Rule 점수이며 최종 종합점수와 다를 수 있습니다.")
    for finding in findings:
        evidence_label = f" · Evidence {finding.evidence_count}건" if triggered else ""
        with st.expander(
            f"{finding.rule_name} · {finding.rule_score_text}{evidence_label}",
            expanded=triggered,
        ):
            st.write(f"Rule ID: {finding.rule_id}")
            st.write(f"Risk type: {finding.risk_type}")
            st.write(f"Execution order: {finding.execution_order}")
            st.write(f"Reason: {finding.reason}")
            st.write(f"Evidence count: {finding.evidence_count}")
            record = finding_by_id.get(finding.rule_id)
            if triggered and record is not None and config is not None:
                _render_evidence(record.evidence, limit=config.max_evidence_rows)


def _render_evidence(evidence, *, limit: int) -> None:
    table = to_evidence_table(evidence, limit=limit)
    st.dataframe(
        [
            {
                "순서": row.evidence_order,
                "거래 ID": row.canonical_transaction_id,
                "단계": row.step,
                "거래 유형": row.action,
                "금액": row.amount_text,
                "출발 계좌": row.actor_account,
                "도착 계좌": row.target_account,
            }
            for row in table.rows
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(table.caption)
    with st.expander("Evidence 상세 필드", expanded=False):
        st.dataframe(
            [
                {
                    "순서": row.evidence_order,
                    "거래 ID": row.canonical_transaction_id,
                    "Source row ID": row.source_row_id,
                    "거래 시각": row.transaction_datetime_text,
                    "상대 계좌": row.counterparty_account,
                    "출발 잔액(전)": row.old_balance_actor,
                    "출발 잔액(후)": row.new_balance_actor,
                    "도착 잔액(전)": row.old_balance_target,
                    "도착 잔액(후)": row.new_balance_target,
                    "설명": row.description,
                    "은행 코드": row.bank_code,
                    "Source format": row.source_format,
                }
                for row in table.rows
            ],
            width="stretch",
            hide_index=True,
        )


def _render_errors(errors) -> None:
    if not errors:
        return
    st.warning(
        "일부 Rule 실행 중 오류가 있었지만 나머지 Rule 분석과 저장은 완료되었습니다."
    )
    for error in errors:
        st.write(
            f"{error.rule_name} ({error.rule_id}) · {error.error_type} · "
            f"순서 {error.execution_order_text}"
        )
        st.write(error.message)


def _render_history_tab(config: AppConfig) -> None:
    st.subheader("Alert 이력")
    selected_detail = None
    try:
        with _open_repository(config) as repository:
            alerts = repository.list_alerts(limit=config.alert_list_limit)
            if not alerts:
                st.session_state.bank_fds_selected_alert_id = None
                st.session_state.pop("bank_fds_alert_selector", None)
            else:
                views = tuple(to_alert_summary(alert) for alert in alerts)
                _render_alert_list(views)
                selected_id = _select_alert(views)
                selected_detail = repository.get_alert_detail(selected_id)
                if selected_detail is None:
                    st.session_state.bank_fds_selected_alert_id = None
                    st.session_state.pop("bank_fds_alert_selector", None)
    except UnsupportedSchemaVersionError as exc:
        _LOGGER.warning("Unsupported history schema: %s", type(exc).__name__)
        st.error("지원하지 않는 데이터베이스 버전입니다. DB 설정을 확인해 주세요.")
        return
    except SchemaValidationError as exc:
        _LOGGER.warning("Invalid stored alert data: %s", type(exc).__name__)
        st.error("저장된 Alert 데이터를 읽을 수 없습니다. 목록을 다시 확인해 주세요.")
        return
    except (sqlite3.Error, OSError) as exc:
        _LOGGER.warning("History database failure: %s", type(exc).__name__)
        st.error("Alert 이력 조회 중 데이터베이스 오류가 발생했습니다. 다시 시도해 주세요.")
        return
    except Exception as exc:  # defensive UI boundary
        _LOGGER.warning("Unexpected history failure: %s", type(exc).__name__)
        st.error("Alert 이력을 불러오지 못했습니다. 다시 시도해 주세요.")
        return

    if not alerts:
        st.info(
            "저장된 Alert가 없습니다. 새 분석에서 위험 패턴이 탐지되면 "
            "여기에 표시됩니다."
        )
        return
    if selected_detail is None:
        st.warning("선택한 Alert를 찾을 수 없습니다. 목록을 새로 확인해 주세요.")
        return
    _render_history_detail(selected_detail, config=config)


def _render_alert_list(views) -> None:
    st.dataframe(
        [
            {
                "분석 시각": view.created_at_text,
                "규칙 기반 위험점수": view.risk_score_text,
                "상태": view.status_text,
                "위험등급": view.risk_level_text,
                "탐지 Rule 수": view.triggered_rule_count,
                "Alert ID": view.alert_id,
                "Analysis Run ID": view.analysis_run_id,
            }
            for view in views
        ],
        width="stretch",
        hide_index=True,
    )


def _select_alert(views) -> str:
    view_by_id = {view.alert_id: view for view in views}
    alert_ids = tuple(view_by_id)
    previous = st.session_state.bank_fds_selected_alert_id
    if previous not in view_by_id:
        previous = alert_ids[0]
        st.session_state.bank_fds_selected_alert_id = previous
        st.session_state.pop("bank_fds_alert_selector", None)
    selected_index = alert_ids.index(previous)
    selected_id = st.selectbox(
        "Alert 선택",
        alert_ids,
        index=selected_index,
        format_func=lambda alert_id: (
            f"{view_by_id[alert_id].created_at_text} · "
            f"{view_by_id[alert_id].risk_score_text} · "
            f"{view_by_id[alert_id].status_text}"
        ),
        key="bank_fds_alert_selector",
    )
    st.session_state.bank_fds_selected_alert_id = selected_id
    return selected_id


def _render_history_detail(detail: AlertDetail, *, config: AppConfig) -> None:
    view = to_alert_detail(detail)
    st.divider()
    st.subheader("Alert 상세")
    score, status, level, triggered = st.columns(4)
    score.metric("규칙 기반 위험점수", view.summary.risk_score_text)
    status.metric("Alert 상태", view.summary.status_text)
    level.metric("위험등급", view.summary.risk_level_text)
    triggered.metric("탐지 Rule 수", view.summary.triggered_rule_count)
    findings, errors, transactions = st.columns(3)
    findings.metric("전체 finding 수", len(detail.findings))
    errors.metric("오류 Rule 수", len(detail.errors))
    transactions.metric("거래 수", view.transaction_count)
    st.caption(RISK_SCORE_CAPTION)
    st.write(f"분석 시각: {view.summary.created_at_text}")
    with st.expander("Alert 기술 정보", expanded=False):
        st.write(f"Alert ID: {view.summary.alert_id}")
        st.write(f"분석 Run ID: {view.analysis_run_id}")
        st.write(f"데이터 Source: {view.source_name}")
        st.write(f"Ruleset: {view.ruleset_version}")
    _render_alert_detail(detail, config=config)


def main() -> None:
    st.set_page_config(
        page_title="Bank FDS Operations",
        page_icon="🔎",
        layout="wide",
    )
    _initialize_session_state()

    try:
        db_path = resolve_bank_fds_db_path()
        config = AppConfig(db_path=db_path)
        prepare_db_parent_directory(config.db_path)
    except (OSError, TypeError, ValueError):
        st.error("앱 설정을 준비하지 못했습니다. DB 경로 설정을 확인해 주세요.")
        st.stop()

    try:
        with _open_repository(config):
            pass
    except _REPOSITORY_ERRORS:
        _render_sidebar(config, database_ready=False)
        st.error("DB 준비에 실패했습니다. DB 경로와 접근 권한을 확인해 주세요.")
        st.stop()

    _render_sidebar(config, database_ready=True)
    st.title("은행 거래 이상징후 탐지 시스템")
    st.write(
        "PaySim 합성 거래 CSV를 설명 가능한 Rule-Based FDS로 점검하고, "
        "분석 결과와 Evidence를 SQLite에 저장합니다."
    )
    st.caption("위험점수는 규칙 기반 판단 지표이며 사기 발생 확률이 아닙니다.")

    analysis_tab, history_tab = st.tabs(["새 분석", "Alert 이력"])
    with analysis_tab:
        _render_analysis_tab(config)
    with history_tab:
        _render_history_tab(config)


if __name__ == "__main__":
    main()

"""High-value contracts that prevent portfolio documentation regressions."""

from __future__ import annotations

from pathlib import Path
import re

import pytest


ROOT = Path(__file__).parents[2]
README = ROOT / "README.md"
ARCHITECTURE = ROOT / "docs" / "architecture.md"
DEMO_GUIDE = ROOT / "docs" / "demo-guide.md"
EXAMPLES_README = ROOT / "kaggle_bank_fds" / "examples" / "README.md"
DOCUMENTS = (README, ARCHITECTURE, DEMO_GUIDE, EXAMPLES_README)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_readme_uses_current_repository_and_bank_fds_entrypoint():
    text = _text(README)
    assert "https://github.com/NoahBhang/FDS_Model.git" in text
    assert "FDS-learning-framework.git" not in text
    assert "python -m streamlit run kaggle_bank_fds/src/ui/streamlit_app.py" in text
    assert "streamlit run scripts/app.py" not in text


def test_readme_covers_demo_scores_files_and_non_probability_contract():
    text = _text(README)
    for filename in (
        "clean.csv", "exact_overlap.csv", "partial_overlap.csv",
        "rounded_full_balance.csv",
    ):
        assert f"kaggle_bank_fds/examples/{filename}" in text
    for score in ("35/100", "65/100", "40/100"):
        assert score in text
    assert "Policy B" in text and "canonical Evidence" in text
    assert "사기 발생 확률이 아닙니다" in text
    assert not re.search(r"(?:사기 확률|사기 발생 확률)(?:입니다|이다|로 사용)", text)


def test_readme_covers_csv_and_sqlite_schema_contracts():
    text = _text(README)
    for column in (
        "step", "type", "amount", "nameOrig", "oldbalanceOrg",
        "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
    ):
        assert re.search(rf"(?m)^{re.escape(column)}$", text)
    for table in (
        "analysis_runs", "transaction_snapshots", "alerts", "rule_findings",
        "finding_evidence", "rule_execution_errors",
    ):
        assert f"`{table}`" in text


def test_readme_links_required_documentation_and_license():
    text = _text(README)
    for target in (
        "docs/architecture.md", "docs/demo-guide.md",
        "kaggle_bank_fds/examples/README.md", "LICENSE",
    ):
        assert f"]({target})" in text


@pytest.mark.parametrize("document", DOCUMENTS)
def test_markdown_structure_has_balanced_fences_valid_images_and_no_raw_html(document):
    text = _text(document)
    assert text.startswith("# ")
    assert text.count("```") % 2 == 0
    for target in re.findall(r"!\[[^]]*]\(([^)]+)\)", text):
        assert (document.parent / target).resolve().is_file()
    assert not re.search(r"</?[A-Za-z][^>]*>", text)


def test_all_local_markdown_links_resolve():
    link_pattern = re.compile(r"(?<!!)\[[^]]+]\(([^)]+)\)")
    for document in DOCUMENTS:
        for raw_target in link_pattern.findall(_text(document)):
            target = raw_target.split("#", 1)[0]
            if not target or "://" in target:
                continue
            assert (document.parent / target).resolve().exists(), (
                f"Broken link in {document.relative_to(ROOT)}: {raw_target}"
            )


def test_architecture_and_demo_guide_keep_key_contracts():
    architecture = _text(ARCHITECTURE)
    guide = _text(DEMO_GUIDE)
    assert "```mermaid" in architecture
    assert "erDiagram" in architecture
    assert "SAVEPOINT" in architecture and "semantic" in architecture.lower()
    assert "exact_overlap.csv" in guide and "35/100" in guide
    assert "Alert 이력" in guide and "canonical Evidence" in guide


def test_screenshot_links_exist_without_placeholders_or_local_user_paths():
    readme = _text(README)
    guide = _text(DEMO_GUIDE)
    image_names = (
        "bank-fds-upload-preview.png", "bank-fds-risk-result.png",
        "bank-fds-alert-history.png",
    )
    assert "docs/images/bank-fds-risk-result.png" in readme
    for name in image_names:
        assert f"images/{name}" in guide
        assert (ROOT / "docs" / "images" / name).is_file()
    combined = "\n".join(_text(document) for document in DOCUMENTS)
    assert "TODO screenshot" not in combined
    assert "placeholder" not in combined.lower()
    assert "/Users/" not in combined

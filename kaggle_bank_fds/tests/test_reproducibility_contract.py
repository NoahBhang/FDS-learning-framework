"""Fresh-environment and delivery-surface reproducibility contracts."""

from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[2]


def _check_ignore(path: str) -> int:
    return subprocess.run(
        ["git", "check-ignore", "-q", path],
        cwd=ROOT,
        check=False,
    ).returncode


def test_requirements_pin_verified_streamlit_and_keep_core_dependencies():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    assert "streamlit==1.60.0" in requirements
    assert "numpy==1.24.3" in requirements
    assert "pandas==2.0.3" in requirements
    assert "SQLAlchemy==2.0.19" in requirements
    assert "pytest" in requirements
    assert (ROOT / "requirements-dev.txt").read_text(encoding="utf-8").splitlines() == [
        "-r requirements-ml.txt"
    ]


def test_readme_quick_start_uses_python311_and_module_entrypoint():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "python3.11 -m venv .venv" in readme
    assert "python -m streamlit run kaggle_bank_fds/src/ui/streamlit_app.py" in readme
    assert "pip install -r requirements-dev.txt" in readme
    assert "1,119개 테스트" in readme
    assert "https://github.com/NoahBhang/FDS_Model.git" in readme
    assert "FDS-learning-framework.git" not in readme
    assert "streamlit run scripts/app.py" not in readme


def test_versioned_demo_and_screenshot_assets_are_not_ignored():
    paths = (
        "kaggle_bank_fds/examples/clean.csv",
        "kaggle_bank_fds/examples/exact_overlap.csv",
        "kaggle_bank_fds/examples/partial_overlap.csv",
        "kaggle_bank_fds/examples/rounded_full_balance.csv",
        "docs/images/bank-fds-upload-preview.png",
        "docs/images/bank-fds-risk-result.png",
        "docs/images/bank-fds-alert-history.png",
    )
    assert all((ROOT / path).is_file() for path in paths)
    assert all(_check_ignore(path) == 1 for path in paths)


def test_runtime_and_secret_artifacts_are_ignored():
    paths = (
        "bank_fds.sqlite3", "bank_fds.sqlite3-wal", "bank_fds.sqlite3-shm",
        "arbitrary.sqlite3", "arbitrary-wal", "arbitrary-shm", ".env",
        "__pycache__/module.pyc", ".pytest_cache/state", ".streamlit/secrets.toml",
    )
    assert all(_check_ignore(path) == 0 for path in paths)


def test_repository_contains_no_runtime_database_or_secret_file():
    forbidden_names = {".env", ".DS_Store", "secrets.toml"}
    forbidden_suffixes = {".db", ".sqlite", ".sqlite3"}
    files = tuple(path for path in ROOT.rglob("*") if path.is_file())
    assert not [path for path in files if path.name in forbidden_names]
    assert not [path for path in files if path.suffix.lower() in forbidden_suffixes]
    assert not [path for path in files if path.name.endswith(("-wal", "-shm"))]

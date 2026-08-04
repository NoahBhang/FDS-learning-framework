"""Generate deterministic, upload-ready PaySim demonstration CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COLUMNS = (
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
)
EXAMPLE_BUILDERS = (
    ("clean.csv", "build_clean_demo"),
    ("exact_overlap.csv", "build_exact_overlap_demo"),
    ("partial_overlap.csv", "build_partial_overlap_demo"),
    ("rounded_full_balance.csv", "build_rounded_full_balance_demo"),
)


def _row(
    step: int,
    action: str,
    amount: float,
    actor: str,
    target: str,
    *,
    actor_balance: float = 500_000.0,
) -> dict[str, object]:
    return {
        "step": step,
        "type": action,
        "amount": amount,
        "nameOrig": actor,
        "oldbalanceOrg": actor_balance,
        "newbalanceOrig": actor_balance - amount,
        "nameDest": target,
        "oldbalanceDest": 0.0,
        "newbalanceDest": amount,
    }


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS)


def build_clean_demo() -> pd.DataFrame:
    """Return a small input that triggers none of the five default rules."""
    return _frame([
        _row(1, "PAYMENT", 1_250.5, "SYNTH_CLEAN_01", "SYNTH_MERCHANT_01"),
        _row(30, "PAYMENT", 2_375.25, "SYNTH_CLEAN_02", "SYNTH_MERCHANT_02"),
    ])


def build_exact_overlap_demo() -> pd.DataFrame:
    """Return three transfers with identical Rapid/Split Evidence sets."""
    return _frame([
        _row(step, "TRANSFER", 70_000.0, "SYNTH_EXACT_ORIG", "SYNTH_EXACT_DEST")
        for step in (1, 2, 3)
    ])


def build_partial_overlap_demo() -> pd.DataFrame:
    """Return Rapid Evidence of three rows and Split Evidence of four rows."""
    rows = [
        _row(step, "TRANSFER", 70_000.0, "SYNTH_PARTIAL_ORIG", "SYNTH_PARTIAL_DEST_A")
        for step in (1, 2, 3)
    ]
    rows.append(
        _row(4, "TRANSFER", 10_000.0, "SYNTH_PARTIAL_ORIG", "SYNTH_PARTIAL_DEST_B")
    )
    return _frame(rows)


def build_rounded_full_balance_demo() -> pd.DataFrame:
    """Return one transfer independently detected by Rounded and FullBalance."""
    return _frame([
        _row(
            1, "TRANSFER", 100_000.0,
            "SYNTH_ROUNDED_ORIG", "SYNTH_ROUNDED_DEST",
            actor_balance=100_000.0,
        )
    ])


def generate_demo_files(output_dir: Path) -> tuple[Path, ...]:
    """Deterministically overwrite all four demo files in ``output_dir``."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    namespace = globals()
    for filename, builder_name in EXAMPLE_BUILDERS:
        path = destination / filename
        namespace[builder_name]().to_csv(path, index=False, encoding="utf-8")
        paths.append(path)
    return tuple(paths)


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    args = parser.parse_args()
    generate_demo_files(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

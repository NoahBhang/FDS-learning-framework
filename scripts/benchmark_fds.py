import platform
import time
from typing import Callable

import pandas as pd


def best_time(func: Callable, repeat: int = 5) -> float:
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)
    return min(times)


def make_transactions(n: int = 100_000) -> pd.DataFrame:
    return pd.DataFrame({
        "transaction_id": [f"T{i:06d}" for i in range(n)],
        "debtor_id": [f"D{i % 1000:04d}" for i in range(n)],
        "receiver": [f"R{i % 5000:04d}" for i in range(n)],
        "amount": [(i % 10_000) + 1 for i in range(n)],
        "transaction_type": ["송금" if i % 3 else "카드결제" for i in range(n)],
    })


def downcast_transactions(df: pd.DataFrame) -> pd.DataFrame:
    optimized = df.copy()
    optimized["amount"] = pd.to_numeric(optimized["amount"], downcast="integer")
    optimized["debtor_id"] = optimized["debtor_id"].astype("category")
    optimized["receiver"] = optimized["receiver"].astype("category")
    optimized["transaction_type"] = optimized["transaction_type"].astype("category")
    return optimized


def accumulate_with_dict(df: pd.DataFrame) -> dict:
    totals = {}
    for debtor_id, amount in zip(df["debtor_id"], df["amount"]):
        totals[debtor_id] = totals.get(debtor_id, 0) + amount
    return totals


def accumulate_with_groupby(df: pd.DataFrame) -> pd.Series:
    return df.groupby("debtor_id", observed=True)["amount"].sum()


def count_blacklist_hits_list(receivers: pd.Series, blacklist: list[str]) -> int:
    return sum(receiver in blacklist for receiver in receivers)


def count_blacklist_hits_set(receivers: pd.Series, blacklist: set[str]) -> int:
    return sum(receiver in blacklist for receiver in receivers)


def main() -> None:
    print("=== FDS Benchmark ===")
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"pandas: {pd.__version__}")
    print()

    df = make_transactions()
    before_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

    optimized = downcast_transactions(df)
    after_mb = optimized.memory_usage(deep=True).sum() / 1024 / 1024

    print("=== Memory ===")
    print(f"before_downcast_mb: {before_mb:.2f}")
    print(f"after_downcast_mb:  {after_mb:.2f}")
    print()

    print("=== Aggregation ===")
    dict_time = best_time(lambda: accumulate_with_dict(df))
    groupby_time = best_time(lambda: accumulate_with_groupby(optimized))
    print(f"dict_accumulation_seconds:    {dict_time:.6f}")
    print(f"groupby_accumulation_seconds: {groupby_time:.6f}")
    print()

    blacklist_list = [f"R{i:04d}" for i in range(100)]
    blacklist_set = set(blacklist_list)

    print("=== Blacklist lookup ===")
    list_time = best_time(lambda: count_blacklist_hits_list(df["receiver"], blacklist_list))
    set_time = best_time(lambda: count_blacklist_hits_set(df["receiver"], blacklist_set))
    print(f"list_lookup_seconds: {list_time:.6f}")
    print(f"set_lookup_seconds:  {set_time:.6f}")

    assert after_mb < before_mb
    assert len(accumulate_with_dict(df)) == len(accumulate_with_groupby(optimized))
    assert count_blacklist_hits_list(df["receiver"], blacklist_list) == count_blacklist_hits_set(df["receiver"], blacklist_set)

    print()
    print("All benchmark assertions passed.")


if __name__ == "__main__":
    main()

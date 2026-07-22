-- verified_window_functions.sql
-- Purpose:
--   Verify rolling 1-hour transaction counts for FDS window-function logic.
--
-- Expected result:
--   T001 -> 1
--   T002 -> 2
--   T003 -> 3
--   T004 -> 3
--   T005 -> 1

WITH transactions(transaction_id, user_id, created_at, amount, expected_count) AS (
    VALUES
        ('T001', 'U001', '2026-07-01 10:00:00', 10000, 1),
        ('T002', 'U001', '2026-07-01 10:30:00', 20000, 2),
        ('T003', 'U001', '2026-07-01 10:50:00', 30000, 3),
        ('T004', 'U001', '2026-07-01 11:10:00', 40000, 3),
        ('T005', 'U001', '2026-07-01 13:00:00', 50000, 1)
),
calculated AS (
    SELECT
        transaction_id,
        user_id,
        created_at,
        amount,
        expected_count,
        COUNT(*) OVER (
            PARTITION BY user_id
            ORDER BY unixepoch(created_at)
            RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
        ) AS tx_count_last_1hour
    FROM transactions
)
SELECT
    transaction_id,
    created_at,
    expected_count,
    tx_count_last_1hour,
    CASE
        WHEN expected_count = tx_count_last_1hour THEN 1
        ELSE 0
    END AS matches_expected
FROM calculated
ORDER BY created_at;

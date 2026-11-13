# glassbox-bench, seed 3

7 of 7 tasks run. Compiled track: 5 of 5 solved (mean circuit recall 1.0, precision 1.0, mechanism match 1.0). Prediction hit rate 0.902. 7 of 7 final findings beat their null controls.

| Task | Track | Family (from the key) | Circuit recall | Circuit precision | Mechanism match | Behavior prediction | Experiments | Minutes | Predictions (confirmed / refuted) | Beats nulls | Final status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t1 | compiled | hist | 1.00 | 1.00 | 1.00 | 1.00 | 13 | 3.43 | 7 / 0 | yes | verified |
| t2 | trained | pair-table | not scored | not scored | 1.00 | 1.00 | 18 | 7.20 | 8 / 2 | yes | verified |
| t3 | compiled | sort-by-key | 1.00 | 1.00 | 1.00 | 1.00 | 16 | 3.68 | 9 / 1 | yes | verified |
| t4 | compiled | lookup2 | 1.00 | 1.00 | 1.00 | 1.00 | 17 | 7.98 | 9 / 1 | yes | verified |
| t5 | compiled | count-compare | 1.00 | 1.00 | 1.00 | 1.00 | 15 | 4.06 | 9 / 0 | yes | verified |
| t6 | compiled | shift-copy | 1.00 | 1.00 | 1.00 | 1.00 | 13 | 2.28 | 7 / 0 | yes | verified |
| t7 | trained | max-rank | not scored | not scored | 1.00 | 1.00 | 18 | 4.62 | 6 / 2 | yes | not-verified |

Trained-track tasks are behaviorally specified with an unknown mechanism: they are scored on behavior prediction only.

Missed and extra components:

- t1 (hist): missed none, extra none
- t3 (sort-by-key): missed none, extra none
- t4 (lookup2): missed none, extra none
- t5 (count-compare): missed none, extra none
- t6 (shift-copy): missed none, extra none

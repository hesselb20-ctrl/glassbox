# glassbox-bench, seed 2

7 of 7 tasks run. Compiled track: 5 of 5 solved (mean circuit recall 1.0, precision 1.0, mechanism match 1.0). Prediction hit rate 0.912. 7 of 7 final findings beat their null controls.

| Task | Track | Family (from the key) | Circuit recall | Circuit precision | Mechanism match | Behavior prediction | Experiments | Minutes | Predictions (confirmed / refuted) | Beats nulls | Final status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t1 | compiled | lookup2 | 1.00 | 1.00 | 1.00 | 1.00 | 16 | 6.41 | 10 / 0 | yes | verified |
| t2 | compiled | shift-copy | 1.00 | 1.00 | 1.00 | 1.00 | 11 | 2.21 | 7 / 0 | yes | verified |
| t3 | compiled | count-compare | 1.00 | 1.00 | 1.00 | 1.00 | 12 | 2.92 | 6 / 0 | yes | verified |
| t4 | trained | pair-table | not scored | not scored | 1.00 | 1.00 | 17 | 7.32 | 7 / 0 | yes | verified |
| t5 | compiled | hist | 1.00 | 1.00 | 1.00 | 1.00 | 15 | 3.41 | 8 / 0 | yes | verified |
| t6 | compiled | sort-by-key | 1.00 | 1.00 | 1.00 | 1.00 | 16 | 5.64 | 7 / 3 | yes | verified |
| t7 | trained | max-rank | not scored | not scored | 1.00 | 1.00 | 15 | 4.51 | 7 / 2 | yes | not-verified |

Trained-track tasks are behaviorally specified with an unknown mechanism: they are scored on behavior prediction only.

Missed and extra components:

- t1 (lookup2): missed none, extra none
- t2 (shift-copy): missed none, extra none
- t3 (count-compare): missed none, extra none
- t5 (hist): missed none, extra none
- t6 (sort-by-key): missed none, extra none

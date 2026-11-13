# glassbox-bench, seed 1

7 of 7 tasks run. Compiled track: 5 of 5 solved (mean circuit recall 1.0, precision 1.0, mechanism match 1.0). Prediction hit rate 0.921. 7 of 7 final findings beat their null controls.

| Task | Track | Family (from the key) | Circuit recall | Circuit precision | Mechanism match | Behavior prediction | Experiments | Minutes | Predictions (confirmed / refuted) | Beats nulls | Final status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t1 | trained | pair-table | not scored | not scored | 1.00 | 1.00 | 16 | 5.08 | 6 / 3 | yes | verified |
| t2 | compiled | sort-by-key | 1.00 | 1.00 | 1.00 | 1.00 | 18 | 4.38 | 10 / 0 | yes | verified |
| t3 | compiled | count-compare | 1.00 | 1.00 | 1.00 | 1.00 | 17 | 3.83 | 9 / 0 | yes | verified |
| t4 | compiled | lookup2 | 1.00 | 1.00 | 1.00 | 1.00 | 15 | 6.80 | 8 / 1 | yes | verified |
| t5 | compiled | shift-copy | 1.00 | 1.00 | 1.00 | 1.00 | 14 | 2.90 | 7 / 0 | yes | verified |
| t6 | compiled | hist | 1.00 | 1.00 | 1.00 | 1.00 | 17 | 5.27 | 9 / 0 | yes | verified |
| t7 | trained | max-rank | not scored | not scored | 1.00 | 1.00 | 16 | 4.18 | 9 / 1 | yes | not-verified |

Trained-track tasks are behaviorally specified with an unknown mechanism: they are scored on behavior prediction only.

Missed and extra components:

- t2 (sort-by-key): missed none, extra none
- t3 (count-compare): missed none, extra none
- t4 (lookup2): missed none, extra none
- t5 (shift-copy): missed none, extra none
- t6 (hist): missed none, extra none

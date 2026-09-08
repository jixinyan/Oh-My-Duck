# Early full-training reward trends — 2026-09-08

Snapshot: **02:08:32 UTC**. All eight runs of `full-local-0908-01` had entered
the full stage. This is early training evidence, not learned behavior acceptance.

Native logged episodic return, averaged over the first ten available records and
the latest ten. RSL begins logging before complete fixed-length episodes; SB3
StandUp first reports returns at iteration 13. Episode initialization, averaging
windows, episode duration and curriculum weights affect these values. Compare
each run to itself; these are not controlled cross-framework rankings.

| Run | Latest iteration | First 10 return | Latest 10 return | Previous 10 return |
|---|---:|---:|---:|---:|
| isaac-newton-rsl-rl-stand_up | 166 | -2.62 | 29.30 | 30.17 |
| isaac-newton-rsl-rl-walking | 263 | 0.27 | 96.21 | 94.29 |
| isaac-newton-sb3-stand_up | 210 | -6.75 | 1.93 | 0.10 |
| isaac-newton-sb3-walking | 187 | 0.20 | 26.14 | 19.62 |
| mujoco-rsl-rl-stand_up | 329 | -2.67 | 38.89 | 38.30 |
| mujoco-rsl-rl-walking | 539 | 0.50 | 102.05 | 101.58 |
| mujoco-sb3-stand_up | 399 | -7.52 | 9.61 | 12.88 |
| mujoco-sb3-walking | 421 | 0.46 | 72.72 | 69.03 |

Walking returns increased in all four combinations. RSL Walking episode length
rose from about 35 to 936 steps in MuJoCo and 33 to 882 in Newton. Linear-velocity
tracking reward terms also increased; command-following quality still requires
fixed evaluation rather than interpreting total return as gait success.

RSL StandUp standing-composite reward rose from 0.067 to 2.045 in MuJoCo and
0.059 to 1.038 in Newton. These are weighted reward terms, not success rates.
A 300-step StandUp episode ends by timeout and does not establish successful recovery.

SB3 StandUp remains unstable: the best ten-record average so far was 17.7
in MuJoCo (iteration 359) and 12.1 in Newton (iteration 59), versus latest averages
9.61 and 1.93. Latest ten-record approximate KL averages are 0.0304 and 0.0619.
Both emit frequent native KL early-stop messages. Inspection of the installed
SB3 PPO implementation confirms these end the current optimization passes; they
do not terminate the training job. This warrants continued stability observation;
no hyperparameters or running source were changed for this inspection.

Evidence (local, ignored generated artifacts):

- `outputs/reports/reward-trends-20260908T020832Z/metrics.json`: extracted complete records, native metric windows and hashes of the bytes read from each log.
- Same directory: `reward-trends.svg`, `reward-trends.png`, and `extract.py`.
- Live inputs: `outputs/experiments/full-local-0908-01/*/full.log`.

All eight workers reported running when inspected. Reward improvement does not
replace normalized export, held-out task checks, sim2sim and CPU/BAM behavior
acceptance after training.

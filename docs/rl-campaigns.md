# Full representative training campaigns

`configs/experiments/representative-full.json` runs two tasks across MuJoCo and
Isaac/Newton, using native RSL-RL PPO and native SB3 PPO. Eight independent learners
share one eight-GPU node, with one learner and 4096 simulated environments per GPU.
The pinned task defaults set Walking to 50,000 PPO iterations and StandUp to
15,000; each iteration collects 24 steps per environment. These are training
budgets, not promises of convergence. There is no added duration limit.

```bash
python omd.py campaign --config configs/experiments/representative-full.json \
  --output outputs/experiments/full-YYYYMMDD-01 --dry-run

python omd.py submit --name omd-rl-full-YYYYMMDD-01 --gpus 8 -- \
  python omd.py campaign --config configs/experiments/representative-full.json \
  --output outputs/experiments/full-YYYYMMDD-01
```

Use a new name and output directory for each attempt. Submission uses an immutable
committed source snapshot. The campaign respects the allocated GPU order, isolates
CUDA visibility, clears inherited distributed ranks and keeps W&B offline. It
uses each framework's own PPO; there is no shared gradient update between tasks.
Native RSL distributed learning remains available separately for a single task.

Each GPU executes:

1. A fresh 64-environment / five-iteration smoke, reward/export audit and video
   CPU/BAM rehearsal. An untrained policy may fail behavior; execution errors stop
   that run before long training.
2. A checkpoint resume check, then 4096 environments for five capacity-check
   iterations and another export audit. Throughput includes startup and is recorded
   explicitly; this fixed-size campaign is not a claim of optimal GPU utilization.
3. Fresh full training using the task budget, with checkpoints every 250 updates.
4. Final audited normalized export, schema-2 local package, both-backend sim2sim
   and CPU/BAM video. MuJoCo rendering explicitly uses OSMesa on this host; Newton
   uses its native renderer. No policy is uploaded.

`campaign.json` records GPU assignment and each worker outcome. Each run directory
has `result.json`, stage logs and exported/evaluation artifacts. RSL checkpoints
retain the native `logs/rsl_rl/<experiment>/...` layout; the campaign records the
exact paths. Other runs continue if one worker fails. Scheduler termination is
forwarded to the worker process groups. Exit code 2 denotes completed execution
with failed behavior gates; exit code 1 denotes a pipeline error.

## Recover a periodic checkpoint

RSL uses native `model_N.pt` checkpoints, including its existing curriculum state.
Run identity is written before learning so an interrupted run can be resumed via
`--agent.resume True --agent.load-run NAME --agent.load-checkpoint model_N.pt`.

SB3 writes `checkpoints/step_N/` with `model.zip`, `vecnormalize.pkl` and `run.json`.
Bundles are finalized after a PPO update, include environment progress and file
hashes, and can be passed directly to `--resume`. Partial directories remain
labelled `.partial` and must not be used as complete checkpoints. Model optimizer
and normalization state are restored; live simulation state and RNG trajectories
are not bitwise restart guarantees. A campaign rerun never silently overwrites or
resumes a previous experiment.

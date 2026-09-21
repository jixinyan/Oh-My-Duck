# Full representative training campaigns

Current scheduling policy (2026-09-13): submit the complete workflow in one job:
required startup gates → full declared training budget → final evaluation/videos.
Reuse configuration-matched gates where verified, and continue automatically
after remaining gates pass. Routine submissions must not use `--prepare-only`;
a failed gate still stops its run. The current environment choice is 8192 per
learner (`configs/experiments/representative-8192.json`). The 4096 configuration
and September 8 local-host instructions below describe historical campaigns;
new GPU work uses the scheduler under the September 12 authorization.

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

The user authorized running the full campaign on the local eight-GPU host after
scheduler failure on 2026-09-08. From an immutable source snapshot, the same CLI
can run directly with `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WANDB_MODE=offline`;
use a detached process with file logs for training that outlives the terminal.
See [the failure and local launch record](reports/full-training-2026-09-07.md).

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

## Explicit GPU sharing and continuation

`--runs-per-gpu N` permits up to N independent native learners on each visible
GPU; the default remains one. With `CUDA_VISIBLE_DEVICES=7 --runs-per-gpu 6`, six
runs share local GPU 7 without shared gradients. Measure aggregate throughput and
memory on the actual host; low VRAM usage alone does not establish spare compute.

A run may contain a `resume` record naming the exact prior run/checkpoint, its
SHA-256, native `completed_iterations`, and `source_campaign`. Continuation checks
identity, native progress, SB3 normalizer hashes and the preceding smoke/export/
rehearsal/resume/capacity gates before bypassing those already completed gates.
Only the remaining task budget is trained. New output directories preserve the
original attempt and operator stop records. Native resume restores optimizer,
normalizer and curriculum state but starts fresh simulation episodes. Checkpoint
intervals bound lost work; native RSL iteration indexing remains unchanged.

SB3 的 `resume.run` 必须直接指定包含 `model.zip`、`vecnormalize.pkl` 和
`run.json` 的目录，`resume.checkpoint` 必须为 `model.zip`。指定最终输出目录
即使用该目录中的最终模型；指定 `checkpoints/step_N` 即使用该周期模型。
Campaign 校验与训练入口检查同一目录的文件 hash，加载后核对模型时间步数，
训练结束时核对完整的剩余预算。恢复检查的 CPU 执行记录见
[训练检查与恢复验证](reports/project-review-2026-09-21.md)。

## 复用训练前检查

`validate_preparation` 与 `reuse_smoke` 检查完整的 `src/oh_my_duck`、
`environments`、`configs`、`pyproject.toml` 和 `omd.py`。来源 commit 与
当前工作目录之间的变化，以及范围内未被 Git 跟踪且未被忽略的文件，均会
阻止复用。检查覆盖 MDP、公共模块、依赖锁文件和训练入口。仅修改 `docs`
允许复用。配置、阶段完成状态、导出文件和回放文件也必须满足各自检查要求。

## Checkpoint video previews

`python omd.py preview --campaign outputs/experiments/RUN --output outputs/previews/NEW \
--gpu 7 --watch` serially exports new saved checkpoints through the official path
and evaluates them in their training backend with 1280×720 video. Open the output
`index.html` for the local gallery; refresh after new renders finish. This is a
checkpoint preview, not a live camera feed or a public service. `--run-id ID` may
be repeated to limit previews, and omitting `--watch` renders one snapshot per run.
Rendering uses only the explicitly selected GPU and W&B stays offline. Failed
exports/renders keep their logs and do not stop training. Every attempt has its
own directory and checkpoint hash; behavior failures remain visibly labelled.

## SB3 learning-rate diagnosis

SB3 新训练默认使用基于前一次 rollout 的 KL 反馈学习率，恢复训练时读取
已保存的 `learning_rate_mode`。`--learning-rate-mode constant` 可明确选择
固定学习率，`--learning-rate 0.0001` 可设置学习率数值。RSL-RL 保持原生的
每 minibatch adaptive-KL 更新方式。学习率设置和实际值写入 run/checkpoint
metadata；调整配置的实验需要独立的检查记录和输出目录。


Campaign run entries may set an optional numeric `learning_rate` for native SB3.
The same value is forwarded to smoke, resume check, capacity and full training,
and is retained in campaign/run provenance. Omitting it preserves the existing
framework default. Values must be finite and positive; RSL-RL entries reject this
field and retain their native adaptive schedule.

[The prepared four-combination SB3 experiment](../configs/experiments/representative-sb3-lr-1e-4.json)
uses `0.0001` with the unchanged official task recipes. It is **prepared, not
launched or behavior-validated**. Inspect allocation without starting training:

```bash
python omd.py campaign --config configs/experiments/representative-sb3-lr-1e-4.json \
  --output outputs/experiments/NEW --runs-per-gpu 4 --dry-run
```

Actual allocation/sharing remains explicit and should follow measured throughput
and the current GPU-7-only constraint. A recovery may reuse preceding gates only
when its learning-rate declaration matches the source campaign; changing or
adding that declaration requires fresh gates. This prevents retuned training from
silently inheriting validation of another setting. The direct native training
CLI still supports deliberate learning-rate overrides on fresh/resumed runs.

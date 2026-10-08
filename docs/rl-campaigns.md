# Full representative training campaigns

GPU acceptance, RL training and background previews are currently stopped.
Future authorized execution may use at most one idle GPU from devices 2–4.
The device must have zero compute processes and sustained zero utilization before
allocation. Preserve existing workloads and every user's processes.

`configs/experiments/representative-8192.json` declares two tasks across MuJoCo and
Isaac/Newton, using native RSL-RL PPO and native SB3 PPO. Its eight independent
learners each use 8192 simulated environments. A queued campaign executes these
complete workflows sequentially on the explicitly allocated device.
The pinned task defaults set Walking to 50,000 PPO iterations and StandUp to
15,000; each iteration collects 24 steps per environment. These are training
budgets, not promises of convergence. There is no added duration limit.

Each workflow runs required startup gates, the declared training budget and final
evaluation/videos. Verified configuration-matched gates can be reused. Failed
execution gates stop the affected workflow. `--prepare-only` explicitly ends after
preparation and produces no full-training acceptance.

## Walking 低速命令因果配对

`configs/experiments/walking-low-speed-boost.json` 是独立的单因素诊断计划：四种后端/原生 PPO 组合各有官方控制组与 `low_speed_tracking_boost=1.0` 组，均使用 8,192 环境和完整 50,000 次更新。该干预只改变 `0.01–0.2 m/s` 非零线速度的 tracking 信号，默认官方配方不变。2026-09-30 已完成全部 8 个 run 的训练前门禁并在门禁后停止，完整训练尚未提交；详情见 [低速命令干预报告](reports/rl-walking-low-speed-intervention-2026-09-30.md) 和 [门禁记录](reports/rl-walking-low-speed-gates-2026-09-30.md)。

```bash
python omd.py campaign --config configs/experiments/representative-8192.json \
  --output outputs/experiments/full-YYYYMMDD-01 --dry-run

CUDA_VISIBLE_DEVICES="$IDLE_GPU_INDEX" python omd.py campaign \
  --config configs/experiments/representative-8192.json \
  --output outputs/experiments/full-YYYYMMDD-01 --queue
```

The dry-run command inspects the declared plan without launching workers.
The execution command requires authorization to resume training and an actual
idle-device check. `IDLE_GPU_INDEX` must identify the allocated device from 2–4.
Use an immutable source snapshot and retained process logs.

Use a new name and output directory for each attempt. Submission uses an immutable
committed source snapshot. The campaign respects the allocated GPU order, isolates
CUDA visibility, clears inherited distributed ranks and propagates the configured
W&B mode. `configs/training.json` selects `online`; `WANDB_MODE` can explicitly
select `offline` for diagnostics. It
uses each framework's own PPO; there is no shared gradient update between tasks.
Native RSL distributed learning remains available separately for a single task.

Each GPU executes:

1. A fresh 64-environment / five-iteration smoke, reward/export audit and video
   CPU/BAM rehearsal. An untrained policy may fail behavior; execution errors stop
   that run before long training.
2. A checkpoint resume check, then the run's declared 8192 environments for five capacity-check
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

## Queued execution and continuation

`--queue` executes at most one learner workflow on each allocated GPU and waits
for its completion before admitting the next run. The current allocation permits
one GPU. Native RSL distributed learning and campaign resource controls remain
available as explicit capabilities with their own authorization and evidence.

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
--gpu "$IDLE_GPU_INDEX" --watch` serially exports new saved checkpoints through the official path
and evaluates them in their training backend with 1280×720 video. Open the output
`index.html` for the local gallery; refresh after new renders finish. This is a
checkpoint preview, not a live camera feed or a public service. `--run-id ID` may
be repeated to limit previews, and omitting `--watch` renders one snapshot per run.
Rendering uses only the explicitly selected GPU and inherits the configured W&B
mode. Previews remain stopped with the current training pause. Failed
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
  --output outputs/experiments/NEW --dry-run
```

Actual allocation follows the current one-idle-GPU restriction. A recovery may reuse preceding gates only
when its learning-rate declaration matches the source campaign; changing or
adding that declaration requires fresh gates. This prevents retuned training from
silently inheriting validation of another setting. The direct native training
CLI still supports deliberate learning-rate overrides on fresh/resumed runs.

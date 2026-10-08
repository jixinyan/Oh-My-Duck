# CPU voice task and provider access

The actual installed voice client submitted a Qwen transcription to the pinned
native Harness with `gpt-6.1-sol` and `high` reasoning. The provider returned HTTP
403 before any tool call. A separate actual Responses request confirmed the
provider message: `This token has no access to model gpt-6.1-sol`.
Native navigation acceptance requires access to that model.

## Verified execution

The source revision was `370755b25b3097a4af2c74b15ed73ab6f4bf3d18`.
An immutable worker checkout ran CPU MuJoCo/BAM with the official apartment,
14 BAM XL330 M6 servos and the pinned official policy catalogue. GPU visibility
was empty. The native Harness revision was
`8a5e685b22d032207f53db20454f0992a4ad60fd`.

Actual Qwen Base synthesis used the existing confirmed profile to produce the
instruction WAV. Qwen ASR transcribed it as:

> 请使用距离和角度工具前往办公室。读取相机和距离传感器，到达后停止。

`omd voice-task` submitted that text unchanged to `navigate-office`. Native run
`a3738a90-306b-4eeb-93cb-9695c5fd72eb` ended as `failed`; the client returned exit
code 2 and generated a status response with the same confirmed voice. The terminal
export preserves all eleven actual events. There were zero executions, policy
actions, sensor frames and formal verdicts.

The ASR model was `Qwen/Qwen3-ASR-0.6B` at
`5eb144179a02acc5e5ba31e748d22b0cf3e303b0`; Base TTS was
`Qwen/Qwen3-TTS-12Hz-0.6B-Base` at
`5d83992436eae1d760afd27aff78a71d676296fc`. Both services ran on CPU in their
separate frozen environments.

Independent decoding verified both complete WAV files as mono PCM16 at 24 kHz,
with finite nonzero samples, matching recorded SHA256 values. The instruction
contains 134400 frames and the status response contains 63360 frames. The confirmed
voice ID, profile revision, reference audio and profile database bytes remained
unchanged. This run used recorded audio without microphone capture or speaker
playback.

Native session `c4265b31-5ded-4c3b-a7d3-23165c768997` reports `closed` and
`resources: released`. All five owned native-server, ASR/TTS-service and SSH
transport processes exited. Cleanup took 0.018–2.843 seconds, without SIGKILL or
a cleanup timeout. Local ports 50427, 50428 and 50429 are closed; the remote native
worker is absent. GPU acceptance and RL remain stopped.

## Retained evidence

Artifacts are under `outputs/acceptance/cpu-voice-loop-20261008-01/` on both the
operator computer and `jd_B300`. The sanitized provider check is stored separately
as `outputs/acceptance/cpu-voice-provider-20261008-01.json`; it contains no token.

| Artifact | SHA256 |
| --- | --- |
| `independent-failure-audit.json` | `6d18b155001b6bb63e11e90a91e6c54640f358ed42ab0f1b0a13c19b6fb8e762` |
| `result.json` | `5aed4205baf229ed809ce0bc2148344524a506f703f0f79497f52799eb1f7c48` |
| `task/result.json` | `a5f038fe5ff4cb291ce73a71da5db895b354cf0b0bb4072979f7f1317c6a7a81` |
| `replay/manifest.json` | `0d49c40c87b8819cfa559c87de10a6593bf668e6c556ec20998fef0336efbd52` |
| Provider check | `a8735c819ebb8611860f579cbacd573957cb281cedf2fcc11cea3750e87e4955` |

The actual model request, terminal failure, fixed-voice response and resource
release are verified. Navigation, formal task success and current-source Newton
execution retain their independent acceptance requirements.

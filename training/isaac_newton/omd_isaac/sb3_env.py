"""Preserve exact pre-reset observations at the pinned Isaac Lab/SB3 boundary."""
import json
from pathlib import Path
import numpy as np
from isaaclab_rl.sb3 import Sb3VecEnvWrapper


class DiagnosticSb3VecEnvWrapper(Sb3VecEnvWrapper):
    def __init__(self, env, fast_variant=True):
        super().__init__(env, fast_variant=fast_variant)
        self.unwrapped._capture_terminal_observations = True
        self._terminal_count = 0
        self._timeout_count = 0

    def step_wait(self):
        self.unwrapped._terminal_snapshot = None
        return super().step_wait()

    def _process_extras(self, obs, terminated, truncated, extras, reset_ids):
        infos = super()._process_extras(obs, terminated, truncated, extras, reset_ids)
        if len(reset_ids):
            snapshot = self.unwrapped._terminal_snapshot
            if snapshot is None:
                raise RuntimeError("SB3 reset is missing the pre-reset terminal observation")
            ids, states = (item.detach().cpu().numpy() for item in snapshot)
            np.testing.assert_array_equal(ids, reset_ids)
            if states.shape != (len(reset_ids), 61) or not np.isfinite(states).all():
                raise ValueError("Malformed or non-finite SB3 terminal observation")
            for row, idx in enumerate(reset_ids):
                infos[idx]["terminal_observation"] = states[row].copy()
                self._terminal_count += 1
                self._timeout_count += int(truncated[idx] and not terminated[idx])
        return infos

    def close(self):
        log_dir = self.unwrapped.cfg.log_dir
        if log_dir:
            report = {"rl_framework": "sb3", "physics": "Newton", "task": "PD diagnostic",
                "terminal_observations": self._terminal_count, "time_limit_truncations": self._timeout_count,
                "terminal_source": "pre-reset snapshot", "validation": "diagnostic_only"}
            Path(log_dir, "adapter.json").write_text(json.dumps(report, indent=2) + "\n")
        super().close()

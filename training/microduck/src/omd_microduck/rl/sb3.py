"""Editable native SB3 policy configuration shared by task launchers.

Return a custom ActorCriticPolicy subclass and kwargs here when a task needs it.
The task recipes own network widths; SB3 continues to own its PPO algorithm.
"""
from dataclasses import dataclass, field
from typing import Any
import torch


@dataclass
class Sb3PolicyCfg:
    policy: str | type = "MlpPolicy"
    kwargs: dict[str, Any] = field(default_factory=dict)


def make_policy_cfg(task_id: str, agent_cfg) -> Sb3PolicyCfg:
    """Override by task_id here; defaults preserve the established ELU policy."""
    return Sb3PolicyCfg(kwargs={
        "activation_fn": torch.nn.ELU,
        "net_arch": {"pi": list(agent_cfg.actor.hidden_dims),
                     "vf": list(agent_cfg.critic.hidden_dims)},
    })

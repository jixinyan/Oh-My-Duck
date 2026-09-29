import torch
from types import SimpleNamespace


def test_low_speed_tracking_boost_only_scales_small_nonzero_linear_commands(monkeypatch):
    from mjlab.tasks.velocity import mdp as velocity_mdp
    from oh_my_duck.rl.mdp.rewards_locomotion import track_linear_velocity_low_speed_boost

    commands = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.0, 0.0, 0.5],
    ])
    env = SimpleNamespace(command_manager=SimpleNamespace(get_command=lambda name: commands))
    monkeypatch.setattr(velocity_mdp, 'track_linear_velocity', lambda *args, **kwargs: torch.ones(4))
    result = track_linear_velocity_low_speed_boost(
        env, std=0.1, command_name='twist', low_speed_threshold=0.2,
        minimum_speed=0.01, boost=1.0,
    )
    assert torch.allclose(result, torch.tensor([1.0, 1.0 + 0.1 / 0.19, 1.0, 1.0]))

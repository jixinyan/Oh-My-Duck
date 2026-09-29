import math
from types import SimpleNamespace

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from oh_my_duck.rl.mdp.rewards_locomotion import (
    body_upright_gaussian,
    upright_gaussian_at_height,
)


def _env_for_tilt(angle: float):
    quat = torch.tensor([[math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0]])
    asset = SimpleNamespace(
        data=SimpleNamespace(
            root_link_quat_w=quat,
            root_link_pos_w=torch.tensor([[0.0, 0.0, 0.115]]),
        )
    )
    class Scene(dict):
        pass

    scene = Scene(
        robot=asset,
        terrain=SimpleNamespace(env_origins=torch.zeros((1, 3))),
    )
    scene.terrain = scene["terrain"]
    return SimpleNamespace(scene=scene, device=quat.device)


def test_upright_gaussian_uses_configured_angle_standard_deviation():
    env = _env_for_tilt(math.radians(30.0))
    expected = math.exp(-(math.radians(30.0) / 0.3) ** 2)
    value = upright_gaussian_at_height(
        env,
        std=0.3,
        height_low=0.060,
        height_high=0.115,
        asset_cfg=SceneEntityCfg("robot"),
    )
    torch.testing.assert_close(value, torch.tensor([expected]), rtol=1e-5, atol=1e-6)


def test_ungated_upright_gaussian_matches_same_angle_scale():
    env = _env_for_tilt(math.radians(30.0))
    expected = math.exp(-(math.radians(30.0) / 0.3) ** 2)
    value = body_upright_gaussian(env, asset_cfg=SceneEntityCfg("robot"), std=0.3)
    torch.testing.assert_close(value, torch.tensor([expected]), rtol=1e-5, atol=1e-6)

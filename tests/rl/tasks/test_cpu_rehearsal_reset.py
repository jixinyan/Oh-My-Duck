"""CPU deployment starts use the same official sampler through real name mappings."""
import mujoco
import numpy as np
import torch
from oh_my_duck.rl.evaluation.rehearsal.infer_policy import MICRODUCK_XML
from oh_my_duck.rl.evaluation.rehearsal.state import sample_ground_pose
from oh_my_duck.rl.tasks.microduck_standup_env_cfg import make_microduck_standup_env_cfg
from oh_my_duck.rl.evaluation.protocols import standup


def test_real_cpu_scene_supports_all_four_official_ground_spawns():
    model=mujoco.MjModel.from_xml_path(MICRODUCK_XML);data=mujoco.MjData(model)
    cfg=make_microduck_standup_env_cfg()
    for scenario in standup().scenarios:
        mujoco.mj_resetData(model,data)
        params={**cfg.events['set_ground_state'].params,**scenario.reset_probabilities}
        torch.manual_seed(42);sample_ground_pose(model,data,params)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        q=model.jnt_qposadr[model.joint('trunk_base_freejoint').id]
        np.testing.assert_allclose(np.linalg.norm(data.qpos[q+3:q+7]),1.,atol=1e-6)
        prefix='prone' if scenario.name.startswith('face_') else scenario.name
        assert params[prefix+'_z_min']<=data.qpos[q+2]<=params[prefix+'_z_max']

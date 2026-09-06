"""Compare the virtual Newton sensor formulas to MuJoCo's native sensor outputs.

These CPU tests exercise moving, rotated bodies and actual accelerations. GPU
Newton state synchronization and contact extraction remain separate job gates.
"""
from types import SimpleNamespace
import unittest
import numpy as np
import torch
import mujoco
from oh_my_duck.rl.backends.isaac_newton.task_binding.sensors import NewtonBuiltinSensor
from oh_my_duck.rl.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg
from mjlab.scene import Scene


class SensorFormulaTests(unittest.TestCase):
    def test_builtin_frames_and_acceleration_match_native_sensors(self):
        scene=Scene(make_microduck_velocity_env_cfg().scene,'cpu')
        model=scene.compile();data=mujoco.MjData(model)
        rng=np.random.default_rng(21)
        body_ids=[i for i in range(model.nbody) if model.body(i).name.startswith('robot/')]
        site_ids=[i for i in range(model.nsite) if model.site(i).name.startswith('robot/')]
        body_names=[model.body(i).name.removeprefix('robot/') for i in body_ids]
        site_names=[model.site(i).name.removeprefix('robot/') for i in site_ids]
        def tensor(value):return torch.tensor(np.array(value),dtype=torch.float64).unsqueeze(0)
        for sample in range(8):
            data.qpos[:]=model.qpos0
            data.qpos[2]=.3
            data.qpos[3:7]=rng.normal(size=4);data.qpos[3:7]/=np.linalg.norm(data.qpos[3:7])
            data.qpos[7:]+=rng.uniform(-.1,.1,size=model.nq-7)
            data.qvel[:]=rng.normal(0,.5,size=model.nv)
            mujoco.mj_forward(model,data)
            site_quats=[]
            for site in site_ids:
                q=np.empty(4);mujoco.mju_mulQuat(q,data.xquat[model.site_bodyid[site]],model.site_quat[site]);site_quats.append(q)
            robot=SimpleNamespace(body_names=body_names,site_names=site_names,indexing=SimpleNamespace(body_ids=torch.tensor(body_ids)),
                data=SimpleNamespace(body_link_quat_w=tensor(data.xquat[body_ids]),
                    site_pose_w=torch.cat((tensor(data.site_xpos[site_ids]),tensor(site_quats)),-1),
                    site_parent_ids=torch.tensor([body_ids.index(int(model.site_bodyid[i])) for i in site_ids])))
            sim=SimpleNamespace(mj_model=model,data=SimpleNamespace(**{name:tensor(getattr(data,name)) for name in ['cvel','cacc','subtree_com','subtree_angmom']}))
            for name in ['robot/orientation','robot/angular-velocity','robot/imu_ang_vel','robot/imu_lin_vel','robot/imu_accel','robot/root_angmom']:
                actual=NewtonBuiltinSensor(name,model,robot,sim).data.numpy()[0]
                sensor=model.sensor(name);expected=data.sensordata[int(sensor.adr[0]):int(sensor.adr[0]+sensor.dim[0])]
                np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=1e-8,err_msg=f'{sample}: {name}')


if __name__=='__main__':unittest.main()

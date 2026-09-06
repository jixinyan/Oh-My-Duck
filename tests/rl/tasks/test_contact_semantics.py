"""Compare Newton's contact aggregation to native MuJoCo contact sensors."""
import numpy as np
import torch
import mujoco
from mjlab.scene import Scene
from oh_my_duck.rl.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg
from oh_my_duck.rl.backends.isaac_newton.task_binding.sensors import aggregate_contacts


def test_contact_netforce_matches_primary_secondary_native_sign():
    model=Scene(make_microduck_velocity_env_cfg().scene,'cpu').compile()
    data=mujoco.MjData(model)
    data.qpos[:]=model.key(0).qpos
    data.qpos[2]=.11
    mujoco.mj_forward(model,data)
    assert data.ncon>0
    geom=torch.tensor([[c.geom1,c.geom2] for c in data.contact],dtype=torch.long)
    forces=[]
    for i,c in enumerate(data.contact):
        local=np.empty(6);mujoco.mj_contactForce(model,data,i,local)
        forces.append(c.frame.reshape(3,3).T@local[:3])
    ids=torch.tensor([model.geom(f'robot/{side}_foot_collision').id for side in ['left','right']])
    # The native reference scene attaches the plane to a fixed terrain body;
    # Newton imports it as world geometry. Normalize only that identity here.
    bodies=torch.tensor(model.geom_bodyid.copy());bodies[bodies==model.body('terrain').id]=0
    found,force=aggregate_contacts(geom,torch.zeros(data.ncon,dtype=torch.long),torch.ones(data.ncon,dtype=torch.bool),
        torch.tensor(np.array(forces)),bodies,ids,1)
    for i,side in enumerate(['left','right']):
        for field,value in [('found',found[0,i:i+1]),('force',force[0,i])]:
            sensor=model.sensor(f'feet_ground_contact_{side}_foot_collision_{field}')
            expected=data.sensordata[int(sensor.adr[0]):int(sensor.adr[0]+sensor.dim[0])]
            np.testing.assert_allclose(value.numpy(),expected,atol=1e-8,rtol=1e-8)
    assert found.sum()>0
    assert force[...,2].sum()<0  # Primary foot -> secondary ground, per official convention.

"""Newton's force-driven solver has no native MuJoCo actuator metadata arrays."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.rl.exporter_utils import get_base_metadata
from oh_my_duck.rl.artifacts.metadata import policy_metadata


def fixture():
    from oh_my_duck.robotics.microduck.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
    entity=MICRODUCK_STANDUP_ROBOT_CFG.build()
    model=entity.compile()
    joint_action=JointPositionAction.__new__(JointPositionAction)
    joint_action._scale=.25
    robot=SimpleNamespace(spec=entity.spec,joint_names=entity.joint_names,
                          data=SimpleNamespace(default_joint_pos=torch.tensor(model.key(0).qpos[7:])[None]))
    env=SimpleNamespace(scene={'robot':robot},sim=SimpleNamespace(mj_model=model),
                         action_manager=SimpleNamespace(get_term=lambda name:joint_action),
                         command_manager=SimpleNamespace(active_terms=['twist','head_pose','body_pose']),
                         observation_manager=SimpleNamespace(active_terms={'actor':['joint_pos']}))
    return env,model


def test_original_metadata_path_is_unchanged():
    env,_=fixture()
    assert policy_metadata(env,'checkpoint')==get_base_metadata(env,'checkpoint')


def test_newton_metadata_uses_spec_matching_reference_without_touching_physics():
    env,model=fixture()
    expected=get_base_metadata(env,'checkpoint')
    empty=SimpleNamespace(actuator_gainprm=np.zeros((0,10)),actuator_biasprm=np.zeros((0,10)))
    env.sim.mj_model=empty
    env.scene['robot'].reference_model=model
    with pytest.raises(IndexError):get_base_metadata(env,'checkpoint')
    actual=policy_metadata(env,'checkpoint')
    assert {k:actual[k] for k in expected}==expected
    assert len(actual['joint_names'])==14
    assert len(actual['joint_stiffness'])==14
    assert actual['actuator_model']=='BAM_XL330_M6'
    assert env.sim.mj_model is empty
    assert empty.actuator_gainprm.shape==(0,10)


def test_newton_save_keeps_native_checkpoint_and_normalized_onnx(tmp_path,capsys):
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict
    from oh_my_duck.rl.learners.rsl_rl.runner import MicroduckOnPolicyRunner
    from oh_my_duck.rl.artifacts.inference import cpu_session
    env,reference=fixture()
    env.scene['robot'].reference_model=reference
    env.sim.mj_model=SimpleNamespace(actuator_gainprm=np.zeros((0,10)),actuator_biasprm=np.zeros((0,10)))
    env.common_step_counter=48000
    torch.manual_seed(19)
    batch=TensorDict({'actor':torch.randn(32,61)*2+3},batch_size=[32])
    actor=MLPModel(batch,{'actor':['actor']},'actor',14,hidden_dims=[16],obs_normalization=True)
    actor.obs_normalizer.update(batch['actor'])
    before={key:value.clone() for key,value in actor.state_dict().items()}
    runner=MicroduckOnPolicyRunner.__new__(MicroduckOnPolicyRunner)
    runner.env=SimpleNamespace(unwrapped=env)
    runner.alg=SimpleNamespace(save=lambda:{'actor_state_dict':actor.state_dict(),'optimizer_state_dict':{'sentinel':123}},get_policy=lambda:actor)
    runner.current_learning_iteration=2000
    runner.cfg={'upload_model':False}
    runner.save(str(tmp_path/'model_2000.pt'),{'test':'retained'})
    checkpoint=torch.load(tmp_path/'model_2000.pt',weights_only=False)
    assert checkpoint['iter']==2000
    assert checkpoint['infos']=={'test':'retained','env_state':{'common_step_counter':48000}}
    assert checkpoint['optimizer_state_dict']=={'sentinel':123}
    exported=tmp_path/(tmp_path.name+'.onnx')
    session=cpu_session(str(exported),providers=['CPUExecutionProvider'])
    assert session.get_modelmeta().custom_metadata_map['simulation_backend']=='isaac-newton'
    with torch.no_grad():expected=actor(batch).numpy()
    actual=np.concatenate([session.run(None,{'obs':row[None]})[0] for row in batch['actor'].numpy()])
    np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-5)
    assert all(torch.equal(before[key],value) for key,value in actor.state_dict().items())
    assert 'ONNX export failed' not in capsys.readouterr().out

    # Exercise the gate with a separate native export, and reject a stale actor.
    from oh_my_duck.rl.artifacts.metadata import verify_periodic_export
    runner.export_policy_to_onnx(str(tmp_path), 'reference.onnx')
    parity = verify_periodic_export(exported, tmp_path / 'reference.onnx')
    assert parity['samples'] == 65 and parity['max_abs_error'] < 1e-6
    with torch.no_grad():
        next(actor.parameters()).add_(.5)
    runner.export_policy_to_onnx(str(tmp_path), 'stale.onnx')
    with pytest.raises(AssertionError):
        verify_periodic_export(exported, tmp_path / 'stale.onnx')

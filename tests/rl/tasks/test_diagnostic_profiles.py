"""Diagnostic interventions must not silently change acceptance or forced resets."""
from types import SimpleNamespace
import numpy as np
import pytest
from oh_my_duck.rl.evaluation.diagnostics import prepare_recipe, summarize_rewards


def test_stage_is_detached_before_protocol_reset_and_original_remains_intact():
    term={'weight_stages':[{'step':0,'weight':2.0}]}
    cfg=SimpleNamespace(curriculum={'reward':term})
    stage=prepare_recipe(cfg,profile='training-stage',curriculum_step=100)
    assert cfg.curriculum == {}
    stage['reward']['weight_stages'][0]['weight']=1.0
    assert term['weight_stages'][0]['weight']==2.0


@pytest.mark.parametrize('profile,step',[('standard',12),('training-stage',None),('training-stage',-1)])
def test_ambiguous_stage_is_rejected(profile,step):
    with pytest.raises(ValueError):prepare_recipe(SimpleNamespace(curriculum={}),profile=profile,curriculum_step=step)


def test_reward_rates_use_final_second_and_preserve_penalty_signs():
    values=np.column_stack([np.r_[np.ones(50),np.full(50,3.)],np.full(100,-2.)])
    result=summarize_rewards({'weighted_reward_rate':values},['task','penalty'])
    assert result['task']['mean_rate']==2
    assert result['task']['final_second_mean_rate']==3
    assert result['penalty']['max_rate']==-2
    values[0,0]=np.nan
    with pytest.raises(ValueError):summarize_rewards({'weighted_reward_rate':values},['task','penalty'])


def test_actual_curriculum_updates_live_terms_before_push_scaling():
    from mjlab.managers.curriculum_manager import CurriculumTermCfg
    from oh_my_duck.rl.mdp.curricula import reward_weight, push_curriculum
    from oh_my_duck.rl.evaluation.diagnostics import apply_stage
    reward=SimpleNamespace(weight=9.)
    push=SimpleNamespace(params={'velocity_range':{'x':(-.1,.1),'y':(-.1,.1)}})
    env=SimpleNamespace(num_envs=1,device='cpu',cfg=SimpleNamespace(events={'push_robot':push}),
                        event_manager=SimpleNamespace(get_term_cfg=lambda name:push),
                        reward_manager=SimpleNamespace(active_terms=['task'],get_term_cfg=lambda name:reward))
    stages={'weight':CurriculumTermCfg(func=reward_weight,params={'reward_name':'task','weight_stages':[{'step':0,'weight':2.}]}),
            'push':CurriculumTermCfg(func=push_curriculum,params={'event_name':'push_robot','push_stages':[{'step':0,'velocity_range':{'x':(-.3,.3),'y':(-.3,.3)}}]})}
    result=apply_stage(env,stages,100,0.)
    assert result['task']==2.
    assert env.common_step_counter==100
    assert push.params['velocity_range']=={'x':(-0.,0.),'y':(-0.,0.)}

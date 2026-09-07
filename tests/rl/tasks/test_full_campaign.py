from pathlib import Path
from oh_my_duck.rl.experiments.campaign import load_plan
from oh_my_duck.rl.training.tasks import project_tasks


def test_full_campaign_matches_registered_task_defaults():
    plan=load_plan(Path(__file__).parents[3]/'configs/experiments/representative-full.json')
    catalog=project_tasks()
    assert len(plan['runs'])==8
    for row in plan['runs']:
        binding=catalog.get(row['task']).binding(row['backend'])
        assert binding.rsl_config.build().max_iterations==row['iterations']
        assert row['num_envs']==4096

"""Static/configuration gates for the experimental protective-fall variant."""

def test_protective_fall_keeps_official_velstand_factory_unchanged():
    from oh_my_duck.rl.tasks.velocity_stand.environment import make_microduck_velstand_env_cfg

    cfg = make_microduck_velstand_env_cfg()
    robot = cfg.scene.entities["robot"]
    assert robot.spec_fn.__name__ == "get_standup_spec"
    assert "servo_impact_contact" not in {sensor.name for sensor in cfg.scene.sensors}


def test_protective_fall_uses_full_collision_and_protection_terms():
    from oh_my_duck.rl.tasks.protective_fall.environment import (
        make_microduck_protective_fall_env_cfg,
    )

    cfg = make_microduck_protective_fall_env_cfg()
    robot = cfg.scene.entities["robot"]
    assert robot.spec_fn.__name__ == "get_allcollisions_spec"
    sensors = {sensor.name: sensor for sensor in cfg.scene.sensors}
    impact = sensors["servo_impact_contact"]
    assert impact.primary.pattern == r".*_servo_collision$"
    assert impact.fields == ("force",)
    assert impact.reduce == "netforce"
    assert {
        "servo_impact_penalty",
        "servo_stall_penalty",
        "servo_acc_spike_penalty",
    } <= set(cfg.rewards)
    assert cfg.rewards["action_rate_l2"].func.__name__ == "action_rate_l2_fallen_scaled"
    assert cfg.rewards["joint_torque_rate_l2"].func.__name__ == "joint_torque_rate_l2_fallen_scaled"


def test_protective_asset_compiles_with_fourteen_servos():
    from oh_my_duck.robotics.microduck.microduck_constants import (
        MICRODUCK_ALLCOLLISIONS_ROBOT_CFG,
    )

    model = MICRODUCK_ALLCOLLISIONS_ROBOT_CFG.build().compile()
    servos = [
        model.joint(i).name
        for i in range(model.njnt)
        if model.jnt_type[i] != 0 and not model.joint(i).name.startswith("passive_")
    ]
    assert len(servos) == 14
    assert all(name for name in servos)

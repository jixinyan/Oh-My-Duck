"""Experimental full-collision protective-fall environment.

The official VelStand recipe remains the reference recovery task. This factory
builds a separate variant so impact sensing and fallen-state regularisation can
be trained without changing the official baseline or its checkpoints.

The variant preserves the canonical observation/action contract by decorating
the existing VelStand configuration. It is deliberately not representative:
its full-collision model and added protection terms require their own Newton,
export, replay and CPU/BAM validation before packaging as a skill.
"""
from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from oh_my_duck.robotics.microduck.microduck_constants import (
    MICRODUCK_ALLCOLLISIONS_ROBOT_CFG,
)
from oh_my_duck.rl import mdp as microduck_mdp
from oh_my_duck.rl.tasks.velocity_stand.environment import (
    REWARD_GATE_TILT_DEG,
    make_microduck_velstand_env_cfg,
)

# These are intentionally conservative starting values. They are task
# configuration, not acceptance thresholds; a learned policy must still pass
# the standard behaviour and transfer gates.
SERVO_STALL_TORQUE_THRESHOLD = 0.4
SERVO_STALL_VELOCITY_THRESHOLD = 0.5
SERVO_ACCELERATION_THRESHOLD = 300.0
SERVO_IMPACT_FORCE_THRESHOLD = 3.0
FALLEN_SMOOTHNESS_SCALE = 0.1


def make_microduck_protective_fall_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the non-representative full-collision recovery variant."""
    cfg = make_microduck_velstand_env_cfg(play=play, rough=rough)

    # Keep the official VelStand factory untouched. The complete collision
    # asset is needed to expose servo housing impacts during a fall.
    cfg.scene.entities = {"robot": MICRODUCK_ALLCOLLISIONS_ROBOT_CFG}
    servo_impact_cfg = ContactSensorCfg(
        name="servo_impact_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r".*_servo_collision$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )
    cfg.scene.sensors = (*cfg.scene.sensors, servo_impact_cfg)

    # Impact and actuator-protection costs are negative weighted rewards. The
    # imported MDP functions use finite, canonical 14-servo views and reset
    # their temporal state at episode boundaries.
    robot = SceneEntityCfg("robot")
    cfg.rewards["servo_impact_penalty"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=-0.02,
        params={
            "sensor_name": servo_impact_cfg.name,
            "threshold": SERVO_IMPACT_FORCE_THRESHOLD,
        },
    )
    cfg.rewards["servo_stall_penalty"] = RewardTermCfg(
        func=microduck_mdp.servo_stall_penalty,
        weight=-0.002,
        params={
            "asset_cfg": robot,
            "torque_thresh": SERVO_STALL_TORQUE_THRESHOLD,
            "vel_thresh": SERVO_STALL_VELOCITY_THRESHOLD,
        },
    )
    cfg.rewards["servo_acc_spike_penalty"] = RewardTermCfg(
        func=microduck_mdp.servo_acc_spike_penalty,
        weight=-0.0005,
        params={
            "asset_cfg": robot,
            "acc_thresh": SERVO_ACCELERATION_THRESHOLD,
        },
    )

    # During a fall, motion needed for recovery should not be suppressed by
    # the walking smoothness costs. Upright behaviour retains the native
    # VelStand weights; only the fallen state is scaled by the shared gate.
    action_rate = cfg.rewards.get("action_rate_l2")
    if action_rate is not None:
        cfg.rewards["action_rate_l2"] = RewardTermCfg(
            func=microduck_mdp.action_rate_l2_fallen_scaled,
            weight=action_rate.weight,
            params={
                "fallen_scale": FALLEN_SMOOTHNESS_SCALE,
                "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
            },
        )
    torque_rate = cfg.rewards.get("joint_torque_rate_l2")
    if torque_rate is not None:
        cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
            func=microduck_mdp.joint_torque_rate_l2_fallen_scaled,
            weight=torque_rate.weight,
            params={
                "asset_cfg": robot,
                "fallen_scale": FALLEN_SMOOTHNESS_SCALE,
                "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
            },
        )

    return cfg

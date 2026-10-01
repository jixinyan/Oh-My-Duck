from copy import deepcopy

from oh_my_duck.robotics.microduck.microduck_constants import (
    MICRODUCK_ALLCOLLISIONS_ROBOT_CFG,
    get_allcollisions_spec,
    MICRODUCK_WALK_ROLLERS_ROBOT_CFG,
    get_walk_rollers_spec,
)


def name_scene_collisions(spec):
    counts = {}
    for geom in spec.geoms:
        if geom.name or not (geom.contype or geom.conaffinity):
            continue
        body = geom.parent.name
        index = counts.get(body, 0)
        counts[body] = index + 1
        geom.name = f"{body}_scene_geom_{index}_collision"
    return spec


def get_isaac_allcollisions_spec():
    return name_scene_collisions(get_allcollisions_spec())


def get_isaac_rollers_spec():
    return name_scene_collisions(get_walk_rollers_spec())


def get_isaac_rollers_cfg():
    cfg = deepcopy(MICRODUCK_WALK_ROLLERS_ROBOT_CFG)
    cfg.spec_fn = get_isaac_rollers_spec
    cfg.collisions = ()
    return cfg


def get_isaac_allcollisions_cfg():
    cfg = deepcopy(MICRODUCK_ALLCOLLISIONS_ROBOT_CFG)
    cfg.spec_fn = get_isaac_allcollisions_spec
    # 外部场景使用原始 MJCF 的全部碰撞属性。
    cfg.collisions = ()
    return cfg

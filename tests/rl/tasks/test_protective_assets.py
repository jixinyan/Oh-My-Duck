import hashlib

import mujoco

from oh_my_duck.robotics.microduck.microduck_constants import (
    MICRODUCK_ALLCOLLISIONS_BACKLASH_XML,
    get_allcollisions_backlash_spec,
    name_servo_collision_geoms,
)


def test_official_full_collision_backlash_asset_is_pinned_and_compilable():
    digest = hashlib.sha256(MICRODUCK_ALLCOLLISIONS_BACKLASH_XML.read_bytes()).hexdigest()
    assert digest == "ccee92e435e39eefc64ee99745823751492ae7f948d835c89701710717e71de8"
    spec = get_allcollisions_backlash_spec()
    names = [geom.name for geom in spec.geoms if geom.name.endswith("_servo_collision")]
    assert len(names) >= 14
    assert len(names) == len(set(names))
    model = spec.compile()
    assert model.nu == 14


def test_servo_geom_naming_is_idempotent_for_a_fresh_spec():
    spec = mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_BACKLASH_XML))
    assert name_servo_collision_geoms(spec) is spec
    first = sorted(geom.name for geom in spec.geoms if geom.name.endswith("_servo_collision"))
    assert name_servo_collision_geoms(spec) is spec
    second = sorted(geom.name for geom in spec.geoms if geom.name.endswith("_servo_collision"))
    assert first == second

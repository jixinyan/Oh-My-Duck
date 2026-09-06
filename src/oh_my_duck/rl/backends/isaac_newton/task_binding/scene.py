"""Shared task scene interfaces backed by the native Isaac scene."""

from types import SimpleNamespace
from mjlab.scene import Scene
from mjlab.sensor import ContactSensor, TerrainHeightSensor, BuiltinSensor
from oh_my_duck.rl.backends.isaac_newton.mdp import as_torch
from oh_my_duck.rl.backends.isaac_newton.task_binding.entity import NewtonEntity
from oh_my_duck.rl.backends.isaac_newton.task_binding.sensors import (
    NewtonContactSensor,
    NewtonBuiltinSensor,
    NewtonFlatHeightSensor,
)


class NewtonScene(Scene):
    def __init__(self, cfg, native, simulation):
        # Compile reference metadata only. It does not instantiate a second simulator.
        reference_scene = Scene(cfg, "cpu")
        reference = reference_scene.compile()
        self._cfg, self._device = cfg, simulation.device
        self._spec = reference_scene.spec
        self._sensor_context = None
        self._default_env_origins = as_torch(native.scene.env_origins)
        self._terrain = SimpleNamespace(
            cfg=cfg.terrain, env_origins=self._default_env_origins, terrain_generator=None
        )
        self._entities = {"robot": NewtonEntity(cfg.entities["robot"], native.scene["robot"], simulation)}
        robot = self._entities["robot"]
        self._sensors = {}
        for name, sensor in reference_scene.sensors.items():
            if isinstance(sensor, ContactSensor):
                bound = NewtonContactSensor(sensor.cfg, robot, simulation)
            elif isinstance(sensor, TerrainHeightSensor):
                bound = NewtonFlatHeightSensor(sensor.cfg, robot, simulation, reference)
            elif isinstance(sensor, BuiltinSensor):
                if int(reference.sensor(name).type[0]) == 42:
                    continue  # internal slots of the already-bound contact sensor
                bound = NewtonBuiltinSensor(name, reference, robot, simulation)
            else:
                raise ValueError(f"Unbound sensor: {name}")
            self._sensors[name] = bound
        simulation.scene = self

"""Unbound USD geometry must not silently index the final solver geom."""
import unittest
from types import SimpleNamespace
import torch
from mjlab.entity import Entity
from oh_my_duck.robotics.microduck.microduck_constants import MICRODUCK_WALK_ROBOT_CFG
from oh_my_duck.rl.backends.isaac_newton.task_binding.entity import NewtonEntity


class GeometrySelectors(unittest.TestCase):
    def test_bound_foot_and_unbound_geometry_are_distinguished(self):
        robot=NewtonEntity.__new__(NewtonEntity)
        Entity.__init__(robot,MICRODUCK_WALK_ROBOT_CFG)
        ids,_=robot.find_geoms('left_foot_collision')
        mapping=torch.full((len(robot.geom_names),),-1,dtype=torch.int32)
        robot.indexing=SimpleNamespace(geom_ids=mapping)
        with self.assertRaises(ValueError):robot.find_geoms('left_foot_collision')
        mapping[ids]=3
        self.assertEqual(robot.find_geoms('left_foot_collision')[0],ids)

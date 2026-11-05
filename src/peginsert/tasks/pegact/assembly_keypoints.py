# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Board, Franka and 4 mm square-peg keypoints (quaternions: x, y, z, w)."""

from __future__ import annotations

from isaaclab.utils import configclass

from peginsert.utils.pose_offset import Offset

__all__ = ["Offset"]


@configclass
class NistBoardKeyPointsCfg:
    """Placement offsets used by PegInset on the NIST board."""
    nist_board_center: Offset = Offset(pos=(0.197176, -0.19145, 0.0))
    rectangular_peg_4mm: Offset = Offset(pos=(0.1971, -0.1915, -0.0003), quat=(0.7071, -0.7071, 0.0, 0.0))
    rectangular_hole_4mm: Offset = Offset(pos=(0.1971, -0.1915, -0.0003), quat=(0.7071, -0.7071, 0.0, 0.0))


@configclass
class RectangularPeg4MMKeyPointsCfg:
    geometry_origin: Offset = Offset(pos=(0.0, 0.0, 0.025))
    peg_tip: Offset = Offset(pos=(0.0, 0.0, 0.0))
    grasp_point: Offset = Offset(pos=(0.0, 0.0, 0.035))
    grasp_diameter: float = 0.004


@configclass
class RectangularHole4MMKeyPointsCfg:
    above_hole_tip_offset: Offset = Offset(pos=(0.0, 0.0, 0.015))
    one_mm_above_hole_tip_offset: Offset = Offset(pos=(0.0, 0.0, 0.01))
    hole_tip_offset: Offset = Offset(pos=(0.0, 0.0, 0.009))
    inserted_peg_base_offset: Offset = Offset(pos=(0.0, 0.0, 0.0))


@configclass
class PandaHandKeyPointsCfg:
    """Grasp keypoints on the Franka Panda hand, relative to the ``panda_hand`` link.

    The 180-degree rotation around y flips the gripper so it faces downward
    for top-down grasps.
    """

    gripper_center_grasp_point: Offset = Offset(pos=(0.0, 0.0, 0.107), quat=(0.0, 1.0, 0.0, 0.0))
    gripper_tip_grasp_point: Offset = Offset(pos=(0.0, 0.0, 0.112), quat=(0.0, 1.0, 0.0, 0.0))


@configclass
class RobotRootKeyPointsCfg:
    base: Offset = Offset(pos=(0.0, 0.0, 0.0), quat=(0.0, 1.0, 0.0, 0.0))

NIST_BOARD_CFG = NistBoardKeyPointsCfg()

RECTANGULAR_PEG_4MM = RectangularPeg4MMKeyPointsCfg()

RECTANGULAR_HOLE_4MM = RectangularHole4MMKeyPointsCfg()

PANDA_HAND = PandaHandKeyPointsCfg()

ROBOT = RobotRootKeyPointsCfg()

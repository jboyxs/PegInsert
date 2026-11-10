# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Concrete assembly parameters for PegInset (NIST PegInsert4MM)."""

import math

from isaaclab.managers import SceneEntityCfg

from peginsert.tasks.pegact import assembly_keypoints as kpts
from peginsert.tasks.pegact.assembly_profile_cfg import (
    AssemblyProfileCfg,
    DiscreteYawCfg,
    IncrementalSegmentCfg,
    UniformPoseNoiseCfg,
)

FIXED_ASSET_MAP = dict(fixed_asset="rectangular_hole_4mm")

# Potential shaping and PPO must use the same per-control-step discount.
DISCOUNT_FACTOR = 0.995

ASSEMBLY_TIP = kpts.RECTANGULAR_HOLE_4MM.hole_tip_offset


def _dist(a: kpts.Offset, b: kpts.Offset) -> tuple[float, float, float]:
    """Position delta from offset *a* to offset *b* [m]."""
    return (b.pos[0] - a.pos[0], b.pos[1] - a.pos[1], b.pos[2] - a.pos[2])


ASSEMBLY_PROFILE = AssemblyProfileCfg(
    segments=[
        IncrementalSegmentCfg(
            fraction=(0.0, 0.7),
            start_sampler=DiscreteYawCfg(yaws=[0.0, math.pi / 2, math.pi, 3 * math.pi / 2]),
            start_pose=kpts.RECTANGULAR_HOLE_4MM.inserted_peg_base_offset,
            distance=_dist(
                kpts.RECTANGULAR_HOLE_4MM.inserted_peg_base_offset, kpts.RECTANGULAR_HOLE_4MM.hole_tip_offset
            ),
        ),
        IncrementalSegmentCfg(
            fraction=(0.7, 1.5),
            start_sampler=UniformPoseNoiseCfg(
                x=(-0.01, 0.01), y=(-0.01, 0.01), roll=(-0.3, 0.3), pitch=(-0.3, 0.3), yaw=(-3.14, 3.14)
            ),
            start_pose=kpts.RECTANGULAR_HOLE_4MM.one_mm_above_hole_tip_offset,
            distance=_dist(
                kpts.RECTANGULAR_HOLE_4MM.one_mm_above_hole_tip_offset,
                kpts.RECTANGULAR_HOLE_4MM.above_hole_tip_offset,
            ),
        ),
    ]
)

HELD_ASSET_ALIGN_OFFSET = kpts.RECTANGULAR_PEG_4MM.peg_tip

HELD_ASSET_GRASP_POINT = kpts.RECTANGULAR_PEG_4MM.grasp_point

HELD_ASSET_GRASP_DIAMETER = kpts.RECTANGULAR_PEG_4MM.grasp_diameter

HELD_ASSET_GRASP_MIDDLE = kpts.RECTANGULAR_PEG_4MM.grasp_point

_INSERT_GRASPED_RANGE = dict(
    x=(-0.005, 0.005),
    y=(-0.005, 0.005),
    z=(0.047, 0.057),
    roll=(3.141, 3.141),
    pitch=(-0.5, 0.5),
    yaw=(-2.09, 2.09),
)

GRASPED_POSE_RANGE = _INSERT_GRASPED_RANGE

RESET_ASSETS = ["nistboard", "fixed_asset", "held_asset", "robot"]

HELD_ASSET_OBSTACLES = [
    SceneEntityCfg("fixed_asset"),
    SceneEntityCfg("robot"),
    SceneEntityCfg("nistboard"),
    SceneEntityCfg("table"),
]

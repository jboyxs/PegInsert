# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka scene for the NIST 4 mm square-peg insertion task."""

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.utils import preset

from . import assets_cfg as assets

_FRANKA_PANDA_PHYSX_CFG = replace(assets.FRANKA_PANDA_PHYSX_CFG, prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class PegInsetSceneBase(InteractiveSceneCfg):
    """Shared Franka and NIST board assets for PegInset."""

    num_envs: int = 4096
    ground = assets.GROUND_CFG
    table = assets.TABLE_CFG
    nistboard = assets.NISTBOARD_CFG
    robot: ArticulationCfg = preset(  # type: ignore[assignment]
        default=_FRANKA_PANDA_PHYSX_CFG,
        isaacsim_physx=_FRANKA_PANDA_PHYSX_CFG,
        physx=_FRANKA_PANDA_PHYSX_CFG,
        newton_mjwarp=replace(assets.FRANKA_PANDA_NEWTON_CFG, prim_path="{ENV_REGEX_NS}/Robot"),
    )
    dome_light = assets.DOMELIGHT_CFG


@configclass
class PegInsetSceneCfg(PegInsetSceneBase):
    env_spacing: float = 2.0

    fixed_asset: RigidObjectCfg = assets.RECTANGULAR_HOLE_4MM_CFG
    held_asset: RigidObjectCfg = assets.RECTANGULAR_PEG_4MM_CFG

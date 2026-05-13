# Copyright (c) 2022-2023, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Modified version of the original FRANKA_PANDA_CFG of Isaac Lab
# Adapted for ARX-X5A arm with fixed XenseWS adapter
#

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from tacex_assets import TACEX_ASSETS_DATA_DIR
##
# Configuration
##

X5A_ARM_XENSEWS_GRIPPER_UIPC_HIGH_RES_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TACEX_ASSETS_DATA_DIR}/Robots/ARX-X5/XenseWS/X5A_XenseWS.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=0.5,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "x5a_joint1": 0.0,
            "x5a_joint2": 0.0,
            "x5a_joint3": 0.0,
            "x5a_joint4": 0.0,
            "x5a_joint5": 0.0,
            "x5a_joint6": 0.0,
            "x5a_adapter_left_mount": 0.02,
            "x5a_adapter_right_mount": 0.02,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "x5a_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["x5a_joint[1-3]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=1000.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "x5a_forearm": ImplicitActuatorCfg(
            joint_names_expr=["x5a_joint[4-6]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=1000.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "x5a_gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "x5a_adapter_left_mount",
                "x5a_adapter_right_mount",
            ],
            effort_limit_sim=1000.0,
            velocity_limit_sim=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# --- Stiffer PD version for Differential IK / Task-space control ---

X5A_ARM_XENSEWS_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG = X5A_ARM_XENSEWS_GRIPPER_UIPC_HIGH_RES_CFG.copy()

X5A_ARM_XENSEWS_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG.spawn.rigid_props.disable_gravity = True
X5A_ARM_XENSEWS_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG.actuators["x5a_shoulder"].stiffness = 400.0
X5A_ARM_XENSEWS_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG.actuators["x5a_shoulder"].damping = 80.0
X5A_ARM_XENSEWS_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG.actuators["x5a_forearm"].stiffness = 400.0
X5A_ARM_XENSEWS_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG.actuators["x5a_forearm"].damping = 80.0

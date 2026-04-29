from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_gripper_uipc_high_res import (
    FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG
)
from tacex_assets.robots.x5a.x5a_xensews_gripper_uipc import (
    X5A_ARM_XENSEWS_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG
)
from tacex_assets.robots.franka.franka_gf225_gripper_uipc import (
    FRANKA_PANDA_ARM_GF225_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG
)

from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from ..sensors.tactile import TactileCfg, create_tactile_cfg

@configclass
class RobotCfg:
    robot: ArticulationCfg = None
    tactiles: list[TactileCfg] = []

    gripper_offset: float = 0.131 # in m
    gripper_max_qpos: float = 0.039 # in m

    tactile_far_plane: float = 30.0 # in mm
    adaptive_grasp_depth_threshold: float = 27.5 # in mm, used for grasping
    contact_threshold: tuple[float, float] = (27.5, 28.0) # in mm, used in `gravity_rotate` api

def create_franka_gsmini_gripper(data_type:list[str]):
    robot = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": 0.0,
                "panda_joint3": 0.0,
                "panda_joint4": -2.46,
                "panda_joint5": 0.0,
                "panda_joint6": 2.5,
                "panda_joint7": 0.741,
                "panda_finger.*": 0.02,
            }
        ),
    )
    tactiles = [
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left",
            gelpad_prim_path="/World/envs/env_.*/Robot/gelpad_left",
            gelpad_attachment_body_name="gelsight_mini_case_left",
            name="left_tactile",
            sensor_type="gsmini",
            data_type=data_type,
        ),
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right",
            gelpad_prim_path="/World/envs/env_.*/Robot/gelpad_right",
            gelpad_attachment_body_name="gelsight_mini_case_right",
            name="right_tactile",
            sensor_type="gsmini",
            data_type=data_type,
        )
    ]
    return RobotCfg(
        robot=robot,
        tactiles=tactiles,
        gripper_offset=0.131,
        gripper_max_qpos=0.039,
        tactile_far_plane=34.0,
        adaptive_grasp_depth_threshold=27.5,
        contact_threshold=(27.5, 28.0)
    )

def create_franka_gf225_gripper(data_type:list[str]):
    robot = FRANKA_PANDA_ARM_GF225_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": 0.0,
                "panda_joint3": 0.0,
                "panda_joint4": -2.46,
                "panda_joint5": 0.0,
                "panda_joint6": 2.5,
                "panda_joint7": 0.741,
                "panda_finger.*": 0.02,
            }
        ), 
    )
    tactiles = [
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/GF225_left",
            gelpad_prim_path="/World/envs/env_.*/Robot/GF225_gelpad_left",
            gelpad_attachment_body_name="GF225_left",
            name="left_tactile",
            sensor_type="gf225",
            data_type=data_type,
        ),
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/GF225_right",
            gelpad_prim_path="/World/envs/env_.*/Robot/GF225_gelpad_right",
            gelpad_attachment_body_name="GF225_right",
            name="right_tactile",
            sensor_type="gf225",
            data_type=data_type,
        )
    ]
    return RobotCfg(
        robot=robot,
        tactiles=tactiles,
        gripper_offset=0.131,
        gripper_max_qpos=0.039,
        tactile_far_plane=26.5,
        adaptive_grasp_depth_threshold=25.3,
        contact_threshold=(25.5, 26.3)
    )
def create_x5a_xensews_gripper(data_type: list[str]):
    """Create RobotCfg for the current X5A + XenseWS USD/URDF layout.

    Expected USD/URDF structure:
      /Robot/x5a_link0 ... /Robot/x5a_link6
      /Robot/x5a_adapter_link
      /Robot/x5a_adapter_left_link/xense_left_mount/XenseWS_gelpad_left
      /Robot/x5a_adapter_right_link/xense_right_mount/XenseWS_gelpad_right
      joints: x5a_joint1..6, x5a_link6_to_adapter,
              x5a_adapter_left_mount, x5a_adapter_right_mount

    The two adapter mount joints are prismatic in the latest URDF:
      x5a_adapter_left_mount:  axis Y,  lower 0.0, upper 0.02
      x5a_adapter_right_mount: axis -Y, lower 0.0, upper 0.02
    """
    robot = X5A_ARM_XENSEWS_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
             pos=(0.0, 0.0, 0.0),
             rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                # 6-DoF X5A arm.
                "x5a_joint1": 0.0,
                "x5a_joint2": 0.0,
                "x5a_joint3": 0.0,
                "x5a_joint4": 0.0,
                "x5a_joint5": 0.0,
                "x5a_joint6": 0.0,
                # XenseWS gripper prismatic joints from X5A.urdf.
                # 0.0 is the closed/default pose; 0.02 is the maximum open pose.
                "x5a_adapter_left_mount": 0.02,
                "x5a_adapter_right_mount": 0.02,
            }
        ),
    )

    robot.spawn.usd_path = f"{TACEX_ASSETS_DATA_DIR}/Robots/ARX-X5/XenseWS/X5A_XenseWS.usd"
    robot.actuators["x5a_gripper"] = ImplicitActuatorCfg(
        joint_names_expr=[
            "x5a_adapter_left_mount",
            "x5a_adapter_right_mount",
        ],
        effort_limit_sim=1000.0,
        velocity_limit_sim=0.2,
        stiffness=5000.0,
        damping=500.0,
    )

    tactiles = [
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/xense_left_mount",
            gelpad_prim_path="/World/envs/env_.*/Robot/xense_left_mount/XenseWS_gelpad_left",
            gelpad_attachment_body_name="xense_left_mount",
            gelpad_attachment_prim_path="/World/envs/env_.*/Robot/xense_left_mount",
            name="left_tactile",
            sensor_type="xensews",
            data_type=data_type,
        ),
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/xense_right_mount",
            gelpad_prim_path="/World/envs/env_.*/Robot/xense_right_mount/XenseWS_gelpad_right",
            gelpad_attachment_body_name="xense_right_mount",
            gelpad_attachment_prim_path="/World/envs/env_.*/Robot/xense_right_mount",
            name="right_tactile",
            sensor_type="xensews",
            data_type=data_type,
        )
    ]

    return RobotCfg(
        robot=robot,
        tactiles=tactiles,
        # Approximate offset from x5a_link6 to the grasp center.
        # Recalibrate this after the final x5a_link6_to_adapter and mount origins are fixed.
        gripper_offset=0.135,
        # Matches the URDF prismatic joint upper limit.
        gripper_max_qpos=0.025,
        tactile_far_plane=30.0,
        adaptive_grasp_depth_threshold=27.3,
        contact_threshold=(27.5, 27.8),
    )

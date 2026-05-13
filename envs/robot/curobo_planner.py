from curobo.geom.transform import pose_multiply
import numpy as np
import transforms3d as t3d
from curobo.types.robot import JointState
from curobo.util.usd_helper import UsdHelper
from curobo.types.math import Pose as CuroboPose
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)
import torch
import yaml
from curobo.util import logger
from copy import deepcopy

from pydantic import constr
logger.setup_logger(level="error", logger_name="curobo")

from pathlib import Path
from ..utils.transforms import *
from isaaclab.utils import configclass

@configclass
class CuroboPlannerCfg:
    dt: float = 1/120
    yaml_path: str = None
    robot_prime_path: str = "/World/robot"

    all_joints_name: list[str] = None
    active_joints_name: list[str] = None

    time_dilation_factor: float = 1.0

class CuroboPlanner:
    def __init__(
        self,
        task: 'BaseTask',
        cfg: CuroboPlannerCfg,
        robot_origin_pose:Pose,
    ):
        super().__init__()
        logger.setup_logger(level="error", logger_name="'curobo")

        self.cfg = cfg
        self.task = task
        self.dt = cfg.dt
        self.robot_prime_path = cfg.robot_prime_path
        self.robot_origin_pose = robot_origin_pose
        self.active_joints_name = cfg.active_joints_name
        self.all_joints = cfg.all_joints_name
        # translate from baselink to arm's base
        with open(self.cfg.yaml_path, "r") as f:
            yml_data = yaml.safe_load(f)

        file_dir = Path(self.cfg.yaml_path).parent
        urdf_path = yml_data['robot_cfg']['kinematics']['urdf_path']
        if not Path(urdf_path).is_absolute():
            yml_data['robot_cfg']['kinematics']['urdf_path'] = str(file_dir / urdf_path)
        collision_spheres = yml_data['robot_cfg']['kinematics']['collision_spheres']
        if not Path(collision_spheres).is_absolute():
            yml_data['robot_cfg']['kinematics']['collision_spheres'] = str(file_dir / collision_spheres)

        self.frame_bias = yml_data["planner"]["frame_bias"]

        self.usd_helper = UsdHelper()
        self.usd_helper.load_stage(self.task.scene.stage)

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg=yml_data,
            world_model=self.get_curr_world_cfg(),
            interpolation_dt=self.dt,
            position_threshold=0.001,
            rotation_threshold=0.01,
            high_precision=True,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_activation_distance=0.4
        )
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup()
    
    def reset(self):
        self.motion_gen.reset()

    def get_curr_world_cfg(self):
        # obstacles = self.usd_helper.get_obstacles_from_stage(
        #     only_paths=["/World"],
        #     reference_prim_path='/World/envs/env_0/Robot',
        #     ignore_paths=[
        #         '/World/visualize',
        #         '/World/envs/env_0/Robot',
                # '/World/envs/env_0/ground_plate',
        #     ]
        # ).get_collision_check_world()
        obstacles = {
            "cuboid": {
                "table": {
                    "dims": [0.5, 0, 0],
                    "pose": [-1000, 0.0, 0.0, 1, 0, 0, 0],
                },
            }
        }
        return obstacles

    def update_world(self):
        self.motion_gen.update_world(self.get_curr_world_cfg())

    def plan_path(
        self,
        curr_joint_pos: torch.Tensor,
        curr_joint_vel: torch.Tensor,
        target_ee_pose,
        real_robot_pose,
        current_ee_pose=None,
        pre_dis=None,
        constraint_pose=None,
        time_dilation_factor=None
    ):
        # self.update_world()

        joint_indices = np.array([
            self.all_joints.index(name)
            for name in self.active_joints_name
            if name in self.all_joints
        ])
        joint_pos = curr_joint_pos[joint_indices].reshape(1, -1)
        joint_vel = curr_joint_vel[joint_indices].reshape(1, -1)

        # Dynamic FK alignment between Isaac's EE frame and cuRobo's EE frame.
        # Reason:
        #   Isaac and cuRobo can report different absolute x5a_link6 positions
        #   for the same joint state. A static world-frame frame_bias only works
        #   near one configuration and can shift targets after grasp.
        # Method:
        #   Convert the Isaac requested motion into a relative transform:
        #       delta = inv(T_isaac_current) @ T_isaac_target
        #   Then apply that same relative motion from cuRobo's current FK pose:
        #       T_curobo_target = T_curobo_current @ delta
        #   This makes target=current stay at current in every configuration,
        #   and makes small EE displacements remain small in cuRobo space.
        try:
            fk_state = self.motion_gen.kinematics.get_state(joint_pos)
            curobo_current_pose = Pose(
                fk_state.ee_position[0].detach().cpu().numpy(),
                fk_state.ee_quaternion[0].detach().cpu().numpy(),
            )

            if current_ee_pose is not None:
                T_isaac_current = current_ee_pose.to_transformation_matrix()
                T_isaac_target = target_ee_pose.to_transformation_matrix()
                T_curobo_current = curobo_current_pose.to_transformation_matrix()

                T_delta = np.linalg.inv(T_isaac_current) @ T_isaac_target
                T_curobo_target = T_curobo_current @ T_delta
                target_pose = Pose.from_matrix(T_curobo_target)

                print(
                    "[DEBUG CUROBO_TARGET_DYNAMIC]",
                    "isaac_current=", current_ee_pose,
                    "isaac_target=", target_ee_pose,
                    "curobo_current=", curobo_current_pose,
                    "curobo_target=", target_pose,
                    "joint_pos=", joint_pos.squeeze(0).detach().cpu().tolist(),
                    flush=True,
                )
            else:
                # Fallback for callers that do not provide Isaac's current EE pose.
                # Keep frame_bias only for backward compatibility; the X5A path should
                # pass current_ee_pose and use the dynamic branch above.
                target_pose = Pose(
                    np.array(target_ee_pose.p).copy(),
                    np.array(target_ee_pose.q).copy(),
                )
                if self.frame_bias is not None:
                    target_pose = target_pose.add_bias(
                        self.frame_bias,
                        coord='world',
                        clone=False,
                    )

                print(
                    "[DEBUG CUROBO_TARGET_STATIC_FALLBACK]",
                    "input_target_pose=", target_ee_pose,
                    "target_pose=", target_pose,
                    "curobo_current=", curobo_current_pose,
                    "joint_pos=", joint_pos.squeeze(0).detach().cpu().tolist(),
                    flush=True,
                )
        except Exception as e:
            print(
                "[DEBUG CUROBO_TARGET_DYNAMIC_ERR]",
                repr(e),
                flush=True,
            )
            # Conservative fallback: do not silently apply a stale frame_bias here.
            target_pose = Pose(
                np.array(target_ee_pose.p).copy(),
                np.array(target_ee_pose.q).copy(),
            )

        goal_pose_of_ee = CuroboPose.from_list(target_pose.tolist())

        start_joint_states = JointState(
            position=joint_pos,
            velocity=joint_vel,
            acceleration=torch.zeros_like(joint_pos),
            jerk=torch.zeros_like(joint_pos),
            joint_names=self.active_joints_name,
        )
        # plan
        if time_dilation_factor is None:
            time_dilation_factor = self.cfg.time_dilation_factor
        plan_config = MotionGenPlanConfig(max_attempts=10, time_dilation_factor=time_dilation_factor)

        pose_cost_metric = None
        if constraint_pose is not None:
            if pre_dis is not None:
                pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=self.motion_gen.tensor_args.to_device(constraint_pose),
                    offset_position=self.motion_gen.tensor_args.to_device([0.0, 0.0, pre_dis])
                )
            else:
                pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=self.motion_gen.tensor_args.to_device(constraint_pose)
                )
        elif pre_dis is not None and pre_dis != 0.0:
            pose_cost_metric = PoseCostMetric.create_grasp_approach_metric(
                offset_position=pre_dis, tstep_fraction=0.6, linear_axis=2)

        if pose_cost_metric is not None:
            plan_config.pose_cost_metric = pose_cost_metric

        return self.motion_gen.plan_single(
            start_joint_states, goal_pose_of_ee, plan_config)

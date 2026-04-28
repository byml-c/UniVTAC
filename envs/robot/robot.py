import yaml
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.utils import configclass

from ..utils.transforms import *
from ..utils.atom import GRASP_DIRECTION_DIC
from .robot_cfg import RobotCfg
from .curobo_planner import CuroboPlanner, CuroboPlannerCfg
from .._global import *

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from curobo.wrap.reacher.motion_gen import MotionGenResult
    from .._base_task import BaseTask

class RobotManager:
    def __init__(self, robot_cfg:RobotCfg, task:'BaseTask', planner_time_dilation_factor:float=1.0):
        self.cfg = robot_cfg
        self.task = task
        self.device = task.device
        self.sensor_type = getattr(task.cfg, 'tactile_sensor_type', 'xsensews')
        
        # 【修改点 1】: 根据你的任务配置获取 robot_type，默认设为 'x5a'
        self.robot_type = getattr(task.cfg, 'robot_type', 'x5a') 

        self.robot = Articulation(self.cfg.robot)
        self.task.scene.articulations['robot'] = self.robot
        self.planner_time_dilation_factor = planner_time_dilation_factor

        # 【修改点 2】: 默认设定为 X5A 的夹爪上限
        self.gripper_max_qpos = 0.044 
        self.last_arm_velocity = None
        self.last_gripper_velocity = None

        if self.robot_type == 'franka_panda':
            self.hand_name = 'panda_hand'
            self._arm_joint_names = [
                'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
                'panda_joint5', 'panda_joint6', 'panda_joint7'
            ]
            self._gripper_joint_names = [
                'panda_finger_joint1', 'panda_finger_joint2'
            ]
            self.gripper_max_qpos = self.cfg.gripper_max_qpos
            self.yaml_path = str(EMBODIMENTS_ROOT / 'franka' / 'curobo.yml')
            offset = self.cfg.gripper_offset
            
        elif self.robot_type == 'x5a':
            # 【修改点 3】: X5A 的末端手掌基座是 x5a_adapter_link
            self.hand_name = 'x5a_link6'
            # 【修改点 4】: X5A 是 6 轴机械臂
            self._arm_joint_names = [
                'x5a_joint1', 'x5a_joint2', 'x5a_joint3', 
                'x5a_joint4', 'x5a_joint5', 'x5a_joint6'
            ]
            # X5A + XenseWS current asset uses the two adapter mount joints
            # as prismatic gripper joints. Their URDF/USD names must match exactly.
            self._gripper_joint_names = [
                'x5a_adapter_left_mount',
                'x5a_adapter_right_mount',
            ]
            self.gripper_max_qpos = getattr(self.cfg, 'gripper_max_qpos', 0.02)
            # 【注意】: 确保你在 EMBODIMENTS_ROOT/ARX-X5/ 目录下有一个配置好的 curobo.yml
            self.yaml_path = str(EMBODIMENTS_ROOT / 'ARX-X5' / 'curobo.yml') 
            offset = self.cfg.gripper_offset
            
        else:
            raise NotImplementedError(f"Robot type {self.robot_type} not implemented.")
 
        # offset from end-effector to gripper center frame
        self._offset = Pose(p=[0, 0, -offset], q=[1, 0, 0, 0])
        self._offset_pos = torch.tensor([0.0, 0.0, offset], device=self.device).repeat(self.task.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.task.num_envs, 1)

    def setup(self):
        """设置机器人属性"""
        body_ids, body_names = self.robot.find_bodies(self.hand_name)
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        self._jacobi_body_idx = self._body_idx - 1

        joint_names = self.robot.joint_names
        self.joint_name_to_id = {name: i for i, name in enumerate(joint_names)}

        missing_arm_joints = [n for n in self._arm_joint_names if n not in self.joint_name_to_id]
        if len(missing_arm_joints) > 0:
            raise KeyError(f"Missing arm joints in articulation: {missing_arm_joints}. Available joints: {joint_names}")

        self._arm_ids = torch.tensor(
            [self.joint_name_to_id[n] for n in self._arm_joint_names],
            dtype=torch.long,
            device=self.device,
        )

        # Fixed-adapter robots, such as the current X5A + XenseWS asset, do not
        # have finger joints. Treat missing/empty gripper joints as a disabled
        # gripper instead of failing during setup.
        missing_gripper_joints = [n for n in self._gripper_joint_names if n not in self.joint_name_to_id]
        if len(missing_gripper_joints) > 0:
            print(f"[WARN] Missing gripper joints: {missing_gripper_joints}. Disable gripper control.")
            self._gripper_joint_names = []

        if len(self._gripper_joint_names) == 0:
            self._gripper_ids = torch.empty(0, dtype=torch.long, device=self.device)
        else:
            self._gripper_ids = torch.tensor(
                [self.joint_name_to_id[n] for n in self._gripper_joint_names],
                dtype=torch.long,
                device=self.device,
            )

        self.origin_pose = self.get_gripper_center_pose()
        self._all_ids = torch.cat([self._arm_ids, self._gripper_ids], dim=0)
 
        self.root_pose = Pose.from_list(self.robot.data.root_link_pos_w[0])
        planner_cfg = CuroboPlannerCfg(
            dt=self.task.cfg.sim.dt,
            all_joints_name=self.robot.joint_names,
            active_joints_name=self._arm_joint_names,
            robot_prime_path=self.cfg.robot.prim_path,
            yaml_path=self.yaml_path
        )
        self.planner = CuroboPlanner(
            task=self.task,
            cfg=planner_cfg,
            robot_origin_pose=self.root_pose,
        )
    
    def ee_to_gripper_center(self, ee_pose:Pose) -> Pose:
        """将夹爪中心位姿转换为末端执行器目标位姿"""
        return ee_pose.add_offset(self._offset.inv())

    def gripper_center_to_ee(self, gripper_center_pose:Pose) -> Pose:
        """将夹爪中心位姿转换为末端执行器目标位姿"""
        return gripper_center_pose.add_offset(self._offset)
    
    def get_gripper_center_pose(self, env_ids:slice=None) -> Pose:
        """获取当前夹爪中心位姿"""
        return self.ee_to_gripper_center(self.get_ee_pose())
    
    def get_inhand_pose(self, actor:'Actor') -> Pose:
        return actor.get_pose().rebase(self.get_gripper_center_pose())
    
    def get_ee_pose(self, env_ids:slice=None) -> Pose:
        """获取当前末端执行器目标位姿（target_pose）"""
        if env_ids is None:
            env_ids = [0]
        ee_pos_w = self.robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self.robot.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self.robot.data.root_link_pos_w
        root_quat_w = self.robot.data.root_link_quat_w
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        return Pose(ee_pose_b[0].cpu().numpy(), ee_quat_b[0].cpu().numpy())

    def get_qpos(self):
        return self.robot.data.joint_pos.clone().cpu()

    def _ids_on_tensor_device(self, ids: torch.Tensor, ref_tensor: torch.Tensor) -> torch.Tensor:
        """Move joint index tensors to the same device as the tensor being indexed.

        get_qpos() intentionally returns a CPU tensor in this project, while
        articulation tensors may live on CUDA. PyTorch requires index tensors to
        be on CPU or on the same device as the indexed tensor, so centralize the
        conversion here.
        """
        return ids.to(device=ref_tensor.device, dtype=torch.long)

    def get_gripper_qpos(self):
        if self._gripper_ids.numel() == 0:
            return 0.0

        qpos = self.get_qpos()
        gripper_ids = self._ids_on_tensor_device(self._gripper_ids, qpos)

        # Use the mean in case the gripper has two symmetric prismatic joints.
        return qpos[0, gripper_ids].mean().detach().cpu().item()

    def get_gripper_percentage(self):
        if self._gripper_ids.numel() == 0 or self.gripper_max_qpos == 0:
            return 0.0
        return self.get_gripper_qpos() / self.gripper_max_qpos

    def set_arm(self, pos:torch.Tensor, vel:torch.Tensor=None, env_ids:slice=None, force:bool=True):
        '''设置目标位姿'''
        self.robot.set_joint_position_target(pos, joint_ids=self._arm_ids, env_ids=env_ids)
        if vel is not None:
            self.robot.set_joint_velocity_target(vel, joint_ids=self._arm_ids, env_ids=env_ids)
        if force:
            self.robot.root_physx_view.set_dof_positions(
                self.robot._data.joint_pos_target,
                self.robot._ALL_INDICES
            )

    def _num_envs_for_target(self, env_ids=None) -> int:
        if env_ids is None:
            return self.task.num_envs
        if isinstance(env_ids, slice):
            return self.task.num_envs
        if torch.is_tensor(env_ids):
            return int(env_ids.numel())
        if isinstance(env_ids, (list, tuple)):
            return len(env_ids)
        return 1

    def _expand_gripper_target(self, value, env_ids=None) -> torch.Tensor:
        """Convert a scalar/single-joint target to [num_envs, num_gripper_joints]."""
        num_gripper_joints = int(self._gripper_ids.numel())
        num_envs = self._num_envs_for_target(env_ids)

        if not torch.is_tensor(value):
            value = torch.tensor(value, dtype=torch.float32, device=self.device)
        else:
            value = value.to(device=self.device, dtype=torch.float32)

        if value.ndim == 0:
            return value.repeat(num_envs, num_gripper_joints)

        if value.ndim == 1:
            if value.numel() == 1:
                return value.repeat(num_envs, num_gripper_joints)
            if value.numel() == num_gripper_joints:
                return value.unsqueeze(0).repeat(num_envs, 1)
            if value.numel() == num_envs:
                return value.view(num_envs, 1).repeat(1, num_gripper_joints)
            return value

        if value.ndim == 2:
            if value.shape[1] == 1 and num_gripper_joints > 1:
                return value.repeat(1, num_gripper_joints)
            return value

        raise ValueError(f"Unsupported gripper target shape: {tuple(value.shape)}")

    def set_gripper(self, pos:torch.Tensor, vel:torch.Tensor=None, env_ids:slice=None, force:bool=True):
        '''设置夹爪目标位置。X5A 使用左右 adapter prismatic joints 时会自动广播到两个关节。'''
        if self._gripper_ids.numel() == 0:
            return

        pos = self._expand_gripper_target(pos, env_ids=env_ids)
        self.robot.set_joint_position_target(pos, joint_ids=self._gripper_ids, env_ids=env_ids)

        if vel is not None:
            vel = self._expand_gripper_target(vel, env_ids=env_ids)
            self.robot.set_joint_velocity_target(vel, joint_ids=self._gripper_ids, env_ids=env_ids)

        if force:
            self.robot.root_physx_view.set_dof_positions(
                self.robot._data.joint_pos_target,
                self.robot._ALL_INDICES
            )

    def plan_arm(self, target_pose:Pose, constraint_pose=None, pre_dis=None, time_dilation_factor=None):
        result:MotionGenResult = self.planner.plan_path(
            # Pass the full articulation joint vector. CuroboPlanner selects
            # active joints by name, so this works for Franka and X5A.
            curr_joint_pos=self.robot.data.joint_pos[0],
            curr_joint_vel=self.robot.data.joint_vel[0],
            target_ee_pose=target_pose,
            real_robot_pose=self.root_pose,
            pre_dis=pre_dis,
            constraint_pose=constraint_pose,
            time_dilation_factor=time_dilation_factor
        )
        
        if result.success.item():
            return {
                'status': 'Success',
                'num_steps': result.interpolated_plan.position.shape[0],
                'position': result.interpolated_plan.position.detach(),
                'velocity': result.interpolated_plan.velocity.detach()
            }
        else:
            return {'status': 'Fail', 'num_steps': 0, 'position': None, 'velocity': None}

    def gripper_percent2qpos(self, percentage:float):
        if self._gripper_ids.numel() == 0 or self.gripper_max_qpos == 0:
            return 0.0
        gripper_range = [0, self.gripper_max_qpos]
        target_pos = gripper_range[0] + (gripper_range[1] - gripper_range[0]) * percentage
        return target_pos

    def plan_gripper(self, pos:float, type:Literal['percent', 'qpos'] = 'percent'):
        # If an asset has no actuated gripper joints, return a zero-step
        # successful plan so higher-level task code can continue.
        if self._gripper_ids.numel() == 0:
            return {
                'status': 'Success',
                'num_steps': 0,
                'position': torch.empty(0, device=self.device),
                'velocity': torch.empty(0, device=self.device),
            }

        if type == 'percent':
            target_pos = self.gripper_percent2qpos(pos)
        else:
            target_pos = pos

        joint_pos = self.robot.data.joint_pos
        gripper_ids = self._ids_on_tensor_device(self._gripper_ids, joint_pos)
        gripper_pos = joint_pos[0, gripper_ids][0]

        start_pos = float(gripper_pos.detach().cpu().item())
        target_pos = float(target_pos)

        num_steps = np.ceil(abs(target_pos - start_pos) / 0.0005).astype(int)
        num_steps = max(int(num_steps), 1)

        position = torch.linspace(start_pos, target_pos, num_steps, device=self.device)
        velocity = torch.clip((position - start_pos) / self.task.cfg.sim.dt, -0.0001, 0.0001)

        return {
            'status': 'Success',
            'num_steps': num_steps,
            'position': position.detach(),
            'velocity': velocity.detach()
        }

    def _reset_idx(self, env_ids: torch.Tensor | None=None):
        """重置环境"""
        if not hasattr(self, 'origin_pose'):
            self._setup_robot_properties()
        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(joint_pos)
        
        self.planner.reset()
        self.robot.set_joint_position_target(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
    
    def get_observations(self, data_type:list[str]=['joint', 'ee']) -> dict:
        obs = {}
        if 'ee' in data_type:
            obs['ee'] = self.get_ee_pose().totensor(device=self.device)
        if 'joint' in data_type:
            obs['joint'] = self.robot.data.joint_pos.squeeze(0)
        return obs
    
    def get_grasp_perfect_direction(self):
        return 'top_down'

from ._base_task import *
import numpy as np


# Tuned X5A parameters. Franka follows the original pull_out_key logic.
X5A_PRE_GRASP_HIGH_PRE_DIS = 0.070
X5A_PRE_GRASP_LOW_PRE_DIS = 0.050
X5A_PRE_GRASP_APPROACH_TIME_DILATION = 0.60
X5A_PRE_GRASP_ALIGN_TIME_DILATION = 0.45
X5A_PRE_GRASP_ALIGN_DELAY_STEPS = 2
X5A_POST_YAW_SETTLE_STEPS = 3
X5A_PRE_GRASP_DESCEND_TIME_DILATION = 0.35
X5A_PRE_CLOSE_SETTLE_STEPS = 8

X5A_CLOSE_DEPTH_THRESHOLD = 28.85
X5A_CLOSE_TIME_DILATION = 1.00
X5A_CLOSE_HOLD_STEPS = 8

X5A_GRASP_YAW_OFFSET = np.pi / 2
X5A_KEY_GRASP_X_BIAS = 0.0005
X5A_KEY_GRASP_Y_BIAS = -0.0015
X5A_KEY_GRASP_Z_BIAS_LOW = -0.0035
X5A_KEY_GRASP_Z_BIAS_HIGH = -0.0025

X5A_PRE_CLOSE_RECENTER_TARGET_GC_X = 0.0035
X5A_PRE_CLOSE_RECENTER_TARGET_GC_Y = -0.0025
X5A_PRE_CLOSE_RECENTER_MAX_STEP_HIGH = 0.004
X5A_PRE_CLOSE_RECENTER_MAX_STEP_LOW = 0.002
X5A_PRE_CLOSE_RECENTER_TIME_DILATION = 0.45
X5A_PRE_CLOSE_RECENTER_SETTLE_STEPS = 4

X5A_ROTATE_FINAL_OFFSET = 0.04
X5A_ROTATE_TIME_DILATION = 0.85
X5A_ROTATE_SETTLE_STEPS = 2
X5A_ROTATE_RECLOSE_HOLD_STEPS = 2

X5A_WORLD_PULL_Z = 0.026
X5A_WORLD_PULL_TIME_DILATION = 1.00
X5A_FINAL_DELAY_STEPS = 6


@configclass
class TaskCfg(BaseTaskCfg):
    pass


class Task(BaseTask):
    def __init__(self, cfg: TaskCfg, mode: Literal['collect', 'eval'] = 'collect', render_mode: str | None = None, **kwargs):
        cfg.sim.physics_material.dynamic_friction = 2.0
        cfg.sim.physics_material.static_friction = 2.0
        cfg.uipc_sim.contact.default_friction_ratio = 2.0
        super().__init__(cfg, mode, render_mode, **kwargs)

    def _is_x5a(self):
        return getattr(self._robot_manager, "robot_type", None) == "x5a"

    def _x5a_grasp_axes(self, pose):
        mat = pose.to_transformation_matrix()
        key_x = np.asarray(mat[:3, 0], dtype=float)
        key_y = np.asarray(mat[:3, 1], dtype=float)
        key_x = key_x / (np.linalg.norm(key_x) + 1e-8)
        key_y = key_y / (np.linalg.norm(key_y) + 1e-8)

        cos_yaw = np.cos(X5A_GRASP_YAW_OFFSET)
        sin_yaw = np.sin(X5A_GRASP_YAW_OFFSET)
        grasp_x = cos_yaw * key_x + sin_yaw * key_y
        grasp_y = -sin_yaw * key_x + cos_yaw * key_y
        grasp_x = grasp_x / (np.linalg.norm(grasp_x) + 1e-8)
        grasp_y = grasp_y / (np.linalg.norm(grasp_y) + 1e-8)
        return key_x, grasp_x, grasp_y

    def _x5a_key_yaw_adjust(self, pose):
        key_x, _, _ = self._x5a_grasp_axes(pose)
        raw_yaw = np.arctan2(key_x[1], key_x[0]) + X5A_GRASP_YAW_OFFSET
        yaw = (raw_yaw + np.pi / 2) % np.pi - np.pi / 2
        return float(np.clip(yaw, -np.pi / 2, np.pi / 2))

    def _x5a_key_grasp_target_pose(self, key_pose):
        _, grasp_x, grasp_y = self._x5a_grasp_axes(key_pose)
        z_bias = self.rng.uniform(X5A_KEY_GRASP_Z_BIAS_LOW, X5A_KEY_GRASP_Z_BIAS_HIGH)
        self.metadata['grasp_z_bias'] = float(z_bias)

        p = (
            np.asarray(key_pose.p, dtype=float)
            + X5A_KEY_GRASP_X_BIAS * grasp_x
            + X5A_KEY_GRASP_Y_BIAS * grasp_y
            + np.asarray([0.0, 0.0, z_bias], dtype=float)
        )
        return Pose(p, key_pose.q)

    def _x5a_pre_close_recenter(self, target_pose, phase: str):
        key_pose = self.key.get_pose()
        gripper_center_pose = self._robot_manager.get_gripper_center_pose()

        key_p = np.asarray(key_pose.p, dtype=float)
        gc_p = np.asarray(gripper_center_pose.p, dtype=float)
        _, grasp_x, grasp_y = self._x5a_grasp_axes(key_pose)

        max_step = (
            X5A_PRE_CLOSE_RECENTER_MAX_STEP_HIGH
            if phase == 'high'
            else X5A_PRE_CLOSE_RECENTER_MAX_STEP_LOW
        )

        gc_delta = gc_p - key_p
        err_x = X5A_PRE_CLOSE_RECENTER_TARGET_GC_X - float(np.dot(gc_delta, grasp_x))
        err_y = X5A_PRE_CLOSE_RECENTER_TARGET_GC_Y - float(np.dot(gc_delta, grasp_y))

        delta = err_x * grasp_x + err_y * grasp_y
        delta = np.asarray(delta, dtype=float)
        delta[2] = 0.0

        norm = float(np.linalg.norm(delta[:2]))
        if norm < 1e-4:
            return target_pose

        delta *= min(1.0, float(max_step) / (norm + 1e-9))
        self.move(
            self.atom.move_by_displacement(
                x=float(delta[0]),
                y=float(delta[1]),
                z=0.0,
            ),
            time_dilation_factor=X5A_PRE_CLOSE_RECENTER_TIME_DILATION,
            constraint_pose=[1, 1, 1, 0, 0, 0],
            delay=False,
        )
        self.delay(X5A_PRE_CLOSE_RECENTER_SETTLE_STEPS, is_save=False)
        return Pose(np.asarray(target_pose.p, dtype=float) + delta, target_pose.q)

    def create_actors(self) -> None:
        base_x = 0.38 if self._is_x5a() else 0.5
        base_pose = Pose([base_x, 0.0, 0.002], [1, 0, 0, 0])
        key_pose = base_pose.add_bias([-0.0025, 0, 0.0785])

        self.slot = self._actor_manager.add_from_usd_file(
            name='slot',
            asset_path="KeySlot.usd",
            pose=base_pose,
            density=1e5,
        )
        self.key = self._actor_manager.add_from_usd_file(
            name='key',
            asset_path="Key.usd",
            pose=key_pose,
            density=10,
        )

    def _reset_actors(self):
        if self._is_x5a():
            random_pose = self.create_noise([0.003, 0.004, 0.0])
            base_x = 0.38
        else:
            random_pose = self.create_noise([0.005, 0.005, 0.0])
            base_x = 0.5

        random_rotate = self.rng.uniform(-np.pi / 4, np.pi / 4)
        base_pose = Pose([base_x, 0.0, 0.002], [1, 0, 0, 0]).add_offset(random_pose)
        key_pose = base_pose.add_bias([-0.0025, 0, 0.0785])

        base_pose = base_pose.add_rotation([0, 0, random_rotate])
        self.key_rotation = self.rng.uniform(-np.pi / 2, -np.pi / 4)
        key_pose = key_pose.add_rotation([0, 0, random_rotate + self.key_rotation])

        self.slot.set_pose(base_pose)
        self.key.set_pose(key_pose)

    def pre_move(self):
        self.delay(10)

        if self._is_x5a():
            target_pose = self._x5a_key_grasp_target_pose(self.key.get_pose())
            cpose = construct_grasp_pose(
                target_pose.p,
                [0, 0, 1],
                [1, 0, 0],
            )
            pre_dis = X5A_PRE_GRASP_HIGH_PRE_DIS
        else:
            z_bias = self.rng.uniform(-0.015, -0.01)
            self.metadata['grasp_z_bias'] = float(z_bias)
            target_pose = self.key.get_pose().add_bias([0, 0, z_bias])
            target_mat = target_pose.to_transformation_matrix()
            cpose = construct_grasp_pose(
                target_pose.p,
                target_mat[:3, 2],
                target_mat[:3, 0],
            )
            pre_dis = 0.08

        self.cid = self.key.register_point(cpose, type='contact')

        if self._is_x5a():
            self.move(
                self.atom.grasp_actor(
                    self.key,
                    contact_point_id=self.cid,
                    pre_dis=pre_dis,
                    dis=0.0,
                    is_close=False,
                ),
                time_dilation_factor=X5A_PRE_GRASP_APPROACH_TIME_DILATION,
            )

            yaw_adjust = self._x5a_key_yaw_adjust(self.key.get_pose())
            if abs(yaw_adjust) > 1e-3:
                self.move(
                    self.atom.move_by_displacement(
                        rpy=[0, 0, yaw_adjust],
                        xyz_coord='local',
                    ),
                    time_dilation_factor=X5A_PRE_GRASP_ALIGN_TIME_DILATION,
                    constraint_pose=[0, 0, 0, 0, 0, 1],
                    delay=False,
                )
                self.delay(X5A_PRE_GRASP_ALIGN_DELAY_STEPS, is_save=False)

            self.delay(X5A_POST_YAW_SETTLE_STEPS, is_save=False)
            target_pose = self._x5a_pre_close_recenter(target_pose, phase='high')

            self.move(
                self.atom.move_by_displacement(
                    z=-(X5A_PRE_GRASP_HIGH_PRE_DIS - X5A_PRE_GRASP_LOW_PRE_DIS),
                ),
                time_dilation_factor=X5A_PRE_GRASP_DESCEND_TIME_DILATION,
                constraint_pose=[1, 1, 1, 0, 0, 0],
                delay=False,
            )
            self.delay(X5A_PRE_CLOSE_SETTLE_STEPS, is_save=False)
            target_pose = self._x5a_pre_close_recenter(target_pose, phase='low')

            self.move(
                self.atom.close_gripper(),
                gripper_depth_threshold=X5A_CLOSE_DEPTH_THRESHOLD,
                time_dilation_factor=X5A_CLOSE_TIME_DILATION,
            )
            self.delay(X5A_CLOSE_HOLD_STEPS, is_save=False)
        else:
            self.move(self.atom.grasp_actor(
                self.key,
                contact_point_id=self.cid,
                pre_dis=pre_dis,
                dis=0.0,
            ))

        self.target_pose = self.key.get_pose()
        self.target_pose[3:] = self.slot.get_pose().add_rotation([0, 0, 0.1])[3:]
        self.slot_init_pose = self.slot.get_pose()

    def _play_once(self):
        if self._is_x5a():
            rotate_rz = -self.key_rotation + X5A_ROTATE_FINAL_OFFSET
            self.metadata['first_rotate_rz'] = float(rotate_rz)
            self.metadata['second_rotate_rz'] = 0.0

            self.move(
                self.atom.move_by_displacement(
                    rpy=[0, 0, rotate_rz],
                    xyz_coord='local',
                ),
                time_dilation_factor=X5A_ROTATE_TIME_DILATION,
                constraint_pose=[0, 0, 0, 1, 1, 1],
                delay=False,
            )
            self.delay(X5A_ROTATE_SETTLE_STEPS, is_save=False)

            self.move(
                self.atom.close_gripper(),
                gripper_depth_threshold=X5A_CLOSE_DEPTH_THRESHOLD,
                time_dilation_factor=X5A_CLOSE_TIME_DILATION,
            )
            self.delay(X5A_ROTATE_RECLOSE_HOLD_STEPS, is_save=False)

            self.move(
                self.atom.move_by_displacement(z=X5A_WORLD_PULL_Z),
                time_dilation_factor=X5A_WORLD_PULL_TIME_DILATION,
            )
            self.delay(X5A_FINAL_DELAY_STEPS)
        else:
            over_rotate = self.rng.uniform(0.09, 0.16)  # 5 deg ~ 9 deg
            self.metadata['over_rotate'] = over_rotate

            self.move(
                self.atom.move_by_displacement(
                    rpy=[0, 0, -self.key_rotation + over_rotate],
                    xyz_coord='local',
                ),
                time_dilation_factor=0.5,
                constraint_pose=[0, 0, 0, 1, 1, 1],
                delay=False,
            )
            self.move(
                self.atom.move_by_displacement(
                    rpy=[0, 0, -over_rotate + 0.05],
                    xyz_coord='local',
                ),
                constraint_pose=[0, 0, 0, 1, 1, 1],
                delay=False,
            )
            self.move(
                self.atom.move_by_displacement(z=-0.03, xyz_coord='local'),
                time_dilation_factor=0.2,
            )
            self.delay(20)

    def check_early_stop(self):
        if self._is_x5a():
            z_dis = np.abs(self._robot_manager.get_gripper_center_pose()[2] - self.key.get_pose()[2])
            z_dis_threshold = 0.055
        else:
            z_dis = np.abs(self._robot_manager.get_ee_pose()[2] - self.key.get_pose()[2])
            z_dis_threshold = 0.14

        slot_rel_pose = self.slot.get_pose().rebase(self.slot_init_pose)
        slot_x_rotate = np.dot(
            slot_rel_pose.to_transformation_matrix()[:3, 0],
            np.array([1, 0, 0]),
        )

        if z_dis >= z_dis_threshold:
            self.metadata['early_stop'] = True
            self.metadata['z_dis'] = float(z_dis)
            return True
        if slot_x_rotate < 0.99:
            self.metadata['early_stop'] = True
            self.metadata['slot_x_rotate'] = float(slot_x_rotate)
            return True
        return False

    def check_success(self):
        target_height = 0.088 if self._is_x5a() else 0.09
        key_pose = self.key.get_pose()
        slot_rel_pose = self.slot.get_pose().rebase(self.slot_init_pose)
        slot_x_rotate = np.dot(
            slot_rel_pose.to_transformation_matrix()[:3, 0],
            np.array([1, 0, 0]),
        )

        if self._is_x5a():
            z_dis = np.abs(self._robot_manager.get_gripper_center_pose()[2] - self.key.get_pose()[2])
            z_dis_threshold = 0.055
        else:
            z_dis = np.abs(self._robot_manager.get_ee_pose()[2] - self.key.get_pose()[2])
            z_dis_threshold = 0.14

        return (
            key_pose.p[2] > target_height
            and np.dot(key_pose.to_transformation_matrix()[:3, 2], np.array([0, 0, 1])) > 0.965
            and slot_x_rotate > 0.99
            and z_dis < z_dis_threshold
        )

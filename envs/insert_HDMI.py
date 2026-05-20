from ._base_task import *
import numpy as np


# X5A HDMI task adaptation. Franka keeps the original task behavior.
X5A_HDMI_SLOT_X = 0.42
X5A_HDMI_PRISM_X = 0.28
X5A_HDMI_USD_SCALE = 1.15

X5A_HDMI_SLOT_ASSET_PATH = "HDMISlot_X5A.usd"
X5A_HDMI_PRISM_ASSET_PATH = "HDMI_X5A.usd"
FRANKA_HDMI_SLOT_ASSET_PATH = "HDMISlot.usd"
FRANKA_HDMI_PRISM_ASSET_PATH = "HDMI.usd"

X5A_HDMI_RESET_NOISE = [0.003, 0.003, 0.0]
X5A_HDMI_PRE_PLACE_NOISE = [0.003, 0.003, 0.0]

X5A_HDMI_GRASP_X_BIAS = -0.003 * X5A_HDMI_USD_SCALE
X5A_HDMI_GRASP_Y_BIAS = -0.001 * X5A_HDMI_USD_SCALE
X5A_HDMI_GRASP_Z_BIAS = 0.0155 * X5A_HDMI_USD_SCALE
X5A_HDMI_GRASP_PITCH_RANGE = [-np.pi / 36, np.pi / 36]

X5A_HDMI_TARGET_Z_BIAS = 0.005 * X5A_HDMI_USD_SCALE
X5A_HDMI_HOLE_Z_BIAS = 0.0128 * X5A_HDMI_USD_SCALE
X5A_HDMI_PLACE_X_BIAS = 0.0

X5A_HDMI_GRASP_TIME_DILATION = 0.60
X5A_HDMI_CLOSE_DEPTH_THRESHOLD = 28.0
X5A_HDMI_CLOSE_REPEAT = 2
X5A_HDMI_AFTER_CLOSE_HOLD_STEPS = 10
X5A_HDMI_LIFT_TIME_DILATION = 0.35
X5A_HDMI_AFTER_LIFT_HOLD_STEPS = 4
X5A_HDMI_RECLOSE_AFTER_LIFT = True
X5A_HDMI_RECLOSE_HOLD_STEPS = 4
X5A_HDMI_PRE_PLACE_TIME_DILATION = 0.30
X5A_HDMI_AFTER_PRE_PLACE_HOLD_STEPS = 4
X5A_HDMI_FINAL_PLACE_TIME_DILATION = 0.45
X5A_HDMI_INSERT_WORLD_Z_STEP_1 = -0.006
X5A_HDMI_INSERT_WORLD_Z_STEP_2 = -0.003
X5A_HDMI_INSERT_TIME_DILATION = 0.35
X5A_HDMI_INSERT_SETTLE_STEPS = 20

X5A_HDMI_SUCCESS_Y_THRESHOLD = 0.005 * X5A_HDMI_USD_SCALE
X5A_HDMI_SUCCESS_Z_THRESHOLD = 0.005 * X5A_HDMI_USD_SCALE
X5A_HDMI_SUCCESS_EE_Z_MIN = 0.145


@configclass
class TaskCfg(BaseTaskCfg):
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(0.74, 0.0, 0.066), rot=(0.512, 0.512, 0.487, 0.487), convention="opengl"),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.5, focus_distance=1.0, horizontal_aperture=3.6, clipping_range=(0.1, 100.0)
            ),
            width=480,
            height=270,
            update_period=1 / 120,
        ),
        CameraCfg(
            name="wrist",
            prim_path="/World/envs/env_.*/Robot/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None,
            width=480,
            height=270,
            update_period=1 / 120,
        ),
    ]
    step_lim = 600


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode: Literal["collect", "eval"] = "collect", render_mode: str | None = None, **kwargs):
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        super().__init__(cfg, mode, render_mode, **kwargs)

    def _is_x5a(self):
        return getattr(getattr(self, "_robot_manager", None), "robot_type", None) == "x5a"

    def create_actors(self):
        if self._is_x5a():
            base_pose = Pose([X5A_HDMI_SLOT_X, 0.0, 0.002], [1, 0, 0, 0])
            prism_pose = Pose([X5A_HDMI_PRISM_X, 0.0, 0.002], [1, 0, 0, 0])
            slot_asset_path = X5A_HDMI_SLOT_ASSET_PATH
            prism_asset_path = X5A_HDMI_PRISM_ASSET_PATH
        else:
            base_pose = Pose([0.55, 0.0, 0.002], [1, 0, 0, 0])
            prism_pose = Pose([0.4, 0.0, 0.002], [1, 0, 0, 0])
            slot_asset_path = FRANKA_HDMI_SLOT_ASSET_PATH
            prism_asset_path = FRANKA_HDMI_PRISM_ASSET_PATH

        self.slot = self._actor_manager.add_from_usd_file(
            name="slot",
            asset_path=slot_asset_path,
            pose=base_pose,
            density=1e5,
        )

        self.prism = self._actor_manager.add_from_usd_file(
            name="prism",
            asset_path=prism_asset_path,
            pose=prism_pose,
        )

    def _reset_actors(self):
        if self._is_x5a():
            base_offset = self.create_noise(list(X5A_HDMI_RESET_NOISE))
            base_x = X5A_HDMI_SLOT_X
        else:
            base_offset = self.create_noise([0.005, 0.005, 0.0])
            base_x = 0.55

        base_pose = Pose([base_x, 0.0, self.slot.get_pose()[2]], [1, 0, 0, 0]).add_offset(base_offset)
        self.slot.set_pose(base_pose)

    def _get_grasp_target_pose(self):
        if self._is_x5a():
            grasp_rotate = self.rng.uniform(
                X5A_HDMI_GRASP_PITCH_RANGE[0],
                X5A_HDMI_GRASP_PITCH_RANGE[1],
            )
            return self.prism.get_pose().add_bias([
                X5A_HDMI_GRASP_X_BIAS,
                X5A_HDMI_GRASP_Y_BIAS,
                X5A_HDMI_GRASP_Z_BIAS,
            ]).add_rotation([0, grasp_rotate, 0])

        grasp_rotate = self.rng.uniform(-np.pi / 18, np.pi / 18)
        return self.prism.get_pose().add_bias([0, 0, 0.012]).add_rotation([0, grasp_rotate, 0])

    def _move_to_grasp_pose(self, contact_point_id):
        grasp_action = self.atom.grasp_actor(
            self.prism,
            contact_point_id=contact_point_id,
            is_close=False,
        )

        if self._is_x5a():
            self.move(grasp_action, time_dilation_factor=X5A_HDMI_GRASP_TIME_DILATION)
        else:
            self.move(grasp_action)

    def _close_and_lift(self):
        if self._is_x5a():
            for _ in range(X5A_HDMI_CLOSE_REPEAT):
                self.move(
                    self.atom.close_gripper(),
                    gripper_depth_threshold=X5A_HDMI_CLOSE_DEPTH_THRESHOLD,
                )
                self.delay(X5A_HDMI_AFTER_CLOSE_HOLD_STEPS, is_save=False)

            self.move(
                self.atom.move_by_displacement(z=0.02),
                time_dilation_factor=X5A_HDMI_LIFT_TIME_DILATION,
                constraint_pose=[1, 1, 1, 0, 0, 0],
            )
            self.delay(X5A_HDMI_AFTER_LIFT_HOLD_STEPS, is_save=False)

            if X5A_HDMI_RECLOSE_AFTER_LIFT:
                self.move(
                    self.atom.close_gripper(),
                    gripper_depth_threshold=X5A_HDMI_CLOSE_DEPTH_THRESHOLD,
                )
                self.delay(X5A_HDMI_RECLOSE_HOLD_STEPS, is_save=False)
        else:
            self.move(self.atom.close_gripper())
            self.move(self.atom.move_by_displacement(z=0.02))

    def _prepare_place_targets(self):
        if self._is_x5a():
            self.target_pose = self.slot.get_pose().add_bias([
                X5A_HDMI_PLACE_X_BIAS,
                0.0,
                X5A_HDMI_TARGET_Z_BIAS,
            ])
            self.hole_pose = self.slot.get_pose().add_bias([
                X5A_HDMI_PLACE_X_BIAS,
                0.0,
                X5A_HDMI_HOLE_Z_BIAS,
            ])
            noise = self.create_noise(list(X5A_HDMI_PRE_PLACE_NOISE))
        else:
            self.target_pose = self.slot.get_pose().add_bias([0.0, 0.0, 0.005])
            self.hole_pose = self.slot.get_pose().add_bias([0.0, 0.0, 0.0128])
            noise = self.create_noise([0.005, 0.005, 0.0])

        self.noise_pose = self.hole_pose.add_offset(noise)

    def _move_to_pre_place(self):
        place_action = self.atom.place_actor(
            self.prism,
            target_pose=self.noise_pose,
            pre_dis=0.02,
            dis=0.01,
            is_open=False,
        )

        if self._is_x5a():
            self.move(place_action, time_dilation_factor=X5A_HDMI_PRE_PLACE_TIME_DILATION)
            self.delay(X5A_HDMI_AFTER_PRE_PLACE_HOLD_STEPS, is_save=False)
        else:
            self.move(place_action)

    def pre_move(self):
        self.delay(10)
        self.move(self.atom.open_gripper(0.5))

        target_pose = self._get_grasp_target_pose()
        target_mat = target_pose.to_transformation_matrix()
        contact_pose = construct_grasp_pose(
            target_pose.p,
            target_mat[:3, 2],
            target_mat[:3, 0],
        )
        contact_point_id = self.prism.register_point(contact_pose, type="contact")

        self._move_to_grasp_pose(contact_point_id)
        self._close_and_lift()
        self._prepare_place_targets()
        self._move_to_pre_place()

    def _insert_x5a(self):
        for z_step in (X5A_HDMI_INSERT_WORLD_Z_STEP_1, X5A_HDMI_INSERT_WORLD_Z_STEP_2):
            self.move(
                self.atom.move_by_displacement(
                    z=z_step,
                    xyz_coord="world",
                ),
                time_dilation_factor=X5A_HDMI_INSERT_TIME_DILATION,
                constraint_pose=[1, 1, 1, 0, 0, 0],
            )
        self.delay(X5A_HDMI_INSERT_SETTLE_STEPS, is_save=True)

    def _insert_franka(self):
        self.move(
            self.atom.move_by_displacement(z=0.005, xyz_coord="local"),
            time_dilation_factor=0.5,
            constraint_pose=[1, 1, 1, 1, 1, 0],
        )
        self.move(
            self.atom.move_by_displacement(z=0.002, xyz_coord="local"),
            time_dilation_factor=0.5,
            constraint_pose=[1, 1, 1, 1, 1, 0],
        )
        self.delay(20, is_save=True)

    def _play_once(self):
        self.move(
            self.atom.place_actor(
                self.prism,
                target_pose=self.hole_pose,
                pre_dis=0.01,
                dis=0.002,
                is_open=False,
            ),
            time_dilation_factor=X5A_HDMI_FINAL_PLACE_TIME_DILATION if self._is_x5a() else 0.5,
        )

        if self._is_x5a():
            self._insert_x5a()
        else:
            self._insert_franka()

    def check_success(self, z_threshold=0.005):
        success_ref_pose = self.hole_pose if self._is_x5a() else self.target_pose
        prism_pose = self.prism.get_pose().rebase(success_ref_pose)
        ee_pose = self._robot_manager.get_ee_pose()

        if self._is_x5a():
            y_threshold = X5A_HDMI_SUCCESS_Y_THRESHOLD
            z_threshold = X5A_HDMI_SUCCESS_Z_THRESHOLD
            ee_z_min = X5A_HDMI_SUCCESS_EE_Z_MIN
        else:
            y_threshold = 0.005
            ee_z_min = 0.145

        self.metadata["rel_pose"] = prism_pose.tolist()
        self.metadata["success_ref_pose"] = "hole_pose" if self._is_x5a() else "target_pose"
        self.metadata["success_y_threshold"] = float(y_threshold)
        self.metadata["success_z_threshold"] = float(z_threshold)

        return (
            np.all(np.abs(prism_pose.p[1:2]) < np.array([y_threshold]))
            and abs(prism_pose.p[2]) < z_threshold
            and ee_pose[2] > ee_z_min
            and np.dot(prism_pose.to_transformation_matrix()[:3, 2], np.array([0, 0, 1])) > 0.965
        )

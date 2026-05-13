from ._base_task import *
import numpy as np
import copy

TASK_VERSION_TAG = "RUNNING_INSERT_TUBE_X5A_STAGE2_HOLE_AXIS"

# X5A-friendly scene scaling.
# Original insert_tube:
#   prism/base x = 0.40
#   slot x       = 0.60
#
# This version keeps both tube start and slot/hole x positions proportional
# to SCENE_X_SCALE:
#   prism/base x = 0.40 * SCENE_X_SCALE = 0.36
#   slot x       = 0.60 * SCENE_X_SCALE = 0.54
#
# The slot/hole ratio is intentionally preserved here.
SCENE_X_SCALE = 0.7
PRISM_X = 0.40 * SCENE_X_SCALE
SLOT_X = 0.60 * SCENE_X_SCALE

# Keep original insert_tube slot noise by default.
# For no-noise bring-up, set this to [0.0, 0.0, 0.0].
SLOT_NOISE_RANGE = [0.005, 0.010, 0.0]

# Try-pose noise for the next robustness step.
# Previous small-noise baseline used 0.5-1.5 mm and reached 11/13.
# This version increases only the upper bound to 2.0 mm.
TRY_NOISE_ENABLED = True
TRY_NOISE_RANGE = [[0.0010, 0.0040], [0.0010, 0.0040], 0]

# Keep SINGLE lift behavior after grasp.
# This is intentionally NOT split into multiple lift motions.
LIFT_AFTER_GRASP_Z = 0.04
LIFT_CONSTRAINT_POSE = [1, 1, 1, 0, 0, 0]
FINE_PLACE_CONSTRAINT_POSE = [1, 1, 1, 0, 0, 0]


DEBUG_STOP_AFTER_PRE_PLACE = False
GRASP_BIAS_RANGE = [0.090, 0.0915]

GRASP_X_OFFSET = 0.0
PRE_PLACE_X_OFFSET = -0.0085

SETTLE_DOWN_Z = -0.003
SETTLE_READY_XY = 0.0048
SETTLE_READY_AXIS_DOT = 0.999

SETTLE_RESCUE_XY = 0.0085
SETTLE_RESCUE_Y = 0.0055
SETTLE_RESCUE_AXIS_DOT = 0.9978
SETTLE_RESCUE_MAX_POSITIVE_Z = 0.002

# First forward is only a light probe. X5A showed that dis=0.01 can
# increase lateral x error before settle_down, so keep this smaller.
FIRST_FORWARD_DIS = 0.006
FIRST_FORWARD_DELTA_D = 0.003

# Gate before first_forward. If pre-place is already bad, do not push the tube
# further; this separates "pre-place already failed" from "first_forward made it worse".
PRE_FIRST_FORWARD_MAX_XY = 0.014
PRE_FIRST_FORWARD_MIN_AXIS_DOT = 0.990
PRE_FIRST_FORWARD_MAX_POSITIVE_Z = 0.006

FINAL_INSERT_STAGE1_Z = -0.026
FINAL_INSERT_STAGE2_Z = -0.011
FINAL_INSERT_TARGET_Z = -0.031
FINAL_INSERT_STAGE2_MAX_XY = 0.0060
FINAL_INSERT_STAGE2_MIN_AXIS_DOT = 0.995
FINAL_INSERT_TIME_DILATION = 0.3
FINAL_INSERT_DELAY_STEPS = 8

PLACE_TIME_DILATION = 0.3
COARSE_PRE_DIS = 0.08
COARSE_DIS = 0.04
FINE_PRE_DIS = 0.04
FINE_DIS = 0.002

# Optional grasp stability debug. Default 0 keeps task flow compact.
GRASP_HOLD_STEPS = 0
DEBUG_STOP_AFTER_GRASP_HOLD = False

# This version is intentionally strict so logs immediately reveal wrong files.
STRICT_PARAM_VERIFY = True


@configclass
class TaskCfg(BaseTaskCfg):
    pass


class Task(BaseTask):
    def __init__(
        self,
        cfg: TaskCfg,
        mode: Literal['collect', 'eval'] = 'collect',
        render_mode: str | None = None,
        **kwargs
    ):
        print(
            "[VERIFY_TASK_FILE_LOADED]",
            TASK_VERSION_TAG,
            "SCENE_X_SCALE=", SCENE_X_SCALE,
            "PRISM_X=", PRISM_X,
            "SLOT_X=", SLOT_X,
            "LIFT_AFTER_GRASP_Z=", LIFT_AFTER_GRASP_Z,
            "PRE_PLACE_X_OFFSET=", PRE_PLACE_X_OFFSET,
            "SETTLE_DOWN_Z=", SETTLE_DOWN_Z,
            "SETTLE_READY_XY=", SETTLE_READY_XY,
            "SETTLE_READY_AXIS_DOT=", SETTLE_READY_AXIS_DOT,
            "SETTLE_RESCUE_XY=", SETTLE_RESCUE_XY,
            "SETTLE_RESCUE_Y=", SETTLE_RESCUE_Y,
            "SETTLE_RESCUE_AXIS_DOT=", SETTLE_RESCUE_AXIS_DOT,
            "SETTLE_RESCUE_MAX_POSITIVE_Z=", SETTLE_RESCUE_MAX_POSITIVE_Z,
            "FIRST_FORWARD_DIS=", FIRST_FORWARD_DIS,
            "FIRST_FORWARD_DELTA_D=", FIRST_FORWARD_DELTA_D,
            "PRE_FIRST_FORWARD_MAX_XY=", PRE_FIRST_FORWARD_MAX_XY,
            "PRE_FIRST_FORWARD_MIN_AXIS_DOT=", PRE_FIRST_FORWARD_MIN_AXIS_DOT,
            "PRE_FIRST_FORWARD_MAX_POSITIVE_Z=", PRE_FIRST_FORWARD_MAX_POSITIVE_Z,
            "FINAL_INSERT_STAGE1_Z=", FINAL_INSERT_STAGE1_Z,
            "FINAL_INSERT_STAGE2_Z=", FINAL_INSERT_STAGE2_Z,
            "FINAL_INSERT_TARGET_Z=", FINAL_INSERT_TARGET_Z,
            "FINAL_INSERT_STAGE2_MAX_XY=", FINAL_INSERT_STAGE2_MAX_XY,
            "FINAL_INSERT_STAGE2_MIN_AXIS_DOT=", FINAL_INSERT_STAGE2_MIN_AXIS_DOT,
            "FINAL_INSERT_TIME_DILATION=", FINAL_INSERT_TIME_DILATION,
            flush=True,
        )

        if STRICT_PARAM_VERIFY:
            assert abs(SCENE_X_SCALE - 0.7) < 1e-9, SCENE_X_SCALE
            assert abs(PRISM_X - 0.28) < 1e-9, PRISM_X
            assert abs(SLOT_X - 0.60 * SCENE_X_SCALE) < 1e-9, SLOT_X
            assert abs(LIFT_AFTER_GRASP_Z - 0.04) < 1e-9, LIFT_AFTER_GRASP_Z
            assert DEBUG_STOP_AFTER_PRE_PLACE is False, DEBUG_STOP_AFTER_PRE_PLACE
            assert abs(PLACE_TIME_DILATION - 0.3) < 1e-9, PLACE_TIME_DILATION
            assert abs(COARSE_PRE_DIS - 0.08) < 1e-9, COARSE_PRE_DIS
            assert abs(COARSE_DIS - 0.04) < 1e-9, COARSE_DIS
            assert abs(FINE_PRE_DIS - 0.04) < 1e-9, FINE_PRE_DIS
            assert abs(FINE_DIS - 0.002) < 1e-9, FINE_DIS
            assert abs(GRASP_X_OFFSET - 0.0) < 1e-9, GRASP_X_OFFSET
            assert abs(PRE_PLACE_X_OFFSET - (-0.0085)) < 1e-9, PRE_PLACE_X_OFFSET
            assert abs(SETTLE_DOWN_Z - (-0.003)) < 1e-9, SETTLE_DOWN_Z
            assert TRY_NOISE_ENABLED is True, TRY_NOISE_ENABLED
            assert TRY_NOISE_RANGE == [[0.0010, 0.0040], [0.0010, 0.0040], 0], TRY_NOISE_RANGE
            assert abs(SETTLE_READY_XY - 0.0048) < 1e-9, SETTLE_READY_XY
            assert abs(SETTLE_READY_AXIS_DOT - 0.999) < 1e-9, SETTLE_READY_AXIS_DOT
            assert abs(SETTLE_RESCUE_XY - 0.0085) < 1e-9, SETTLE_RESCUE_XY
            assert abs(SETTLE_RESCUE_Y - 0.0055) < 1e-9, SETTLE_RESCUE_Y
            assert abs(SETTLE_RESCUE_AXIS_DOT - 0.9978) < 1e-9, SETTLE_RESCUE_AXIS_DOT
            assert abs(SETTLE_RESCUE_MAX_POSITIVE_Z - 0.002) < 1e-9, SETTLE_RESCUE_MAX_POSITIVE_Z
            assert abs(FIRST_FORWARD_DIS - 0.006) < 1e-9, FIRST_FORWARD_DIS
            assert abs(FIRST_FORWARD_DELTA_D - 0.003) < 1e-9, FIRST_FORWARD_DELTA_D
            assert abs(PRE_FIRST_FORWARD_MAX_XY - 0.014) < 1e-9, PRE_FIRST_FORWARD_MAX_XY
            assert abs(PRE_FIRST_FORWARD_MIN_AXIS_DOT - 0.990) < 1e-9, PRE_FIRST_FORWARD_MIN_AXIS_DOT
            assert abs(PRE_FIRST_FORWARD_MAX_POSITIVE_Z - 0.006) < 1e-9, PRE_FIRST_FORWARD_MAX_POSITIVE_Z
            assert abs(FINAL_INSERT_STAGE1_Z - (-0.026)) < 1e-9, FINAL_INSERT_STAGE1_Z
            assert abs(FINAL_INSERT_STAGE2_Z - (-0.011)) < 1e-9, FINAL_INSERT_STAGE2_Z
            assert abs(FINAL_INSERT_TARGET_Z - (-0.031)) < 1e-9, FINAL_INSERT_TARGET_Z
            assert abs(FINAL_INSERT_STAGE2_MAX_XY - 0.0060) < 1e-9, FINAL_INSERT_STAGE2_MAX_XY
            assert abs(FINAL_INSERT_STAGE2_MIN_AXIS_DOT - 0.995) < 1e-9, FINAL_INSERT_STAGE2_MIN_AXIS_DOT

        super().__init__(cfg=cfg, mode=mode, render_mode=render_mode, **kwargs)

    def create_actors(self):
        print(
            "[VERIFY_CREATE_ACTORS]",
            TASK_VERSION_TAG,
            "slot_x=", SLOT_X,
            "prism_x=", PRISM_X,
            flush=True,
        )

        slot_pose = Pose([SLOT_X, 0.0, 0.002], [1, 0, 0, 0])
        base_pose = Pose([PRISM_X, 0.0, 0.002], [1, 0, 0, 0])
        prism_pose = Pose([PRISM_X, 0.0, 0.005], [1, 0, 0, 0])

        self.slot = self._actor_manager.add_from_usd_file(
            name='slot',
            asset_path="TestTubeSlot.usd",
            pose=slot_pose,
            density=1e5
        )
        self.prism_base = self._actor_manager.add_from_usd_file(
            name='prism_base',
            asset_path="TestTubeBase.usd",
            pose=base_pose,
            density=1e5
        )
        self.prism = self._actor_manager.add_from_usd_file(
            name='prism',
            asset_path="TestTube.usd",
            pose=prism_pose,
            density=10
        )

    def _reset_actors(self):
        print(
            "[VERIFY_RESET_ACTORS]",
            TASK_VERSION_TAG,
            "slot_base_x=", SLOT_X,
            "slot_noise_range=", SLOT_NOISE_RANGE,
            flush=True,
        )

        slot_offset = self.create_noise(SLOT_NOISE_RANGE.copy())
        slot_pose = Pose(
            [SLOT_X, 0.0, self.slot.get_pose()[2]],
            [1, 0, 0, 0]
        ).add_offset(slot_offset)
        self.slot.set_pose(slot_pose)

        self.metadata["task_version_tag"] = TASK_VERSION_TAG
        self.metadata["scene_x_scale"] = SCENE_X_SCALE
        self.metadata["prism_x"] = PRISM_X
        self.metadata["slot_x"] = SLOT_X
        self.metadata["lift_after_grasp_z"] = LIFT_AFTER_GRASP_Z
        self.metadata["slot_noise_range"] = list(SLOT_NOISE_RANGE)

    def _print_inhand(self, tag):
        prism_inhand_pose = self.prism.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose()
        )

        if hasattr(self, "origin_inhand_pose"):
            delta_xyz = np.abs(
                np.array(prism_inhand_pose[:3]) -
                np.array(self.origin_inhand_pose[:3])
            )
        else:
            delta_xyz = None

        print(
            "[DEBUG INHAND]",
            "version=", TASK_VERSION_TAG,
            "tag=", tag,
            "prism_pose=", self.prism.get_pose(),
            "gripper_center_pose=", self._robot_manager.get_gripper_center_pose(),
            "inhand_pose=", prism_inhand_pose,
            "origin_inhand_pose=", getattr(self, "origin_inhand_pose", None),
            "delta_xyz=", delta_xyz,
            flush=True,
        )

        return prism_inhand_pose

    def _print_grasp_world_delta(self, tag):
        prism_p = np.array(self.prism.get_pose().p)
        gc_p = np.array(self._robot_manager.get_gripper_center_pose().p)
        print(
            "[DEBUG GRASP_WORLD_DELTA]",
            "version=", TASK_VERSION_TAG,
            "tag=", tag,
            "gc_minus_prism=", gc_p - prism_p,
            "prism_pose=", self.prism.get_pose(),
            "gripper_center_pose=", self._robot_manager.get_gripper_center_pose(),
            flush=True,
        )

    def pre_move(self):
        print(
            "[VERIFY_PRE_MOVE_ENTER]",
            TASK_VERSION_TAG,
            "SCENE_X_SCALE=", SCENE_X_SCALE,
            "PRISM_X=", PRISM_X,
            "SLOT_X=", SLOT_X,
            "SLOT_NOISE_RANGE=", SLOT_NOISE_RANGE,
            "TRY_NOISE_ENABLED=", TRY_NOISE_ENABLED,
            "TRY_NOISE_RANGE=", TRY_NOISE_RANGE,
            "LIFT_AFTER_GRASP_Z=", LIFT_AFTER_GRASP_Z,
            "LIFT_CONSTRAINT_POSE=", LIFT_CONSTRAINT_POSE,
            "FINE_PLACE_CONSTRAINT_POSE=", FINE_PLACE_CONSTRAINT_POSE,
            "DEBUG_STOP_AFTER_PRE_PLACE=", DEBUG_STOP_AFTER_PRE_PLACE,
            "GRASP_BIAS_RANGE=", GRASP_BIAS_RANGE,
            "GRASP_X_OFFSET=", GRASP_X_OFFSET,
            "PRE_PLACE_X_OFFSET=", PRE_PLACE_X_OFFSET,
            "SETTLE_DOWN_Z=", SETTLE_DOWN_Z,
            "SETTLE_READY_XY=", SETTLE_READY_XY,
            "SETTLE_READY_AXIS_DOT=", SETTLE_READY_AXIS_DOT,
            "SETTLE_RESCUE_XY=", SETTLE_RESCUE_XY,
            "SETTLE_RESCUE_Y=", SETTLE_RESCUE_Y,
            "SETTLE_RESCUE_AXIS_DOT=", SETTLE_RESCUE_AXIS_DOT,
            "SETTLE_RESCUE_MAX_POSITIVE_Z=", SETTLE_RESCUE_MAX_POSITIVE_Z,
            "FIRST_FORWARD_DIS=", FIRST_FORWARD_DIS,
            "FIRST_FORWARD_DELTA_D=", FIRST_FORWARD_DELTA_D,
            "PRE_FIRST_FORWARD_MAX_XY=", PRE_FIRST_FORWARD_MAX_XY,
            "PRE_FIRST_FORWARD_MIN_AXIS_DOT=", PRE_FIRST_FORWARD_MIN_AXIS_DOT,
            "PRE_FIRST_FORWARD_MAX_POSITIVE_Z=", PRE_FIRST_FORWARD_MAX_POSITIVE_Z,
            "FINAL_INSERT_STAGE1_Z=", FINAL_INSERT_STAGE1_Z,
            "FINAL_INSERT_STAGE2_Z=", FINAL_INSERT_STAGE2_Z,
            "FINAL_INSERT_TARGET_Z=", FINAL_INSERT_TARGET_Z,
            "FINAL_INSERT_STAGE2_MAX_XY=", FINAL_INSERT_STAGE2_MAX_XY,
            "FINAL_INSERT_STAGE2_MIN_AXIS_DOT=", FINAL_INSERT_STAGE2_MIN_AXIS_DOT,
            "FINAL_INSERT_TIME_DILATION=", FINAL_INSERT_TIME_DILATION,
            "PLACE_TIME_DILATION=", PLACE_TIME_DILATION,
            "COARSE_PRE_DIS=", COARSE_PRE_DIS,
            "COARSE_DIS=", COARSE_DIS,
            "FINE_PRE_DIS=", FINE_PRE_DIS,
            "FINE_DIS=", FINE_DIS,
            "GRASP_HOLD_STEPS=", GRASP_HOLD_STEPS,
            "DEBUG_STOP_AFTER_GRASP_HOLD=", DEBUG_STOP_AFTER_GRASP_HOLD,
            flush=True,
        )

        self.delay(10)

        grasp_bias = self.rng.uniform(GRASP_BIAS_RANGE[0], GRASP_BIAS_RANGE[1])
        target_pose = self.prism.get_pose().add_bias([GRASP_X_OFFSET, 0, grasp_bias])

        cpose = construct_grasp_pose(
            target_pose.p,
            [0, 0, 1],
            [1, 0, 0]
        )

        print(
            "[DEBUG PRE_GRASP_POSE]",
            "version=", TASK_VERSION_TAG,
            "grasp_bias=", grasp_bias,
            "grasp_x_offset=", GRASP_X_OFFSET,
            "prism_pose=", self.prism.get_pose(),
            "target_pose=", target_pose,
            "cpose=", cpose,
            flush=True,
        )

        self.cid = self.prism.register_point(cpose, type='contact')

        # Before the composite grasp_actor action starts.
        # grasp_actor includes both arm motion and gripper close, so the exact
        # "arm reached target but before close" moment is inside Atom/BaseTask.
        self._print_grasp_world_delta("before_grasp_actor")

        ok = self.move(
            self.atom.grasp_actor(
                self.prism,
                contact_point_id=self.cid,
                pre_dis=0.0,
                dis=0.0
            ),
            tag="pre_grasp_prism_x5a_verify"
        )
        if not ok:
            print("[DEBUG PRE_MOVE_STOP] failed at pre_grasp_prism_x5a_verify", flush=True)
            return

        self.origin_inhand_pose = self.prism.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose()
        )
        self.metadata["grasp_origin_inhand_pose"] = self.origin_inhand_pose.tolist()
        self._print_inhand("after_grasp")
        self._print_grasp_world_delta("after_grasp")

        if GRASP_HOLD_STEPS > 0:
            print(
                "[DEBUG HOLD_TEST]",
                "version=", TASK_VERSION_TAG,
                "after grasp, hold still steps=",
                GRASP_HOLD_STEPS,
                flush=True,
            )
            self.delay(GRASP_HOLD_STEPS, is_save=False)
            self._print_inhand("after_grasp_hold")
            self._print_grasp_world_delta("after_grasp_hold")

        if DEBUG_STOP_AFTER_GRASP_HOLD:
            print(
                "[DEBUG STOP_AFTER_GRASP_HOLD] skip lift/place for grasp stability test",
                flush=True,
            )
            return

        base_pose = self.slot.get_pose()

        if TRY_NOISE_ENABLED:
            self.random_noise = self.create_noise(copy.deepcopy(TRY_NOISE_RANGE))
            self.random_noise[:2] *= np.sign(self.rng.uniform(-1, 1, size=2))
        else:
            self.random_noise = Pose([0.0, 0.0, 0.0], [1, 0, 0, 0])

        self.metadata['random_noise'] = self.random_noise.tolist()

        # Keep original insert_tube hole definition.
        self.hole_pose = base_pose.add_bias([-0.008, 0, 0.077]).add_rotation([0, -np.pi / 6, 0])
        try_pose = self.hole_pose.add_offset(self.random_noise)

        # Use a slightly x-compensated target only for pre-place approach.
        # Keep self.hole_pose unchanged so insertion checks remain against the real hole.
        pre_place_pose = try_pose.add_bias([PRE_PLACE_X_OFFSET, 0, 0])

        print(
            "[DEBUG TUBE_TARGET]",
            "version=", TASK_VERSION_TAG,
            "slot_pose=", self.slot.get_pose(),
            "hole_pose=", self.hole_pose,
            "try_pose=", try_pose,
            "pre_place_pose=", pre_place_pose,
            "pre_place_x_offset=", PRE_PLACE_X_OFFSET,
            "random_noise=", self.random_noise,
            flush=True,
        )

        # Single lift only. This should produce target_z = current_z + 0.08.
        lift_start_ee_pose = self._robot_manager.get_ee_pose()
        lift_start_gripper_center_pose = self._robot_manager.get_gripper_center_pose()
        lift_start_prism_pose = self.prism.get_pose()
        expected_target_z = float(lift_start_gripper_center_pose.p[2] + LIFT_AFTER_GRASP_Z)

        print(
            "[VERIFY_BEFORE_SINGLE_LIFT]",
            TASK_VERSION_TAG,
            "lift_z=", LIFT_AFTER_GRASP_Z,
            "lift_constraint_pose=", LIFT_CONSTRAINT_POSE,
            "start_ee_pose=", lift_start_ee_pose,
            "start_gripper_center_pose=", lift_start_gripper_center_pose,
            "start_prism_pose=", lift_start_prism_pose,
            "expected_gripper_center_target_z=", expected_target_z,
            flush=True,
        )

        ok = self.move(
            self.atom.move_by_displacement(z=LIFT_AFTER_GRASP_Z),
            tag=f"lift_after_grasp_single_z{int(LIFT_AFTER_GRASP_Z * 1000):03d}_scale{int(SCENE_X_SCALE * 100):03d}",
            constraint_pose=LIFT_CONSTRAINT_POSE,
        )
        if not ok:
            print(
                "[DEBUG PRE_MOVE_STOP]",
                "version=", TASK_VERSION_TAG,
                "failed_at=lift_after_grasp_single",
                "LIFT_AFTER_GRASP_Z=", LIFT_AFTER_GRASP_Z,
                "start_ee_pose=", lift_start_ee_pose,
                "start_gripper_center_pose=", lift_start_gripper_center_pose,
                "start_prism_pose=", lift_start_prism_pose,
                "current_ee_pose=", self._robot_manager.get_ee_pose(),
                "current_gripper_center_pose=", self._robot_manager.get_gripper_center_pose(),
                "current_prism_pose=", self.prism.get_pose(),
                flush=True,
            )
            return

        lift_end_ee_pose = self._robot_manager.get_ee_pose()
        lift_end_gripper_center_pose = self._robot_manager.get_gripper_center_pose()
        lift_end_prism_pose = self.prism.get_pose()

        print(
            "[VERIFY_AFTER_SINGLE_LIFT]",
            TASK_VERSION_TAG,
            "requested_lift_z=", LIFT_AFTER_GRASP_Z,
            "ee_delta=", np.array(lift_end_ee_pose.p) - np.array(lift_start_ee_pose.p),
            "gripper_center_delta=", np.array(lift_end_gripper_center_pose.p) - np.array(lift_start_gripper_center_pose.p),
            "prism_delta=", np.array(lift_end_prism_pose.p) - np.array(lift_start_prism_pose.p),
            "end_ee_pose=", lift_end_ee_pose,
            "end_gripper_center_pose=", lift_end_gripper_center_pose,
            "end_prism_pose=", lift_end_prism_pose,
            flush=True,
        )

        self._print_inhand("after_lift_after_grasp_single")
        self._print_grasp_world_delta("after_lift_after_grasp_single")

        # Keep original insert_tube place distances.
        ok = self.move(
            self.atom.place_actor(
                self.prism,
                target_pose=pre_place_pose,
                pre_dis=COARSE_PRE_DIS,
                dis=COARSE_DIS,
                is_open=False
            ),
            tag="pre_place_coarse_x5a_verify",
            time_dilation_factor=PLACE_TIME_DILATION
        )
        if not ok:
            print("[DEBUG PRE_MOVE_STOP] failed at pre_place_coarse_x5a_verify", flush=True)
            return

        self._print_inhand("after_pre_place_coarse")
        self._print_grasp_world_delta("after_pre_place_coarse")

        ok = self.move(
            self.atom.place_actor(
                self.prism,
                target_pose=pre_place_pose,
                pre_dis=FINE_PRE_DIS,
                dis=FINE_DIS,
                is_open=False
            ),
            tag="pre_place_fine_x5a_verify",
            constraint_pose=FINE_PLACE_CONSTRAINT_POSE,
            time_dilation_factor=PLACE_TIME_DILATION
        )
        if not ok:
            print("[DEBUG PRE_MOVE_STOP] failed at pre_place_fine_x5a_verify", flush=True)
            return

        # Reset in-hand reference after reaching the insertion start pose.
        # This makes early_stop measure insertion-phase slip, not transfer-phase slip.
        self.origin_inhand_pose = self.prism.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose()
        )
        self.metadata["place_origin_inhand_pose"] = self.origin_inhand_pose.tolist()
        self._print_inhand("after_pre_place")
        self._print_grasp_world_delta("after_pre_place")
        print(
            "[DEBUG PRE_PLACE_READY]",
            "version=", TASK_VERSION_TAG,
            "debug_stop_after_pre_place=", DEBUG_STOP_AFTER_PRE_PLACE,
            "prism_x=", PRISM_X,
            "slot_x=", SLOT_X,
            "hole_x=", self.hole_pose.p[0],
            "pre_place_x=", pre_place_pose.p[0],
            "pre_place_x_offset=", PRE_PLACE_X_OFFSET,
            "fine_dis=", FINE_DIS,
            flush=True,
        )

    def _play_once(self):
        if DEBUG_STOP_AFTER_PRE_PLACE:
            print(
                "[DEBUG STOP_AFTER_PRE_PLACE] skip play_once / insertion phase for no-penetration stability test",
                flush=True,
            )
            self.metadata['debug_stop_after_pre_place'] = True
            return

        def get_insert_state(tag):
            rel_pose = self.prism.get_pose().rebase(self.hole_pose)
            xy_err = np.abs(rel_pose.p[:2])
            z_err = rel_pose.p[2]
            axis_dot = np.dot(
                rel_pose.to_transformation_matrix()[:3, 2],
                np.array([0, 0, 1])
            )
            print(
                tag,
                "rel_p=", rel_pose.p,
                "xy_err=", xy_err,
                "z_err=", z_err,
                "axis_dot=", axis_dot,
                flush=True,
            )
            return rel_pose, xy_err, z_err, axis_dot

        def is_aligned_enough(xy_err, axis_dot):
            return np.all(xy_err < np.array([0.006, 0.006])) and axis_dot > 0.99

        # 1. Before any forward insertion, check whether pre-place is already usable.
        # If it is already too far from the hole, first_forward will only expose/amplify
        # the bad pre-place state. Stop here so the log points to pre-place, not final insert.
        rel_pose, xy_err, z_err, axis_dot = get_insert_state("[DEBUG BEFORE_FIRST_FORWARD]")
        if (
            np.any(xy_err > np.array([PRE_FIRST_FORWARD_MAX_XY, PRE_FIRST_FORWARD_MAX_XY]))
            or axis_dot < PRE_FIRST_FORWARD_MIN_AXIS_DOT
            or z_err > PRE_FIRST_FORWARD_MAX_POSITIVE_Z
        ):
            print(
                "[DEBUG PLAY_ONCE_STOP] bad pre-place before first_forward",
                "z_err=", z_err,
                "xy_err=", xy_err,
                "axis_dot=", axis_dot,
                "max_xy=", PRE_FIRST_FORWARD_MAX_XY,
                "min_axis_dot=", PRE_FIRST_FORWARD_MIN_AXIS_DOT,
                "max_positive_z=", PRE_FIRST_FORWARD_MAX_POSITIVE_Z,
                flush=True,
            )
            self._print_inhand("bad_pre_place_before_first_forward")
            self._print_grasp_world_delta("bad_pre_place_before_first_forward")
            return

        # 2. First conservative forward. This should not trigger a realign
        # just because insertion depth is still shallow.
        print(
            "[DEBUG FIRST_FORWARD_CONFIG]",
            "dis=", FIRST_FORWARD_DIS,
            "delta_d=", FIRST_FORWARD_DELTA_D,
            flush=True,
        )
        self.try_forward(
            self.prism,
            dis=FIRST_FORWARD_DIS,
            delta_d=FIRST_FORWARD_DELTA_D,
        )
        rel_pose, xy_err, z_err, axis_dot = get_insert_state("[DEBUG AFTER_FIRST_FORWARD]")

        aligned_enough = is_aligned_enough(xy_err, axis_dot)
        deep_enough = z_err < -0.02

        strict_settle_ready = (
            aligned_enough
            and not deep_enough
            and np.all(xy_err < np.array([SETTLE_READY_XY, SETTLE_READY_XY]))
            and axis_dot > SETTLE_READY_AXIS_DOT
        )
        rescue_settle_ready = (
            not deep_enough
            and xy_err[0] < SETTLE_RESCUE_XY
            and xy_err[1] < SETTLE_RESCUE_Y
            and axis_dot > SETTLE_RESCUE_AXIS_DOT
            and z_err < SETTLE_RESCUE_MAX_POSITIVE_Z
        )

        # 2. If first_forward is shallow and still well-oriented, do a small
        # settle-down move. The rescue band allows x-boundary cases around
        # 5-6.5 mm to try settle_down instead of stopping immediately.
        if strict_settle_ready or rescue_settle_ready:
            if strict_settle_ready:
                print(
                    "[DEBUG INSERT_STAGE] very aligned but shallow, do settle_down instead of stage_forward",
                    "z_err=", z_err,
                    "xy_err=", xy_err,
                    "axis_dot=", axis_dot,
                    "settle_down_z=", SETTLE_DOWN_Z,
                    flush=True,
                )
            else:
                print(
                    "[DEBUG INSERT_STAGE] rescue settle_down after first_forward",
                    "z_err=", z_err,
                    "xy_err=", xy_err,
                    "axis_dot=", axis_dot,
                    "settle_down_z=", SETTLE_DOWN_Z,
                    "rescue_xy=", SETTLE_RESCUE_XY,
                    "rescue_y=", SETTLE_RESCUE_Y,
                    "rescue_axis_dot=", SETTLE_RESCUE_AXIS_DOT,
                    flush=True,
                )

            ok = self.move(
                self.atom.move_by_displacement(
                    z=SETTLE_DOWN_Z,
                    xyz_coord=self.prism.get_pose()
                ),
                tag="settle_down_x5a_verify",
                time_dilation_factor=PLACE_TIME_DILATION,
                delay=False,
            )
            if not ok:
                print("[DEBUG PLAY_ONCE_STOP] failed at settle_down_x5a_verify", flush=True)
                return

            rel_pose, xy_err, z_err, axis_dot = get_insert_state("[DEBUG AFTER_SETTLE_DOWN]")

            print(
                "[DEBUG BEFORE_FINAL_GATE]",
                "z_err=", z_err,
                "xy_err=", xy_err,
                "axis_dot=", axis_dot,
                flush=True,
            )

        # 3. If xy / axis are still bad and outside the rescue band after the
        # first forward, stop. The previous mid_realign recovery was observed
        # to amplify a small rim-contact error on X5A.
        elif not aligned_enough:
            print(
                "[DEBUG PLAY_ONCE_STOP] first forward outside rescue band, skip mid realign",
                "xy_err=", xy_err,
                "z_err=", z_err,
                "axis_dot=", axis_dot,
                "rescue_xy=", SETTLE_RESCUE_XY,
                "rescue_y=", SETTLE_RESCUE_Y,
                "rescue_axis_dot=", SETTLE_RESCUE_AXIS_DOT,
                flush=True,
            )
            return

        elif aligned_enough and not deep_enough:
            print(
                "[DEBUG PLAY_ONCE_STOP] aligned but not precise enough for settle_down; skip stage_forward",
                "z_err=", z_err,
                "xy_err=", xy_err,
                "axis_dot=", axis_dot,
                "settle_ready_xy=", SETTLE_READY_XY,
                "settle_ready_axis_dot=", SETTLE_READY_AXIS_DOT,
                "rescue_xy=", SETTLE_RESCUE_XY,
                flush=True,
            )
            return

        else:
            print(
                "[DEBUG INSERT_STAGE] first forward already deep enough",
                "z_err=", z_err,
                "xy_err=", xy_err,
                "axis_dot=", axis_dot,
                flush=True,
            )

        # 4. Do not hard final-insert unless the tube is already partially
        # seated and still aligned. This prevents pushing down while the tube
        # is only touching the rim / not actually inside the hole.
        if z_err > -0.004 or xy_err[0] > 0.0045 or axis_dot < 0.995:
            print(
                "[DEBUG PLAY_ONCE_STOP] not deep/aligned enough for final_insert_down",
                "z_err=", z_err,
                "xy_err=", xy_err,
                "axis_dot=", axis_dot,
                flush=True,
            )
            return

        print(
            "[DEBUG FINAL_INSERT_STAGE1_CONFIG]",
            "z=", FINAL_INSERT_STAGE1_Z,
            "target_z=", FINAL_INSERT_TARGET_Z,
            "time_dilation=", FINAL_INSERT_TIME_DILATION,
            flush=True,
        )
        ok = self.move(
            self.atom.move_by_displacement(
                z=FINAL_INSERT_STAGE1_Z,
                xyz_coord=self.prism.get_pose()
            ),
            tag="final_insert_stage1_x5a_verify",
            time_dilation_factor=FINAL_INSERT_TIME_DILATION,
            delay=False,
        )
        if not ok:
            print("[DEBUG PLAY_ONCE_STOP] failed at final_insert_stage1_x5a_verify", flush=True)
            return
        self.delay(FINAL_INSERT_DELAY_STEPS, is_save=False)

        rel_pose, xy_err, z_err, axis_dot = get_insert_state("[DEBUG AFTER_FINAL_INSERT_STAGE1]")

        if z_err <= FINAL_INSERT_TARGET_Z:
            print(
                "[DEBUG FINAL_INSERT_STAGE1_DONE] deep enough, skip stage2",
                "z_err=", z_err,
                "xy_err=", xy_err,
                "axis_dot=", axis_dot,
                flush=True,
            )
            return

        if (
            np.any(xy_err > np.array([FINAL_INSERT_STAGE2_MAX_XY, FINAL_INSERT_STAGE2_MAX_XY]))
            or axis_dot < FINAL_INSERT_STAGE2_MIN_AXIS_DOT
        ):
            print(
                "[DEBUG PLAY_ONCE_STOP] final stage1 not safe for stage2",
                "z_err=", z_err,
                "xy_err=", xy_err,
                "axis_dot=", axis_dot,
                "stage2_max_xy=", FINAL_INSERT_STAGE2_MAX_XY,
                "stage2_min_axis_dot=", FINAL_INSERT_STAGE2_MIN_AXIS_DOT,
                flush=True,
            )
            return

        print(
            "[DEBUG FINAL_INSERT_STAGE2_CONFIG]",
            "z=", FINAL_INSERT_STAGE2_Z,
            "target_z=", FINAL_INSERT_TARGET_Z,
            "time_dilation=", FINAL_INSERT_TIME_DILATION,
            flush=True,
        )
        ok = self.move(
            self.atom.move_by_displacement(
                z=FINAL_INSERT_STAGE2_Z,
                xyz_coord=self.prism.get_pose()
            ),
            tag="final_insert_stage2_x5a_verify",
            time_dilation_factor=FINAL_INSERT_TIME_DILATION,
            delay=False,
        )
        if not ok:
            print("[DEBUG PLAY_ONCE_STOP] failed at final_insert_stage2_x5a_verify", flush=True)
            return
        self.delay(FINAL_INSERT_DELAY_STEPS, is_save=False)
        get_insert_state("[DEBUG AFTER_FINAL_INSERT_STAGE2]")

    def check_mid_success(self):
            prism_pose = self.prism.get_pose().rebase(self.hole_pose)
            xy_err = np.abs(prism_pose.p[:2])
            z_err = prism_pose.p[2]
            axis_dot = np.dot(
                prism_pose.to_transformation_matrix()[:3, 2],
                np.array([0, 0, 1])
            )

            print(
                "[DEBUG CHECK_MID_SUCCESS]",
                "rel_p=", prism_pose.p,
                "xy_err=", xy_err,
                "z_err=", z_err,
                "axis_dot=", axis_dot,
                flush=True,
            )

            return (
                np.all(xy_err < np.array([0.005, 0.005]))
                and z_err < -0.02
                and axis_dot > 0.99
            )
    

    def check_early_stop(self):
        prism_inhand_pose = self.prism.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose()
        )
        inhand_bias = np.abs(self.origin_inhand_pose[2] - prism_inhand_pose[2])
        if inhand_bias > 0.03:
            self.metadata['early_stop'] = True
            self.metadata['inhand_bias'] = float(inhand_bias)
            return True

    def check_success(self, z_threshold=0.03):
        prism_pose = self.prism.get_pose().rebase(self.hole_pose)
        prism_inhand_pose = self.prism.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose()
        )
        xy_err = np.abs(prism_pose.p[:2])
        z_err = prism_pose.p[2]
        axis_dot = np.dot(prism_pose.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))
        inhand_bias = np.abs(self.origin_inhand_pose[2] - prism_inhand_pose[2])

        self.metadata['rel_pose'] = prism_pose.tolist()
        self.metadata['inhand_bias'] = inhand_bias

        success = (
            np.all(xy_err < np.array([0.005, 0.005]))
            and z_err < -z_threshold
            and axis_dot > 0.965
            and inhand_bias < 0.03
        )

        print(
            "[DEBUG CHECK_SUCCESS]",
            "success=", success,
            "rel_p=", prism_pose.p,
            "xy_err=", xy_err,
            "z_err=", z_err,
            "axis_dot=", axis_dot,
            "inhand_bias=", inhand_bias,
            "z_threshold=", z_threshold,
            flush=True,
        )

        return success

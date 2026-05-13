from ._base_task import *
import numpy as np
import copy

TASK_VERSION_TAG = "RUNNING_INSERT_HOLE_X5A_ZERO_PERTURB_V4_P30"

SCENE_X_SCALE = 0.8
PRISM_X = 0.40 * SCENE_X_SCALE
HOLE_X = 0.60 * SCENE_X_SCALE

# Zero-perturbation diagnostic baseline. Use this before adding slot/try/grasp noise.
# FORCE_ROTATE=np.pi means target_pose uses +30 degrees around y.
SLOT_NOISE_RANGE = [0.0, 0.0, 0.0]
ROTATE_RANDOM_ENABLED = False
ROTATE_CHOICES = [0, np.pi]
FORCE_ROTATE = np.pi

TRY_NOISE_ENABLED = False
TRY_NOISE_RANGE = [[0.0, 0.0], [0.0, 0.0], 0.0]

GRASP_BIAS_RANGE = [0.091, 0.091]
GRASP_X_OFFSET = 0.0
# The v2 run slipped along the tube during insertion. Close a little deeper
# than the default 24.9 mm tactile threshold so the tube is held before pushing.
GRASP_DEPTH_THRESHOLD = 24.75
LIFT_AFTER_GRASP_Z = 0.04
LIFT_CONSTRAINT_POSE = [1, 1, 1, 0, 0, 0]

HOLE_BIAS = [0.0, 0.0, 0.1]
# X5A place_actor tends to leave the tube about +1 cm in target-frame x
# for this scene. This mirrors the insert_tube compensation idea.
PRE_PLACE_X_OFFSET = -0.008
PRE_PLACE_Z_OFFSET = 0.0

PLACE_TIME_DILATION = 0.3
COARSE_PRE_DIS = 0.05
COARSE_DIS = 0.012
FINE_PRE_DIS = 0.012
FINE_DIS = 0.002
FINE_PLACE_CONSTRAINT_POSE = [1, 1, 1, 0, 0, 0]

PRE_FIRST_DOWN_MAX_XY = 0.014
# Before first_down the tube is intentionally not yet aligned to the final tilted target.
# Seed 3 reached axis_dot=0.896856 here; the next correction gate is 0.88,
# so this pre-gate should not reject a pose that the correction phase can handle.
PRE_FIRST_DOWN_MIN_AXIS_DOT = 0.89
PRE_FIRST_DOWN_MAX_POSITIVE_Z = 0.010

FIRST_DOWN_Z = -0.018
FIRST_DOWN_TIME_DILATION = 0.25

# Seed 6 showed the large one-shot tilt correction (-14.7 mm x) is where the
# gripper starts sliding along the tube. Do correction as a damped, capped
# motion and guard the in-hand pose before going into final insertion.
TILT_CORRECTION_GAIN = 0.55
TILT_CORRECTION_MAX_X = 0.008
TILT_CORRECTION_MAX_Z = 0.0012
POST_FIRST_DOWN_MAX_INHAND_BIAS = 0.018
POST_TILT_MAX_INHAND_BIAS = 0.025

CORRECT_MAX_XY = 0.014
# This gate is before tilt correction, so it must be loose enough to allow correction.
CORRECT_MIN_AXIS_DOT = 0.88
CORRECT_TIME_DILATION = 0.4

FINAL_INSERT_STAGE1_Z = -0.030
FINAL_INSERT_STAGE2_Z = -0.008
FINAL_INSERT_TARGET_Z = -0.043
FINAL_INSERT_STAGE2_MAX_XY = 0.008
FINAL_INSERT_STAGE2_MIN_AXIS_DOT = 0.985
FINAL_INSERT_TIME_DILATION = 0.35
FINAL_INSERT_DELAY_STEPS = 8

# Keep success thresholds in one place so insertion-phase early exits and
# check_success() use exactly the same gates. The failing seed-2 run reached
# z/axis/inhand gates but stopped with x just outside the 1 cm success gate.
SUCCESS_MAX_XY = 0.010
SUCCESS_Z_THRESHOLD = 0.040
SUCCESS_MIN_AXIS_DOT = 0.990
SUCCESS_MAX_INHAND_BIAS = 0.025

# Before the final deep insertion, recenter the tube if it is already safe but
# close to the final xy boundary. This avoids the prism-local final-z motion
# pushing x/y outside the success gate. Correction is capped to avoid a large
# lateral scrape while partially inserted.
PRE_FINAL_XY_TARGET = 0.006
PRE_FINAL_XY_CORRECTION_MAX = 0.0015
# Only do lateral xy correction before the tube is meaningfully inside the hole.
# The v2 seed-5 run did xy correction at z_err about -0.021, which side-loaded
# the rod and the gripper then slid down along it.
PRE_FINAL_XY_MIN_Z_ERR = -0.012
PRE_FINAL_XY_CORRECTION_TIME_DILATION = 0.35
PRE_FINAL_XY_CORRECTION_DELAY_STEPS = 6

# If stage1 is deep enough but still barely outside xy success, allow one tiny
# target-frame settle instead of declaring the stage done based on z only.
FINAL_XY_SETTLE_ENABLED = True
FINAL_XY_SETTLE_MAX_XY = 0.011
FINAL_XY_SETTLE_CORRECTION_MAX = 0.001
FINAL_XY_SETTLE_DELAY_STEPS = 6

# Split the final insertion into small guarded pushes. If the gripper moves down
# relative to the tube without corresponding insertion progress, stop immediately
# instead of letting the fingers tunnel/slide along the rod.
FINAL_INSERT_STEP_Z = -0.004
FINAL_INSERT_MAX_STEPS = 10
FINAL_INSERT_GUARD_DELAY_STEPS = 6
FINAL_INSERT_MIN_Z_PROGRESS = 0.0008
FINAL_INSERT_MAX_STEP_INHAND_BIAS = 0.0045
FINAL_INSERT_MAX_TOTAL_INHAND_BIAS = 0.025

GRASP_HOLD_STEPS = 12
DEBUG_STOP_AFTER_PRE_PLACE = False
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
            "HOLE_X=", HOLE_X,
            "SLOT_NOISE_RANGE=", SLOT_NOISE_RANGE,
            "ROTATE_RANDOM_ENABLED=", ROTATE_RANDOM_ENABLED,
            "FORCE_ROTATE=", FORCE_ROTATE,
            "TRY_NOISE_ENABLED=", TRY_NOISE_ENABLED,
            "TRY_NOISE_RANGE=", TRY_NOISE_RANGE,
            "GRASP_BIAS_RANGE=", GRASP_BIAS_RANGE,
            "GRASP_DEPTH_THRESHOLD=", GRASP_DEPTH_THRESHOLD,
            "LIFT_AFTER_GRASP_Z=", LIFT_AFTER_GRASP_Z,
            "FIRST_DOWN_Z=", FIRST_DOWN_Z,
            "TILT_CORRECTION_GAIN=", TILT_CORRECTION_GAIN,
            "TILT_CORRECTION_MAX_X=", TILT_CORRECTION_MAX_X,
            "POST_FIRST_DOWN_MAX_INHAND_BIAS=", POST_FIRST_DOWN_MAX_INHAND_BIAS,
            "POST_TILT_MAX_INHAND_BIAS=", POST_TILT_MAX_INHAND_BIAS,
            "PRE_FIRST_DOWN_MIN_AXIS_DOT=", PRE_FIRST_DOWN_MIN_AXIS_DOT,
            "FINAL_INSERT_STAGE1_Z=", FINAL_INSERT_STAGE1_Z,
            "FINAL_INSERT_STAGE2_Z=", FINAL_INSERT_STAGE2_Z,
            "SUCCESS_MAX_XY=", SUCCESS_MAX_XY,
            "PRE_FINAL_XY_TARGET=", PRE_FINAL_XY_TARGET,
            "PRE_FINAL_XY_CORRECTION_MAX=", PRE_FINAL_XY_CORRECTION_MAX,
            "PRE_FINAL_XY_MIN_Z_ERR=", PRE_FINAL_XY_MIN_Z_ERR,
            "FINAL_INSERT_STEP_Z=", FINAL_INSERT_STEP_Z,
            "FINAL_INSERT_MAX_TOTAL_INHAND_BIAS=", FINAL_INSERT_MAX_TOTAL_INHAND_BIAS,
            flush=True,
        )

        if STRICT_PARAM_VERIFY:
            assert abs(SCENE_X_SCALE - 0.8) < 1e-9, SCENE_X_SCALE
            assert abs(PRISM_X - 0.32) < 1e-9, PRISM_X
            assert abs(HOLE_X - 0.48) < 1e-9, HOLE_X
            assert SLOT_NOISE_RANGE == [0.0, 0.0, 0.0], SLOT_NOISE_RANGE
            assert TRY_NOISE_ENABLED is False, TRY_NOISE_ENABLED
            assert GRASP_BIAS_RANGE == [0.091, 0.091], GRASP_BIAS_RANGE
            assert DEBUG_STOP_AFTER_PRE_PLACE is False, DEBUG_STOP_AFTER_PRE_PLACE
            assert abs(LIFT_AFTER_GRASP_Z - 0.04) < 1e-9, LIFT_AFTER_GRASP_Z
            assert abs(FIRST_DOWN_Z - (-0.018)) < 1e-9, FIRST_DOWN_Z
            assert abs(PRE_PLACE_X_OFFSET - (-0.008)) < 1e-9, PRE_PLACE_X_OFFSET
            assert abs(PRE_FIRST_DOWN_MIN_AXIS_DOT - 0.89) < 1e-9, PRE_FIRST_DOWN_MIN_AXIS_DOT
            assert abs(CORRECT_MIN_AXIS_DOT - 0.88) < 1e-9, CORRECT_MIN_AXIS_DOT
            assert abs(FINAL_INSERT_STAGE1_Z - (-0.030)) < 1e-9, FINAL_INSERT_STAGE1_Z
            assert abs(FINAL_INSERT_STAGE2_MIN_AXIS_DOT - 0.985) < 1e-9, FINAL_INSERT_STAGE2_MIN_AXIS_DOT
            assert abs(TILT_CORRECTION_GAIN - 0.55) < 1e-9, TILT_CORRECTION_GAIN
            assert abs(TILT_CORRECTION_MAX_X - 0.008) < 1e-9, TILT_CORRECTION_MAX_X
            assert abs(POST_FIRST_DOWN_MAX_INHAND_BIAS - 0.018) < 1e-9, POST_FIRST_DOWN_MAX_INHAND_BIAS
            assert abs(POST_TILT_MAX_INHAND_BIAS - 0.025) < 1e-9, POST_TILT_MAX_INHAND_BIAS
            assert abs(SUCCESS_MAX_XY - 0.010) < 1e-9, SUCCESS_MAX_XY
            assert abs(SUCCESS_Z_THRESHOLD - 0.040) < 1e-9, SUCCESS_Z_THRESHOLD
            assert abs(SUCCESS_MAX_INHAND_BIAS - 0.025) < 1e-9, SUCCESS_MAX_INHAND_BIAS
            assert abs(GRASP_DEPTH_THRESHOLD - 24.75) < 1e-9, GRASP_DEPTH_THRESHOLD
            assert abs(PRE_FINAL_XY_TARGET - 0.006) < 1e-9, PRE_FINAL_XY_TARGET
            assert abs(PRE_FINAL_XY_CORRECTION_MAX - 0.0015) < 1e-9, PRE_FINAL_XY_CORRECTION_MAX
            assert abs(PRE_FINAL_XY_MIN_Z_ERR - (-0.012)) < 1e-9, PRE_FINAL_XY_MIN_Z_ERR
            assert abs(FINAL_XY_SETTLE_MAX_XY - 0.011) < 1e-9, FINAL_XY_SETTLE_MAX_XY
            assert abs(FINAL_INSERT_STEP_Z - (-0.004)) < 1e-9, FINAL_INSERT_STEP_Z
            assert abs(FINAL_INSERT_MAX_TOTAL_INHAND_BIAS - 0.025) < 1e-9, FINAL_INSERT_MAX_TOTAL_INHAND_BIAS

        super().__init__(cfg=cfg, mode=mode, render_mode=render_mode, **kwargs)

    def create_actors(self):
        slot_pose = Pose([HOLE_X, 0.0, 0.002], [1, 0, 0, 0])
        base_pose = Pose([PRISM_X, 0.0, 0.002], [1, 0, 0, 0])
        prism_pose = Pose([PRISM_X, 0.0, 0.005], [1, 0, 0, 0])

        self.slot = self._actor_manager.add_from_usd_file(
            name='slot',
            asset_path="TestTubeHoleSlot.usd",
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
        if ROTATE_RANDOM_ENABLED:
            self.rotate = self.rng.choice(ROTATE_CHOICES)
        else:
            self.rotate = FORCE_ROTATE

        base_offset = self.create_noise(SLOT_NOISE_RANGE.copy()).add_rotation([0, 0, self.rotate])
        base_pose = Pose([HOLE_X, 0.0, 0.002], [1, 0, 0, 0]).add_offset(base_offset)
        self.slot.set_pose(base_pose)

        self.metadata["task_version_tag"] = TASK_VERSION_TAG
        self.metadata["scene_x_scale"] = SCENE_X_SCALE
        self.metadata["prism_x"] = PRISM_X
        self.metadata["hole_x"] = HOLE_X
        self.metadata["slot_noise_range"] = list(SLOT_NOISE_RANGE)
        self.metadata["rotate_random_enabled"] = bool(ROTATE_RANDOM_ENABLED)
        self.metadata["force_rotate"] = float(FORCE_ROTATE)
        self.metadata["try_noise_enabled"] = bool(TRY_NOISE_ENABLED)
        self.metadata["grasp_bias_range"] = list(GRASP_BIAS_RANGE)
        self.metadata["rotate"] = float(self.rotate)
        self.metadata["pre_first_down_min_axis_dot"] = float(PRE_FIRST_DOWN_MIN_AXIS_DOT)
        self.metadata["grasp_depth_threshold"] = float(GRASP_DEPTH_THRESHOLD)
        self.metadata["final_insert_step_z"] = float(FINAL_INSERT_STEP_Z)
        self.metadata["final_insert_max_total_inhand_bias"] = float(FINAL_INSERT_MAX_TOTAL_INHAND_BIAS)
        self.metadata["tilt_correction_gain"] = float(TILT_CORRECTION_GAIN)
        self.metadata["tilt_correction_max_x"] = float(TILT_CORRECTION_MAX_X)
        self.metadata["post_first_down_max_inhand_bias"] = float(POST_FIRST_DOWN_MAX_INHAND_BIAS)
        self.metadata["post_tilt_max_inhand_bias"] = float(POST_TILT_MAX_INHAND_BIAS)

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

    def _get_insert_state(self, tag):
        rel_pose = self.prism.get_pose().rebase(self.target_pose)
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

    def _get_inhand_bias(self):
        prism_inhand_pose = self.prism.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose()
        )
        inhand_bias = np.abs(self.origin_inhand_pose[2] - prism_inhand_pose[2])
        return prism_inhand_pose, inhand_bias

    def _is_success_state(self, rel_pose, axis_dot, z_threshold=SUCCESS_Z_THRESHOLD):
        _, inhand_bias = self._get_inhand_bias()
        xy_err = np.abs(rel_pose.p[:2])
        z_err = rel_pose.p[2]
        success = (
            np.all(xy_err < np.array([SUCCESS_MAX_XY, SUCCESS_MAX_XY]))
            and z_err < -z_threshold
            and axis_dot > SUCCESS_MIN_AXIS_DOT
            and inhand_bias < SUCCESS_MAX_INHAND_BIAS
        )
        return success, xy_err, z_err, inhand_bias

    def _target_frame_xy_correction(self, rel_pose, max_step):
        xy_move = -np.array(rel_pose.p[:2], dtype=float)
        xy_move = np.clip(xy_move, -max_step, max_step)
        return float(xy_move[0]), float(xy_move[1])

    def pre_move(self):
        self.delay(10)

        grasp_bias = self.rng.uniform(GRASP_BIAS_RANGE[0], GRASP_BIAS_RANGE[1])
        target_pose = self.prism.get_pose().add_bias([GRASP_X_OFFSET, 0, grasp_bias])

        cpose = construct_grasp_pose(
            target_pose.p,
            [0, 0, 1],
            [1, 0, 0]
        )
        self.cid = self.prism.register_point(cpose, type='contact')

        print(
            "[DEBUG PRE_GRASP_POSE]",
            "version=", TASK_VERSION_TAG,
            "grasp_bias=", grasp_bias,
            "target_pose=", target_pose,
            "cpose=", cpose,
            flush=True,
        )

        grasp_actions = self.atom.grasp_actor(
            self.prism,
            contact_point_id=self.cid,
            pre_dis=0.0,
            dis=0.0,
        )
        for action in grasp_actions:
            if action.action in ["gripper", "all"]:
                action.args["gripper_depth_threshold"] = GRASP_DEPTH_THRESHOLD
        ok = self.move(
            grasp_actions,
            tag="pre_grasp_prism_x5a_insert_hole",
        )
        if not ok:
            print("[DEBUG PRE_MOVE_STOP] failed at pre_grasp_prism_x5a_insert_hole", flush=True)
            return

        self.origin_inhand_pose = self.prism.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose()
        )
        self.metadata["grasp_origin_inhand_pose"] = self.origin_inhand_pose.tolist()
        self._print_inhand("after_grasp")

        if GRASP_HOLD_STEPS > 0:
            self.delay(GRASP_HOLD_STEPS, is_save=False)
            self._print_inhand("after_grasp_hold")

        base_pose = self.slot.get_pose()
        base_pose[3:] = (1, 0, 0, 0)

        self.hole_pose = base_pose.add_bias(HOLE_BIAS)
        if self.rotate == 0:
            self.target_pose = self.hole_pose.add_rotation([0, -np.pi / 6, 0])
        else:
            self.target_pose = self.hole_pose.add_rotation([0, np.pi / 6, 0])

        if TRY_NOISE_ENABLED:
            self.random_noise = self.create_noise(copy.deepcopy(TRY_NOISE_RANGE))
            self.random_noise[:2] *= np.sign(self.rng.uniform(-1, 1, size=2))
        else:
            self.random_noise = Pose([0.0, 0.0, 0.0], [1, 0, 0, 0])

        self.metadata["random_noise"] = self.random_noise.tolist()
        try_pose = self.hole_pose.add_offset(self.random_noise)
        pre_place_pose = try_pose.add_bias([PRE_PLACE_X_OFFSET, 0, PRE_PLACE_Z_OFFSET])

        print(
            "[DEBUG HOLE_TARGET]",
            "version=", TASK_VERSION_TAG,
            "slot_pose=", self.slot.get_pose(),
            "hole_pose=", self.hole_pose,
            "target_pose=", self.target_pose,
            "try_pose=", try_pose,
            "pre_place_pose=", pre_place_pose,
            "random_noise=", self.random_noise,
            flush=True,
        )

        ok = self.move(
            self.atom.move_by_displacement(z=LIFT_AFTER_GRASP_Z),
            tag=f"lift_after_grasp_single_z{int(LIFT_AFTER_GRASP_Z * 1000):03d}_insert_hole",
            constraint_pose=LIFT_CONSTRAINT_POSE,
        )
        if not ok:
            print("[DEBUG PRE_MOVE_STOP] failed at lift_after_grasp_single_insert_hole", flush=True)
            return
        self._print_inhand("after_lift_after_grasp_single")

        ok = self.move(
            self.atom.place_actor(
                self.prism,
                target_pose=pre_place_pose,
                pre_dis=COARSE_PRE_DIS,
                dis=COARSE_DIS,
                is_open=False,
            ),
            tag="pre_place_coarse_insert_hole_x5a",
            time_dilation_factor=PLACE_TIME_DILATION,
        )
        if not ok:
            print("[DEBUG PRE_MOVE_STOP] failed at pre_place_coarse_insert_hole_x5a", flush=True)
            return

        ok = self.move(
            self.atom.place_actor(
                self.prism,
                target_pose=pre_place_pose,
                pre_dis=FINE_PRE_DIS,
                dis=FINE_DIS,
                is_open=False,
            ),
            tag="pre_place_fine_insert_hole_x5a",
            constraint_pose=FINE_PLACE_CONSTRAINT_POSE,
            time_dilation_factor=PLACE_TIME_DILATION,
        )
        if not ok:
            print("[DEBUG PRE_MOVE_STOP] failed at pre_place_fine_insert_hole_x5a", flush=True)
            return

        self.origin_inhand_pose = self.prism.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose()
        )
        self.metadata["place_origin_inhand_pose"] = self.origin_inhand_pose.tolist()
        self._print_inhand("after_pre_place")
        self._get_insert_state("[DEBUG AFTER_PRE_PLACE]")

    def _guarded_final_insert(self):
        """Do final insertion as small pushes with an in-hand slip guard."""
        rel_pose, xy_err, z_err, axis_dot = self._get_insert_state("[DEBUG BEFORE_GUARDED_FINAL_INSERT]")
        _, prev_inhand_bias = self._get_inhand_bias()
        if prev_inhand_bias > FINAL_INSERT_MAX_TOTAL_INHAND_BIAS:
            self.metadata["slip_abort"] = True
            self.metadata["slip_abort_stage"] = "before_guarded_final_insert"
            self.metadata["slip_abort_inhand_bias"] = float(prev_inhand_bias)
            print(
                "[DEBUG PLAY_ONCE_STOP] slip already too large before guarded final insert",
                "inhand_bias=", prev_inhand_bias,
                "limit=", FINAL_INSERT_MAX_TOTAL_INHAND_BIAS,
                "xy_err=", xy_err,
                "z_err=", z_err,
                "axis_dot=", axis_dot,
                flush=True,
            )
            return False
        prev_z_err = z_err

        for step_idx in range(FINAL_INSERT_MAX_STEPS):
            success_now, xy_err, z_err, inhand_bias = self._is_success_state(rel_pose, axis_dot)
            if success_now:
                print(
                    "[DEBUG GUARDED_FINAL_INSERT_DONE] success before next push",
                    "step_idx=", step_idx,
                    "xy_err=", xy_err,
                    "z_err=", z_err,
                    "axis_dot=", axis_dot,
                    "inhand_bias=", inhand_bias,
                    flush=True,
                )
                return True

            if (
                np.any(xy_err > np.array([FINAL_INSERT_STAGE2_MAX_XY, FINAL_INSERT_STAGE2_MAX_XY]))
                or axis_dot < FINAL_INSERT_STAGE2_MIN_AXIS_DOT
            ):
                print(
                    "[DEBUG PLAY_ONCE_STOP] guarded final insert unsafe before push",
                    "step_idx=", step_idx,
                    "xy_err=", xy_err,
                    "z_err=", z_err,
                    "axis_dot=", axis_dot,
                    "inhand_bias=", inhand_bias,
                    flush=True,
                )
                return False

            ok = self.move(
                self.atom.move_by_displacement(
                    z=FINAL_INSERT_STEP_Z,
                    xyz_coord=self.prism.get_pose(),
                ),
                tag=f"guarded_final_insert_{step_idx:02d}_insert_hole_x5a",
                time_dilation_factor=FINAL_INSERT_TIME_DILATION,
                delay=False,
            )
            if not ok:
                print(
                    "[DEBUG PLAY_ONCE_STOP] failed at guarded final insert",
                    "step_idx=", step_idx,
                    flush=True,
                )
                return False

            self.delay(FINAL_INSERT_GUARD_DELAY_STEPS, is_save=False)
            rel_pose, xy_err, z_err, axis_dot = self._get_insert_state(
                f"[DEBUG AFTER_GUARDED_FINAL_INSERT_{step_idx:02d}]"
            )
            _, inhand_bias = self._get_inhand_bias()
            z_progress = float(prev_z_err - z_err)
            step_inhand_bias = float(abs(inhand_bias - prev_inhand_bias))

            print(
                "[DEBUG FINAL_INSERT_SLIP_GUARD]",
                "step_idx=", step_idx,
                "z_progress=", z_progress,
                "step_inhand_bias=", step_inhand_bias,
                "total_inhand_bias=", inhand_bias,
                "xy_err=", xy_err,
                "z_err=", z_err,
                "axis_dot=", axis_dot,
                flush=True,
            )

            success_now, xy_err, z_err, inhand_bias = self._is_success_state(rel_pose, axis_dot)
            if success_now:
                print(
                    "[DEBUG GUARDED_FINAL_INSERT_DONE] success after push",
                    "step_idx=", step_idx,
                    "xy_err=", xy_err,
                    "z_err=", z_err,
                    "axis_dot=", axis_dot,
                    "inhand_bias=", inhand_bias,
                    flush=True,
                )
                return True

            if (
                inhand_bias > FINAL_INSERT_MAX_TOTAL_INHAND_BIAS
                or (
                    step_inhand_bias > FINAL_INSERT_MAX_STEP_INHAND_BIAS
                    and z_progress < FINAL_INSERT_MIN_Z_PROGRESS
                )
            ):
                self.metadata["slip_abort"] = True
                self.metadata["slip_abort_step"] = int(step_idx)
                self.metadata["slip_abort_inhand_bias"] = float(inhand_bias)
                self.metadata["slip_abort_z_progress"] = float(z_progress)
                print(
                    "[DEBUG PLAY_ONCE_STOP] slip guard triggered",
                    "step_idx=", step_idx,
                    "z_progress=", z_progress,
                    "step_inhand_bias=", step_inhand_bias,
                    "total_inhand_bias=", inhand_bias,
                    flush=True,
                )
                return False

            if z_err <= FINAL_INSERT_TARGET_Z and FINAL_XY_SETTLE_ENABLED:
                if (
                    np.all(xy_err < np.array([FINAL_XY_SETTLE_MAX_XY, FINAL_XY_SETTLE_MAX_XY]))
                    and axis_dot > FINAL_INSERT_STAGE2_MIN_AXIS_DOT
                    and inhand_bias < FINAL_INSERT_MAX_TOTAL_INHAND_BIAS
                ):
                    x_move, y_move = self._target_frame_xy_correction(
                        rel_pose,
                        FINAL_XY_SETTLE_CORRECTION_MAX,
                    )
                    print(
                        "[DEBUG FINAL_XY_SETTLE]",
                        "xy_err=", xy_err,
                        "z_err=", z_err,
                        "axis_dot=", axis_dot,
                        "inhand_bias=", inhand_bias,
                        "x_move=", x_move,
                        "y_move=", y_move,
                        flush=True,
                    )
                    ok = self.move(
                        self.atom.move_by_displacement(
                            x=x_move,
                            y=y_move,
                            xyz_coord=self.target_pose,
                        ),
                        tag="final_xy_settle_insert_hole_x5a",
                        constraint_pose=FINE_PLACE_CONSTRAINT_POSE,
                        time_dilation_factor=FINAL_INSERT_TIME_DILATION,
                        delay=False,
                    )
                    if not ok:
                        print("[DEBUG PLAY_ONCE_STOP] failed at final_xy_settle_insert_hole_x5a", flush=True)
                        return False
                    self.delay(FINAL_XY_SETTLE_DELAY_STEPS, is_save=False)
                    rel_pose, xy_err, z_err, axis_dot = self._get_insert_state("[DEBUG AFTER_FINAL_XY_SETTLE]")
                    success_now, xy_err, z_err, inhand_bias = self._is_success_state(rel_pose, axis_dot)
                    if success_now:
                        print("[DEBUG FINAL_XY_SETTLE_DONE] success gates satisfied", flush=True)
                        return True

                print(
                    "[DEBUG GUARDED_FINAL_INSERT_DEEP_BUT_NOT_SUCCESS]",
                    "xy_err=", xy_err,
                    "z_err=", z_err,
                    "axis_dot=", axis_dot,
                    "inhand_bias=", inhand_bias,
                    flush=True,
                )
                return False

            prev_z_err = z_err
            prev_inhand_bias = inhand_bias

        print(
            "[DEBUG PLAY_ONCE_STOP] guarded final insert exhausted",
            "max_steps=", FINAL_INSERT_MAX_STEPS,
            "last_xy_err=", xy_err,
            "last_z_err=", z_err,
            "last_axis_dot=", axis_dot,
            "last_inhand_bias=", inhand_bias,
            flush=True,
        )
        return False

    def _play_once(self):
        if DEBUG_STOP_AFTER_PRE_PLACE:
            self.metadata["debug_stop_after_pre_place"] = True
            print("[DEBUG STOP_AFTER_PRE_PLACE] skip insertion phase", flush=True)
            return

        rel_pose, xy_err, z_err, axis_dot = self._get_insert_state("[DEBUG BEFORE_FIRST_DOWN]")
        if (
            np.any(xy_err > np.array([PRE_FIRST_DOWN_MAX_XY, PRE_FIRST_DOWN_MAX_XY]))
            or axis_dot < PRE_FIRST_DOWN_MIN_AXIS_DOT
            or z_err > PRE_FIRST_DOWN_MAX_POSITIVE_Z
        ):
            print(
                "[DEBUG PLAY_ONCE_STOP] bad pre-place before first_down",
                "xy_err=", xy_err,
                "z_err=", z_err,
                "axis_dot=", axis_dot,
                "min_axis_dot=", PRE_FIRST_DOWN_MIN_AXIS_DOT,
                "max_xy=", PRE_FIRST_DOWN_MAX_XY,
                "max_positive_z=", PRE_FIRST_DOWN_MAX_POSITIVE_Z,
                flush=True,
            )
            return

        ok = self.move(
            self.atom.move_by_displacement(z=FIRST_DOWN_Z),
            tag="first_down_insert_hole_x5a",
            time_dilation_factor=FIRST_DOWN_TIME_DILATION,
        )
        if not ok:
            print("[DEBUG PLAY_ONCE_STOP] failed at first_down_insert_hole_x5a", flush=True)
            return

        rel_pose, xy_err, z_err, axis_dot = self._get_insert_state("[DEBUG AFTER_FIRST_DOWN]")
        _, inhand_bias = self._get_inhand_bias()
        print(
            "[DEBUG POST_FIRST_DOWN_INHAND]",
            "inhand_bias=", inhand_bias,
            "limit=", POST_FIRST_DOWN_MAX_INHAND_BIAS,
            "xy_err=", xy_err,
            "z_err=", z_err,
            "axis_dot=", axis_dot,
            flush=True,
        )
        if inhand_bias > POST_FIRST_DOWN_MAX_INHAND_BIAS:
            self.metadata["slip_abort"] = True
            self.metadata["slip_abort_stage"] = "after_first_down"
            self.metadata["slip_abort_inhand_bias"] = float(inhand_bias)
            print(
                "[DEBUG PLAY_ONCE_STOP] inhand slip after first_down",
                "inhand_bias=", inhand_bias,
                "limit=", POST_FIRST_DOWN_MAX_INHAND_BIAS,
                flush=True,
            )
            return

        if np.any(xy_err > np.array([CORRECT_MAX_XY, CORRECT_MAX_XY])) or axis_dot < CORRECT_MIN_AXIS_DOT:
            print(
                "[DEBUG PLAY_ONCE_STOP] first_down outside correction gate",
                "xy_err=", xy_err,
                "z_err=", z_err,
                "axis_dot=", axis_dot,
                flush=True,
            )
            return

        gripper_dis = self._robot_manager.get_gripper_center_pose().rebase(self.prism.get_pose())[2]
        raw_x_move = -gripper_dis * np.sin(rel_pose.euler[1])
        raw_z_move = gripper_dis * (np.cos(rel_pose.euler[1]) - 1)
        x_move = float(np.clip(raw_x_move * TILT_CORRECTION_GAIN, -TILT_CORRECTION_MAX_X, TILT_CORRECTION_MAX_X))
        z_move = float(np.clip(raw_z_move * TILT_CORRECTION_GAIN, -TILT_CORRECTION_MAX_Z, TILT_CORRECTION_MAX_Z))

        print(
            "[DEBUG TILT_CORRECTION]",
            "gripper_dis=", gripper_dis,
            "rel_euler=", rel_pose.euler,
            "raw_x_move=", raw_x_move,
            "raw_z_move=", raw_z_move,
            "gain=", TILT_CORRECTION_GAIN,
            "x_move=", x_move,
            "z_move=", z_move,
            "coord=target_pose",
            flush=True,
        )

        ok = self.move(
            self.atom.move_by_displacement(
                x=x_move,
                z=z_move,
                xyz_coord=self.target_pose,
            ),
            tag="tilt_correction_insert_hole_x5a",
            time_dilation_factor=CORRECT_TIME_DILATION,
        )
        if not ok:
            print("[DEBUG PLAY_ONCE_STOP] failed at tilt_correction_insert_hole_x5a", flush=True)
            return

        rel_pose, xy_err, z_err, axis_dot = self._get_insert_state("[DEBUG AFTER_TILT_CORRECTION]")
        _, inhand_bias = self._get_inhand_bias()
        print(
            "[DEBUG POST_TILT_INHAND]",
            "inhand_bias=", inhand_bias,
            "limit=", POST_TILT_MAX_INHAND_BIAS,
            "xy_err=", xy_err,
            "z_err=", z_err,
            "axis_dot=", axis_dot,
            flush=True,
        )
        if inhand_bias > POST_TILT_MAX_INHAND_BIAS:
            self.metadata["slip_abort"] = True
            self.metadata["slip_abort_stage"] = "after_tilt_correction"
            self.metadata["slip_abort_inhand_bias"] = float(inhand_bias)
            print(
                "[DEBUG PLAY_ONCE_STOP] inhand slip after tilt_correction",
                "inhand_bias=", inhand_bias,
                "limit=", POST_TILT_MAX_INHAND_BIAS,
                flush=True,
            )
            return

        if z_err > -0.004 or np.any(xy_err > np.array([FINAL_INSERT_STAGE2_MAX_XY, FINAL_INSERT_STAGE2_MAX_XY])) or axis_dot < FINAL_INSERT_STAGE2_MIN_AXIS_DOT:
            print(
                "[DEBUG PLAY_ONCE_STOP] not safe for final insert",
                "xy_err=", xy_err,
                "z_err=", z_err,
                "axis_dot=", axis_dot,
                flush=True,
            )
            return

        if (
            z_err > PRE_FINAL_XY_MIN_Z_ERR
            and np.any(xy_err > np.array([PRE_FINAL_XY_TARGET, PRE_FINAL_XY_TARGET]))
        ):
            x_move, y_move = self._target_frame_xy_correction(
                rel_pose,
                PRE_FINAL_XY_CORRECTION_MAX,
            )
            print(
                "[DEBUG PRE_FINAL_XY_CORRECTION]",
                "xy_err=", xy_err,
                "x_move=", x_move,
                "y_move=", y_move,
                "coord=target_pose",
                flush=True,
            )
            ok = self.move(
                self.atom.move_by_displacement(
                    x=x_move,
                    y=y_move,
                    xyz_coord=self.target_pose,
                ),
                tag="pre_final_xy_correction_insert_hole_x5a",
                constraint_pose=FINE_PLACE_CONSTRAINT_POSE,
                time_dilation_factor=PRE_FINAL_XY_CORRECTION_TIME_DILATION,
                delay=False,
            )
            if not ok:
                print("[DEBUG PLAY_ONCE_STOP] failed at pre_final_xy_correction_insert_hole_x5a", flush=True)
                return
            self.delay(PRE_FINAL_XY_CORRECTION_DELAY_STEPS, is_save=False)
            rel_pose, xy_err, z_err, axis_dot = self._get_insert_state("[DEBUG AFTER_PRE_FINAL_XY_CORRECTION]")

            if z_err > -0.004 or np.any(xy_err > np.array([FINAL_INSERT_STAGE2_MAX_XY, FINAL_INSERT_STAGE2_MAX_XY])) or axis_dot < FINAL_INSERT_STAGE2_MIN_AXIS_DOT:
                print(
                    "[DEBUG PLAY_ONCE_STOP] pre-final xy correction not safe for final insert",
                    "xy_err=", xy_err,
                    "z_err=", z_err,
                    "axis_dot=", axis_dot,
                    flush=True,
                )
                return

        self._guarded_final_insert()

    def check_early_stop(self):
        prism_inhand_pose = self.prism.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose())
        inhand_bias = np.abs(self.origin_inhand_pose[2] - prism_inhand_pose[2])
        if inhand_bias > FINAL_INSERT_MAX_TOTAL_INHAND_BIAS:
            self.metadata['early_stop'] = True
            self.metadata['inhand_bias'] = float(inhand_bias)
            return True
        return False

    def check_success(self, z_threshold=SUCCESS_Z_THRESHOLD):
        prism_pose = self.prism.get_pose().rebase(self.target_pose)
        axis_dot = np.dot(
            prism_pose.to_transformation_matrix()[:3, 2],
            np.array([0, 0, 1])
        )
        success, xy_err, z_err, inhand_bias = self._is_success_state(
            prism_pose,
            axis_dot,
            z_threshold=z_threshold,
        )

        self.metadata['rel_pose'] = prism_pose.tolist()
        self.metadata['inhand_bias'] = float(inhand_bias)
        self.metadata['xy_err'] = xy_err.tolist()
        self.metadata['z_err'] = float(z_err)
        self.metadata['axis_dot'] = float(axis_dot)

        print(
            "[DEBUG CHECK_SUCCESS]",
            "success=", success,
            "rel_p=", prism_pose.p,
            "xy_err=", xy_err,
            "z_err=", z_err,
            "axis_dot=", axis_dot,
            "inhand_bias=", inhand_bias,
            "xy_threshold=", SUCCESS_MAX_XY,
            "z_threshold=", z_threshold,
            "axis_threshold=", SUCCESS_MIN_AXIS_DOT,
            "inhand_threshold=", SUCCESS_MAX_INHAND_BIAS,
            flush=True,
        )

        return success

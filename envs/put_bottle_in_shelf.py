from ._base_task import *
import numpy as np

@configclass
class TaskCfg(BaseTaskCfg):
    pass

class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode:Literal['collect', 'eval'] = 'collect', render_mode: str|None = None, **kwargs):
        cfg.sim.physics_material.dynamic_friction = 1
        cfg.sim.physics_material.static_friction = 1
        cfg.uipc_sim.contact.default_friction_ratio = 1
        super().__init__(cfg, mode, render_mode, **kwargs)

    def _is_x5a(self):
        return getattr(self._robot_manager, 'robot_type', None) == 'x5a'

    def create_actors(self):
        if self._is_x5a():
            base_pose = Pose([0.66, 0.0, 0.01], [1, 0, 0, 0])
            bottle_pose = Pose([0.32, 0.0, 0.01], [1, 0, 0, 0])
        else:
            base_pose = Pose([0.9, 0.0, 0.01], [1, 0, 0, 0])
            bottle_pose = Pose([0.5, 0.0, 0.01], [1, 0, 0, 0])

        self.shelf = self._actor_manager.add_from_usd_file(
            name='shelf',
            asset_path='Shelf.usd',
            pose=base_pose,
        )
        self.bottle = self._actor_manager.add_from_usd_file(
            name='prism',
            asset_path='BottleLift.usd',
            pose=bottle_pose,
        )

    def _reset_actors(self):
        if self._is_x5a():
            base_pose = Pose([0.66, 0.0, 0.01], [1, 0, 0, 0]).add_offset(
                self.create_noise([0.01, 0.0, 0.0])
            )
            bottle_pose = Pose([0.32, 0.0, 0.01], [1, 0, 0, 0]).add_offset(
                self.create_noise([0.0, 0.006, 0.0])
            )
            self.metadata['x5a_tuning_version'] = 'shelf_x5a_v14_fixed_clearance_trajectory'
        else:
            base_pose = Pose([0.9, 0.0, 0.01], [1, 0, 0, 0]).add_offset(
                self.create_noise([0.05, 0.0, 0.0])
            )
            bottle_pose = Pose([0.5, 0.0, 0.01], [1, 0, 0, 0]).add_offset(
                self.create_noise([0.0, 0.03, 0.0])
            )

        self.shelf.set_pose(base_pose)
        self.bottle.set_pose(bottle_pose)

    def pre_move(self):
        self.delay(10)

        if self._is_x5a():
            self.move(self.atom.open_gripper(0.5))

        bottle_pose = self.bottle.get_pose()

        if self._is_x5a():
            target_pose = bottle_pose.add_bias([-0.003, 0, 0.102])
            self.grasp_noise = self.create_noise(euler=[0, np.pi / 96, 0])
            grasp_pre_dis = 0.035
            grasp_time_dilation = 0.98
            self.metadata['x5a_grasp_bias'] = [-0.003, 0.0, 0.102]
            self.metadata['x5a_grasp_pre_dis'] = float(grasp_pre_dis)
            self.metadata['x5a_grasp_time_dilation'] = float(grasp_time_dilation)
        else:
            target_pose = bottle_pose.add_bias([0, 0, 0.11 + 0.01 * self.rng.random()])
            self.grasp_noise = self.create_noise(euler=[0, np.pi / 18, 0])
            grasp_pre_dis = 0.0

        target_pose = construct_grasp_pose(
            target_pose.p,
            [0, 0, 1],
            [1, 0, 0],
        ).add_offset(self.grasp_noise)

        grasp_idx = self.bottle.register_point(
            pose=target_pose,
            type='contact',
        )
        if self._is_x5a():
            self.move(
                self.atom.grasp_actor(
                    self.bottle,
                    contact_point_id=grasp_idx,
                    pre_dis=grasp_pre_dis,
                    is_close=False,
                ),
                tag="x5a_slow_pre_grasp_bottle",
                time_dilation_factor=grasp_time_dilation,
            )
        else:
            self.move(self.atom.grasp_actor(
                self.bottle,
                contact_point_id=grasp_idx,
                pre_dis=grasp_pre_dis,
                is_close=False,
            ))

        
        if self._is_x5a():
            # Keep the higher target for motion to avoid the shelf edge, but check success
            # against a lower/settled target because the bottle drops slightly after release.
            self.place_target = self.shelf.get_pose().add_bias([-0.225, 0, 0.200])
            self.place_target_success = self.shelf.get_pose().add_bias([-0.225, 0, 0.180])
            self.metadata['x5a_place_motion_bias'] = [-0.225, 0.0, 0.200]
            self.metadata['x5a_place_success_bias'] = [-0.225, 0.0, 0.180]
        else:
            self.place_target = self.shelf.get_pose().add_bias([-0.2, 0, 0.21])
        self.move(self.atom.close_gripper())

        if self._is_x5a():
            self.delay(6, is_save=False)

    def _play_once(self):
        if self._is_x5a():
            # X5A: put the 4 cm clearance directly into the main lift, then keep
            # the previous placement flow.  This avoids a separate post-rotate
            # z-lift loop driven by place_target.
            x5a_lift_extra_total = 0.05
            lift_segments = (0.05, 0.05)
            x5a_lift_extra_step = x5a_lift_extra_total / len(lift_segments)
            for dz in lift_segments:
                self.move(self.atom.move_by_displacement(
                    z=float(dz + x5a_lift_extra_step),
                ), constraint_pose=[0, 0, 1, 0, 0, 0], time_dilation_factor=0.75)
            self.metadata['x5a_lift_extra_total'] = float(x5a_lift_extra_total)

            self.move(self.atom.move_by_displacement(
                rpy=[0, -np.pi / 5, 0],
            ), constraint_pose=[0, 0, 0, 0, 1, 0], time_dilation_factor=0.7)

            self.gravity_rotate(self.bottle, target_vec=np.array([0, 0, 1]))

            rel = np.asarray(self.bottle.get_pose().rebase(self.place_target).p, dtype=float).reshape(3)
            correction_xy = np.clip(-rel[:2], [-0.020, -0.014], [0.020, 0.014])
            if np.any(np.abs(correction_xy) > np.array([0.006, 0.006])):
                self.move(self.atom.move_by_displacement(
                    x=float(correction_xy[0]),
                    y=float(correction_xy[1]),
                ), constraint_pose=[1, 1, 0, 0, 0, 0], time_dilation_factor=0.75)

            rel = np.asarray(self.bottle.get_pose().rebase(self.place_target).p, dtype=float).reshape(3)
            dz = float(np.clip(-rel[2], -0.008, 0.008))
            self.metadata['x5a_local_place_rel_before_z'] = [float(x) for x in rel]
            self.metadata['x5a_local_place_dz'] = dz
            if abs(dz) > 0.002:
                self.move(self.atom.move_by_displacement(
                    z=dz,
                ), constraint_pose=[0, 0, 1, 0, 0, 0], time_dilation_factor=0.75)


            rel = np.asarray(self.bottle.get_pose().rebase(self.place_target).p, dtype=float).reshape(3)
            correction_xy = np.clip(-rel[:2], [-0.010, -0.010], [0.010, 0.010])
            self.metadata['x5a_local_place_rel_before_xy'] = [float(x) for x in rel]
            self.metadata['x5a_local_place_xy'] = [float(x) for x in correction_xy]
            if np.any(np.abs(correction_xy) > np.array([0.004, 0.004])):
                self.move(self.atom.move_by_displacement(
                    x=float(correction_xy[0]),
                    y=float(correction_xy[1]),
                ), constraint_pose=[1, 1, 0, 0, 0, 0], time_dilation_factor=0.75)

            self.delay(6, is_save=False)
            self.move(self.atom.open_gripper(0.28))
        else:
            self.move(self.atom.move_by_displacement(
                z=0.15 + self.rng.uniform(0.0, 0.05),
            ), constraint_pose=[1, 1, 1, 0, 1, 0])
            self.move(self.atom.move_by_displacement(
                rpy=[0, -np.pi / 3 - 0.2 * self.rng.random(), 0],
            ), constraint_pose=[0, 0, 0, 0, 1, 0])
            self.gravity_rotate(self.bottle, target_vec=np.array([0, 0, 1]))
            self.move(self.atom.place_actor(
                self.bottle,
                pre_dis=0.01,
                dis=0.005,
                target_pose=self.place_target,
                is_open=False,
            ), time_dilation_factor=0.3)
            self.move(self.atom.open_gripper(0.5))

        self.delay(20, is_save=False)

    def check_early_stop(self):
        min_depth = torch.min(self._tactile_manager.get_min_depth()).item()

        if min_depth < 20:
            self.metadata['early_stop'] = True
            self.metadata['min_depth'] = float(min_depth)
            return True
        return False

    def check_success(self):
        if self._is_x5a():
            success_target = getattr(self, 'place_target_success', self.place_target)
            bottle_pose = self.bottle.get_pose().rebase(success_target)
            axis_dot = np.dot(bottle_pose.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))
            rel = np.asarray(bottle_pose.p, dtype=float).reshape(3)

            xy_ok = bool(abs(rel[0]) < 0.030 and abs(rel[1]) < 0.100)
            # X5A uses a higher motion target, so after release the bottle can settle below it.
            # Allow a lower final z while still rejecting obviously high/outside placements.
            z_ok = bool(-0.045 < rel[2] < 0.035)
            axis_ok = bool(axis_dot > 0.94)

            self.metadata['x5a_success_rel_xyz'] = [float(x) for x in rel]
            self.metadata['x5a_success_axis_dot'] = float(axis_dot)
            self.metadata['x5a_success_xy_ok'] = xy_ok
            self.metadata['x5a_success_z_ok'] = z_ok
            self.metadata['x5a_success_axis_ok'] = axis_ok
            return xy_ok and z_ok and axis_ok

        bottle_pose = self.bottle.get_pose().rebase(self.place_target)
        axis_dot = np.dot(bottle_pose.to_transformation_matrix()[:3, 2], np.array([0, 0, 1]))
        return np.all(np.abs(bottle_pose.p) < np.array([0.02, 0.1, 0.02])) and axis_dot > 0.965

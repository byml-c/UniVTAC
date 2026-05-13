import torch
import os
from envs.utils import data
import numpy as np
from tacex import GelSightSensor, GelSightSensorCfg
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.sensors.gf225.gf225_cfg import GF225Cfg
from tacex.simulation_approaches.fem_based import ManiSkillSimulatorCfg
from tacex.simulation_approaches.fots import FOTSMarkerSimulatorCfg

from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg

from tacex_uipc import (
    UipcRLEnv,
    UipcIsaacAttachments,
    UipcIsaacAttachmentsCfg,
    UipcObject,
    UipcObjectCfg,
    UipcSimCfg
)

from ..utils.transforms import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .._base_task import BaseTask
    from tacex_uipc.sim import UipcIsaacAttachmentsCfg, UipcSim
    from tacex_uipc import UipcInteractiveScene


@configclass
class TactileCfg:
    name: str = 'tactile_sensor'
    sensor_cfg = None
    gelpad_cfg: UipcObjectCfg = None
    gelpad_attachment_cfg: UipcIsaacAttachmentsCfg = None

# Marker-only visual/observation offset for XenseWS.
# Positive x moves markers right; negative x moves markers left.
# Positive y moves markers down; negative y moves markers up.
# This does not change depth, gelpad attachment, grasp control, or physics.
XENSEWS_MARKER_SHIFT_PX = {
    "left_tactile": (-8, -10),
    "right_tactile": (-8, -10),
}

def create_gelsight_mini_cfg(
    prim_path: str,
    gelpad_prim_path: str,
    gelpad_attachment_body_name: str,
    name: str = "tactile_sensor",
    resolution = (320, 240),
    update_period = 1/120,
    data_type:list[str] = ["camera_depth", "tactile_rgb"],
):
    from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg
    sensor_cfg = GelSightMiniCfg(
        prim_path=prim_path,
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            resolution=resolution,
            update_period=update_period,
            data_types=["depth", "rgb"],
            clipping_range=(0.024, 0.034),
        ),
        device="cuda",
        debug_vis=False,  # for rendering sensor output in the gui
        update_period=1/120,
        marker_motion_sim_cfg=ManiSkillSimulatorCfg(
            tactile_img_res=resolution,
            marker_shape=(9, 7),
            marker_interval=(2.40625, 2.45833),
            sub_marker_num=0,
            marker_radius=6,
            camera_to_surface=0.0283,
            real_size=(0.0266, 0.0209),
            sensor_type='gsmini',
        ),
        data_types=data_type
    )
    sensor_cfg.marker_motion_sim_cfg.marker_params.num_markers = 64
    sensor_cfg.optical_sim_cfg = sensor_cfg.optical_sim_cfg.replace(
        with_shadow=False,
        tactile_img_res=resolution,
        device="cuda",
    )

    cfg = TactileCfg(
        name=name,
        sensor_cfg=sensor_cfg,
        gelpad_cfg=UipcObjectCfg(
            prim_path=gelpad_prim_path,
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(youngs_modulus=0.1),
            mass_density=1e4
        ),
        gelpad_attachment_cfg=UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=1e4,
            body_name=gelpad_attachment_body_name,
            debug_vis=False,
        ),
    )
    return cfg

def create_gf225_cfg(
    prim_path: str,
    gelpad_prim_path: str,
    gelpad_attachment_body_name: str,
    gelpad_attachment_prim_path: str = None,
    name: str = "tactile_sensor",
    data_type: list[str] = ["camera_depth", "tactile_rgb"],
) -> TactileCfg:
    resolution = (480, 480)  # GF225 resolution
    update_period = 1/120
    
    sensor_cfg = GF225Cfg(
        prim_path=prim_path,
        sensor_camera_cfg=GF225Cfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            resolution=resolution,
            update_period=update_period,
            data_types=["depth"],
            clipping_range=(0.02, 0.0265),
        ),
        device="cuda",
        debug_vis=False,
        update_period=1/120,
        marker_motion_sim_cfg=ManiSkillSimulatorCfg(
            tactile_img_res=resolution,
            sub_marker_num=0,
            marker_radius=8,
            marker_shape=(9, 9),
            marker_interval=(2.0, 2.0),
            camera_to_surface=0.0265,
            real_size = (0.0235, 0.0250),
            sensor_type='gf225',
        ),
        data_types=data_type
    )
    
    from tacex.simulation_approaches.mlp_fots import MLPFOTSSimulatorCfg
    from tacex_assets import TACEX_ASSETS_DATA_DIR

    sensor_cfg.marker_motion_sim_cfg.marker_params.num_markers = 81
    sensor_cfg.optical_sim_cfg = MLPFOTSSimulatorCfg(
        calib_folder_path=f"{TACEX_ASSETS_DATA_DIR}/Sensors/GF225/calibs/480x480",
        tactile_img_res=resolution,
        device="cuda",
    )
    
    cfg = TactileCfg(
        name=name,
        sensor_cfg=sensor_cfg,
        gelpad_cfg=UipcObjectCfg(
            prim_path=gelpad_prim_path,
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(youngs_modulus=0.1),
            mass_density=1e4
        ),
        gelpad_attachment_cfg=UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=1e4,
            body_name=gelpad_attachment_body_name,
            isaac_rigid_prim_path=gelpad_attachment_prim_path,
            debug_vis=False,
        ),
    )
    return cfg

def create_xensews_cfg(
    prim_path: str,
    gelpad_prim_path: str,
    gelpad_attachment_body_name: str,
    gelpad_attachment_prim_path: str = None,
    name: str = "tactile_sensor",
    resolution = (320, 240),
    update_period = 1/120,
    data_type:list[str] = ["camera_depth", "tactile_rgb"],
) -> TactileCfg:
    from tacex_assets.sensors.xensews.xensews_cfg import XenseWSCfg

    sensor_cfg = XenseWSCfg(
        prim_path=prim_path,
        sensor_camera_cfg=XenseWSCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            update_period=update_period,
            resolution=resolution,
            data_types=["depth", "rgb"],
            clipping_range=(0.025, 0.029),  # (0.024, 0.034),
        ),
        device="cuda",
        debug_vis=True,  # for rendering sensor output in the gui
        update_period=update_period,
        marker_motion_sim_cfg=ManiSkillSimulatorCfg(
            tactile_img_res=resolution,
            marker_shape=(9, 7),
            marker_interval=(2.0, 2.0),
            sub_marker_num=0,
            marker_radius=2,
            camera_to_surface=0.0245,
            real_size=(0.0285, 0.0166),
            sensor_type='xensews',
        ),
        data_types=data_type
    )
    sensor_cfg.marker_motion_sim_cfg.marker_params.num_markers = 63
    sensor_cfg.optical_sim_cfg = sensor_cfg.optical_sim_cfg.replace(
        with_shadow=False,
        tactile_img_res=resolution,
        device="cuda",
    )

    cfg = TactileCfg(
        name=name,
        sensor_cfg=sensor_cfg,
        gelpad_cfg=UipcObjectCfg(
            prim_path=gelpad_prim_path,
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(youngs_modulus=0.1),
            mass_density=1e4
        ),
        gelpad_attachment_cfg=UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=2e4,
            body_name=gelpad_attachment_body_name,
            isaac_rigid_prim_path=gelpad_attachment_prim_path,
            debug_vis=False,
        ),
    )
    return cfg

def create_tactile_cfg(
    prim_path: str,
    gelpad_prim_path: str,
    gelpad_attachment_body_name: str,
    gelpad_attachment_prim_path: str = None,
    name: str = "tactile_sensor",
    sensor_type:Literal['gsmini', 'xensews', 'gf225'] = "xensews",
    data_type:list[str] = ["camera_depth", "tactile_rgb"],
) -> TactileCfg:
    if sensor_type == "gsmini":
        return create_gelsight_mini_cfg(
            prim_path=prim_path,
            gelpad_prim_path=gelpad_prim_path,
            gelpad_attachment_body_name=gelpad_attachment_body_name,
            name=name,
            data_type=data_type,
        )
    elif sensor_type == "xensews":
        return create_xensews_cfg(
            prim_path=prim_path,
            gelpad_prim_path=gelpad_prim_path,
            gelpad_attachment_body_name=gelpad_attachment_body_name,
            gelpad_attachment_prim_path=gelpad_attachment_prim_path,
            name=name,
            data_type=data_type,
        )
    elif sensor_type == "gf225":
        return create_gf225_cfg(
            prim_path=prim_path,
            gelpad_prim_path=gelpad_prim_path,
            gelpad_attachment_body_name=gelpad_attachment_body_name,
            gelpad_attachment_prim_path=gelpad_attachment_prim_path,
            name=name,
            data_type=data_type,
        )
    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")


class VisualTactileSensor:
    def __init__(self, name:str, cfg:TactileCfg, robot, scene: 'UipcInteractiveScene', uipc_sim:'UipcSim'):
        self.cfg = cfg
        self.name = name
        self.scene = scene
        self.robot = robot
        self.uipc_sim = uipc_sim
	
        print("\n" + "=" * 80, flush=True)
        print("[DEBUG tactile cfg]", flush=True)
        print("name:", self.name, flush=True)  
        print("gelpad_cfg.prim_path:", self.cfg.gelpad_cfg.prim_path, flush=True)
        print("attachment body_name:", self.cfg.gelpad_attachment_cfg.body_name, flush=True)
        print("attachment rigid prim:", self.cfg.gelpad_attachment_cfg.isaac_rigid_prim_path, flush=True)
        print("sensor prim_path:", self.cfg.sensor_cfg.prim_path, flush=True)
        print("=" * 80 + "\n", flush=True)

        self.gelpad = UipcObject(self.cfg.gelpad_cfg, self.uipc_sim)
        self.attachment = UipcIsaacAttachments(
            self.cfg.gelpad_attachment_cfg, self.gelpad, self.robot
        )
        self.sensor = GelSightSensor(self.cfg.sensor_cfg, self.gelpad)

        # XenseWS ROI/baseline delta state.
        # The baseline is captured explicitly at the start of a grasp action
        # or lazily from the first ROI metric after reset. It is NOT a rolling
        # previous-frame baseline, so contact delta will not be cancelled out.
        self.depth_baseline_metric = None
        self.last_depth_metric = None
        self.last_depth_baseline_metric = None
        self.last_depth_delta_metric = None
        self.last_depth_roi_tag = None
        self.depth_debug_count = 0

        # self.scene.sensors[f'tactile_{self.cfg.name}'] = self.sensor

    def _get_marker_shift_px(self):
        sensor_type = getattr(
            self.cfg.sensor_cfg.marker_motion_sim_cfg,
            "sensor_type",
            None
        )
        if sensor_type != "xensews":
            return (0, 0)
        return XENSEWS_MARKER_SHIFT_PX.get(self.name, (0, 0))

    def _shift_image_xy(self, img, dx: int, dy: int):
        if dx == 0 and dy == 0:
            return img

        out = torch.zeros_like(img)

        if img.ndim == 2:
            h, w = img.shape
            dst_y0, dst_y1 = max(dy, 0), min(h + dy, h)
            src_y0, src_y1 = max(-dy, 0), min(h - dy, h)
            dst_x0, dst_x1 = max(dx, 0), min(w + dx, w)
            src_x0, src_x1 = max(-dx, 0), min(w - dx, w)
            if dst_y1 > dst_y0 and dst_x1 > dst_x0:
                out[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
            return out

        if img.ndim == 3:
            # HWC layout: last dimension is channel.
            if img.shape[-1] in (1, 3, 4):
                h, w = img.shape[0], img.shape[1]
                dst_y0, dst_y1 = max(dy, 0), min(h + dy, h)
                src_y0, src_y1 = max(-dy, 0), min(h - dy, h)
                dst_x0, dst_x1 = max(dx, 0), min(w + dx, w)
                src_x0, src_x1 = max(-dx, 0), min(w - dx, w)
                if dst_y1 > dst_y0 and dst_x1 > dst_x0:
                    out[dst_y0:dst_y1, dst_x0:dst_x1, :] = img[src_y0:src_y1, src_x0:src_x1, :]
                return out

            # CHW layout.
            h, w = img.shape[-2], img.shape[-1]
            dst_y0, dst_y1 = max(dy, 0), min(h + dy, h)
            src_y0, src_y1 = max(-dy, 0), min(h - dy, h)
            dst_x0, dst_x1 = max(dx, 0), min(w + dx, w)
            src_x0, src_x1 = max(-dx, 0), min(w - dx, w)
            if dst_y1 > dst_y0 and dst_x1 > dst_x0:
                out[:, dst_y0:dst_y1, dst_x0:dst_x1] = img[:, src_y0:src_y1, src_x0:src_x1]
            return out

        return img
    
    def setup(self):
        self.device = self.uipc_sim.cfg.device
        init_pts = self.gelpad._data.nodal_pos_w[self.attachment.attachment_points_idx].cpu().numpy()
        init_world_trans = self.gelpad.init_world_transform.cpu().numpy()
        self.origin_pts = (init_pts - init_world_trans[:3, 3]) @ (init_world_trans[:3, :3].T).T
        attach_pts = self.attachment.attachment_offsets
        init_trans = estimate_rigid_transform(self.origin_pts, attach_pts)
        self.attach_to_init = np.linalg.inv(init_trans)
        self.attach_to_init = torch.tensor(self.attach_to_init, dtype=torch.float64, device=self.device)

        self.sensor.marker_motion_simulator.marker_motion_sim.init_vertices()

    def get_attach_pose(self):
        if type(self.attachment.isaaclab_rigid_object) is Articulation:
            # this only works when rigid body is an articulation
            # self.attachment.isaaclab_rigid_object._physics_sim_view.update_articulations_kinematic()
            # read data from simulation
            poses = self.attachment.isaaclab_rigid_object._root_physx_view.get_link_transforms().clone()
            poses[..., 3:7] = math_utils.convert_quat(poses[..., 3:7], to="wxyz")
            pose = poses[:, self.attachment.rigid_body_id, 0:7].clone()
        elif type(self.attachment.isaaclab_rigid_object) is RigidObject:
            # only works with rigid body
            pose = self.attachment.isaaclab_rigid_object._root_physx_view.root_state_w.view(-1, 1, 13)
            pose = pose[:, self.attachment.rigid_body_id, 0:7].clone()
        else:
            raise RuntimeError("Need an Articulation or a RigidBody object for the Isaac X UIPC attachment.")
        return Pose.from_list(pose.flatten().tolist())

    def get_init_pts(self):
        curr_attach_pose = self.get_attach_pose()
        trans_to_attach = np.linalg.inv(curr_attach_pose.to_transformation_matrix())
        trans_to_attach = torch.tensor(trans_to_attach, dtype=torch.float64, device=self.device)
        trans_to_init = self.attach_to_init @ trans_to_attach
        return self.gelpad.data.nodal_pos_w @ trans_to_init[:3, :3].T + trans_to_init[:3, 3]
 
    def update(self, dt, force_recompute=False):
        self.gelpad.update(dt=dt)
        self.sensor.update(dt=dt, force_recompute=force_recompute)
    
    def set_debug_vis(self):
        if not self.sensor.cfg.debug_vis:
            return 
        for data_type in ['marker_motion']:
            self.sensor._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)
    
    def get_observations(self, data_types: list[str] = None):
        obs = {}
        if data_types is None:
            data_types = ['rgb', 'rgb_marker', 'depth', 'points', 'pose', 'flow']
        for data_type in data_types:
            if data_type == 'rgb':
                obs['rgb'] = self.sensor.data.output['tactile_rgb'].squeeze(0)
            elif data_type == 'rgb_marker':
                rgb_marker = self.sensor.data.output['marker_rgb'].squeeze(0)
                dx, dy = self._get_marker_shift_px()
                obs['rgb_marker'] = self._shift_image_xy(rgb_marker, dx, dy)
            elif data_type == 'depth':
                obs['depth'] = self.sensor.data.output['height_map'].squeeze(0)
            elif data_type == 'marker':
                marker = self.sensor.data.output['marker_motion'].squeeze(0).clone()
                dx, dy = self._get_marker_shift_px()
                if dx != 0 or dy != 0:
                    marker[..., 0] = marker[..., 0] + dx
                    marker[..., 1] = marker[..., 1] + dy
                obs['marker'] = marker
            elif data_type == 'points':
                obs['points'] = self.get_init_pts()
            elif data_type == 'pose':
                obs['pose'] = self.get_attach_pose().totensor()
        return obs
    
    def _reset_idx(self):
        self.init_pose_mat = self.get_attach_pose().to_transformation_matrix()
        self.depth_baseline_metric = None
        self.last_depth_metric = None
        self.last_depth_baseline_metric = None
        self.last_depth_delta_metric = None
        self.last_depth_roi_tag = None
        self.depth_debug_count = 0
        # self.gelpad.write_vertex_positions_to_sim(vertex_positions=self.gelpad.init_vertex_pos)
    
    def render_depth_debug(self, depth_img):
        import cv2
        if isinstance(depth_img, torch.Tensor):
            depth_img = depth_img.cpu().numpy()
        depth_img = depth_img.astype(np.float32)
        d_min, d_max = np.min(depth_img), np.max(depth_img)
        depth_img = ((depth_img - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
        cv2.imwrite(f"debug_{self.name}_depth.png", depth_vis)
    
    # def get_min_depth(self):
    #     return torch.min(self.sensor.data.output[ 'height_map']).item()
    def _get_xensews_roi_box(self, h: int, w: int):
        """Return the ROI box used by this XenseWS sensor.

        The default keeps your existing split ROI:
        - left_tactile  -> right_center  (x: 40%~90%)
        - right_tactile -> left_center   (x: 10%~60%)

        Runtime overrides, without editing code:
            XENSEWS_ROI=center60
            XENSEWS_LEFT_ROI=right_center
            XENSEWS_RIGHT_ROI=left_center
        """
        boxes = {
            "center60": (int(h * 0.20), int(h * 0.80), int(w * 0.20), int(w * 0.80)),
            "center40": (int(h * 0.30), int(h * 0.70), int(w * 0.30), int(w * 0.70)),
            "center30": (int(h * 0.35), int(h * 0.65), int(w * 0.35), int(w * 0.65)),
            "left_center": (int(h * 0.25), int(h * 0.75), int(w * 0.10), int(w * 0.60)),
            "right_center": (int(h * 0.25), int(h * 0.75), int(w * 0.40), int(w * 0.90)),
            "upper_center": (int(h * 0.10), int(h * 0.60), int(w * 0.25), int(w * 0.75)),
            "lower_center": (int(h * 0.40), int(h * 0.90), int(w * 0.25), int(w * 0.75)),
        }

        default_roi = os.environ.get("XENSEWS_ROI", "").strip()
        if self.name == "left_tactile":
            roi_tag = os.environ.get("XENSEWS_LEFT_ROI", default_roi or "right_center").strip()
        elif self.name == "right_tactile":
            roi_tag = os.environ.get("XENSEWS_RIGHT_ROI", default_roi or "left_center").strip()
        else:
            roi_tag = default_roi or "center40"

        if roi_tag not in boxes:
            raise ValueError(
                f"Unknown XenseWS ROI tag: {roi_tag}. "
                "Use center60/center40/center30/left_center/right_center/"
                "upper_center/lower_center."
            )
        return roi_tag, boxes[roi_tag]

    def _compute_depth_metric(self):
        """Compute the tactile depth metric used by adaptive grasp.

        For XenseWS, this is the ROI mean you were already using. For other
        sensors, this preserves the original global min behavior.
        """
        depth = self.sensor.data.output['height_map'].squeeze(0).float()

        sensor_type = getattr(
            self.cfg.sensor_cfg.marker_motion_sim_cfg,
            "sensor_type",
            None,
        )

        if sensor_type == "xensews":
            h, w = depth.shape[-2], depth.shape[-1]
            roi_tag, (y0, y1, x0, x1) = self._get_xensews_roi_box(h, w)
            roi = depth[y0:y1, x0:x1].flatten()
            metric = torch.mean(roi)

            self.last_depth_roi_tag = roi_tag
            self.last_depth_metric = float(metric.item())

            debug_roi = os.environ.get("DEBUG_TACTILE_ROI", "0") == "1"
            debug_every = int(os.environ.get("DEBUG_TACTILE_ROI_EVERY", "20"))
            do_print = debug_roi and (
                self.depth_debug_count < 5
                or (debug_every > 0 and self.depth_debug_count % debug_every == 0)
            )
            if do_print:
                print(
                    "[DBG_XENSEWS_DEPTH_METRIC]",
                    "name=", self.name,
                    "roi_tag=", roi_tag,
                    "shape=", tuple(depth.shape),
                    "box_y=", (y0, y1),
                    "box_x=", (x0, x1),
                    "global_min=", torch.min(depth).item(),
                    "roi_min=", torch.min(roi).item(),
                    "roi_p05=", torch.quantile(roi, 0.05).item(),
                    "roi_mean=", metric.item(),
                    "roi_max=", torch.max(roi).item(),
                    flush=True,
                )
            self.depth_debug_count += 1
            return metric

        metric = torch.min(depth)
        self.last_depth_metric = float(metric.item())
        return metric

    def reset_depth_baseline(self):
        """Capture a fixed baseline for baseline-relative delta.

        This should be called right before adaptive close starts. If it is not
        called, get_depth_delta() will lazily capture the first metric as the
        baseline after reset.
        """
        metric = self._compute_depth_metric().detach().clone()
        self.depth_baseline_metric = metric
        self.last_depth_baseline_metric = float(metric.item())
        self.last_depth_delta_metric = 0.0
        return metric

    def get_depth_delta(self):
        """Return baseline - current metric.

        Positive delta means the ROI depth moved closer / compressed relative
        to the fixed baseline.
        """
        metric = self._compute_depth_metric()
        if self.depth_baseline_metric is None:
            self.depth_baseline_metric = metric.detach().clone()

        baseline = self.depth_baseline_metric.to(metric.device)
        delta = baseline - metric

        self.last_depth_baseline_metric = float(baseline.item())
        self.last_depth_delta_metric = float(delta.item())

        debug_delta = os.environ.get("DEBUG_TACTILE_DELTA", "0") == "1"
        debug_every = int(os.environ.get("DEBUG_TACTILE_DELTA_EVERY", "20"))
        do_print = debug_delta and (
            self.depth_debug_count < 5
            or (debug_every > 0 and self.depth_debug_count % debug_every == 0)
        )
        if do_print:
            print(
                "[DBG_XENSEWS_ROI_DELTA]",
                "name=", self.name,
                "roi_tag=", self.last_depth_roi_tag,
                "baseline=", self.last_depth_baseline_metric,
                "current=", self.last_depth_metric,
                "delta=", self.last_depth_delta_metric,
                flush=True,
            )
        return delta

    def get_min_depth(self):
        # Kept for compatibility with the original base task. For XenseWS this
        # still returns your ROI mean, not global min.
        return self._compute_depth_metric().item()

class TactileManager:
    def __init__(self, cfg_list: list[TactileCfg], task:'BaseTask'):
        self.task = task
        self.scene = task.scene
        self.uipc_sim = task.uipc_sim
        self.robot = task._robot_manager.robot
        
        self.tactiles = {
            cfg.name: VisualTactileSensor(
                cfg.name, cfg, self.robot, self.scene, self.uipc_sim
            ) for cfg in cfg_list
        }

    def update(self, dt, force_recompute=False):
        for tact in self.tactiles.values():
            tact.update(dt=dt, force_recompute=force_recompute)
 
    def set_debug_vis(self, debug_vis):
        if not debug_vis: return
        for tact in self.tactiles.values():
            tact.set_debug_vis()

    def get_observations(self, data_types: list[str] = None):
        obs = {}
        for name, tact in self.tactiles.items():
            obs[name] = tact.get_observations(data_types)
        return obs

    def get_min_depth(self):
        self.task._update_render()
        depth = []
        for tact in self.tactiles.values():
            depth.append(tact.get_min_depth())
        return torch.tensor(depth, dtype=torch.float32, device=self.task.device)

    def reset_depth_baseline(self):
        self.task._update_render()
        baselines = []
        for tact in self.tactiles.values():
            if hasattr(tact, "reset_depth_baseline"):
                baselines.append(tact.reset_depth_baseline().item())
            else:
                baselines.append(tact.get_min_depth())
        return torch.tensor(baselines, dtype=torch.float32, device=self.task.device)

    def get_depth_delta(self):
        self.task._update_render()
        deltas = []
        for tact in self.tactiles.values():
            if hasattr(tact, "get_depth_delta"):
                deltas.append(tact.get_depth_delta().item())
            else:
                deltas.append(0.0)
        return torch.tensor(deltas, dtype=torch.float32, device=self.task.device)

    def get_depth_debug_state(self):
        state = {}
        for name, tact in self.tactiles.items():
            state[name] = {
                "roi": getattr(tact, "last_depth_roi_tag", None),
                "baseline": getattr(tact, "last_depth_baseline_metric", None),
                "metric": getattr(tact, "last_depth_metric", None),
                "delta": getattr(tact, "last_depth_delta_metric", None),
            }
        return state

    def _reset_idx(self):
        for tact in self.tactiles.values():
            tact._reset_idx()

    def setup(self):
        for tact in self.tactiles.values():
            tact.setup()

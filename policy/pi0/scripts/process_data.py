import sys
from pathlib import Path
sys.path.append(str((Path(__file__).parent / '../..' ).resolve()))

from _base_data_preprocessor import *

import shutil
import h5py
import numpy as np
import cv2
import argparse
import json
from typing import Literal

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

import compute_norm_stats as _compute_norm_stats

class Pi0DataPreprocessor(BaseDataPreprocessor):
    MOTORS = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_finger"
    ]

    LEROBOT_CAMERAS = ("cam_head", "cam_wrist", "tactile_left", "tactile_right")

    def __init__(self, task_name, task_config):
        super().__init__(task_name, task_config)
    
    def images_encoding(self, imgs):
        encode_data = []
        padded_data = []
        max_len = 0
        for i in range(len(imgs)):
            success, encoded_image = cv2.imencode(".jpg", imgs[i])
            jpeg_data = encoded_image.tobytes()
            encode_data.append(jpeg_data)
            max_len = max(max_len, len(jpeg_data))
        # padding
        for i in range(len(imgs)):
            padded_data.append(encode_data[i].ljust(max_len, b"\0"))
        return encode_data, max_len

    def export_to_lerobot(
        self,
        repo_id: str,
        *,
        root: str | Path | None = None,
        task_prompt: str | None = None,
        robot_type: str = "franka",
        fps: int = 50,
        mode: Literal["image", "video"] = "image",
        use_videos: bool = False,
        overwrite: bool = True,
    ) -> dict:
        assert self._data is not None, "Data not loaded. Please call load_data() before export_to_lerobot()."

        state_all = np.asarray(self._data["embodiment/joint_state"], dtype=np.float32)
        action_all = np.asarray(self._data["embodiment/joint_action"], dtype=np.float32)
        if state_all.shape[-1] != len(self.MOTORS) or action_all.shape[-1] != len(self.MOTORS):
            raise ValueError(
                "Pi0 Franka config expects 8-D state/action. "
                f"Got state={state_all.shape[-1]}, action={action_all.shape[-1]}."
            )

        dataset_root = self._lerobot_root(repo_id, root)
        if dataset_root.exists():
            if not overwrite:
                raise FileExistsError(f"LeRobot dataset already exists: {dataset_root}")
            shutil.rmtree(dataset_root)

        dataset = self._create_lerobot_dataset(
            repo_id=repo_id,
            root=dataset_root,
            robot_type=robot_type,
            fps=fps,
            mode=mode,
            use_videos=use_videos,
        )
        images_all = self._lerobot_image_arrays()

        start_idx = 0
        for ep_idx in tqdm(range(len(self.selected_raw_hdf5_paths)), desc="Writing LeRobot episodes"):
            end_idx = self._data["episode_ends"][ep_idx]
            instruction = self._sample_instruction(task_prompt)
            for frame_idx in range(start_idx, end_idx):
                frame = {
                    "observation.state": state_all[frame_idx],
                    "action": action_all[frame_idx],
                    "task": instruction,
                }
                for camera in self.LEROBOT_CAMERAS:
                    frame[f"observation.images.{camera}"] = images_all[camera][frame_idx]
                dataset.add_frame(frame)
            dataset.save_episode()
            start_idx = end_idx

        self.save_root_path = dataset.root
        metadata = {
            "repo_id": repo_id,
            "dataset_dir": str(dataset.root),
            "num_episodes": len(self.selected_raw_hdf5_paths),
            "camera_names": list(self.LEROBOT_CAMERAS),
            "episode_map": {i: str(path) for i, path in enumerate(self.selected_raw_hdf5_paths)},
        }
        with open(dataset.root / "source_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
        return metadata

    def run(
        self,
        repo_id: str,
        root: str | Path | None = None,
        task_prompt: str | None = None,
        visual_cameras=("head",),
        tactile_cameras=("left", "right"),
        downsample_factor=1,
        episode_num=50,
        random_select=False,
    ) -> dict:
        self.load_data(
            visual_cameras=visual_cameras,
            tactile_cameras=tactile_cameras,
            downsample_factor=downsample_factor,
            episode_num=episode_num,
            random_select=random_select,
        )
        return self.export_to_lerobot(repo_id, root=root, task_prompt=task_prompt)

    def _create_lerobot_dataset(
        self,
        repo_id: str,
        root: Path,
        robot_type: str,
        fps: int,
        mode: Literal["image", "video"],
        use_videos: bool,
    ) -> LeRobotDataset:
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(self.MOTORS),),
                "names": [self.MOTORS],
            },
            "action": {
                "dtype": "float32",
                "shape": (len(self.MOTORS),),
                "names": [self.MOTORS],
            },
        }
        image_shape = self._image_shape()
        for camera in self.LEROBOT_CAMERAS:
            features[f"observation.images.{camera}"] = {
                "dtype": mode,
                "shape": (3, image_shape[0], image_shape[1]),
                "names": ["channels", "height", "width"],
            }

        return LeRobotDataset.create(
            repo_id=repo_id,
            root=root,
            fps=fps,
            robot_type=robot_type,
            features=features,
            use_videos=use_videos,
        )
    
    def joint_transform(self, joints):
        return joints[:, :len(self.MOTORS)]

    def _lerobot_root(self, repo_id: str, root: str | Path | None) -> Path:
        if root is None:
            return Path(HF_LEROBOT_HOME) / repo_id
        return Path(root).expanduser().resolve()

    def _image_shape(self) -> tuple[int, int]:
        images = self._lerobot_image_arrays()
        first = images[self.LEROBOT_CAMERAS[0]]
        return int(first.shape[1]), int(first.shape[2])

    def _lerobot_image_arrays(self) -> dict[str, np.ndarray]:
        source_map = self._available_image_sources()
        cam_head = source_map['visual/head']
        cam_wrist = source_map['visual/wrist']
        tactile_left = source_map['tactile/left']
        tactile_right = source_map['tactile/right']
        return {
            "cam_head": cam_head,
            "cam_wrist": cam_wrist,
            "tactile_left": tactile_left,
            "tactile_right": tactile_right,
        }

    def _available_image_sources(self) -> dict[str, np.ndarray]:
        sources = {}
        for cam in self.visual_cameras:
            key = f"visual/{cam}"
            raw_key = self.camera_key_map[key]["raw_key"]
            sources[key] = np.asarray(self._data[raw_key], dtype=np.uint8)
        for cam in self.tactile_cameras:
            key = f"tactile/{cam}"
            raw_key = self.camera_key_map[key]["raw_key"]
            sources[key] = np.asarray(self._data[raw_key], dtype=np.uint8)
        return sources

    def _sample_instruction(self, task_prompt: str | None) -> str:
        if task_prompt:
            return task_prompt

        for path in (self.raw_root_path / "instructions.json", self.raw_root_path.parent / "instructions.json"):
            if path.exists():
                with open(path, "r") as f:
                    instructions = json.load(f).get("instructions", [])
                if instructions:
                    return str(np.random.choice(instructions))

        return self.task_name.replace("_", " ")


def main(
    task_name,
    task_config,
    expert_data_num,
    repo_id=None,
    task_prompt=None,
    config_names=None,
    max_norm_frames=None,
):
    repo_id = repo_id or f"univtac/{task_name}-{task_config}-{expert_data_num}"
    
    task_settings_path = Path(__file__).parent / '../../task_settings.json'
    if task_settings_path.exists():
        with open(task_settings_path, 'r') as f:
            task_settings = json.load(f)
    else:
        task_settings = {}
        print('Warning: task_settings.json not found. Using default settings for all tasks.')
    
    camera_type = task_settings.get(task_name, {}).get('camera_type', 'head')
    # if camera_type == 'all':
    #     visual_cameras = ['head', 'wrist']
    # else:
    #     visual_cameras = [camera_type]
    visual_cameras = ['head', 'wrist']
    tactile_cameras = ['left', 'right']
    downsample_factor = task_settings.get(task_name, {}).get('downsample_factor', 1)
 
    processor = Pi0DataPreprocessor(task_name, task_config)
    metadata = processor.run(
        repo_id=repo_id,
        task_prompt=task_prompt,
        visual_cameras=visual_cameras,
        tactile_cameras=tactile_cameras,
        downsample_factor=downsample_factor,
        episode_num=expert_data_num,
        random_select=False,
    )
    print(f"LeRobot repo_id: {metadata['repo_id']}")
    print(f"LeRobot dataset_dir: {metadata['dataset_dir']}")

    for config_name in config_names or ():
        stats_dir = _compute_norm_stats.compute_and_save(
            config_name,
            max_frames=max_norm_frames,
            repo_id=repo_id,
            asset_id=repo_id,
        )
        print(f"Norm stats for {config_name}: {stats_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        default="beat_block_hammer",
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("setting", type=str)
    parser.add_argument(
        "expert_data_num",
        type=int,
        default=50,
        help="Number of episodes to process (e.g., 50)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="LeRobot repo id. Defaults to local/<task_name>-<setting>-<expert_data_num>.",
    )
    parser.add_argument(
        "--task-prompt",
        type=str,
        default=None,
        help="Language instruction stored in the LeRobot task field. Defaults to instructions.json or task name.",
    )
    parser.add_argument(
        "--config-name",
        action="append",
        default=[],
        help=(
            "Training config name to compute norm stats for after export. "
            "Can be passed multiple times. Stats are saved to assets/<config-name>/<repo-id>."
        ),
    )
    parser.add_argument(
        "--max-norm-frames",
        type=int,
        default=None,
        help="Optional max frame count passed to compute_norm_stats.",
    )
    args = parser.parse_args()

    task_name = args.task_name
    setting = args.setting
    expert_data_num = args.expert_data_num

    main(
        task_name,
        setting,
        expert_data_num,
        repo_id=args.repo_id,
        task_prompt=args.task_prompt,
        config_names=args.config_name,
        max_norm_frames=args.max_norm_frames,
    )

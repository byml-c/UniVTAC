import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from .._base_policy import BasePolicy

import numpy as np
import torch
import dill
import os, sys
from torchvision import transforms

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from pi_model import *

class Policy(BasePolicy):
    def __init__(self, args):
        self.train_config_name = args.get('train_config', 'train_config')
        self.model_name = args.get('model_name', 'model')
        self.checkpoint_id = args.get('checkpoint_id', 'latest')
        self.pi0_step = args.get('pi0_step', 5)
        
        self.visual_cameras = args.get('visual_cameras', ['head'])
        self.tactile_cameras = args.get('tactile_cameras', ['left', 'right'])
        
        self.model = PI0(
            self.train_config_name,
            self.model_name,
            self.checkpoint_id,
            self.pi0_step
        )

    # Encode observation for the model
    def encode_obs(self, observation):
        """
        Encode UniVTAC observation to Pi0 input format
        
        Input (TacArena):
            observation = {
                "observation": {"head": {"rgb": torch.Tensor([H, W, 3])}},  # HWC, 0-255
                "joint_action": torch.Tensor([9])  # [arm(7), gripper(1), extra(1)]
            }
            camera: 480x270
            tactile: 320x240
        
        Output (ACT):
            obs = {
                "qpos": torch.Tensor([8])  # [arm(7), gripper(1)]
                "cam_head": torch.Tensor([3, 256, 256]),  # CHW, 0-1
                "tac_left": torch.Tensor([3, 256, 256]),  # CHW, 0-1
                "tac_right": torch.Tensor([3, 256, 256]),  # CHW, 0-1
            }
        """
        def preprocess_image(img):
            img = transforms.Resize((256, 256))(img.permute(2, 0, 1))  # HWC -> CHW
            img = img / 255.0  # Normalize to [0, 1]

        input_images = {}
        for cam in self.visual_cameras:
            input_images[f'cam_{cam}'] = preprocess_image(
                observation["observation"][f"{cam}"]["rgb"])
        for cam in self.tactile_cameras:
            input_images[f'tac_{cam}'] = preprocess_image(
                observation["observation"][f"{cam}_tactile"]["rgb"])

        input_state = observation["joint_action"]

        return input_images, input_state


    def eval(self, task, observation):
        if self.model.observation_window is None:
            instruction = task.instruction
            self.model.set_language(instruction)

        input_images, input_state = self.encode_obs(observation)
        self.model.update_observation_window(
            input_images, input_state)

        # ======== Get Action ========

        actions = self.model.get_action()[:self.model.pi0_step]
        for action in actions:
            exec_succ, eval_succ = task.take_action(action)
            observation = task._get_observations()
            input_images, input_state = self.encode_obs(observation)
            self.model.update_observation_window(input_images, input_state)
        # ============================

    def reset(self):
        self.model.reset_obsrvationwindows()

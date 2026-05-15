import secrets
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))

from .._base_policy import BasePolicy
from .pi0_ipc import recv_msg, send_msg


class _RemotePI0:
    def __init__(self, args):
        self.pi0_dir = Path(__file__).resolve().parent
        self.repo_root = self.pi0_dir.parents[1]
        self.python = Path(args.get("pi0_python", self.pi0_dir / ".venv" / "bin" / "python"))
        self.startup_timeout = float(args.get("pi0_startup_timeout", 300))
        self.request_timeout = float(args.get("pi0_request_timeout", 300))
        self.pi0_step = args["pi0_step"]
        self._request_id = 0
        self._closed = False

        if not self.python.exists():
            raise FileNotFoundError(f"pi0 python not found: {self.python}")

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(1)
        self._server.settimeout(self.startup_timeout)
        host, port = self._server.getsockname()
        self._authkey = secrets.token_hex(16)

        worker = self.pi0_dir / "pi0_inference_worker.py"
        self._process = subprocess.Popen(
            [
                str(self.python),
                str(worker),
                "--host",
                host,
                "--port",
                str(port),
                "--authkey",
                self._authkey,
            ],
            cwd=str(self.repo_root),
        )

        try:
            self._sock, _ = self._server.accept()
            hello = recv_msg(self._sock)
            if hello.get("authkey") != self._authkey:
                raise RuntimeError("pi0 worker authentication failed")
            self._sock.settimeout(self.request_timeout)
            self._request(
                "init",
                config={
                    "train_config_name": args["train_config_name"],
                    "model_name": args["model_name"],
                    "checkpoint_id": args["checkpoint_id"],
                    "pi0_step": args["pi0_step"],
                },
            )
        except Exception:
            self.close()
            raise
        finally:
            self._server.close()

    def _request(self, command, **payload):
        if self._closed:
            raise RuntimeError("pi0 worker is already closed")
        self._request_id += 1
        request_id = self._request_id
        send_msg(self._sock, {"request_id": request_id, "command": command, **payload})
        response = recv_msg(self._sock)
        if response.get("request_id") != request_id:
            raise RuntimeError(f"pi0 worker returned mismatched response: {response}")
        if response.get("type") == "error":
            raise RuntimeError(
                "pi0 worker error:\n"
                f"{response.get('error')}\n"
                f"{response.get('traceback')}"
            )
        return response

    def set_language(self, instruction):
        self._request("set_language", instruction=instruction)

    def get_action(self, images, state):
        response = self._request("infer", images=images, state=state)
        return response["actions"]

    def reset(self):
        self._request("reset")

    def close(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        try:
            if hasattr(self, "_sock"):
                send_msg(self._sock, {"request_id": -1, "command": "close"})
                recv_msg(self._sock)
        except Exception:
            pass
        try:
            if hasattr(self, "_sock"):
                self._sock.close()
        except Exception:
            pass
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()

class Policy(BasePolicy):
    def __init__(self, args):
        self.train_config_name = args.get('train_config_base_name', 'pi0_fast_franka_univtac_lora')
        self.task_name = args.get('task_name', 'demo')
        self.task_config_name = args.get('task_config_name', args.get('task_config', 'demo'))
        self.train_config_name = f"{self.train_config_name}_{self.task_name}-{self.task_config_name}-50"
        self.model_name = args.get('model_name', 'model')
        self.checkpoint_id = args.get('checkpoint_id', 'latest')
        self.pi0_step = args.get('pi0_step', 5)
        
        self.visual_cameras = args.get('visual_cameras', ['head'])
        self.tactile_cameras = args.get('tactile_cameras', ['left', 'right'])
        self._language_set = False

        self.model = _RemotePI0({
            **args,
            "train_config_name": self.train_config_name,
            "model_name": self.model_name,
            "checkpoint_id": self.checkpoint_id,
            "pi0_step": self.pi0_step,
        })

    def _to_numpy(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _encode_image(self, image):
        image = self._to_numpy(image)
        if image.ndim != 3:
            raise ValueError(f"expected image with 3 dims, got shape {image.shape}")
        if image.shape[-1] in (1, 3, 4):
            image = np.transpose(image[..., :3], (2, 0, 1))
        elif image.shape[0] == 4:
            image = image[:3]
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating):
                max_value = float(np.nanmax(image)) if image.size else 1.0
                if max_value <= 1.0:
                    image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image)

    def _get_tactile_image(self, observation, cam):
        candidates = [
            ("observation", f"{cam}_tactile", "rgb"),
            ("tactile", f"{cam}_tactile", "rgb"),
            ("tactile", f"{cam}_tactile", "rgb_marker"),
        ]
        for root, camera, key in candidates:
            if root in observation and camera in observation[root] and key in observation[root][camera]:
                return observation[root][camera][key]
        raise KeyError(f"cannot find tactile image for camera '{cam}'")

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
        
        Output (pi0 worker):
            input_images = {
                "cam_head": np.ndarray([3, H, W], dtype=np.uint8),
                "tactile_left": np.ndarray([3, H, W], dtype=np.uint8),
                "tactile_right": np.ndarray([3, H, W], dtype=np.uint8),
            }
            input_state = np.ndarray([8], dtype=np.float32)
        """
        input_images = {}
        for cam in self.visual_cameras:
            input_images[f'cam_{cam}'] = self._encode_image(
                observation["observation"][f"{cam}"]["rgb"])
        for cam in self.tactile_cameras:
            input_images[f'tactile_{cam}'] = self._encode_image(
                self._get_tactile_image(observation, cam))

        if "joint_action" in observation:
            input_state = observation["joint_action"]
        else:
            input_state = observation["embodiment"]["joint"]
        input_state = np.asarray(self._to_numpy(input_state)[:8], dtype=np.float32)

        return input_images, input_state


    def eval(self, task, observation):
        if not self._language_set:
            instruction = task.instruction
            self.model.set_language(instruction)
            self._language_set = True

        input_images, input_state = self.encode_obs(observation)

        # ======== Get Action ========

        actions = self.model.get_action(input_images, input_state)[:self.pi0_step]
        for action in actions:
            action = torch.from_numpy(np.asarray(action)).to(task.device).float()
            exec_succ, eval_succ = task.take_action(action)
        # ============================

    def reset(self):
        self._language_set = False
        self.model.reset()

    def close(self):
        self.model.close()

import argparse
import os
import socket
import sys
import traceback
from pathlib import Path

from pi0_ipc import recv_msg, send_msg


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _connect(host: str, port: int, authkey: str) -> socket.socket:
    sock = socket.create_connection((host, port), timeout=30)
    sock.settimeout(None)
    send_msg(sock, {"type": "hello", "authkey": authkey, "pid": os.getpid()})
    return sock


def _send_error(sock: socket.socket, request_id, exc: BaseException) -> None:
    send_msg(
        sock,
        {
            "type": "error",
            "request_id": request_id,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="pi0 inference worker")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--authkey", required=True)
    args = parser.parse_args()

    root = _repo_root()
    pi0_dir = Path(__file__).resolve().parent
    os.chdir(root)
    sys.path.insert(0, str(pi0_dir / "src"))
    sys.path.insert(0, str(pi0_dir))
    sys.path.insert(0, str(root))

    sock = _connect(args.host, args.port, args.authkey)
    model = None

    while True:
        msg = recv_msg(sock)
        request_id = msg.get("request_id")
        command = msg.get("command")

        try:
            if command == "init":
                from pi_model import PI0

                config = msg["config"]
                model = PI0(
                    config["train_config_name"],
                    config["model_name"],
                    config["checkpoint_id"],
                    config["pi0_step"],
                )
                send_msg(sock, {"type": "ok", "request_id": request_id})

            elif command == "set_language":
                assert model is not None, "worker is not initialized"
                model.set_language(msg["instruction"])
                send_msg(sock, {"type": "ok", "request_id": request_id})

            elif command == "infer":
                assert model is not None, "worker is not initialized"
                model.update_observation_window(msg["images"], msg["state"])
                actions = model.get_action()[: model.pi0_step]
                send_msg(sock, {"type": "ok", "request_id": request_id, "actions": actions})

            elif command == "reset":
                if model is not None:
                    model.reset_obsrvationwindows()
                send_msg(sock, {"type": "ok", "request_id": request_id})

            elif command == "close":
                send_msg(sock, {"type": "ok", "request_id": request_id})
                break

            else:
                raise ValueError(f"unknown command: {command}")

        except BaseException as exc:
            _send_error(sock, request_id, exc)

    sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

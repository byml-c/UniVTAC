import argparse
import os
import socket
import sys
import traceback
from pathlib import Path

from pi0_ipc import recv_msg, send_msg


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _setup_paths() -> None:
    root = _repo_root()
    pi0_dir = Path(__file__).resolve().parent
    os.chdir(root)
    sys.path.insert(0, str(pi0_dir / "src"))
    sys.path.insert(0, str(pi0_dir))
    sys.path.insert(0, str(root))


def _full_train_config_name(args) -> str:
    if args.train_config_name:
        return args.train_config_name
    if not args.task_name or not args.task_config:
        raise ValueError(
            "provide --train-config-name, or provide both --task-name and --task-config"
        )
    return f"{args.train_config_base_name}_{args.task_name}-{args.task_config}-{args.expert_data_num}"


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


def _handle_client(sock: socket.socket, model, config: dict) -> bool:
    hello = recv_msg(sock)
    if hello.get("authkey", "") != config.get("authkey", ""):
        send_msg(sock, {"type": "error", "error": "authentication failed"})
        return False
    send_msg(sock, {"type": "ok", "service": "pi0"})

    while True:
        msg = recv_msg(sock)
        request_id = msg.get("request_id")
        command = msg.get("command")

        try:
            if command == "describe":
                send_msg(
                    sock,
                    {
                        "type": "ok",
                        "request_id": request_id,
                        "config": {
                            "train_config_name": config["train_config_name"],
                            "model_name": config["model_name"],
                            "checkpoint_id": config["checkpoint_id"],
                            "pi0_step": config["pi0_step"],
                        },
                    },
                )

            elif command == "set_language":
                model.set_language(msg["instruction"])
                send_msg(sock, {"type": "ok", "request_id": request_id})

            elif command == "infer":
                model.update_observation_window(msg["images"], msg["state"])
                actions = model.get_action()[: model.pi0_step]
                send_msg(sock, {"type": "ok", "request_id": request_id, "actions": actions})

            elif command == "reset":
                model.reset_obsrvationwindows()
                send_msg(sock, {"type": "ok", "request_id": request_id})

            elif command == "disconnect":
                send_msg(sock, {"type": "ok", "request_id": request_id})
                return False

            elif command == "shutdown":
                send_msg(sock, {"type": "ok", "request_id": request_id})
                return True

            else:
                raise ValueError(f"unknown command: {command}")

        except BaseException as exc:
            _send_error(sock, request_id, exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Long-running pi0 inference service")
    parser.add_argument("--host", default=os.environ.get("PI0_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=os.environ.get("PI0_PORT"))
    parser.add_argument("--authkey", default=os.environ.get("PI0_AUTHKEY", ""))
    parser.add_argument("--train-config-name", default=os.environ.get("PI0_TRAIN_CONFIG_NAME"))
    parser.add_argument(
        "--train-config-base-name",
        default=os.environ.get("PI0_TRAIN_CONFIG_BASE_NAME", "pi0_fast_franka_univtac_lora"),
    )
    parser.add_argument("--task-name", default=os.environ.get("PI0_TASK_NAME"))
    parser.add_argument("--task-config", default=os.environ.get("PI0_TASK_CONFIG"))
    parser.add_argument("--expert-data-num", default=os.environ.get("PI0_EXPERT_DATA_NUM", "50"))
    parser.add_argument("--model-name", default=os.environ.get("PI0_MODEL_NAME", "pi0"))
    parser.add_argument("--checkpoint-id", default=os.environ.get("PI0_CHECKPOINT_ID", "latest"))
    parser.add_argument("--pi0-step", type=int, default=os.environ.get("PI0_STEP", 5))
    args = parser.parse_args()

    if args.port is None:
        raise ValueError("PI0_PORT or --port is required")
    args.pi0_step = int(args.pi0_step)

    _setup_paths()
    from pi_model import PI0

    config = {
        "authkey": args.authkey,
        "train_config_name": _full_train_config_name(args),
        "model_name": args.model_name,
        "checkpoint_id": args.checkpoint_id,
        "pi0_step": args.pi0_step,
    }

    model = PI0(
        config["train_config_name"],
        config["model_name"],
        config["checkpoint_id"],
        config["pi0_step"],
    )

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, int(args.port)))
    server.listen(1)
    print(f"pi0 inference service listening on {args.host}:{args.port}", flush=True)

    try:
        while True:
            client, address = server.accept()
            print(f"pi0 client connected from {address}", flush=True)
            try:
                should_shutdown = _handle_client(client, model, config)
            except ConnectionError:
                should_shutdown = False
            finally:
                client.close()
            if should_shutdown:
                break
    finally:
        server.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

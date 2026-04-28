import os
import sys
import time
import yaml
import json
import torch
import argparse
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Literal

sys.path.append('.')

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Collect data"
)
parser.add_argument(
    "task",
    type=str,
    help="Task file name",
)
parser.add_argument(
    "config",
    type=str,
    help="Config file name",
    default="demo.yml"
)
parser.add_argument(
    "--episode_num",
    type=int,
    default=-1,
)
parser.add_argument(
    "--start_seed",
    type=int,
    default=-1,
)
parser.add_argument(
    "--max_seed",
    type=int,
    default=-1,
)
parser.add_argument(
    "--gpu",
    type=str,
    default=None,
)
parser.add_argument(
    "--visual",
    action="store_true",
    help="Launch Isaac Sim with local GUI window instead of headless/livestream mode.",
)
parser.add_argument(
    "--debug_hold",
    action="store_true",
    help="Initialize the task once, then keep Isaac Sim open without seed/action/collection/reset loop.",
)
parser.add_argument(
    "--debug_step",
    action="store_true",
    help="With --debug_hold, advance physics using task.sim.step(render=True) instead of render-only.",
)
parser.add_argument(
    "--debug_update_obs",
    action="store_true",
    help="With --debug_hold, call task._get_observations() every frame to update tactile/sensor/UIPC buffers without reset/action/save.",
)
parser.add_argument(
    "--debug_hold_sleep",
    type=float,
    default=0.02,
    help="Sleep time in seconds between debug hold frames.",
)
parser.add_argument(
    "--debug_stop_on_error",
    action="store_true",
    help="Stop at the first failed seed instead of continuing to the next seed.",
)
parser.add_argument(
    "--debug_hold_on_error",
    action="store_true",
    help="Keep Isaac Sim window open at the failed scene for inspection after an exception or failed episode.",
)
parser.add_argument(
    "--debug_save_error_video",
    action="store_true",
    help="Try to close/save current cache/video with result='error' or result='fail' before stopping on error.",
)
parser.add_argument(
    "--debug_error_sleep",
    type=float,
    default=0.02,
    help="Sleep time in seconds between frames while holding after an error.",
)

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)

# parse arguments after AppLauncher adds Isaac Sim / Isaac Lab CLI options
args_cli = parser.parse_args()
if args_cli.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args_cli.gpu

# force cameras on for data collection
args_cli.enable_cameras = True
args_cli.num_envs = 1

# optional local GUI mode
if args_cli.visual:
    args_cli.headless = False
    args_cli.livestream = 0

def get_config(file, default_root:Path, type:Literal['yaml', 'json']):
    if type == 'yaml':
        if file.endswith('.yml') or file.endswith('.yaml'):
            file = Path(file)
        else:
            file = default_root / f'{file}.yml'
        with open(file, 'r') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config, file
    else:
        if file.endswith('.json'):
            file = Path(file)
        else:
            file = default_root / f'{file}.json'
        with open(file, 'r') as f:
            config = json.load(f)
        return config, file

task_config, task_config_file = get_config(
    args_cli.config, 
    default_root=Path(__file__).parent.parent / 'task_config', 
    type='yaml'
)

if task_config.get('render_frequency', 1) == 0 and not args_cli.visual:
    # Default behavior for non-visual data collection: enable livestream when rendering is disabled.
    args_cli.livestream = 2

# launch omniverse app, must done before importing anything from omni.isaac
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib
if TYPE_CHECKING:
    from envs._base_task import BaseTask, BaseTaskCfg

log_path = Path('./log')
def log(msg):
    global log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    msg = f"[{time.strftime(r'%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(log_path, 'a') as f:
        f.write(msg + '\n')
    print(msg)

def _write_suc_map(task: 'BaseTask', suc_map):
    """Write suc_map safely so debug stop mode still records the failed seed."""
    try:
        task.save_root.mkdir(parents=True, exist_ok=True)
        with open(task.save_root / 'suc_map.txt', 'w') as f:
            f.write(' '.join([s for s in suc_map]))
    except Exception:
        print("[WARN] Failed to write suc_map.txt")
        traceback.print_exc()


def _safe_clean_cache(task: 'BaseTask', mean_steps: float, result: str):
    """Call task.clean_cache without letting cleanup errors hide the real failure."""
    try:
        task.clean_cache(mean_steps=mean_steps, result=result)
    except Exception:
        print(f"[WARN] task.clean_cache(result={result!r}) failed. Keeping the original error visible.")
        traceback.print_exc()


def debug_hold_after_error(task: 'BaseTask', seed: int, result: str = 'error'):
    """Keep Isaac Sim alive at the failed scene so the final pose/path can be inspected."""
    print("\n" + "=" * 100)
    print(f"[DEBUG_ERROR] Holding failed scene. seed={seed}, result={result}")
    print("[DEBUG_ERROR] No next seed will be started. Inspect the viewport now.")
    print("[DEBUG_ERROR] Close Isaac Sim window or press Ctrl+C in the terminal to exit.")
    print("=" * 100 + "\n")

    fallback_to_app_update = False
    try:
        while simulation_app.is_running():
            try:
                if fallback_to_app_update:
                    simulation_app.update()
                else:
                    # Render-only hold: do not advance physics, do not reset, do not start another action.
                    task.sim.render()
            except Exception:
                print("[DEBUG_ERROR] task.sim.render() failed once; falling back to simulation_app.update().")
                traceback.print_exc()
                fallback_to_app_update = True
                simulation_app.update()

            if args_cli.debug_error_sleep > 0:
                time.sleep(args_cli.debug_error_sleep)
    except KeyboardInterrupt:
        print("[DEBUG_ERROR] Interrupted by user.")


def run(task: 'BaseTask', episode_num, use_seed, start_seed, max_seed):
    suc_num, seed = 0, 0
    suc_map = []
    should_stop = False

    if start_seed != -1:
        seed = start_seed
        log(f"Starting from seed {seed}.")
    elif use_seed:
        suc_map_path = task.save_root / 'suc_map.txt'
        if suc_map_path.exists():
            with open(suc_map_path, 'r') as f:
                raw = f.read().strip()
            suc_map = raw.split(' ') if raw else []
            suc_num = sum([1 for s in suc_map if s == '1'])
            seed = len(suc_map)
            log(f"Use seed with {suc_num} successful episodes. Starting from seed {seed}.")

    mean_steps = 0.0
    while suc_num < episode_num and (max_seed == -1 or seed <= max_seed):
        try:
            start_t = time.perf_counter()
            print("\n" + "=" * 100)
            print(f"[DEBUG_SEED] Starting seed {seed}")
            print("=" * 100)
            task.reset(seed=seed)
            task.play_once()
            cost_t = time.perf_counter() - start_t
        except Exception:
            tb = traceback.format_exc()
            log(f"[{suc_num:<3d}] Seed {seed} failed with error: {tb}")
            suc_map.append('0')
            _write_suc_map(task, suc_map)

            if args_cli.debug_stop_on_error:
                print("\n" + "=" * 100)
                print(f"[DEBUG_ERROR] Stop on first exception enabled. seed={seed}")
                print("[DEBUG_ERROR] The next seed will NOT be started.")
                print("=" * 100 + "\n")

                if args_cli.debug_save_error_video:
                    print("[DEBUG_ERROR] Trying to save/close current error cache/video before holding.")
                    _safe_clean_cache(task, mean_steps=mean_steps, result='error')
                else:
                    print("[DEBUG_ERROR] Skipping task.clean_cache() to keep the failed scene/cache untouched.")
                    print("[DEBUG_ERROR] Use --debug_save_error_video if you want to try saving the error video/cache.")

                if args_cli.debug_hold_on_error:
                    debug_hold_after_error(task, seed=seed, result='error')

                should_stop = True
                break

            _safe_clean_cache(task, mean_steps=mean_steps, result='error')
        else:
            if task.plan_success and task.check_success() and not task.check_early_stop():
                task.save_to_hdf5()
                log(f"[{suc_num:<3d}] Seed {seed} success in {cost_t:.2f} s.\n"
                    f"steps: {task.step_count:<5d}, save frames: {task.save_count:<5d}.\n")
                suc_num += 1
                suc_map.append('1')
                if mean_steps > 0: 
                    mean_steps = ((suc_num - 1) * mean_steps + task.step_count) / suc_num
                else:
                    mean_steps = task.step_count
                _safe_clean_cache(task, mean_steps=mean_steps, result='success')
            else:
                log(f"[{suc_num:<3d}] Seed {seed} failed in {cost_t:.2f} s.\n"
                    f"Plan {task.plan_success}, Check {task.check_success()}, EarlyStop {task.check_early_stop()}")
                suc_map.append('0')
                _write_suc_map(task, suc_map)

                if args_cli.debug_stop_on_error:
                    print("\n" + "=" * 100)
                    print(f"[DEBUG_FAIL] Stop on first failed episode enabled. seed={seed}")
                    print("[DEBUG_FAIL] This is a non-exception failure: plan/check/early-stop failed.")
                    print("[DEBUG_FAIL] The next seed will NOT be started.")
                    print("=" * 100 + "\n")

                    if args_cli.debug_save_error_video:
                        print("[DEBUG_FAIL] Trying to save/close current fail cache/video before holding.")
                        _safe_clean_cache(task, mean_steps=mean_steps, result='fail')
                    else:
                        print("[DEBUG_FAIL] Skipping task.clean_cache() to keep the failed scene/cache untouched.")
                        print("[DEBUG_FAIL] Use --debug_save_error_video if you want to try saving the fail video/cache.")

                    if args_cli.debug_hold_on_error:
                        debug_hold_after_error(task, seed=seed, result='fail')

                    should_stop = True
                    break

                _safe_clean_cache(task, mean_steps=mean_steps, result='fail')

        _write_suc_map(task, suc_map)
        seed += 1

    if should_stop:
        log(f'Debug stop after seed {seed}. Current success count: {suc_num}.')
    else:
        denom = max(seed, 1)
        log(f'Complete collection, success rate: {suc_num}/{seed} ({(suc_num / denom) * 100:.2f}%)')

    try:
        task.close()
    except Exception:
        traceback.print_exc()
    simulation_app.close()



def debug_dump_stage_on_failure():
    """Print USD/PhysX schema, joint body0/body1, articulation roots, and keep Isaac Sim alive."""
    print("\n[DEBUG] Task 初始化失败，但保持 Isaac Sim 窗口打开。")
    print("[DEBUG] 请在 Stage 中检查 /World/envs/env_0/Robot 以及 /World/envs/env_0/Robot/x5a_link0。")
    print("[DEBUG] 关闭 Isaac Sim 窗口后脚本才会退出。\n")

    try:
        import omni.usd
        from pxr import UsdPhysics, PhysxSchema

        stage = omni.usd.get_context().get_stage()
        robot_prefix = "/World/envs/env_0/Robot"

        prim_paths = [
            f"{robot_prefix}",
            f"{robot_prefix}/x5a_link0",
            f"{robot_prefix}/x5a_link1",
            f"{robot_prefix}/x5a_link2",
            f"{robot_prefix}/x5a_link3",
            f"{robot_prefix}/x5a_link4",
            f"{robot_prefix}/x5a_link5",
            f"{robot_prefix}/x5a_link6",
            f"{robot_prefix}/x5a_adapter_link",
            f"{robot_prefix}/x5a_adapter_left_link",
            f"{robot_prefix}/x5a_adapter_right_link",
            f"{robot_prefix}/x5a_adapter_left_link/xense_left_mount",
            f"{robot_prefix}/x5a_adapter_right_link/xense_right_mount",
            f"{robot_prefix}/XenseWS_gelpad_left",
            f"{robot_prefix}/XenseWS_gelpad_right",
            f"{robot_prefix}/x5a_camera_base",
            f"{robot_prefix}/x5a_camera",
            f"{robot_prefix}/joints/x5a_joint1",
        ]

        print("\n" + "#" * 100)
        print("[DEBUG] PRIM SCHEMA CHECK")
        print("#" * 100)
        for path in prim_paths:
            prim = stage.GetPrimAtPath(path)
            print("=" * 100)
            print("PATH:", path)
            print("valid:", prim.IsValid())
            if not prim.IsValid():
                continue
            print("type:", prim.GetTypeName())
            print("schemas:", prim.GetAppliedSchemas())
            print("has ArticulationRootAPI:", prim.HasAPI(UsdPhysics.ArticulationRootAPI))
            print("has PhysxArticulationAPI:", prim.HasAPI(PhysxSchema.PhysxArticulationAPI))
            print("has RigidBodyAPI:", prim.HasAPI(UsdPhysics.RigidBodyAPI))
            print("has MassAPI:", prim.HasAPI(UsdPhysics.MassAPI))

        joint_paths = [
            f"{robot_prefix}/joints/x5a_joint1",
            f"{robot_prefix}/joints/x5a_joint2",
            f"{robot_prefix}/joints/x5a_joint3",
            f"{robot_prefix}/joints/x5a_joint4",
            f"{robot_prefix}/joints/x5a_joint5",
            f"{robot_prefix}/joints/x5a_joint6",
            f"{robot_prefix}/joints/x5a_hand_to_camera_mount",
            f"{robot_prefix}/joints/x5a_camera_joint",
            f"{robot_prefix}/joints/x5a_link6_to_adapter",
            f"{robot_prefix}/joints/x5a_adapter_left_mount",
            f"{robot_prefix}/joints/x5a_adapter_right_mount",
        ]

        print("\n" + "#" * 100)
        print("[DEBUG] JOINT BODY0/BODY1 CHECK")
        print("#" * 100)
        for joint_path in joint_paths:
            joint_prim = stage.GetPrimAtPath(joint_path)
            print("=" * 100)
            print("JOINT:", joint_path)
            print("valid:", joint_prim.IsValid())
            if not joint_prim.IsValid():
                continue

            print("type:", joint_prim.GetTypeName())
            print("schemas:", joint_prim.GetAppliedSchemas())

            body0_targets = joint_prim.GetRelationship("physics:body0").GetTargets()
            body1_targets = joint_prim.GetRelationship("physics:body1").GetTargets()

            print("body0:", body0_targets)
            print("body1:", body1_targets)

            for label, targets in (("body0", body0_targets), ("body1", body1_targets)):
                if not targets:
                    print(label, "EMPTY")
                    continue

                target_path = targets[0]
                target_prim = stage.GetPrimAtPath(target_path)

                print(label, "target:", target_path)
                print(label, "target valid:", target_prim.IsValid())

                if target_prim.IsValid():
                    print(label, "target type:", target_prim.GetTypeName())
                    print(label, "target schemas:", target_prim.GetAppliedSchemas())
                    print(label, "target has RigidBodyAPI:", target_prim.HasAPI(UsdPhysics.RigidBodyAPI))
                    print(label, "target has MassAPI:", target_prim.HasAPI(UsdPhysics.MassAPI))

        print("\n" + "#" * 100)
        print("[DEBUG] ARTICULATION ROOT CHECK UNDER /World/envs/env_0/Robot")
        print("#" * 100)
        articulation_roots = []
        for prim in stage.Traverse():
            path_str = str(prim.GetPath())
            if path_str.startswith(robot_prefix) and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_roots.append(path_str)

        print("articulation roots found:", articulation_roots)
        print("articulation root count:", len(articulation_roots))
        if len(articulation_roots) != 1:
            print("[WARN] Articulation root 数量不是 1。理想状态应该只有 /World/envs/env_0/Robot/x5a_link0。")

        print("\n" + "#" * 100)
        print("[DEBUG] RIGID BODY SUMMARY UNDER /World/envs/env_0/Robot")
        print("#" * 100)
        rigid_bodies = []
        for prim in stage.Traverse():
            path_str = str(prim.GetPath())
            if path_str.startswith(robot_prefix) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_bodies.append(path_str)

        for rb in rigid_bodies:
            print("RigidBody:", rb)
        print("rigid body count:", len(rigid_bodies))

    except Exception:
        print("[DEBUG] 打印 Stage 调试信息时出错：")
        traceback.print_exc()

    while simulation_app.is_running():
        simulation_app.update()

    simulation_app.close()

def debug_hold_task(task: 'BaseTask'):
    """Keep Isaac Sim open after one task initialization without reset/action/collection loop.

    Render-only mode inspects the spawned USD without advancing physics.
    With --debug_step, physics advances so you can check whether Play/physics itself
    causes flying/drifting, without the collection script repeatedly resetting seeds.
    With --debug_update_obs, call task._get_observations() every frame so tactile
    sensors / UIPC buffers are refreshed without running task actions or saving data.
    """
    print("\n" + "=" * 100)
    print("[DEBUG_HOLD] Task initialized successfully.")
    print("[DEBUG_HOLD] No task.reset(seed=...), no task.play_once(), no cache cleaning, no HDF5/video saving.")
    if args_cli.debug_step:
        print("[DEBUG_HOLD] Mode: physics STEP enabled. This tests whether Play/physics itself causes flying/drifting.")
    else:
        print("[DEBUG_HOLD] Mode: render-only. This tests the static spawned USD without advancing physics.")
    if args_cli.debug_update_obs:
        print("[DEBUG_HOLD] Sensor update enabled: calling task._get_observations() every frame.")
    print("[DEBUG_HOLD] Close Isaac Sim window or press Ctrl+C in terminal to exit.")
    print("=" * 100 + "\n")

    fallback_to_app_update = False
    try:
        while simulation_app.is_running():
            try:
                if fallback_to_app_update:
                    simulation_app.update()
                elif args_cli.debug_step:
                    # Advance physics but do not run task actions or collection/reset logic.
                    try:
                        task.sim.step(render=True)
                    except TypeError:
                        task.sim.step()
                        task.sim.render()
                else:
                    # Keep GUI responsive and show initialized scene without advancing physics.
                    task.sim.render()

                if args_cli.debug_update_obs:
                    # Refresh tactile/sensor/UIPC buffers without calling reset(), play_once(), or save logic.
                    try:
                        task._get_observations()
                    except Exception:
                        print("[DEBUG_HOLD] task._get_observations() failed once; continuing to keep the window alive.")
                        traceback.print_exc()
                        args_cli.debug_update_obs = False
            except Exception:
                print("[DEBUG_HOLD] task.sim render/step failed once; falling back to simulation_app.update().")
                traceback.print_exc()
                fallback_to_app_update = True
                simulation_app.update()

            if args_cli.debug_hold_sleep > 0:
                time.sleep(args_cli.debug_hold_sleep)
    except KeyboardInterrupt:
        print("[DEBUG_HOLD] Interrupted by user.")
    finally:
        try:
            task.close()
        except Exception:
            pass
        simulation_app.close()


def main():
    global args_cli, task_config, task_config_file, log_path
    task_file_name = args_cli.task

    episode_num = task_config.get("episode_num", -1)
    if args_cli.episode_num != -1:
        episode_num = args_cli.episode_num
    start_seed = task_config.get("start_seed", -1)
    if args_cli.start_seed != -1:
        start_seed = args_cli.start_seed
    max_seed = task_config.get("max_seed", -1)
    if args_cli.max_seed != -1:
        max_seed = args_cli.max_seed
    
    task_config.update({
        "episode_num": episode_num,
        "start_seed": start_seed,
        "max_seed": max_seed,
    })

    task_module = importlib.import_module(f"envs.{task_file_name}")
    env_cfg:'BaseTaskCfg' = task_module.TaskCfg()
    env_cfg.tactile_sensor_type = task_config.get('sensor_type', 'gsmini')
    env_cfg.save_dir = Path(task_config.get("save_dir", "./data")) / task_file_name / task_config_file.stem
    env_cfg.decimation = task_config.get("decimation", env_cfg.decimation)
    env_cfg.save_frequency = task_config.get("save_frequency", env_cfg.save_frequency)
    env_cfg.video_frequency = task_config.get("video_frequency", env_cfg.video_frequency)
    env_cfg.render_frequency = task_config.get("render_frequency", env_cfg.render_frequency)
    env_cfg.obs_data_type = task_config.get("observations", {})
    env_cfg.random_texture = task_config.get("random_texture", False)

    env_cfg.scene.num_envs = 1
    
    init_start = time.perf_counter()
    try:
        task: 'BaseTask' = task_module.Task(env_cfg, mode='collect')
    except Exception:
        traceback.print_exc()
        debug_dump_stage_on_failure()
        return
    init_cost = time.perf_counter() - init_start
    
    log_path = task.save_root / f"{time.strftime(r'%Y-%m-%d_%H:%M:%S')}.log"
    log(f"Task Name: {task_file_name}")
    log(f"Config Name: {task_config_file.stem}")
    log(f"Task Config: \n{json.dumps(task_config, ensure_ascii=False, indent=4)}\n{'-' * 20}\n")
    log(f"Env Config: \n{env_cfg}\n{'-' * 20}\n")
    log(f"Init cost {init_cost:.2f} seconds, devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    if args_cli.debug_hold:
        debug_hold_task(task)
        return

    run(
        task,
        episode_num=episode_num,
        use_seed=task_config.get("use_seed", True),
        start_seed=start_seed,
        max_seed=max_seed,
    )

if __name__ == "__main__":
    main()
import threading
import evdev
from evdev import ecodes

def init_keyboard_listener():
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }

    def listen():
        # 同时监听笔记本键盘和外接键盘
        devices = []
        for path in ["/dev/input/event2", "/dev/input/event6"]:
            try:
                devices.append(evdev.InputDevice(path))
            except Exception:
                pass

        if not devices:
            print("Warning: No keyboard device found for evdev listener")
            return

        from selectors import DefaultSelector, EVENT_READ
        selector = DefaultSelector()
        for dev in devices:
            selector.register(dev, EVENT_READ)

        while True:
            for key, mask in selector.select():
                device = key.fileobj
                for ev in device.read():
                    if ev.type == ecodes.EV_KEY and ev.value == 1:  # key down
                        if ev.code == ecodes.KEY_RIGHT:
                            print("Right arrow key pressed. Exiting loop...")
                            events["exit_early"] = True
                        elif ev.code == ecodes.KEY_LEFT:
                            print("Left arrow key pressed. Rerecording...")
                            events["rerecord_episode"] = True
                            events["exit_early"] = True
                        elif ev.code == ecodes.KEY_ESC:
                            print("Escape key pressed. Stopping recording...")
                            events["stop_recording"] = True
                            events["exit_early"] = True

    t = threading.Thread(target=listen, daemon=True)
    t.start()
    return None, events

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101LeaderConfig
from lerobot.teleoperators.so_leader import SO101Leader
#from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors
import os
import shutil  # 修复：添加缺失的shutil导入
from pathlib import Path

# 配置参数
NUM_EPISODES = 10000
FPS = 30
EPISODE_TIME_SEC = 30
RESET_TIME_SEC = 10
# TASK_DESCRIPTION = "Put all the balls into the plate."
TASK_DESCRIPTION = "stack the cubes."
VIDEO_CODEC = "h264"

def tee(msg: str):
    print(msg)
    log_say(msg)

def main():
    try:
        # Create robot configuration
        robot_config = SO101FollowerConfig(
            id="pzj_follower_arm",
            cameras={
                # 建议：先验证相机索引是否正确，可改为0/1/2测试
                 "wrist": OpenCVCameraConfig(index_or_path=4, width=320, height=240, fps=FPS),
                 "external": OpenCVCameraConfig(
                            index_or_path=10,
                            fps=FPS,
                            width=640,
                            height=480
                        )
            },
            port="/dev/ttyACM0",
        )

        teleop_config = SO101LeaderConfig(
            id="pzj_leader_arm",
            port="/dev/ttyACM1",
        )

        # 初始化硬件（添加异常处理）
        log_say("初始化机器人和遥操作器...")
        robot = SO101Follower(robot_config)
        teleop = SO101Leader(teleop_config)

        # Configure the dataset features
        action_features = hw_to_dataset_features(robot.action_features, "action")
        obs_features = hw_to_dataset_features(robot.observation_features, "observation")
        dataset_features = {**action_features, **obs_features}
        # ↓ 添加这段，手动覆盖图像 shape 为实际分辨率

        # 1. 路径定义
        repo_id = "zhenqis123/0514_stack_the_cubes"
        current_dir = Path(__file__).parent if "__file__" in locals() else Path(".")
        root_path = current_dir / "outputs_recordings_new" 
        dataset_path = root_path / repo_id

        # 确保父目录存在
        root_path.mkdir(parents=True, exist_ok=True)

        # --- 2. 加载/创建数据集 (带自动重置功能) ---
        dataset = None
        episode_idx = 0

        try:
            # 尝试加载现有数据集
            if (dataset_path / "meta/info.json").exists():
                log_say(f"Attempting to load local dataset: {dataset_path}")
                os.environ["HF_HUB_OFFLINE"] = "1"
                dataset = LeRobotDataset(
                    repo_id=repo_id,
                    root=dataset_path,
                    vcodec=VIDEO_CODEC,
                )
                episode_idx = dataset.num_episodes
                log_say(f"Load successful. Current episodes: {episode_idx}")
            else:
                # 如果 info.json 都不存在，直接触发异常进入创建流程
                raise FileNotFoundError("info.json missing")
                
        except Exception as e:
            # 如果加载过程中发生任何错误（如 FileNotFoundError, ValueError 等）
            log_say(f"Dataset corrupted or incomplete ({type(e).__name__}). Resetting...")
            
            # 彻底删除旧目录
            if dataset_path.exists():
                shutil.rmtree(dataset_path)
                log_say(f"Deleted corrupted directory: {dataset_path}")

            # 重新创建数据集
            log_say("Initializing a fresh dataset...")
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                root=dataset_path,
                fps=FPS,
                features=dataset_features,
                robot_type=robot.name,
                use_videos=True,
                vcodec=VIDEO_CODEC,
                # image_writer_processes=4,
                # batch_encoding_size=16,
            )
            episode_idx = 0
            log_say("New dataset initialized successfully.")

        # 初始化键盘监听和可视化
        _, events = init_keyboard_listener()
        init_rerun(session_name="recording")

        # 连接硬件
        log_say("连接机器人和遥操作器...")
        robot.connect()
        
        
        
        teleop.connect()

        # 创建处理器
        teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

        # 修复：不再重复定义episode_idx，保留原有计数
        log_say(f"开始录制，共需录制 {NUM_EPISODES} 个 Episode（当前已录制 {episode_idx} 个）")
        while episode_idx < NUM_EPISODES and not events["stop_recording"]:
            log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                dataset=dataset,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
            )

            # 重置环境
            if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
                log_say("Reset the environment")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    control_time_s=RESET_TIME_SEC,
                    single_task=TASK_DESCRIPTION,
                    display_data=True,
                )

            if events["rerecord_episode"]:
                log_say("Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            episode_idx += 1

    except Exception as e:
        # 捕获所有异常，避免进程直接崩溃
        log_say(f"运行出错：{type(e).__name__}: {str(e)}")
        raise  # 重新抛出异常，方便定位问题
    finally:
        # 确保硬件资源被释放
        try:
            log_say("Cleaning up resources...")
            if dataset is not None:
                dataset.finalize()
            if robot is not None:
                robot.disconnect()
            if teleop is not None:
                teleop.disconnect()
        except:
            pass
        log_say("Stop recording successfully")

if __name__ == "__main__":
    main()


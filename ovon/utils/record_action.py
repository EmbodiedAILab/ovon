import json
import os
import gzip
from habitat.core.dataset import BaseEpisode


def get_habitat_sim_action_too_str(action):
    if action == 0:
        return "STOP"
    elif action == 1:
        return "MOVE_FORWARD"
    elif action == 2:
        return "TURN_LEFT"
    elif action == 3:
        return "TURN_RIGHT"
    elif action == 4:
        return "LOOK_UP"
    elif action == 5:
        return "LOOK_DOWN"

all_datasets = {}

def get_scene_name(scene_id):
    folder_part = scene_id.split('/')[-2]
    scene_name = folder_part.split('-')[1]
    return scene_name

def record_trajectory_start(input_path, split_name):
    formatted_data_path = input_path.format(split=split_name)
    data_file_dir = os.path.dirname(formatted_data_path)
    dataset_path_constructed = os.path.join(data_file_dir, "content")
    
    json_gz_files = [f for f in os.listdir(dataset_path_constructed) if f.endswith('.json.gz')]
    
    for file_name in json_gz_files:
        dataset_path = os.path.join(dataset_path_constructed, file_name)
        with gzip.open(dataset_path, 'rt', encoding='utf-8') as f_in: # 'rt' 表示以文本模式读取
            origin_dataset = json.load(f_in)
            for ep in origin_dataset["episodes"]:
                ep["reference_replay"] = []
            
            scene_name = get_scene_name(origin_dataset["episodes"][0]["scene_id"])
            
            all_datasets[scene_name] = origin_dataset


def record_trajectory_process(actions, current_episodes:BaseEpisode):
    for idx in range(len(current_episodes)):
        current_episode = current_episodes[idx]
        action = actions[idx]
        scene_name = get_scene_name(current_episode.scene_id)
        episode_id = int(current_episode.episode_id)
        record = {}
        action_str = get_habitat_sim_action_too_str(action)
        record["action"] = action_str
        record["agent_state"] = {}
        all_datasets[scene_name]["episodes"][episode_id]["reference_replay"].append(record)
            
    return

def record_trajectory_close(traj_dir):

    if not os.path.exists(traj_dir):  # 检查路径是否存在
        os.makedirs(traj_dir)
    
    for scene_name, dataset in all_datasets.items():
        with open(traj_dir + scene_name + ".json.gz", "wt") as f:
            json.dump(dataset, f, indent=2)  # indent参数保持可读性
    
    return
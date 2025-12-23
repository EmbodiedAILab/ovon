from ovon.models.transformer_policy import OVONTransformerPolicy
from ovon.algos.dagger import DAgger, DAggerPolicy
from ovon.trainers.dagger_trainer import VERDAggerTrainer
import habitat
from habitat import get_config
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.construct_vector_env import construct_envs
from habitat_baselines.rl.ddppo.ddp_utils import is_slurm_batch_job
import torch
from habitat_baselines.rl.ppo import PPO
from ovon.utils.utils import load_pickle
from typing import TYPE_CHECKING, Any, Dict, Optional
from PIL import Image
import numpy as np
from habitat_baselines.utils.common import (
    batch_obs,
    inference_mode
)
from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch
import argparse
from gym import spaces
from habitat.config import read_write
import cv2
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat.core.spaces import ActionSpace, EmptySpace, ListSpace
from omegaconf import OmegaConf
import random
import os
import gzip
import json

class ClipObjectGoalEmbeding:
    def __init__(
        self,
        config: "DictConfig",
    ):
        self.cache = load_pickle("/workspace/ovon/ovon/limo/siglip_7.pkl")
        k = list(self.cache.keys())[0]
        self._embed_dim = self.cache[k].shape[0]
        for v in self.cache.values():
            assert self._embed_dim == v.shape[0] and v.ndim == 1

    def get_observation(
        self,
        object_category: str,
    ) -> Optional[int]:
        if object_category not in self.cache:
            print("Missing category: {}".format(object_category))
        return self.cache[object_category]


def create_episode(file_path, start_position, start_rotation, object_category):
    episode_info = {
        "start_position": [1.0342, 0.0, -1.5636],
        "start_rotation": [0.0, 0.7071, 0.0, 0.7071],
        "object_category": "sofa",
    }
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        dataset = json.load(f)
        dataset["episodes"][0]["object_category"] = episode_info["object_category"]
        dataset["episodes"][0]["start_position"] = episode_info["start_position"]
        dataset["episodes"][0]["start_rotation"] = episode_info["start_rotation"]
        
        with gzip.open(file_path, 'wt', encoding='utf-8') as f:
            json.dump(dataset, f, indent=4)


class OVON2Limo:
    def __init__(
        self,
        save_dir,
        obj_category,
    ) -> None:
        self.save_dir = save_dir
        self.obj_category = obj_category
        config_dir = "/workspace/ovon/ovon/app/v1/transformer_il_3dgs.yaml"
        config = get_config(config_dir)
        
        with read_write(config):
            if hasattr(
                config.habitat_baselines.rl.policy.obs_transforms, "relabel_teacher_actions"
            ):
                print("[run.py]: Removing relabel_teacher_actions from config for eval.")
                config.habitat_baselines.rl.policy.obs_transforms.pop(
                    "relabel_teacher_actions"
                )

            for k in ["look_up", "look_down"]:
                if k in config.habitat.task.actions:
                    config.habitat.task.actions.pop(k)

        self.env = habitat.Env(config=config)
        obs = self.env.reset()
        
        self.current_obs = obs
        self.current_action = 0

        self.config = config
        print("config:", self.config)
        ppo_cfg = self.config.habitat_baselines.rl.ppo

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_transforms = get_active_obs_transforms(self.config)

        action_space = spaces.Discrete(4)

        # 手动定义观测空间（示例：RGB 图像 + CLIP 嵌入）
        observation_space = spaces.Dict({
            "rgb": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            "clip_objectgoal": spaces.Box(low=-np.inf, high=np.inf, shape=(768,), dtype=np.float32),
            "step_id": spaces.Box(low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max, shape=(1,), dtype=np.int64),
            "next_actions": spaces.Discrete(1)
        })

        self.orig_policy_action_space = spaces.Dict({
            'stop': EmptySpace(),
            "move_forward": EmptySpace(),
            'turn_left': EmptySpace(),
            'turn_right': EmptySpace()
        })
        
        random.seed(config.habitat.seed)
        np.random.seed(config.habitat.seed)
        torch.manual_seed(config.habitat.seed)
        print("seed:", config.habitat.seed)
        if (
            config.habitat_baselines.force_torch_single_threaded
            and torch.cuda.is_available()
        ):
            torch.set_num_threads(1)
        
        trainer_init = baseline_registry.get_trainer(
            config.habitat_baselines.trainer_name
        )

        self.trainer = trainer_init(config)
        self.trainer.obs_space = observation_space
        self.trainer.policy_action_space = action_space
        self.trainer.orig_policy_action_space = self.orig_policy_action_space
        self.trainer.device = self.device
        self.trainer._setup_actor_critic_agent(ppo_cfg)

        checkpoint_path = "/workspace/ovon/ovon/app/v1/ckpt_3dgs.pth"
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        self.trainer.agent.load_state_dict(ckpt["state_dict"], strict=False)
        self.actor_critic = self.trainer.agent.actor_critic
        self.actor_critic.eval()

        prev_actions = 0
        self.prev_actions_tensor = torch.tensor([[prev_actions]], device=self.device)
        self.not_done_masks = torch.zeros(
                                self.config.habitat_baselines.num_environments,
                                1,
                                device=self.device,
                                dtype=torch.bool,
                            )
        self.test_recurrent_hidden_states = torch.zeros(
                                                self.config.habitat_baselines.num_environments,
                                                self.actor_critic.num_recurrent_layers,
                                                self.config.habitat_baselines.rl.ppo.hidden_size,
                                                device=self.device,
                                            )
        self.object_goal_cache = ClipObjectGoalEmbeding(config=config)
        self.step_id = 0

    def act(self):
        object_goal_embeding = self.object_goal_cache.get_observation(self.obj_category)

        observation = {}
        observation['rgb'] = np.array(self.current_obs['rgb'])
        observation['clip_objectgoal'] = object_goal_embeding
        observation['step_id'] = self.step_id
        observation['next_actions'] = 0

        cur_img_name = self.obj_category + "_" + str(self.step_id) + "_" + str(self.current_action) + ".jpg"
        save_path = os.path.join(self.save_dir, cur_img_name)
        print("save_path:", save_path)
        cv2.imwrite(save_path, np.array(self.current_obs['rgb']))


        if self.step_id != 0:
            self.not_done_masks = torch.ones(
                                self.config.habitat_baselines.num_environments,
                                1,
                                device=self.device,
                                dtype=torch.bool,
                            )
        self.step_id += 1

        batch = batch_obs([observation], device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        with inference_mode():
            (
                _,
                action,
                _,
                self.test_recurrent_hidden_states,
            ) = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions_tensor,
                self.not_done_masks,
                deterministic=False,
            )

        action_value = action.item()

        self.prev_actions_tensor.copy_(action)  # 更新历史动作

        step_data = action.cpu().item()   # 获取可执行动作（0-3）
        self.current_action = step_data

        print("action:", action, ", type:", type(action))
        # print("self.test_recurrent_hidden_states:", self.test_recurrent_hidden_states)
        # return step_data

        obs = self.env.step(step_data)
        self.current_obs = obs

        # import time  
        # time.sleep(0.5)  # 暂停 0.5 秒以模拟处理时间  

        # metrics = env.get_metrics()
        # print("metrics:", metrics)
        
        print("step action:", step_data)
        if step_data == 0:
            self.env.close()
            return True
            
        return False

if __name__ == "__main__":

    file_path = "/workspace/ovon/data/task_datasets/objectnav/cloudrobo_v1/demo_1230/val/content/shenzhen-room_ziwei_20250724-metacam.json.gz"
    object_category = "sofa"
    start_position =  [
                        -0.016438689082860947,
                        0.05,
                        4.700855445861816
                    ]
    start_rotation = [
                        0.0,
                        0.0,
                        0.0,
                        1.0
                    ]
    create_episode(file_path, start_position, start_rotation, object_category)

    save_dir="/workspace/ovon/ovon/app/v1/save_img"

    ovon = OVON2Limo(save_dir, object_category)

    finish = False
    while not finish:    
        finish = ovon.act()
    print("Finished!")
        
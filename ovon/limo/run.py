from ovon.models.transformer_policy import OVONTransformerPolicy
from habitat import get_config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.construct_vector_env import construct_envs
from habitat_baselines.rl.ddppo.ddp_utils import is_slurm_batch_job
import torch
from habitat_baselines.rl.ppo import PPO
from ovon.utils.utils import load_pickle
from typing import TYPE_CHECKING, Any, Dict, Optional
from PIL import Image
import numpy as np
from habitat_baselines.utils.common import batch_obs
from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch
import argparse
from gym import spaces

class ClipObjectGoalEmbeding:
    def __init__(
        self,
        config: "DictConfig",
    ):
        self.cache = load_pickle(config.habitat.task.lab_sensors.clip_objectgoal_sensor.cache)
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



config_dir = "/home/fsq/ovon_upload/ovon/ovon/limo/transformer_il.yaml"
config = get_config(config_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = baseline_registry.get_policy("OVONTransformerPolicy")

# 手动定义动作空间（例如离散动作）
action_space = spaces.Discrete(6)  # 假设有 6 个离散动作

# 手动定义观测空间（示例：RGB 图像 + CLIP 嵌入）
observation_space = spaces.Dict({
    "rgb": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
    "clip_objectgoal": spaces.Box(low=-np.inf, high=np.inf, shape=(768,), dtype=np.float32),
    "step_id": spaces.Box(low=0, high=np.iinfo(np.int64).max, shape=(1,), dtype=np.int64),
    "next_actions": spaces.Discrete(1)
})

actor_critic = policy.from_config(
                    config,
                    observation_space,
                    action_space,
                    orig_action_space=action_space,
                )

checkpoint_path = "/home/fsq/ovon_upload/ovon/output/checkpoints/ckpt.1.pth"
ckpt = torch.load(checkpoint_path, map_location="cpu")

ppo_cfg = config.habitat_baselines.rl.ppo
agent = PPO.from_config(actor_critic, ppo_cfg)
agent.actor_critic.load_state_dict(ckpt["state_dict"], strict=False)
actor_critic = agent.actor_critic.to(device)

image_path = "/home/fsq/ovon_upload/ovon/ovon/limo/Snipaste_2025-10-13_11-06-46.png"
image = Image.open(image_path).convert('RGB')
rgb_array = np.array(image)

object_goal_cache = ClipObjectGoalEmbeding(config=config)
object_goal_embeding = object_goal_cache.get_observation("chair")


observation = {}
observation['rgb'] = rgb_array
observation['clip_objectgoal'] = object_goal_embeding
observation['step_id'] = 0
observation['next_actions'] = 0

# print("observation:", observation)


batch = batch_obs([observation], device=device)  # 返回字典：{"rgb": tensor, ...}

prev_actions = 1

# 转换为张量并添加批次维度
prev_actions_tensor = torch.tensor([[prev_actions]], dtype=torch.int, device=device)
not_done_masks = torch.zeros(
                    config.habitat_baselines.num_environments,
                    1,
                    device=device,
                    dtype=torch.bool,
                )
test_recurrent_hidden_states = torch.zeros(
                                config.habitat_baselines.num_environments,
                                actor_critic.num_recurrent_layers,
                                config.habitat_baselines.rl.ppo.hidden_size,
                                device=device,
                            )

(
    _,
    action,
    _,
    test_recurrent_hidden_states,
) = actor_critic.act(
    batch,
    test_recurrent_hidden_states,
    prev_actions_tensor,
    not_done_masks,
    deterministic=False,
)

action_value = action.item()
if action_value >= 4:
    new_action = torch.randint(0, 4, (1, 1), device=action.device)
    action.copy_(new_action)

prev_actions_tensor.copy_(action)  # 更新历史动作
step_data = action.cpu().item()   # 获取可执行动作（0-3）

print("action:", action, ", type:", type(action))

print("step_data:", step_data)


class OVON2Limo:
    def __init__(
        self,
        config_dir: str,
        checkpoint_path: str,
    ) -> None:
        # config_dir = "/home/fsq/ovon_upload/ovon/ovon/limo/transformer_il.yaml"
        self.config = get_config(config_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = baseline_registry.get_policy("OVONTransformerPolicy")

        # 手动定义动作空间（例如离散动作）
        action_space = spaces.Discrete(6)  # 假设有 6 个离散动作

        # 手动定义观测空间（示例：RGB 图像 + CLIP 嵌入）
        observation_space = spaces.Dict({
            "rgb": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            "clip_objectgoal": spaces.Box(low=-np.inf, high=np.inf, shape=(768,), dtype=np.float32),
            "step_id": spaces.Box(low=0, high=np.iinfo(np.int64).max, shape=(1,), dtype=np.int64),
            "next_actions": spaces.Discrete(1)
        })

        actor_critic = policy.from_config(
                            config,
                            observation_space,
                            action_space,
                            orig_action_space=action_space,
                        )

        # checkpoint_path = "/home/fsq/ovon_upload/ovon/output/checkpoints/ckpt.1.pth"
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        ppo_cfg = self.config.habitat_baselines.rl.ppo
        agent = PPO.from_config(actor_critic, ppo_cfg)

        agent.actor_critic.load_state_dict(ckpt["state_dict"], strict=False)

        self.actor_critic = agent.actor_critic.to(device)

        prev_actions = 0
        self.prev_actions_tensor = torch.tensor([[prev_actions]], dtype=torch.int, device=self.device)
        self.not_done_masks = torch.zeros(
                                self.config.habitat_baselines.num_environments,
                                1,
                                device=self.device,
                                dtype=torch.bool,
                            )
        self.test_recurrent_hidden_states = torch.zeros(
                                                selfconfig.habitat_baselines.num_environments,
                                                selfactor_critic.num_recurrent_layers,
                                                selfconfig.habitat_baselines.rl.ppo.hidden_size,
                                                device=selfdevice,
                                            )
        self.object_goal_cache = ClipObjectGoalEmbeding(config=config)

    def act(image_array: np.array, object_goal: str):
        object_goal_embeding = self.object_goal_cache.get_observation("chair")

        observation = {}
        observation['rgb'] = image_array
        observation['clip_objectgoal'] = object_goal_embeding
        observation['step_id'] = 0
        observation['next_actions'] = 0

        print("observation:", observation)


        batch = batch_obs([observation], device=device)  # 返回字典：{"rgb": tensor, ...}

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
        if action_value >= 4:
            new_action = torch.randint(0, 4, (1, 1), device=action.device)
            action.copy_(new_action)

        self.prev_actions_tensor.copy_(action)  # 更新历史动作
        step_data = action.cpu().item()   # 获取可执行动作（0-3）

        print("action:", action, ", type:", type(action))
        print("step_data:", step_data)
        return step_data


def init_ovon():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config_dir",
        "-c",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--ckpt_dir",
        "-k",
        type=str,
        required=True,
        help="path to ckpt about experiment",
    )

    args = parser.parse_args()
    return OVON2Limo(args.config_dir, args.ckpt_dir)



# if __name__ == "__main__":
#     main()
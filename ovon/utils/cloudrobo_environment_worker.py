#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from multiprocessing.context import BaseContext
from typing import (
    TYPE_CHECKING,
    List,
    Optional,
)
import attr
from habitat_baselines.rl.ver.environment_worker import (
    EnvironmentWorker,
    EnvironmentWorkerProcess,
    infinite_shuffling_iterator,
    _make_proc_config,
)
from habitat_baselines.rl.ver.worker_common import WorkerBase, WorkerQueues

if TYPE_CHECKING:
    from omegaconf import DictConfig
from habitat_baselines.rl.ddppo.ddp_utils import rank0_only

from habitat_baselines.rl.ver.timing import Timing
from habitat_baselines.rl.ver.task_enums import ReportWorkerTasks
from habitat import RLEnv, logger, make_dataset
from habitat_baselines.rl.ddppo.ddp_utils import get_distrib_size, EXIT
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict

MIN_SCENES_PER_ENV = 1

@dataclass
class CloudRoboEpisodeCounter:
    max_episodes: int = 0
    episodes_completed: int = 0
    scene_used: int = 0
    scene_used_dict: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )


@attr.s(auto_attribs=True, auto_detect=True)
class CloudRoboEnvironmentWorkerProcess(EnvironmentWorkerProcess):
    teacher_label: Optional[str] = None
    episode_counter: CloudRoboEpisodeCounter = attr.ib(
        default=attr.Factory(CloudRoboEpisodeCounter)
    )
    cycle_round: int = 0

    def start(self):
        super().start()
        self.episode_counter.max_episodes += self.env.number_of_episodes
        self.episode_counter.episodes_completed = 0
        self.episode_counter.cycle_finish = False

    def step(self):
        with self.timer.avg_time("process actions"):
            action = self.action_plugin(self.actions[self.env_idx])

        self._last_obs, reward, done, info, episodes = self._step_env(action)

        with self.timer.avg_time("enqueue env"):
            self.send_transfer_buffers[self.env_idx] = dict(
                observations=self._last_obs,
                rewards=reward,
                masks=not done,
                episode_ids=self._episode_id,
                step_ids=self._step_id,
            )

        self.queues.inference.put(self.env_idx)

        self.total_reward += reward
        self.episode_length += 1

        if done:
            self.queues.report.put_many(
                [
                    (
                        ReportWorkerTasks.episode_end,
                        dict(
                            env_idx = self.env_idx,
                            length = self.episode_length,
                            reward = self.total_reward,
                            info = info,
                            episodes = episodes,
                        ),
                    ),
                    (
                        ReportWorkerTasks.env_timing, 
                            self.timer
                    ),
                ]
            )
            self.timer = Timing()
            self.total_reward = 0.0
            self.episode_length = 0


    def _step_env(self, action):
        with self.timer.avg_time("step env"):
            obs, reward, done, info = self.env.step(action)
            self._step_id += 1

            if not math.isfinite(reward):
                reward = -1.0

        episodes = {}
        with self.timer.avg_time("reset env"):
            if done:
                # add episodes done info 
                self.episode_counter.episodes_completed += 1
                self.episode_counter.scene_used_dict[str(self.env.current_episode().scene_id)][str(self.env.current_episode().episode_id)] += 1
                self.episode_counter.scene_used = len(self.episode_counter.scene_used_dict)

                if self.episode_counter.episodes_completed == self.episode_counter.max_episodes:
                    self.cycle_round += 1
                    if self.cycle_round >= self.env_config.habitat_baselines.rl.policy.cycle_round:
                        self.episode_counter.cycle_finish = True
                
                episodes["rank_env"] = str(self.env_config.habitat_baselines.torch_gpu_id) + str("_") + str(self.env_idx)
                episodes["max_episodes"] = int(self.episode_counter.max_episodes)
                episodes["episodes_completed"] = int(self.episode_counter.episodes_completed)
                episodes["scenes_used"] = int(self.episode_counter.scene_used)
                episodes["cycle_finish"] = self.episode_counter.cycle_finish
                
                self._episode_id += 1
                self._step_id = 0
                if self.auto_reset_done:
                    obs = self.env.reset()

        return obs, reward, done, info, episodes


class CloudRoboEnvironmentWorker(EnvironmentWorker):
    def __init__(
        self,
        mp_ctx: BaseContext,
        env_idx: int,
        env_config,
        auto_reset_done,
        queues: WorkerQueues,
    ):
        teacher_label = None
        obs_trans_conf = env_config.habitat_baselines.rl.policy.obs_transforms
        if hasattr(env_config.habitat_baselines.rl.policy, "obs_transforms"):
            for obs_transform_config in obs_trans_conf.values():
                if hasattr(obs_transform_config, "teacher_label"):
                    teacher_label = obs_transform_config.teacher_label
                    break
        assert teacher_label is not None, "teacher_label not found in config"
        WorkerBase.__init__(
            self,
            mp_ctx,
            CloudRoboEnvironmentWorkerProcess,
            env_idx,
            env_config,
            auto_reset_done,
            queues,
            teacher_label=teacher_label,
        )
        self.env_worker_queue = queues.environments[env_idx]


def _construct_cloudrobo_environment_workers_impl(
    configs,
    auto_reset_done,
    mp_ctx: BaseContext,
    queues: WorkerQueues,
):
    num_environments = len(configs)
    num_environments = configs[0].habitat_baselines.num_environments
    workers = []
    for i in range(num_environments):
        w = CloudRoboEnvironmentWorker(mp_ctx, i, configs[i], auto_reset_done, queues)
        workers.append(w)

    return workers


def construct_cloudrobo_environment_workers(
    config: "DictConfig",
    mp_ctx: BaseContext,
    worker_queues: WorkerQueues,
) -> List[EnvironmentWorker]:
    configs = _create_cloudrobo_worker_configs(config)

    return _construct_cloudrobo_environment_workers_impl(configs, True, mp_ctx, worker_queues)


def _create_cloudrobo_worker_configs(config: "DictConfig"):
    num_environments = config.habitat_baselines.num_environments
    _, world_rank, world_rank_size = get_distrib_size()
    splits_num = num_environments * world_rank_size

    dataset = make_dataset(config.habitat.dataset.type)
    scenes = config.habitat.dataset.content_scenes
    if "*" in config.habitat.dataset.content_scenes:
        scenes = dataset.get_scenes_to_load(config.habitat.dataset)

    # We use a minimum number of scenes per environment to reduce bias
    scenes_per_env = max(
        int(math.ceil(len(scenes) / splits_num)), MIN_SCENES_PER_ENV
    )

    scene_splits: List[List[str]] = [[] for _ in range(splits_num)]
    for idx, scene in enumerate(scenes):
        scene_splits[idx % len(scene_splits)].append(scene)
        if len(scene_splits[-1]) == scenes_per_env:
            break

    assert len(set().union(*(set(scenes) for scenes in scene_splits))) == len(
        scenes
    )

    args = [
        _make_proc_config(config, rank, scenes, scene_splits)
        for rank in range(splits_num)
    ]

    sub_args = args[world_rank*num_environments:(world_rank+1)*num_environments]
    return sub_args
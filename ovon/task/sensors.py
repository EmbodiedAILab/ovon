import hashlib
import random
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
from gym import Space, spaces
from habitat.core.registry import registry
from habitat.core.simulator import RGBSensor, Sensor, SensorTypes, Simulator, Observations
from habitat.core.utils import try_cv2_import
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from ovon.task.object_nav_task import ObjectNavigationTask
from ovon.utils.utils import load_pickle
import os 

from pano_gs.gs_simulator import GsSimulator
from pano_gs.gs_simulator.sim_camera.habitat_cam import (
    habitat_sensor_to_CamSettings,
    construct_viewpoint_cam_from_agent_state
)
from habitat.tasks.nav.nav import NavigationEpisode

if TYPE_CHECKING:
    from omegaconf import DictConfig


@registry.register_sensor
class ClipObjectGoalSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id, and we will generate the prompt corresponding to it
    so that it's usable by CLIP's text encoder.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalPromptSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """

    cls_uuid: str = "clip_objectgoal"

    def __init__(
        self,
        *args: Any,
        config: "DictConfig",
        **kwargs: Any,
    ):
        self.cache = load_pickle(config.cache)
        k = list(self.cache.keys())[0]
        self._embed_dim = self.cache[k].shape[0]
        for v in self.cache.values():
            assert self._embed_dim == v.shape[0] and v.ndim == 1
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._embed_dim,), dtype=np.float32
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: Any,
        **kwargs: Any,
    ) -> Optional[int]:
        category = (
            episode.object_category if hasattr(episode, "object_category") else ""
        )
        if category not in self.cache:
            print("Missing category: {}".format(category))
        return self.cache[category]


@registry.register_sensor
class ClipImageGoalSensor(Sensor):
    cls_uuid: str = "clip_imagegoal"

    def __init__(
        self,
        sim: HabitatSim,
        config: "DictConfig",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        sensors = self._sim.sensor_suite.sensors
        rgb_sensor_uuids = [
            uuid for uuid, sensor in sensors.items() if isinstance(sensor, RGBSensor)
        ]
        if len(rgb_sensor_uuids) != 1:
            raise ValueError(
                "ImageGoalNav requires one RGB sensor,"
                f" {len(rgb_sensor_uuids)} detected"
            )
        (self._rgb_sensor_uuid,) = rgb_sensor_uuids
        super().__init__(config=config)
        self._curr_ep_id = None
        self.image_goal = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return self._sim.sensor_suite.observation_spaces.spaces[self._rgb_sensor_uuid]

    def _reset(self, episode):
        self._curr_ep_id = episode.episode_id
        sampled_goal = random.choice(episode.goals)
        sampled_viewpoint = random.choice(sampled_goal.view_points)
        observations = self._sim.get_observations_at(
            position=sampled_viewpoint.agent_state.position,
            rotation=sampled_viewpoint.agent_state.rotation,
            keep_agent_at_new_pose=False,
        )
        assert observations is not None
        self.image_goal = observations["rgb"]
        # Mutate the episode
        episode.goals = [sampled_goal]

    def get_observation(
        self,
        observations,
        episode: Any,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        if self.image_goal is None or self._curr_ep_id != episode.episode_id:
            self._reset(episode)
        assert self.image_goal is not None
        return self.image_goal


@registry.register_sensor
class ClipGoalSelectorSensor(Sensor):
    cls_uuid: str = "clip_goal_selector"

    def __init__(
        self,
        config: "DictConfig",
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(config=config)
        self._image_sampling_prob = config.image_sampling_probability
        self._curr_ep_id = None
        self._use_image_goal = True

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=np.bool,
        )

    def _reset(self, episode):
        self._curr_ep_id = episode.episode_id
        self._use_image_goal = random.random() < self._image_sampling_prob

    def get_observation(
        self,
        observations,
        episode: Any,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        if self._curr_ep_id != episode.episode_id:
            self._reset(episode)
        return np.array([self._use_image_goal], dtype=np.bool)


@registry.register_sensor
class ImageGoalRotationSensor(Sensor):
    r"""Sensor for ImageGoal observations which are used in ImageGoal Navigation.
    RGBSensor needs to be one of the Simulator sensors.
    This sensor return the rgb image taken from the goal position to reach with
    random rotation.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the ImageGoal sensor.
    """

    cls_uuid: str = "image_goal_rotation"

    def __init__(self, *args: Any, sim: Simulator, config: "DictConfig", **kwargs: Any):
        self._sim = sim
        sensors = self._sim.sensor_suite.sensors
        rgb_sensor_uuids = [
            uuid for uuid, sensor in sensors.items() if isinstance(sensor, RGBSensor)
        ]
        if len(rgb_sensor_uuids) != 1:
            raise ValueError(
                "ImageGoalNav requires one RGB sensor,"
                f" {len(rgb_sensor_uuids)} detected"
            )

        (self._rgb_sensor_uuid,) = rgb_sensor_uuids
        self._current_episode_id: Optional[str] = None
        self._current_image_goal = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return self._sim.sensor_suite.observation_spaces.spaces[self._rgb_sensor_uuid]

    def _get_pointnav_episode_image_goal(self, episode: NavigationEpisode):
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        # Add rotation to episode
        if self.config.sample_angle:
            angle = np.random.uniform(0, 2 * np.pi)
        else:
            # to be sure that the rotation is the same for the same episode_id
            # since the task is currently using pointnav Dataset.
            seed = abs(hash(episode.episode_id)) % (2**32)
            rng = np.random.RandomState(seed)
            angle = rng.uniform(0, 2 * np.pi)
        source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        episode.goals[0].rotation = source_rotation

        goal_observation = self._sim.get_observations_at(
            position=goal_position.tolist(), rotation=source_rotation
        )
        return goal_observation[self._rgb_sensor_uuid]

    def get_observation(
        self,
        *args: Any,
        observations,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_image_goal

        self._current_image_goal = self._get_pointnav_episode_image_goal(episode)
        self._current_episode_id = episode_uniq_id

        return self._current_image_goal


@registry.register_sensor
class CurrentEpisodeUUIDSensor(Sensor):
    r"""Sensor for current episode uuid observations.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the ImageGoal sensor.
    """

    cls_uuid: str = "current_episode_uuid"

    def __init__(self, *args: Any, sim: Simulator, config: "DictConfig", **kwargs: Any):
        self._sim = sim
        self._current_episode_id: Optional[str] = None

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.iinfo(np.int64).min,
            high=np.iinfo(np.int64).max,
            shape=(1,),
            dtype=np.int64,
        )

    def get_observation(
        self,
        *args: Any,
        observations,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        episode_uuid = (
            int(hashlib.sha1(episode_uniq_id.encode("utf-8")).hexdigest(), 16) % 10**8
        )
        return episode_uuid


@registry.register_sensor
class StepIDSensor(Sensor):
    cls_uuid: str = "step_id"
    curr_ep_id: str = ""
    _elapsed_steps: int = 0

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.iinfo(np.int64).min,
            high=np.iinfo(np.int64).max,
            shape=(1,),
            dtype=np.int64,
        )

    def get_observation(
        self,
        *args: Any,
        observations,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if self.curr_ep_id != episode_uniq_id:
            self.curr_ep_id = episode_uniq_id
            self._elapsed_steps = 0
        else:
            self._elapsed_steps += 1
        return self._elapsed_steps


def get_habitat_sim_action(action):
    if action == "TURN_RIGHT":
        return HabitatSimActions.turn_right     # 3
    elif action == "TURN_LEFT":
        return HabitatSimActions.turn_left      # 2
    elif action == "MOVE_FORWARD":
        return HabitatSimActions.move_forward   # 1
    elif action == "LOOK_UP":
        return HabitatSimActions.look_up        # 4
    elif action == "LOOK_DOWN":
        return HabitatSimActions.look_down      # 5
    return HabitatSimActions.stop               # 0

@registry.register_sensor
class DemonstrationSensor(Sensor):
    def __init__(self, **kwargs):
        self.uuid = "next_actions"
        self.observation_space = spaces.Discrete(1)
        self.timestep = 0
        self.prev_action = 0

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode,
        task,
        **kwargs
    ):
        # Fetch next action as observation
        if task._is_resetting:  # reset
            self.timestep = 1

        if hasattr(episode, 'reference_replay') and episode.reference_replay is not None and\
             self.timestep < len(episode.reference_replay)  and len(episode.reference_replay) > 0:
            action_name = episode.reference_replay[self.timestep].action
            action = get_habitat_sim_action(action_name)
        else:
            action = 0

        self.timestep += 1
        return action

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)

import time
@registry.register_sensor
class GaussianSplattingRGBSensor(Sensor):
    cls_uuid: str = "rgb"

    def __init__(self, *args: Any, sim: Simulator, config: "DictConfig", **kwargs: Any):
        super().__init__(config=config)

        # 存储基础配置路径，而不是具体的场景路径
        self._sim = sim
        self.gs_sim = None
        self.cam_setting = None
        self.current_gs_sim = None
        self.current_scene_name = None
        self.gs_sim_cache = {}

        # 初始化相机设置（使用配置中的参数）
        width = getattr(config, 'width', 640)
        height = getattr(config, 'height', 480)
        hfov = getattr(config, 'hfov', 79)
        self.cam_setting = habitat_sensor_to_CamSettings(width, height, hfov)

        data_platform_dir = getattr(config, 'data_platform_dir', './data/3dgs')
        self.scene_base_path = getattr(config, 'reconstruction_scene_assets_dir', 'data/reconstruction_scene_assets/')

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args, **kwargs) -> Space:
        # 返回观测空间定义
        # 假设是640x480的RGB图像
        height = getattr(self.config, 'height', 480)
        width = getattr(self.config, 'width', 640)
        return spaces.Box(
            low=0,
            high=255,
            shape=(height, width, 3),
            dtype=np.uint8,
        )

    
    def _set_scene(self, scene_name: str):
        """设置当前场景（使用缓存）"""
        if self.current_scene_name == scene_name:
            return  # 场景未改变
        
        self.current_gs_sim = self._get_scene_gs_sim(scene_name)
        self.current_scene_name = scene_name
   
        
    def _get_scene_gs_sim(self, scene_name: str):
        """获取指定场景的GS模拟器实例(带缓存)"""
        if scene_name in self.gs_sim_cache:
            # 场景已在缓存中
            return self.gs_sim_cache[scene_name]

        print("Loading new 3dgs scene:", scene_name)

        splat_ply_path = os.path.join(
            self.scene_base_path,
            scene_name,
            "semantic",
            "splat.semantic.ply"
        )
        if not os.path.exists(splat_ply_path):
            raise FileNotFoundError(f"PLY file not found: {splat_ply_path}")

        habitat_transform_path = os.path.join(
            self.scene_base_path,
            scene_name,
            "anno_res",
            "to_habitat.txt"
        )
        if not os.path.exists(habitat_transform_path):
            raise FileNotFoundError(f"To habitat file not found: {habitat_transform_path}")

        floor_transform_path = os.path.join(
            self.scene_base_path,
            scene_name,
            "anno_res",
            "floor_transform.txt"
        )
        if not os.path.exists(floor_transform_path):
            raise FileNotFoundError(f"Floor transform file not found: {floor_transform_path}")

        anno_res_dir = os.path.join(
            self.scene_base_path,
            scene_name,
            "anno_res"
        )
        if not os.path.exists(anno_res_dir):
            raise FileNotFoundError(f"Anno file not found: {anno_res_dir}")

        gs_sim = GsSimulator(
            splat_ply_path=splat_ply_path,
            habitat_transform_path=habitat_transform_path,
            floor_transform_path=floor_transform_path,
            anno_res_dir=anno_res_dir
        )

        gs_sim.set_default_cam_setting(self.cam_setting)
        self.gs_sim_cache[scene_name] = gs_sim
        return gs_sim
        
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def get_observation(
        self, 
        observations, 
        episode: NavigationEpisode, 
        *args: Any, 
        **kwargs: Any
    ):
        scene_id  = episode.scene_id
        scene_name = os.path.basename(scene_id).replace('.glb', '')
        self._set_scene(scene_name)
        
        agent_state = self._sim.get_agent_state().sensor_states["rgb"]
        vpc = construct_viewpoint_cam_from_agent_state(
            agent_state = agent_state,
            cam_setting = self.current_gs_sim.default_cam_setting,
            gs_habitat_transform = self.current_gs_sim.habitat_transform,
            gs_floor_transform = self.current_gs_sim.floor_transform
        )

        gs_image = self.current_gs_sim.get_observations(
                    vpc,
                    request_semantic=False,
                    request_semantic_rgb=False,
                    request_instance=False,
                    request_instance_rgb=False,
                )

        return np.array(gs_image["rgb"])

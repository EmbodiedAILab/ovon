import copy
import os
import random
import time
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from habitat import logger
from habitat.config import read_write
from habitat.utils import profiling_wrapper
from habitat_baselines import VERTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.rl.ddppo.algo import DDPPO
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    add_signal_handlers,
    get_distrib_size,
    get_free_port_distributed,
    get_main_addr,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet  # noqa: F401.
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.rl.ver.environment_worker import (
    build_action_plugin_from_policy_action_space,
    construct_environment_workers,
)
from ovon.utils.cloudrobo_environment_worker import (
    construct_cloudrobo_environment_workers,
)
from habitat_baselines.rl.ver.preemption_decider import PreemptionDeciderWorker
from habitat_baselines.rl.ver.task_enums import ReportWorkerTasks
from ovon.utils.cloudrobo_report_worker import CloudRoboReportWorker
from habitat_baselines.rl.ver.timing import Timing
from habitat_baselines.rl.ver.ver_rollout_storage import VERRolloutStorage
from habitat_baselines.rl.ver.worker_common import (
    InferenceWorkerSync,
    WorkerBase,
    WorkerQueues,
)
from habitat_baselines.utils.common import (
    cosine_decay,
    get_num_actions,
    inference_mode,
    is_continuous_action_space,
    batch_obs,
    generate_video,
)
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LambdaLR

from ovon.algos.ppo import DDPPO as DDPPOPirlNav
from ovon.algos.ppo import PPO as PPOPirlNav
from ovon.trainers.inference_worker_with_kv import (
    InferenceWorkerWithKV,
    InferenceWorkerWithKVProcess,
)
from ovon.trainers.ver_rollout_storage_with_kv import VERRolloutStorageWithKVCache
from ovon.utils.lr_scheduler import PIRLNavLRScheduler


import numpy as np
import tqdm
from omegaconf import OmegaConf
from collections import defaultdict

from habitat.config.default import get_agent_config
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils.render_wrapper import overlay_frame
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,

)
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
)


from ovon.utils.record_action import record_trajectory_start, record_trajectory_process, record_trajectory_close

try:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
except AttributeError:
    pass


@baseline_registry.register_trainer(name="ver_transformer")
class VERTransformerTrainer(VERTrainer):
    """
    If the actor critic is NOT a transformer, the trainer will be the same as the
    VERTrainer.
    """
    episode_finish: bool = False

    def _setup_actor_critic_agent(self, ppo_cfg: "DictConfig") -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        policy_cfg = self.config.habitat_baselines.rl.policy
        if not policy_cfg.finetune.enabled:
            return super()._setup_actor_critic_agent(ppo_cfg)

        logger.add_filehandler(self.config.habitat_baselines.log_file)

        policy = baseline_registry.get_policy(
            self.config.habitat_baselines.rl.policy.name
        )
        observation_space = self.obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        self.actor_critic = policy.from_config(
            self.config,
            observation_space,
            self.policy_action_space,
            orig_action_space=self.orig_policy_action_space,
        )
        self.obs_space = observation_space
        self.actor_critic.to(self.device)

        if (
            self.config.habitat_baselines.rl.ddppo.pretrained_encoder
            or self.config.habitat_baselines.rl.ddppo.pretrained
        ):
            pretrained_state = torch.load(
                self.config.habitat_baselines.rl.ddppo.pretrained_weights,
                map_location="cpu",
            )

        if self.config.habitat_baselines.rl.ddppo.pretrained:
            missing_keys = self.actor_critic.load_state_dict(
                {  # type: ignore
                    k[len("actor_critic.") :]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
            logger.info("Loading pretrained weights: {}".format(missing_keys))
        elif self.config.habitat_baselines.rl.ddppo.pretrained_encoder:
            prefix = "actor_critic.net.visual_encoder."
            self.actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

        if not self.config.habitat_baselines.rl.ddppo.train_encoder:
            self._static_encoder = True
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        policy_cfg = self.config.habitat_baselines.rl.policy
        if hasattr(policy_cfg, "finetune") and policy_cfg.finetune.enabled:
            self.actor_critic.freeze_visual_encoders()
            self.actor_critic.freeze_state_encoder()
            self.actor_critic.freeze_actor()
            self.actor_critic.freeze_new_params()
            logger.info("Freezing all parameters for PIRLNav finetuning")

        if self.config.habitat_baselines.rl.ddppo.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        if hasattr(policy_cfg, "finetune") and policy_cfg.finetune.enabled:
            self.agent = (
                DDPPOPirlNav if self._is_distributed else PPOPirlNav
            ).from_config(self.actor_critic, ppo_cfg)
        else:
            self.agent = (DDPPO if self._is_distributed else PPO).from_config(
                self.actor_critic, ppo_cfg
            )

    def _init_train(self, resume_state):
        r"""Copy of VERTrainer._init_train, but the rollout storage will use the
        VERRolloutStorageWithKV rollout class if the policy is transformer-based. This
        method will also change the inference workers to a different variant if the
        policy is transformer-based. These different workers will provide a KV cache
        during onling inference instead of the usual RNN hidden state."""
        if self._is_distributed:
            local_rank, world_rank, _ = get_distrib_size()

            with read_write(self.config):
                self.config.habitat_baselines.torch_gpu_id = local_rank
                self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = local_rank
                # Multiply by the number of simulators to make sure they also get unique
                # seeds
                self.config.habitat.seed += (
                    world_rank * self.config.habitat_baselines.num_environments
                )

        random.seed(self.config.habitat.seed)
        np.random.seed(self.config.habitat.seed)
        torch.manual_seed(self.config.habitat.seed)

        self.mp_ctx = torch.multiprocessing.get_context("forkserver")
        self.queues = WorkerQueues(self.config.habitat_baselines.num_environments)
        self.environment_workers = construct_cloudrobo_environment_workers(
            self.config, self.mp_ctx, self.queues
        )
        [ew.start() for ew in self.environment_workers]
        [ew.reset() for ew in self.environment_workers]

        if self.config.habitat_baselines.rl.ddppo.force_distributed:
            self._is_distributed = True

        if is_slurm_batch_job():
            add_signal_handlers()

        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.habitat_baselines.rl.ddppo.distrib_backend
            )
            if rank0_only():
                logger.info(
                    "Initialized VER+DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )
        else:
            logger.info("Initialized VER")
            tcp_store = None

        self._last_should_end_val = None
        self._last_should_end_calc_time = 0

        if rank0_only() and self.config.habitat_baselines.verbose:
            logger.info(f"config: {self.config}")

        profiling_wrapper.configure(
            capture_start_step=self.config.habitat_baselines.profiling.capture_start_step,
            num_steps_to_capture=self.config.habitat_baselines.profiling.num_steps_to_capture,
        )

        self._all_workers: List[WorkerBase] = []

        if self._is_distributed:
            world_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            world_rank = 0
            world_size = 1

        self._my_t_zero = time.perf_counter()
        self.preemption_decider = PreemptionDeciderWorker(
            self.mp_ctx,
            get_main_addr(),
            get_free_port_distributed("preemption", tcp_store),
            world_rank,
            world_size,
            self.config,
            self.queues,
            self._my_t_zero,
        )

        has_report_resume_state = (
            resume_state is not None
            and "report_worker_state" in resume_state["requeue_stats"]
        )
        run_id = None
        if (
            has_report_resume_state
            and resume_state["requeue_stats"]["report_worker_state"] is not None
        ):
            run_id = resume_state["requeue_stats"]["report_worker_state"]["run_id"]

        self.report_worker = CloudRoboReportWorker(
            self.mp_ctx,
            get_free_port_distributed("report", tcp_store),
            self.config,
            self.queues.report,
            self._my_t_zero,
            self.num_steps_done,
            run_id=run_id,
        )

        if has_report_resume_state:
            self.report_worker.load_state_dict(
                resume_state["requeue_stats"]["report_worker_state"]
            )

        init_reports = [self.environment_workers[0].get_init_report()]

        action_space = init_reports[0]["act_space"]

        self.policy_action_space = action_space
        self.orig_policy_action_space = None

        [
            ew.set_action_plugin(
                build_action_plugin_from_policy_action_space(self.policy_action_space)
            )
            for ew in self.environment_workers
        ]
        if is_continuous_action_space(action_space):
            # Assume ALL actions are NOT discrete
            action_shape = (get_num_actions(action_space),)
            discrete_actions = False
        else:
            # For discrete pointnav
            action_shape = (1,)
            discrete_actions = True

        ppo_cfg = self.config.habitat_baselines.rl.ppo
        if torch.cuda.is_available():
            self.device = torch.device(
                "cuda", self.config.habitat_baselines.torch_gpu_id
            )
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if rank0_only() and not os.path.exists(
            self.config.habitat_baselines.checkpoint_folder
        ):
            os.makedirs(self.config.habitat_baselines.checkpoint_folder)

        actor_obs_space = init_reports[0]["obs_space"]
        self.obs_space = copy.deepcopy(actor_obs_space)
        self.ver_config = self.config.habitat_baselines.rl.ver

        self._setup_actor_critic_agent(ppo_cfg)
        """BEGINNING OF CHANGES RELATIVE TO VERTrainer._init_train"""
        self._is_transformer = getattr(self.actor_critic, "is_transformer", False)
        """END OF CHANGES RELATIVE TO VERTrainer._init_train"""
        if resume_state is not None:
            self.agent.load_state_dict(resume_state["state_dict"])
            self.agent.optimizer.load_state_dict(resume_state["optim_state"])

        rollouts_obs_space = copy.deepcopy(self.obs_space)
        if self._static_encoder and hasattr(self.actor_critic, "net"):
            self._encoder = self.actor_critic.net.visual_encoder
            rollouts_obs_space = spaces.Dict(
                {
                    PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY: spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self._encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **rollouts_obs_space.spaces,
                }
            )

        n_inference_workers = self.ver_config.num_inference_workers
        main_is_iw = not self.ver_config.overlap_rollouts_and_learn
        with inference_mode():
            storage_kwargs = dict(
                variable_experience=self.ver_config.variable_experience,
                numsteps=ppo_cfg.num_steps,
                num_envs=len(self.environment_workers),
                action_space=self.policy_action_space,
                recurrent_hidden_state_size=ppo_cfg.hidden_size,
                num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
                action_shape=action_shape,
                discrete_actions=discrete_actions,
                observation_space=rollouts_obs_space,
            )
            """BEGINNING OF CHANGES RELATIVE TO VERTrainer._init_train"""
            if self._is_transformer:
                tf_cfg = self.config.habitat_baselines.rl.policy.transformer_config
                storage_kwargs["num_layers"] = tf_cfg.n_layers
                storage_kwargs["num_heads"] = tf_cfg.n_heads
                storage_kwargs["max_context_length"] = tf_cfg.max_context_length
                storage_kwargs["head_dim"] = tf_cfg.n_hidden // tf_cfg.n_heads
                storage_kwargs["recurrent_hidden_state_size"] = 1
                storage_kwargs["num_recurrent_layers"] = 1

                rollout_storage_class = VERRolloutStorageWithKVCache
            else:
                rollout_storage_class = VERRolloutStorage
            """END OF CHANGES RELATIVE TO VERTrainer._init_train"""

            self.rollouts = rollout_storage_class(**storage_kwargs)
            self.rollouts.to(self.device)
            self.rollouts.share_memory_()
            if self.ver_config.overlap_rollouts_and_learn:
                self.learning_rollouts = rollout_storage_class(**storage_kwargs)
                self.learning_rollouts.to(self.device)
            else:
                self.learning_rollouts = self.rollouts

            storage_kwargs["observation_space"] = actor_obs_space
            storage_kwargs["numsteps"] = 1

            self._transfer_buffers = (
                rollout_storage_class(**storage_kwargs)
                .buffers.slice_keys(
                    "rewards",
                    "masks",
                    "observations",
                    "episode_ids",
                    "environment_ids",
                    "actions",
                    "step_ids",
                )
                .map_in_place(lambda t: t.share_memory_())
            )[
                (
                    slice(0, len(self.environment_workers))
                    if self.ver_config.variable_experience
                    else 0
                )
            ]

        self.actor_critic.share_memory()

        if self._is_distributed:
            self.agent.init_distributed(find_unused_params=True)

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.actor_critic.parameters())
            )
        )
        self._iw_sync = InferenceWorkerSync(
            self.mp_ctx,
            n_inference_workers,
        )

        inference_worker_args = (
            n_inference_workers,
            self.config,
            self.queues,
            self._iw_sync,
            self._transfer_buffers,
            self.config.habitat_baselines.rl.policy.name,
            (self.config, self.obs_space, self.policy_action_space),
            self.device,
            self.preemption_decider.rollout_ends,
        )

        self._transfer_policy_tensors = list(self.actor_critic.all_policy_tensors())

        self.inference_workers = [
            InferenceWorkerWithKV(
                self.mp_ctx, self._is_transformer, i, *inference_worker_args
            )
            for i in range(1 if main_is_iw else 0, n_inference_workers)
        ]
        if main_is_iw:
            self._inference_worker_impl = InferenceWorkerWithKVProcess(
                None,
                None,
                None,
                0,
                *inference_worker_args,
            )
            self._inference_worker_impl.set_actor_critic_tensors(
                self._transfer_policy_tensors
            )
            self._inference_worker_impl.set_rollouts(self.rollouts)
        else:
            self._inference_worker_impl = None

        for iw in self.inference_workers:
            # We send the policy weights and the rollouts
            # via a torch.multiprocessing.SimpleQueue instead
            # of in the constructor as otherwise the shared
            # cuda tensors don't get properly freed on
            # destruction which causes an error.
            iw.set_actor_critic_tensors(self._transfer_policy_tensors)
            iw.set_rollouts(self.rollouts)
            iw.start()

        ews_to_wait = []
        for i, ew in enumerate(self.environment_workers):
            ew.set_transfer_buffers(self._transfer_buffers)
            if i > 0:
                init_reports.append(ew.get_init_report())

            ew.wait_start()
            ews_to_wait.append(ew)
            if len(ews_to_wait) >= 4:
                [a.wait_sync() for a in ews_to_wait]
                ews_to_wait = []

        [a.wait_sync() for a in ews_to_wait]
        ews_to_wait = []

        if self._is_distributed:
            torch.distributed.barrier()
        [aw.start_experience_collection() for aw in self.environment_workers]
        self.report_worker.start_collection()

        self.timer = Timing()

        self._all_workers.extend(self.environment_workers)
        self._all_workers.extend(self.inference_workers)
        self._all_workers.append(self.report_worker)
        self._all_workers.append(self.preemption_decider)

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training VER using PIRLNav LR scheduler.

        Returns:
            None
        """
        policy_cfg = self.config.habitat_baselines.rl.policy
        if not policy_cfg.finetune.enabled:
            return self.ver_train()
            # return super().train()

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        self.num_steps_done = 0
        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.config = resume_state["config"]

            requeue_stats = resume_state["requeue_stats"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]

        self._init_train(resume_state)

        count_checkpoints = 0
        policy_cfg = self.config.habitat_baselines.rl.policy

        if hasattr(policy_cfg, "finetune") and policy_cfg.finetune.enabled:
            lr_scheduler = PIRLNavLRScheduler(
                optimizer=self.agent.optimizer,
                agent=self.agent,
                num_updates=self.config.habitat_baselines.num_updates,
                base_lr=self.config.habitat_baselines.rl.ppo.lr,
                finetuning_lr=policy_cfg.finetune.lr,
                ppo_eps=self.config.habitat_baselines.rl.ppo.eps,
                start_actor_update_at=policy_cfg.finetune.start_actor_update_at,
                start_actor_warmup_at=policy_cfg.finetune.start_actor_warmup_at,
                start_critic_update_at=policy_cfg.finetune.start_critic_update_at,
                start_critic_warmup_at=policy_cfg.finetune.start_critic_warmup_at,
            )
            logger.info("Setting up for PIRLNav scheduler")
        else:
            lr_scheduler = LambdaLR(
                optimizer=self.agent.optimizer,
                lr_lambda=lambda x: cosine_decay(self.percent_done()),
            )
            logger.info("LambdaLR scheduler")

        if resume_state is not None:
            lr_scheduler.load_state_dict(resume_state["lr_sched_state"])

            requeue_stats = resume_state["requeue_stats"]
            self._last_checkpoint_percent = requeue_stats["_last_checkpoint_percent"]
            count_checkpoints = requeue_stats["count_checkpoints"]

        ppo_cfg = self.config.habitat_baselines.rl.ppo
        if self.ver_config.overlap_rollouts_and_learn:
            self.preemption_decider.start_rollout()

        while not self.is_done():
            profiling_wrapper.on_start_step()

            if ppo_cfg.use_linear_clip_decay:
                self.agent.clip_param = ppo_cfg.clip_param * (1 - self.percent_done())

            if rank0_only() and self._should_save_resume_state():
                requeue_stats = dict(
                    count_checkpoints=count_checkpoints,
                    num_steps_done=self.num_steps_done,
                    num_updates_done=self.num_updates_done,
                    _last_checkpoint_percent=self._last_checkpoint_percent,
                    report_worker_state=self.report_worker.state_dict(),
                )
                resume_state = dict(
                    state_dict=self.agent.state_dict(),
                    optim_state=self.agent.optimizer.state_dict(),
                    lr_sched_state=lr_scheduler.state_dict(),
                    config=self.config,
                    requeue_stats=requeue_stats,
                )

                save_resume_state(
                    resume_state,
                    self.config,
                )

            if EXIT.is_set():
                profiling_wrapper.range_pop()  # train update
                [w.close() for w in self._all_workers]
                [w.join() for w in self._all_workers]

                requeue_job()
                break

            with inference_mode():
                if not self.ver_config.overlap_rollouts_and_learn:
                    self.preemption_decider.start_rollout()
                    while not self.rollouts.rollout_done:
                        self._inference_worker_impl.try_one_step()

                    self._inference_worker_impl.finish_rollout()

                self._iw_sync.rollout_done.wait()
                self._iw_sync.rollout_done.clear()

                if self._iw_sync.all_workers.n_waiting > 0:
                    raise RuntimeError(
                        f"{self._iw_sync.all_workers.n_waiting} inference worker(s)"
                        " is(are) still waiting on the IW barrier. Likely they never"
                        " waited on it.\n"
                    )

                self.rollouts.after_rollout()

                if self.ver_config.overlap_rollouts_and_learn:
                    with self.timer.avg_time("overlap_transfers"):
                        self.learning_rollouts.copy(self.rollouts)

                self.preemption_decider.end_rollout(self.rollouts.num_steps_to_collect)

                self.queues.report.put(
                    (
                        ReportWorkerTasks.num_steps_collected,
                        int(self.rollouts.num_steps_collected),
                    )
                )

                if self.ver_config.overlap_rollouts_and_learn:
                    with self.timer.avg_time("overlap_transfers"):
                        self.rollouts.after_update()
                        self._iw_sync.should_start_next.set()
                        self.preemption_decider.start_rollout()

            losses = self._update_agent()
            lrs = {}
            for i, param_group in enumerate(self.agent.optimizer.param_groups):
                lrs["lr_{}".format(i)] = param_group["lr"]

            learner_metrics = {
                **losses,
                **lrs,
            }

            self.preemption_decider.learner_time(self._learning_time)

            self.queues.report.put_many(
                (
                    (
                        ReportWorkerTasks.learner_timing,
                        self.timer,
                    ),
                    (
                        ReportWorkerTasks.learner_update,
                        learner_metrics,
                    ),
                )
            )
            self.timer = Timing()

            if ppo_cfg.use_linear_lr_decay:
                lr_scheduler.step()  # type: ignore

            self.num_steps_done = int(self.report_worker.num_steps_done)

            self.num_updates_done += 1
            # checkpoint model
            if rank0_only() and self.should_checkpoint():
                self.save_checkpoint(
                    f"ckpt.{count_checkpoints}.pth",
                    dict(
                        step=self.num_steps_done,
                        wall_time=self.report_worker.time_taken,
                    ),
                )
                count_checkpoints += 1

        self.window_episode_stats = self.report_worker.get_window_episode_stats()

        [w.close() for w in self._all_workers]
        [w.join() for w in self._all_workers]

        if self._is_distributed:
            torch.distributed.barrier()

    @rank0_only
    def _training_log(self, writer, losses: Dict[str, float], prev_time: int = 0):
        deltas = {
            k: (v[-1] - v[0]).sum().item() if len(v) > 1 else v[0].sum().item()
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        writer.add_scalar(
            "reward",
            deltas["reward"] / deltas["count"],
            self.num_steps_done,
        )

        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }

        for k, v in metrics.items():
            writer.add_scalar(f"metrics/{k}", v, self.num_steps_done)
        for k, v in losses.items():
            writer.add_scalar(f"learner/{k}", v, self.num_steps_done)

        fps = self.num_steps_done / ((time.time() - self.t_start) + prev_time)
        writer.add_scalar("perf/fps", fps, self.num_steps_done)

        lrs = {}
        for i, param_group in enumerate(self.agent.optimizer.param_groups):
            lrs["pg_{}".format(i)] = param_group["lr"]
        writer.add_scalars("learning_rate", lrs, self.num_steps_done)

        # log stats
        if self.num_updates_done % self.config.habitat_baselines.log_interval == 0:
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    fps,
                )
            )

            logger.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\tframes: {}".format(
                    self.num_updates_done,
                    self.env_time,
                    self.pth_time,
                    self.num_steps_done,
                )
            )

            logger.info(
                "Average window size: {}  {}".format(
                    len(self.window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
                    ),
                )
            )

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Map location CPU is almost always better than mapping to a CUDA device.
        if self.config.habitat_baselines.eval.should_load_ckpt:
            ckpt_dict = self.load_checkpoint(
                checkpoint_path, map_location="cpu"
            )
            step_id = ckpt_dict["extra_state"]["step"]
            print(step_id)
        else:
            ckpt_dict = {"config": None}

        config = self._get_resume_state_config_or_new_config(
            ckpt_dict["config"]
        )

        ppo_cfg = config.habitat_baselines.rl.ppo

        with read_write(config):
            config.habitat.dataset.split = config.habitat_baselines.eval.split

        if (
            len(config.habitat_baselines.video_render_views) > 0
            and len(self.config.habitat_baselines.eval.video_option) > 0
        ):
            agent_config = get_agent_config(config.habitat.simulator)
            agent_sensors = agent_config.sim_sensors
            render_view_uuids = [
                agent_sensors[render_view].uuid
                for render_view in config.habitat_baselines.video_render_views
                if render_view in agent_sensors
            ]
            assert len(render_view_uuids) > 0, (
                f"Missing render sensors in agent config: "
                f"{config.habitat_baselines.video_render_views}."
            )
            with read_write(config):
                for render_view_uuid in render_view_uuids:
                    if render_view_uuid not in config.habitat.gym.obs_keys:
                        config.habitat.gym.obs_keys.append(render_view_uuid)
                config.habitat.simulator.debug_render = True

        if config.habitat_baselines.verbose:
            logger.info(f"env config: {OmegaConf.to_yaml(config)}")

        self._init_envs(config, is_eval=True)

        action_space = self.envs.action_spaces[0]
        self.policy_action_space = action_space
        self.orig_policy_action_space = self.envs.orig_action_spaces[0]
        if is_continuous_action_space(action_space):
            # Assume NONE of the actions are discrete
            action_shape = (get_num_actions(action_space),)
            discrete_actions = False
        else:
            # For discrete pointnav
            action_shape = (1,)
            discrete_actions = True

        self._setup_actor_critic_agent(ppo_cfg)

        if self.agent.actor_critic.should_load_agent_state:
            self.agent.load_state_dict(ckpt_dict["state_dict"], strict=False)
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        
        record_trajectory_start(self.config.habitat.dataset.data_path, self.config.habitat_baselines.eval.split)

        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device="cpu"
        )

        test_recurrent_hidden_states = torch.zeros(
            self.config.habitat_baselines.num_environments,
            self.actor_critic.num_recurrent_layers,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.habitat_baselines.num_environments,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            self.config.habitat_baselines.num_environments,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        rgb_frames = [
            [] for _ in range(self.config.habitat_baselines.num_environments)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(self.config.habitat_baselines.video_dir, exist_ok=True)

        number_of_eval_episodes = (
            self.config.habitat_baselines.test_episode_count
        )
        evals_per_ep = self.config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        self.actor_critic.eval()
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and self.envs.num_envs > 0
        ):
            current_episodes_info = self.envs.current_episodes()

            with inference_mode():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                prev_actions.copy_(actions)  # type: ignore
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if is_continuous_action_space(self.policy_action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        self.policy_action_space.low,
                        self.policy_action_space.high,
                    )
                    for a in actions.cpu()
                ]
            else:
                step_data = [a.item() for a in actions.cpu()]

            record_trajectory_process(step_data, self.envs.current_episodes())

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            policy_info = self.actor_critic.get_policy_info(infos, dones)
            for i in range(len(policy_info)):
                infos[i].update(policy_info[i])
            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)

                if len(self.config.habitat_baselines.eval.video_option) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, infos[i]
                    )
                    if not not_done_masks[i].item():
                        # The last frame corresponds to the first frame of the next episode
                        # but the info is correct. So we use a black frame
                        frame = observations_to_image(
                            {k: v[i] * 0.0 for k, v in batch.items()}, infos[i]
                        )
                    frame = overlay_frame(frame, infos[i])
                    rgb_frames[i].append(frame)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    if (
                        len(self.config.habitat_baselines.eval.video_option)
                        > 0
                    ):
                        generate_video(
                            video_option=self.config.habitat_baselines.eval.video_option,
                            video_dir=self.config.habitat_baselines.video_dir,
                            images=rgb_frames[i],
                            episode_id=current_episodes_info[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            fps=self.config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=self.config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        rgb_frames[i] = []

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            self.config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        pbar.close()
        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values()]
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        self.envs.close()

        record_trajectory_close(self.config.habitat_baselines.eval.traj_dir)


    def is_done(self) -> bool:
        local_done = self.episode_finish or (self.percent_done() >= 1.0)
        local_rank, world_size, _ = get_distrib_size() # 假设这个函数能返回rank和world_size

        if local_rank == 0:
            # Rank 0 收集所有信息或自行判断全局条件
            # 这里简化处理，假设Rank 0根据自身状态和已知信息决定是否全部结束
            global_done = local_done # 更复杂的逻辑可以在这里实现
            signal_tensor = torch.tensor([global_done], dtype=torch.int, device='cuda')
        else:
            signal_tensor = torch.tensor([0], dtype=torch.int, device='cuda')

        # Rank 0 将决策广播给所有其他进程
        torch.distributed.broadcast(signal_tensor, src=0)
        
        if signal_tensor.item() == 1:
            print(f"Rank {local_rank} received broadcast termination signal.")
            torch.distributed.barrier()
            return True
        else:
            return False

    @profiling_wrapper.RangeContext("train")
    def ver_train(self) -> None:
        r"""Main method for training DD/PPO.

        Returns:
            None
        """
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        self.num_steps_done = 0
        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.config = resume_state["config"]

            requeue_stats = resume_state["requeue_stats"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]

        self._init_train(resume_state)

        count_checkpoints = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: cosine_decay(self.percent_done()),
        )

        if resume_state is not None:
            lr_scheduler.load_state_dict(resume_state["lr_sched_state"])

            requeue_stats = resume_state["requeue_stats"]
            self._last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"
            ]
            count_checkpoints = requeue_stats["count_checkpoints"]

        ppo_cfg = self.config.habitat_baselines.rl.ppo
        if self.ver_config.overlap_rollouts_and_learn:
            self.preemption_decider.start_rollout()

        while not self.is_done():
            profiling_wrapper.on_start_step()

            if ppo_cfg.use_linear_clip_decay:
                self.agent.clip_param = ppo_cfg.clip_param * (
                    1 - self.percent_done()
                )

            if rank0_only() and self._should_save_resume_state():
                requeue_stats = dict(
                    count_checkpoints=count_checkpoints,
                    num_steps_done=self.num_steps_done,
                    num_updates_done=self.num_updates_done,
                    _last_checkpoint_percent=self._last_checkpoint_percent,
                    report_worker_state=self.report_worker.state_dict(),
                )
                resume_state = dict(
                    state_dict=self.agent.state_dict(),
                    optim_state=self.agent.optimizer.state_dict(),
                    lr_sched_state=lr_scheduler.state_dict(),
                    config=self.config,
                    requeue_stats=requeue_stats,
                )

                save_resume_state(
                    resume_state,
                    self.config,
                )

            if EXIT.is_set():
                profiling_wrapper.range_pop()  # train update
                [w.close() for w in self._all_workers]
                [w.join() for w in self._all_workers]

                requeue_job()
                break

            with inference_mode():
                if not self.ver_config.overlap_rollouts_and_learn:
                    self.preemption_decider.start_rollout()
                    while not self.rollouts.rollout_done:
                        self._inference_worker_impl.try_one_step()

                    self._inference_worker_impl.finish_rollout()

                self._iw_sync.rollout_done.wait()
                self._iw_sync.rollout_done.clear()

                if self._iw_sync.all_workers.n_waiting > 0:
                    raise RuntimeError(
                        f"{self._iw_sync.all_workers.n_waiting} inference worker(s) is(are) still waiting on the IW barrier. "
                        "Likely they never waited on it.\n"
                    )

                self.rollouts.after_rollout()

                if self.ver_config.overlap_rollouts_and_learn:
                    with self.timer.avg_time("overlap_transfers"):
                        self.learning_rollouts.copy(self.rollouts)

                self.preemption_decider.end_rollout(
                    self.rollouts.num_steps_to_collect
                )

                self.queues.report.put(
                    (
                        ReportWorkerTasks.num_steps_collected,
                        int(self.rollouts.num_steps_collected),
                    )
                )

                if self.ver_config.overlap_rollouts_and_learn:
                    with self.timer.avg_time("overlap_transfers"):
                        self.rollouts.after_update()
                        self._iw_sync.should_start_next.set()
                        self.preemption_decider.start_rollout()

            losses = self._update_agent()

            self.preemption_decider.learner_time(self._learning_time)

            self.queues.report.put_many(
                (
                    (
                        ReportWorkerTasks.learner_timing,
                        self.timer,
                    ),
                    (
                        ReportWorkerTasks.learner_update,
                        losses,
                    ),
                )
            )
            self.timer = Timing()

            if ppo_cfg.use_linear_lr_decay:
                lr_scheduler.step()  # type: ignore

            self.num_steps_done = int(self.report_worker.num_steps_done)

            self.num_updates_done += 1
            
            if rank0_only() and self.report_worker.get_env_status():
                print("episode finish")
                print("All processes reported episode finish. Saving checkpoint and exiting...")
                self.save_checkpoint(
                    f"ckpt.{count_checkpoints}.pth",
                    dict(
                        step=self.num_steps_done,
                        wall_time=self.report_worker.time_taken,
                    ),
                )
                count_checkpoints += 1
                self.episode_finish = True

            # checkpoint model
            if rank0_only() and self.should_checkpoint():
                self.save_checkpoint(
                    f"ckpt.{count_checkpoints}.pth",
                    dict(
                        step=self.num_steps_done,
                        wall_time=self.report_worker.time_taken,
                    ),
                )
                count_checkpoints += 1

        self.window_episode_stats = (
            self.report_worker.get_window_episode_stats()
        )

        [w.close() for w in self._all_workers]
        [w.join() for w in self._all_workers]

        if self._is_distributed:
            torch.distributed.barrier()
# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional

from .data_plane import TransferQueueDataPlane
from .registry import AdapterRegistry
from .staleness import StalenessManager
from .types import PartitionMetadata, SampleRecord, TrainingContext
from .workers import (
    AdvantageWorker,
    AsyncRollouter,
    RewardWorker,
    ToolManagerFactory,
    TrainerScheduler,
    TrainerStepResult,
    TrainerWorker,
)


@dataclass
class BaseRLPipelineConfig:
    """Runtime knobs for the MVP async RL pipeline.

    The first version follows the short_math_grpo client pattern:
    train one LoRA adapter with MultiLoraTransformersModel, save adapter
    weights after each train partition, and pass that saved path to rollout.
    """

    tenant_id: str = 'default_tenant'
    training_run_id: str = 'default_run'
    base_model_id: str = ''
    adapter_name: str = 'default'
    reward_type: str = 'default'
    loss_type: str = 'grpo'
    algorithm: str = 'grpo'
    env_type: str = 'tool_calling'
    tool_profile: str = 'default'
    max_staleness: int = 0
    max_concurrent_groups: int = 16
    target_groups_per_partition: int = 1
    reward_batch_size: int = 1024
    advantage_batch_size: int = 1024
    max_train_partitions: Optional[int] = None
    save_name_prefix: str = 'async-rl-sampler-weights'
    save_optimizer: bool = False
    is_sampler_checkpoint: bool = True
    max_grad_norm: float = 1.0
    norm_type: int = 2
    train_kwargs: Dict[str, Any] = field(default_factory=dict)


class BaseRLPipeline:
    """Compose rollout, TQ data plane, reward, advantage, and trainer workers.

    This class is intentionally a thin orchestrator. Vertical tasks still own
    rollout behavior, reward logic, advantage logic, and model configuration.
    The default train step is compatible with `MultiLoraTransformersModel`.
    """

    def __init__(
        self,
        *,
        config: BaseRLPipelineConfig,
        model: Any,
        rollout: Any,
        reward_registry: Dict[str, Callable[..., list[float]]],
        data_plane: Optional[TransferQueueDataPlane] = None,
        adapter_registry: Optional[AdapterRegistry] = None,
        staleness_manager: Optional[StalenessManager] = None,
        tool_manager_factory: Optional[ToolManagerFactory] = None,
        advantage_fn: Optional[Callable[[list[SampleRecord], TrainingContext], tuple[list[float], list[float]]]] = None,
        rollout_policy: Optional[Any] = None,
        train_policy: Optional[Any] = None,
        train_partition_fn: Optional[Callable[[TrainingContext, str, Any], TrainerStepResult | Dict[str, Any] | None]] = None,
        receive_weights_fn: Optional[Callable[[TrainingContext], None]] = None,
    ):
        self.config = config
        self.model = model
        self.context = TrainingContext(
            tenant_id=config.tenant_id,
            training_run_id=config.training_run_id,
            base_model_id=config.base_model_id,
            adapter_name=config.adapter_name,
            reward_type=config.reward_type,
            loss_type=config.loss_type,
            algorithm=config.algorithm,
            env_type=config.env_type,
            tool_profile=config.tool_profile,
        )

        self.data_plane = data_plane or TransferQueueDataPlane()
        self.adapter_registry = adapter_registry or AdapterRegistry()
        self.staleness_manager = staleness_manager or StalenessManager(
            max_staleness=config.max_staleness,
            target_groups_per_partition=config.target_groups_per_partition,
        )
        self.adapter_registry.register(self.context)
        self.data_plane.init_namespace(self.context)

        self.rollouter = AsyncRollouter(
            data_plane=self.data_plane,
            adapter_registry=self.adapter_registry,
            staleness_manager=self.staleness_manager,
            rollout=rollout,
            tool_manager_factory=tool_manager_factory,
            rollout_policy=rollout_policy,
            max_concurrent_groups=config.max_concurrent_groups,
            target_groups_per_partition=config.target_groups_per_partition,
        )
        self.reward_worker = RewardWorker(data_plane=self.data_plane, reward_registry=reward_registry)
        self.advantage_worker = AdvantageWorker(data_plane=self.data_plane, advantage_fn=advantage_fn)
        self.trainer_scheduler = TrainerScheduler(adapter_registry=self.adapter_registry, train_policy=train_policy)
        self.trainer_worker = TrainerWorker(
            data_plane=self.data_plane,
            adapter_registry=self.adapter_registry,
            scheduler=self.trainer_scheduler,
            train_partition_fn=train_partition_fn or self.train_partition,
            receive_weights_fn=receive_weights_fn,
        )

    @classmethod
    def build_multilora_model(
        cls,
        *,
        model_id: str,
        adapter_name: str,
        lora_config: Any,
        loss_cls: str = 'GRPOLoss',
        optimizer_cls: str = 'Adam',
        learning_rate: float = 2e-5,
        template_cls: str = 'Qwen3_5Template',
        processor_cls: str = 'InputProcessor',
        gradient_accumulation_steps: int = 1,
        loss_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        template_kwargs: Optional[Dict[str, Any]] = None,
        processor_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Create and configure MultiLoraTransformersModel like the GRPO cookbook."""

        from twinkle_client.model import MultiLoraTransformersModel

        model = MultiLoraTransformersModel(model_id=model_id)
        model.add_adapter_to_model(
            adapter_name,
            lora_config,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        model.set_loss(loss_cls, **(loss_kwargs or {'epsilon': 0.2, 'beta': 0.0}))
        optimizer_params = {'lr': learning_rate}
        optimizer_params.update(optimizer_kwargs or {})
        model.set_optimizer(optimizer_cls, **optimizer_params)
        model.set_processor(processor_cls, **(processor_kwargs or {}))
        model.set_template(template_cls, model_id=model_id, **(template_kwargs or {}))
        return model

    def add_pending(self, samples: Iterable[SampleRecord], context: Optional[TrainingContext] = None) -> None:
        self.submit_rollout_samples(samples, context=context)

    def submit_rollout_samples(self, samples: Iterable[SampleRecord], context: Optional[TrainingContext] = None) -> None:
        """Submit prompt groups for rollout.

        These are not trainer batches. The trainer only reads samples that have
        already passed rollout/reward/advantage stages from TransferQueue.
        """
        self.rollouter.add_pending(context or self.current_context(), samples)

    async def step_async(self) -> Dict[str, Optional[PartitionMetadata]]:
        rollout_meta = await self.rollouter.step()
        reward_meta = self._run_reward_if_ready()
        advantage_meta = self._run_advantage_if_ready()
        train_meta = self.trainer_worker.run_once()
        return {
            'rollout': rollout_meta,
            'reward': reward_meta,
            'advantage': advantage_meta,
            'train': train_meta,
        }

    def step(self) -> Dict[str, Optional[PartitionMetadata]]:
        return asyncio.run(self.step_async())

    async def run_async(
        self,
        rollout_samples: Optional[Iterable[SampleRecord]] = None,
        *,
        max_steps: Optional[int] = None,
    ) -> list[Dict[str, Optional[PartitionMetadata]]]:
        if rollout_samples is not None:
            self.submit_rollout_samples(rollout_samples)
        limit = max_steps if max_steps is not None else self.config.max_train_partitions
        history: list[Dict[str, Optional[PartitionMetadata]]] = []
        trained = 0
        idle_steps = 0
        while limit is None or trained < limit:
            result = await self.step_async()
            history.append(result)
            if result['train'] is not None:
                trained += 1
                idle_steps = 0
            elif any(value is not None for value in result.values()):
                idle_steps = 0
            else:
                idle_steps += 1
                if self._is_drained() or idle_steps >= 3:
                    break
                await asyncio.sleep(0)
        return history

    def run(
        self,
        rollout_samples: Optional[Iterable[SampleRecord]] = None,
        *,
        max_steps: Optional[int] = None,
    ) -> list[Dict[str, Optional[PartitionMetadata]]]:
        """Drive the async RL loop.

        `rollout_samples` is an optional convenience feed for rollout prompts.
        Training batches are always read from TransferQueue by TrainerWorker.
        """
        return asyncio.run(self.run_async(rollout_samples, max_steps=max_steps))

    def run_until_idle(self, *, max_steps: Optional[int] = None) -> list[Dict[str, Optional[PartitionMetadata]]]:
        """Advance workers without adding new rollout prompts."""
        return self.run(max_steps=max_steps)

    def train_partition(self, context: TrainingContext, partition_id: str, dataloader: Any) -> TrainerStepResult:
        batch = list(dataloader)
        inputs = [sample.get('trajectory', sample) for sample in batch]
        advantages = [sample.get('advantages') for sample in batch if 'advantages' in sample]
        old_logps = [sample.get('old_logps') for sample in batch if 'old_logps' in sample]

        kwargs = dict(self.config.train_kwargs)
        if advantages and len(advantages) == len(inputs):
            kwargs.setdefault('advantages', advantages)
        if old_logps and len(old_logps) == len(inputs):
            kwargs.setdefault('old_logps', old_logps)

        self.model.forward_backward(inputs=inputs, adapter_name=context.adapter_name, **kwargs)
        self.model.clip_grad_and_step(
            adapter_name=context.adapter_name,
            max_grad_norm=self.config.max_grad_norm,
            norm_type=self.config.norm_type,
        )
        save_result = self.model.save(
            name=f'{self.config.save_name_prefix}-{context.training_run_id}-{context.adapter_name}',
            adapter_name=context.adapter_name,
            save_optimizer=self.config.save_optimizer,
            is_sampler=self.config.is_sampler_checkpoint,
        )
        adapter_revision = getattr(save_result, 'twinkle_path', None)
        if adapter_revision is None and isinstance(save_result, dict):
            adapter_revision = save_result.get('twinkle_path') or save_result.get('path')
        return TrainerStepResult(adapter_revision=adapter_revision)

    def _run_reward_if_ready(self) -> Optional[PartitionMetadata]:
        try:
            return self.reward_worker.run_once(self.current_context(), batch_size=self.config.reward_batch_size)
        except LookupError:
            return None

    def _run_advantage_if_ready(self) -> Optional[PartitionMetadata]:
        try:
            return self.advantage_worker.run_once(self.current_context(), batch_size=self.config.advantage_batch_size)
        except LookupError:
            return None

    def current_context(self) -> TrainingContext:
        record = self.adapter_registry.get(self.context)
        return self.context.with_policy_version(record.policy_version, record.adapter_revision)

    def _is_drained(self) -> bool:
        context = self.current_context()
        pending = self.rollouter.pending_by_context.get(context.key)
        has_pending = bool(pending)
        active_partitions = [
            partition for partition in self.data_plane.get_metadata(context)
            if partition.status.value not in {'CLEARED', 'FAILED', 'CANCELLED'}
        ]
        return not has_pending and not active_partitions

# Copyright (c) ModelScope Contributors. All rights reserved.
"""Server-side async GRPO cookbook with TransferQueue + Multi-LoRA.

This entrypoint is the in-process/server MVP:

  dataset prompts -> AsyncRollouter -> TransferQueue -> RewardWorker
  -> AdvantageWorker -> TrainerWorker -> LoRA save -> vLLM adapter_path

It does not use the Twinkle client/server submission path. The script owns
resource initialization, model construction, sampler construction, and the
pipeline run loop.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import OmegaConf
from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import MultiLoraMegatronModel
from twinkle.processor import InputProcessor
from twinkle.reward import GSM8KAccuracyReward
from twinkle.reward.base import Reward
from twinkle.sampler import vLLMSampler
from twinkle.preprocessor.llm import GSM8KProcessor
from twinkle_agentic.async_rl import (
    AsyncMultiLoraGRPOPipeline as RuntimeAsyncMultiLoraGRPOPipeline,
    BaseRLPipeline,
    BaseRLPipelineConfig,
    PromptFeeder,
    TrainingContext,
    TransferQueueDataPlane,
    TransferQueueRuntimeConfig,
)
from twinkle_agentic.async_rl.workers import TrainerStepResult

logger = get_logger()


class GSM8KBrevityReward(Reward):
    """Reward valid, shorter answers."""

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for traj in trajectories:
            messages = traj.get('messages', [])
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break
            has_answer = bool(
                re.search(r'\\boxed\{[^}]+\}', completion)
                or re.search(r'####\s*[\-\d,\.]+', completion)
            )
            if not has_answer:
                rewards.append(0.0)
                continue
            length = len(completion)
            rewards.append(1.0 if length <= 300 else max(0.0, 1.0 - (length - 300) / 3000))
        return rewards


class GSM8KReward:

    def __init__(self):
        self.accuracy = GSM8KAccuracyReward()
        self.brevity = GSM8KBrevityReward()

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        accuracy = self.accuracy(trajectories)
        brevity = self.brevity(trajectories)
        return [a + b for a, b in zip(accuracy, brevity)]


class ServerSingleTurnRollout:
    """One prompt-group rollout adapter for local vLLMSampler."""

    def __init__(self, sampler: vLLMSampler, *, sampling_params: SamplingParams, num_generations: int):
        self.sampler = sampler
        self.sampling_params = sampling_params
        self.num_generations = num_generations

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        adapter_path = kwargs.get('adapter_path')
        adapter_name = kwargs.get('adapter_name', '')
        expanded = []
        for prompt_idx, trajectory in enumerate(trajectories):
            group_id = trajectory.get('group_id') or trajectory.get('sample_id') or f'prompt_{prompt_idx}'
            for generation_idx in range(self.num_generations):
                item = dict(trajectory)
                item['group_id'] = group_id
                item['generation_idx'] = generation_idx
                expanded.append(item)

        responses = self.sampler.sample(
            expanded,
            self.sampling_params,
            adapter_name=adapter_name,
            adapter_path=adapter_path,
        )
        rows: list[dict[str, Any]] = []
        for source, response in zip(expanded, responses):
            for sequence in response.sequences:
                row = dict(sequence.new_input_feature or source)
                row.setdefault('group_id', source['group_id'])
                row.setdefault('generation_idx', source['generation_idx'])
                row['old_logps'] = self._extract_logps(sequence.logprobs)
                row['stop_reason'] = sequence.stop_reason
                row['policy_version'] = kwargs.get('policy_version')
                rows.append(row)
        return rows

    @staticmethod
    def _extract_logps(logprobs) -> list[float]:
        values = []
        for item in logprobs or []:
            if not item:
                values.append(0.0)
            else:
                values.append(float(item[0][1]))
        return values


def load_config(path: str):
    return OmegaConf.load(path)


def training_context_configs(cfg):
    if cfg.get('training_contexts'):
        return list(cfg.training_contexts)
    return [cfg.training_context]


def primary_training_context(cfg):
    return training_context_configs(cfg)[0]


def build_training_contexts(cfg) -> list[TrainingContext]:
    contexts = []
    for context_cfg in training_context_configs(cfg):
        contexts.append(
            TrainingContext(
                tenant_id=context_cfg.tenant_id,
                training_run_id=context_cfg.training_run_id,
                base_model_id=context_cfg.base_model_id,
                adapter_name=context_cfg.adapter_name,
                reward_type=context_cfg.reward_type,
                loss_type=context_cfg.loss_type,
                tool_profile=context_cfg.get('tool_profile', 'default'),
            )
        )
    base_models = {context.base_model_id for context in contexts}
    if len(base_models) != 1:
        raise ValueError(f'one async multi-LoRA job must use one base model, got {sorted(base_models)}')
    return contexts


def context_dataset_config(cfg, context_cfg):
    return context_cfg.get('dataset', cfg.dataset)


def create_gsm8k_dataset(cfg, context_cfg):
    dataset_cfg = context_dataset_config(cfg, context_cfg)
    data_slice = range(int(dataset_cfg.data_num)) if dataset_cfg.get('data_num') else None
    dataset = Dataset()
    dataset.add_dataset(
        DatasetMeta(
            dataset_cfg.dataset_id,
            subset_name=dataset_cfg.get('subset_name'),
            split=dataset_cfg.get('split', 'train'),
            data_slice=data_slice,
        )
    )
    template_cfg = cfg.model.template
    dataset.set_template(
        template_cfg.cls,
        model_id=context_cfg.base_model_id,
        max_length=template_cfg.get('max_length', 4096),
        truncation_strategy=template_cfg.get('truncation_strategy', 'delete'),
        enable_thinking=template_cfg.get('enable_thinking', False),
    )
    dataset.map(GSM8KProcessor(system=dataset_cfg.system_prompt))
    dataset.encode(add_generation_prompt=True)
    return dataset


def build_device_meshes(cfg):
    runtime = cfg.runtime
    model_gpus = int(runtime.model_gpus)
    sampler_gpus = int(runtime.sampler_gpus)
    sampler_tp = int(runtime.sampler_tp)
    total_gpus = model_gpus + sampler_gpus
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(model_gpus)), device_type='GPU'),
        DeviceGroup(
            name='sampler',
            ranks=list(range(model_gpus, total_gpus)),
            device_type='GPU',
            gpus_per_worker=sampler_tp,
        ),
    ]
    model_mesh_cfg = cfg.model.mesh
    model_mesh = DeviceMesh.from_sizes(
        world_size=model_gpus,
        tp_size=int(model_mesh_cfg.get('tp_size', 1)),
        ep_size=int(model_mesh_cfg.get('ep_size', 1)),
        pp_size=int(model_mesh_cfg.get('pp_size', 1)),
        sequence_parallel=bool(model_mesh_cfg.get('sequence_parallel', False)),
    )
    sampler_mesh = DeviceMesh.from_sizes(
        world_size=sampler_gpus,
        dp_size=max(1, sampler_gpus // sampler_tp),
        tp_size=sampler_tp,
    )
    return total_gpus, device_groups, model_mesh, sampler_mesh


class AsyncMultiLoraGRPOPipeline(BaseRLPipeline):

    def __init__(self, cfg, *, model_mesh: DeviceMesh, sampler_mesh: DeviceMesh):
        self.cfg = cfg
        self.model_mesh = model_mesh
        self.sampler_mesh = sampler_mesh
        contexts = build_training_contexts(cfg)
        primary_context = contexts[0]
        super().__init__(
            config=BaseRLPipelineConfig(
                training_contexts=contexts,
                tenant_id=primary_context.tenant_id,
                training_run_id=primary_context.training_run_id,
                base_model_id=primary_context.base_model_id,
                adapter_name=primary_context.adapter_name,
                reward_type=primary_context.reward_type,
                loss_type=primary_context.loss_type,
                tool_profile=primary_context.tool_profile,
                max_staleness=int(cfg.pipeline.max_staleness),
                target_groups_per_partition=int(cfg.pipeline.target_groups_per_partition),
                max_concurrent_groups=int(cfg.pipeline.max_concurrent_groups),
                reward_batch_size=int(cfg.pipeline.reward_batch_size),
                advantage_batch_size=int(cfg.pipeline.advantage_batch_size),
                max_train_partitions=int(cfg.pipeline.max_steps),
                save_name_prefix=cfg.pipeline.save_name_prefix,
                adapter_checkpoint_dir=cfg.model.adapter_checkpoint_dir,
                is_sampler_checkpoint=bool(cfg.pipeline.is_sampler_checkpoint),
                save_optimizer=bool(cfg.pipeline.save_optimizer),
                max_grad_norm=float(cfg.pipeline.max_grad_norm),
                norm_type=int(cfg.pipeline.norm_type),
            )
        )

    def build_model(self):
        primary_context = primary_training_context(self.cfg)
        lora_cfg = self.cfg.model.lora
        lora_config = LoraConfig(
            target_modules=lora_cfg.target_modules,
            r=int(lora_cfg.r),
            lora_alpha=int(lora_cfg.lora_alpha),
            lora_dropout=float(lora_cfg.lora_dropout),
        )
        model = MultiLoraMegatronModel(
            model_id=primary_context.base_model_id,
            device_mesh=self.model_mesh,
            remote_group='model',
            mixed_precision=self.cfg.model.mixed_precision,
        )
        loss_kwargs = {k: v for k, v in self.cfg.model.loss.items() if k != 'cls'}
        template_kwargs = {k: v for k, v in self.cfg.model.template.items() if k not in {'cls', 'max_length', 'truncation_strategy'}}
        for context_cfg in training_context_configs(self.cfg):
            adapter_name = context_cfg.adapter_name
            model.add_adapter_to_model(
                adapter_name,
                lora_config,
                gradient_accumulation_steps=int(self.cfg.model.gradient_accumulation_steps),
            )
            model.set_optimizer(
                self.cfg.model.optimizer.cls,
                lr=float(self.cfg.model.optimizer.lr),
                adapter_name=adapter_name,
            )
            model.set_lr_scheduler(
                self.cfg.model.lr_scheduler.cls,
                lr_decay_steps=int(self.cfg.model.lr_scheduler.lr_decay_steps),
                max_lr=float(self.cfg.model.lr_scheduler.max_lr),
                adapter_name=adapter_name,
            )
            model.set_loss(self.cfg.model.loss.cls, adapter_name=adapter_name, **loss_kwargs)
            model.set_processor(InputProcessor, adapter_name=adapter_name)
            model.set_template(
                self.cfg.model.template.cls,
                model_id=primary_context.base_model_id,
                adapter_name=adapter_name,
                **template_kwargs,
        )
        return model

    def build_prompt_feeders(self):
        feeders = []
        max_pending_groups = self.cfg.pipeline.get('prompt_max_pending_groups')
        for context_cfg, context in zip(training_context_configs(self.cfg), self.contexts):
            dataset_cfg = context_dataset_config(self.cfg, context_cfg)
            dataloader = DataLoader(
                dataset=lambda context_cfg=context_cfg: create_gsm8k_dataset(self.cfg, context_cfg),
                batch_size=int(dataset_cfg.batch_size),
                min_batch_size=int(dataset_cfg.batch_size),
                device_mesh=self.model_mesh,
                remote_group='model',
            )
            feeders.append(
                PromptFeeder(
                    context=context,
                    dataloader=dataloader,
                    rollouter=self.rollouter,
                    max_pending_groups=max_pending_groups,
                )
            )
        return feeders

    def build_rollout(self):
        primary_context = primary_training_context(self.cfg)
        engine_args = OmegaConf.to_container(self.cfg.sampler.engine_args, resolve=True)
        engine_args.setdefault('tensor_parallel_size', int(self.cfg.runtime.sampler_tp))
        sampler = vLLMSampler(
            model_id=primary_context.base_model_id,
            engine_args=engine_args,
            device_mesh=self.sampler_mesh,
            remote_group='sampler',
        )
        sampler_template_kwargs = {k: v for k, v in self.cfg.sampler.template.items() if k != 'cls'}
        sampler.set_template(
            self.cfg.sampler.template.cls,
            model_id=primary_context.base_model_id,
            **sampler_template_kwargs,
        )
        sampling_params = SamplingParams.from_dict(
            OmegaConf.to_container(self.cfg.sampler.sampling_params, resolve=True)
        )
        return ServerSingleTurnRollout(
            sampler,
            sampling_params=sampling_params,
            num_generations=int(self.cfg.sampler.num_generations),
        )

    def build_data_plane(self):
        tq_cfg = self.cfg.transfer_queue
        return TransferQueueDataPlane(
            tq_config=TransferQueueRuntimeConfig(
                init=bool(tq_cfg.get('init', True)),
                total_storage_size=tq_cfg.get('total_storage_size'),
                max_rows=tq_cfg.get('max_rows'),
                max_rows_per_context=tq_cfg.get('max_rows_per_context'),
                num_data_storage_units=int(tq_cfg.get('num_data_storage_units', 4)),
                storage_backend=tq_cfg.get('storage_backend', 'SimpleStorage'),
            )
        )

    def build_reward_registry(self):
        return {context_cfg.reward_type: GSM8KReward() for context_cfg in training_context_configs(self.cfg)}

    def build_advantage_fn(self):
        return grpo_advantage_fn

    def train_partition(self, context, partition_id: str, dataloader) -> TrainerStepResult:
        batch = list(dataloader)
        mini_batch_size = int(self.cfg.pipeline.mini_batch_size)
        micro_batch_size = int(self.cfg.pipeline.micro_batch_size)
        for mb_start in range(0, len(batch), mini_batch_size):
            mini_batch = batch[mb_start:mb_start + mini_batch_size]
            inputs = [sample.get('trajectory', sample) for sample in mini_batch]
            old_logps = [sample.get('old_logps', []) for sample in mini_batch]
            advantages = [sample.get('advantages', 0.0) for sample in mini_batch]
            self.model.forward_backward(
                inputs=inputs,
                old_logps=old_logps,
                advantages=advantages,
                micro_batch_size=micro_batch_size,
                adapter_name=context.adapter_name,
            )
            self.model.clip_grad_and_step(
                adapter_name=context.adapter_name,
                max_grad_norm=float(self.cfg.pipeline.max_grad_norm),
                norm_type=int(self.cfg.pipeline.norm_type),
            )

        save_result = self.model.save(
            f'{self.cfg.pipeline.save_name_prefix}-{context.training_run_id}-{context.adapter_name}-v{context.policy_version + 1}',
            output_dir=self.cfg.model.adapter_checkpoint_dir,
            adapter_name=context.adapter_name,
            save_optimizer=bool(self.cfg.pipeline.save_optimizer),
            is_sampler=bool(self.cfg.pipeline.is_sampler_checkpoint),
        )
        adapter_revision = save_result if isinstance(save_result, str) else getattr(save_result, 'twinkle_path', None)
        return TrainerStepResult(adapter_revision=adapter_revision)


def grpo_advantage_fn(samples: List[Dict[str, Any]], context) -> tuple[list[float], list[float]]:
    rewards = [float(sample.get('rewards', sample.get('reward', 0.0))) for sample in samples]
    if not rewards:
        return [], []
    num_generations = max(1, max(int(sample.get('generation_idx', 0)) for sample in samples) + 1)
    advantages = GRPOAdvantage()(rewards, num_generations=num_generations, scale='group').tolist()
    return advantages, rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default=Path(__file__).with_suffix('.yaml').as_posix(),
        help='Path to server-side async multi-LoRA GRPO YAML config.',
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    total_gpus, device_groups, model_mesh, sampler_mesh = build_device_meshes(cfg)
    twinkle.initialize(
        mode=cfg.runtime.mode,
        nproc_per_node=total_gpus,
        groups=device_groups,
        lazy_collect=bool(cfg.runtime.get('lazy_collect', False)),
    )

    pipeline = RuntimeAsyncMultiLoraGRPOPipeline(cfg, model_mesh=model_mesh, sampler_mesh=sampler_mesh)

    logger.info('Starting server-side async multi-LoRA GRPO')
    logger.info(get_device_placement())
    history = pipeline.run_until_idle(max_steps=int(cfg.pipeline.max_steps))
    trained = sum(1 for item in history if item.get('train') is not None)
    logger.info('async_multi_lora_grpo progress: trained_partitions=%s', trained)

    pipeline.shutdown()
    for context in pipeline.current_contexts():
        final_name = f'async-grpo-final-{context.training_run_id}-{context.adapter_name}'
        pipeline.model.save(
            final_name,
            output_dir=cfg.model.adapter_checkpoint_dir,
            adapter_name=context.adapter_name,
            save_optimizer=bool(cfg.pipeline.save_optimizer),
        )
    logger.info('Training completed. trained_partitions=%s', trained)


if __name__ == '__main__':
    main()

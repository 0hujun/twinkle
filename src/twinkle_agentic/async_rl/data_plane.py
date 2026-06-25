# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

from .types import PartitionMetadata, PartitionStatus, SampleRecord, TrainingContext


class InMemoryTransferQueueBackend:
    """Small backend with TransferQueue-like semantics for tests and local MVP runs."""

    def __init__(self):
        self.partitions: Dict[str, Dict[str, SampleRecord]] = defaultdict(dict)
        self.partition_meta: Dict[str, PartitionMetadata] = {}
        self._lock = threading.RLock()

    def put_samples(
        self,
        partition: PartitionMetadata,
        samples: Iterable[SampleRecord],
        *,
        key_prefix: str = 'sample',
    ) -> list[str]:
        with self._lock:
            self.partition_meta[partition.partition_id] = partition
            keys = []
            bucket = self.partitions[partition.partition_id]
            for sample in samples:
                key = sample.get('sample_id') or f'{key_prefix}_{len(bucket)}'
                bucket[key] = dict(sample)
                keys.append(key)
            partition.num_rows = len(bucket)
            partition.touch()
            return keys

    def get_samples(self, partition_id: str, keys: Optional[list[str]] = None) -> list[SampleRecord]:
        with self._lock:
            bucket = self.partitions.get(partition_id, {})
            selected = keys if keys is not None else list(bucket)
            return [dict(bucket[k]) for k in selected if k in bucket]

    def update_samples(self, partition_id: str, updates: Dict[str, Dict[str, Any]]) -> None:
        with self._lock:
            bucket = self.partitions[partition_id]
            for key, fields in updates.items():
                bucket[key].update(fields)
            if partition_id in self.partition_meta:
                self.partition_meta[partition_id].touch()

    def list_partition_ids(self) -> list[str]:
        with self._lock:
            return list(self.partition_meta)

    def clear_partition(self, partition_id: str) -> None:
        with self._lock:
            self.partitions.pop(partition_id, None)
            meta = self.partition_meta.get(partition_id)
            if meta is not None:
                meta.status = PartitionStatus.CLEARED
                meta.touch()


class TransferQueueDataPlane:
    """Context-safe data-plane boundary around TransferQueue.

    The MVP implementation stores metadata locally and can use an in-memory
    backend. A production deployment can pass a backend object with compatible
    methods or extend this class to call `transfer_queue` directly.
    """

    def __init__(self, backend: Optional[InMemoryTransferQueueBackend] = None):
        self.backend = backend or InMemoryTransferQueueBackend()
        self._meta: Dict[str, PartitionMetadata] = self.backend.partition_meta
        self._next_train_id: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()

    def init_namespace(self, context: TrainingContext) -> None:
        # Namespace is implicit in the partition id. This method is kept as the
        # single initialization hook for future real TransferQueue setup.
        context.metadata()

    def next_partition_id(self, context: TrainingContext) -> str:
        with self._lock:
            train_id = self._next_train_id[context.key]
            self._next_train_id[context.key] += 1
            return context.partition_id(train_id)

    def create_partition(
        self,
        context: TrainingContext,
        *,
        target_groups: int,
        partition_id: Optional[str] = None,
    ) -> PartitionMetadata:
        partition_id = partition_id or self.next_partition_id(context)
        meta = PartitionMetadata(
            context=context,
            partition_id=partition_id,
            policy_version=context.policy_version,
            target_groups=target_groups,
        )
        with self._lock:
            self._meta[partition_id] = meta
        return meta

    def put_rollout_batch(
        self,
        context: TrainingContext,
        partition_id: str,
        trajectories: List[SampleRecord],
        *,
        ready_groups: int = 1,
        seal: bool = False,
    ) -> PartitionMetadata:
        with self._lock:
            meta = self._meta.get(partition_id)
            if meta is None:
                meta = self.create_partition(context, target_groups=ready_groups, partition_id=partition_id)
            if meta.context.key != context.key:
                raise ValueError(f'partition {partition_id} belongs to {meta.context.key}, not {context.key}')
            samples = []
            for idx, trajectory in enumerate(trajectories):
                sample = dict(trajectory)
                sample_meta = dict(sample.get('metadata') or {})
                sample_meta.update(context.metadata())
                sample_meta.setdefault('partition_id', partition_id)
                context.validate_metadata(sample_meta)
                sample['metadata'] = sample_meta
                sample.setdefault('sample_id', f'{partition_id}/sample_{meta.num_rows + idx}')
                samples.append(sample)
            self.backend.put_samples(meta, samples)
            meta.ready_groups += ready_groups
            if seal or (meta.target_groups and meta.ready_groups >= meta.target_groups):
                meta.status = PartitionStatus.ROLLOUT_DONE
            meta.touch()
            return meta

    def list_partitions(
        self,
        context: Optional[TrainingContext] = None,
        *,
        statuses: Optional[Iterable[PartitionStatus]] = None,
    ) -> list[PartitionMetadata]:
        status_set = set(statuses) if statuses is not None else None
        with self._lock:
            partitions = list(self._meta.values())
        if context is not None:
            partitions = [p for p in partitions if p.context.key == context.key]
        if status_set is not None:
            partitions = [p for p in partitions if p.status in status_set]
        return sorted(partitions, key=lambda p: (p.created_at, p.partition_id))

    def get_metadata(self, context: Optional[TrainingContext] = None) -> list[PartitionMetadata]:
        return self.list_partitions(context)

    def check_capacity(self, context: TrainingContext) -> bool:
        # Real TQ capacity is enforced by the storage backend; the MVP data
        # plane keeps this hook explicit for AsyncRollouter gating.
        return True

    def claim_reward_batch(self, context: TrainingContext, batch_size: int) -> tuple[PartitionMetadata, list[SampleRecord]]:
        partitions = self.list_partitions(context, statuses=[PartitionStatus.ROLLOUT_DONE])
        if not partitions:
            raise LookupError(f'no rollout-ready partition for {context.key}')
        meta = partitions[0]
        samples = self.backend.get_samples(meta.partition_id)[:batch_size]
        return meta, samples

    def append_rewards(
        self,
        context: TrainingContext,
        partition_id: str,
        rewards: list[float],
        *,
        field_name: str = 'rewards',
    ) -> PartitionMetadata:
        samples = self.backend.get_samples(partition_id)
        if len(rewards) != len(samples):
            raise ValueError(f'reward count {len(rewards)} does not match sample count {len(samples)}')
        updates = {}
        for sample, reward in zip(samples, rewards):
            context.validate_metadata(sample.get('metadata') or {})
            updates[sample['sample_id']] = {field_name: reward}
        self.backend.update_samples(partition_id, updates)
        meta = self._meta[partition_id]
        meta.status = PartitionStatus.REWARD_DONE
        meta.touch()
        return meta

    def claim_advantage_batch(self, context: TrainingContext, batch_size: int) -> tuple[PartitionMetadata, list[SampleRecord]]:
        partitions = self.list_partitions(context, statuses=[PartitionStatus.REWARD_DONE])
        if not partitions:
            raise LookupError(f'no reward-ready partition for {context.key}')
        meta = partitions[0]
        samples = self.backend.get_samples(meta.partition_id)[:batch_size]
        return meta, samples

    def append_advantages(
        self,
        context: TrainingContext,
        partition_id: str,
        advantages: list[float],
        returns: Optional[list[float]] = None,
    ) -> PartitionMetadata:
        samples = self.backend.get_samples(partition_id)
        if len(advantages) != len(samples):
            raise ValueError(f'advantage count {len(advantages)} does not match sample count {len(samples)}')
        if returns is None:
            returns = advantages
        updates = {}
        for sample, advantage, ret in zip(samples, advantages, returns):
            context.validate_metadata(sample.get('metadata') or {})
            updates[sample['sample_id']] = {'advantages': advantage, 'returns': ret}
        self.backend.update_samples(partition_id, updates)
        meta = self._meta[partition_id]
        meta.status = PartitionStatus.TRAIN_READY
        meta.touch()
        return meta

    def list_train_ready_partitions(self) -> list[PartitionMetadata]:
        return self.list_partitions(statuses=[PartitionStatus.TRAIN_READY])

    def mark_training(self, context: TrainingContext, partition_id: str) -> PartitionMetadata:
        meta = self._meta[partition_id]
        if meta.context.key != context.key:
            raise ValueError(f'partition {partition_id} belongs to {meta.context.key}, not {context.key}')
        meta.status = PartitionStatus.TRAINING
        meta.touch()
        return meta

    def mark_trained(self, context: TrainingContext, partition_id: str) -> PartitionMetadata:
        meta = self._meta[partition_id]
        if meta.context.key != context.key:
            raise ValueError(f'partition {partition_id} belongs to {meta.context.key}, not {context.key}')
        meta.status = PartitionStatus.TRAIN_DONE
        meta.touch()
        return meta

    def build_streaming_dataloader(self, context: TrainingContext, partition_id: str):
        meta = self._meta[partition_id]
        if meta.context.key != context.key:
            raise ValueError(f'partition {partition_id} belongs to {meta.context.key}, not {context.key}')
        return self.backend.get_samples(partition_id)

    def clear_partition(self, context: TrainingContext, partition_id: str) -> None:
        meta = self._meta.get(partition_id)
        if meta is not None and meta.context.key != context.key:
            raise ValueError(f'partition {partition_id} belongs to {meta.context.key}, not {context.key}')
        self.backend.clear_partition(partition_id)

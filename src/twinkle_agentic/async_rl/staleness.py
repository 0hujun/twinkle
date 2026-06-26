# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass

from .types import PartitionMetadata, PartitionStatus, RolloutCapacity, TrainingContext


@dataclass
class StalenessManager:
    max_staleness: int = 0
    target_groups_per_partition: int = 1
    throttle_watermark: float = 0.8
    sleep_seconds: float = 1.0

    @property
    def max_live_partitions(self) -> int:
        return self.max_staleness + 1

    def get_rollout_capacity(
        self,
        context: TrainingContext,
        partitions: list[PartitionMetadata],
    ) -> RolloutCapacity:
        scoped = [p for p in partitions if p.context.key == context.key and p.status != PartitionStatus.CLEARED]
        live_count = len(scoped)

        open_partitions = [p for p in scoped if p.status == PartitionStatus.OPEN]
        if open_partitions:
            current = sorted(open_partitions, key=lambda p: p.created_at)[0]
            remaining = max(0, self.target_groups_per_partition - current.ready_groups)
            if remaining > 0:
                return RolloutCapacity(remaining, action='submit')

        if live_count >= self.max_live_partitions:
            return RolloutCapacity(0, action='sleep', reason='max_staleness_reached', sleep_seconds=self.sleep_seconds)

        capacity = (self.max_live_partitions - live_count) * self.target_groups_per_partition
        if capacity <= 0:
            return RolloutCapacity(0, action='sleep', reason='no_partition_capacity', sleep_seconds=self.sleep_seconds)

        if live_count >= int(self.max_live_partitions * self.throttle_watermark):
            return RolloutCapacity(capacity, action='submit', reason='near_staleness_limit')
        return RolloutCapacity(capacity, action='submit')

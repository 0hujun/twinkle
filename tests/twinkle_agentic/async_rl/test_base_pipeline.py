from types import SimpleNamespace

from twinkle_agentic.async_rl import BaseRLPipeline, BaseRLPipelineConfig, PartitionStatus


def make_sample(i=0):
    return {
        'sample_id': f'sample_{i}',
        'messages': [{'role': 'user', 'content': f'q{i}'}],
        'group_id': f'g{i}',
        'generation_idx': 0,
        'old_logps': [-0.1],
    }


class EchoRollout:

    def __init__(self):
        self.calls = []

    def __call__(self, trajectories, **kwargs):
        self.calls.append(kwargs)
        out = []
        for trajectory in trajectories:
            copied = dict(trajectory)
            copied['messages'] = list(copied.get('messages', [])) + [{'role': 'assistant', 'content': 'ok'}]
            out.append(copied)
        return out


class FakeMultiLoraModel:

    def __init__(self):
        self.forward_backward_calls = []
        self.step_calls = []
        self.save_calls = []

    def forward_backward(self, **kwargs):
        self.forward_backward_calls.append(kwargs)

    def clip_grad_and_step(self, **kwargs):
        self.step_calls.append(kwargs)

    def save(self, **kwargs):
        self.save_calls.append(kwargs)
        return SimpleNamespace(twinkle_path=f"/tmp/{kwargs['adapter_name']}-v{len(self.save_calls)}")


def test_base_pipeline_runs_one_multilora_grpo_partition():
    model = FakeMultiLoraModel()
    rollout = EchoRollout()
    received = []
    pipeline = BaseRLPipeline(
        config=BaseRLPipelineConfig(
            tenant_id='tenant',
            training_run_id='run',
            base_model_id='base',
            adapter_name='lora_a',
            reward_type='constant',
            max_train_partitions=1,
        ),
        model=model,
        rollout=rollout,
        reward_registry={'constant': lambda trajectories, **_: [1.0 for _ in trajectories]},
        receive_weights_fn=lambda context: received.append(context),
    )

    history = pipeline.run([make_sample(0)])

    assert history[-1]['train'] is not None
    assert model.forward_backward_calls[0]['adapter_name'] == 'lora_a'
    assert model.forward_backward_calls[0]['advantages'] == [0.0]
    assert model.forward_backward_calls[0]['old_logps'] == [[-0.1]]
    assert model.step_calls[0]['adapter_name'] == 'lora_a'
    assert model.save_calls[0]['adapter_name'] == 'lora_a'
    assert model.save_calls[0]['is_sampler'] is True
    assert received[0].policy_version == 1
    assert received[0].adapter_revision == '/tmp/lora_a-v1'
    assert pipeline.data_plane.list_partitions(pipeline.context)[0].status == PartitionStatus.CLEARED


def test_base_pipeline_uses_latest_adapter_revision_for_next_rollout():
    model = FakeMultiLoraModel()
    rollout = EchoRollout()
    pipeline = BaseRLPipeline(
        config=BaseRLPipelineConfig(
            tenant_id='tenant',
            training_run_id='run',
            base_model_id='base',
            adapter_name='lora_a',
            reward_type='constant',
            max_train_partitions=1,
        ),
        model=model,
        rollout=rollout,
        reward_registry={'constant': lambda trajectories, **_: [1.0 for _ in trajectories]},
    )

    pipeline.run([make_sample(0)], max_steps=1)
    pipeline.run([make_sample(1)], max_steps=1)

    assert rollout.calls[0]['adapter_name'] == 'lora_a'
    assert 'adapter_path' not in rollout.calls[0]
    assert rollout.calls[1]['adapter_path'] == '/tmp/lora_a-v1'

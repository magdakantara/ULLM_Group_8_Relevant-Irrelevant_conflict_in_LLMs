"""Explicit, separate Qwen2.5 YaRN experiment configuration."""
def configure_long_context(model_config, experiment):
    expected = {'rope_type': 'yarn', 'factor': 4.0, 'original_max_position_embeddings': 32768}
    if experiment.get('model') != 'Qwen/Qwen2.5-7B-Instruct' or model_config.model_type != 'qwen2':
        raise ValueError('This long-context experiment supports Qwen2.5-7B-Instruct only.')
    if experiment.get('rope_scaling') != expected:
        raise ValueError('Expected the explicit factor-4 YaRN configuration.')
    if experiment.get('max_input_tokens') != 131072:
        raise ValueError('Expected total context budget 131072.')
    if experiment.get('attention_backend') != 'sdpa_flash_only' or experiment.get('logits_to_keep') != 1:
        raise ValueError('Long-context memory settings must be sdpa_flash_only and logits_to_keep=1.')
    model_config.rope_scaling = dict(expected)
    model_config.max_position_embeddings = 131072

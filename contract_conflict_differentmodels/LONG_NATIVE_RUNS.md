# Long-context runs for Llama-3.1 / Granite-3.1 and Llama-3.1-70B

`src/run_model_long_native.py` is a copy of Dilek's `src/run_model_long.py` with only two changes:

1. No YaRN / RoPE rescaling. Llama-3.1 and Granite-3.1 support 128k context natively; the script
   refuses to run if a model would need rope scaling (use `run_model_long.py` for Qwen).
2. Optional multi-GPU sharding (`"device_map": "auto"` in the config, used for 70B). Layers are
   spread evenly over the visible GPUs and the run aborts if anything is offloaded to CPU/disk.

Everything else is identical to the Qwen long run: same prompts, schema, bf16, seed 42,
`max_input_tokens` 131072, `max_new_tokens` 512/1024, `sdpa_flash_only`, `logits_to_keep=1`,
resume behaviour and output files (`responses.jsonl`, `run_info.json`, `reference_snapshot.json`).
`run_info.json` additionally records `gpu_count` and `device_map`; peak GPU memory is summed over GPUs.

Configs: `config/experiment_long_native.{llama8b,granite8b,llama70b}.json`.
SLURM: `snellius/run_long_native.sh` (1x A100, 8h) and `snellius/run_70b.sh` (4x H100, 12h).

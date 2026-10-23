# Raw model outputs

One folder per model, prompt and (for Qwen) context route. Each folder contains:

- `responses.jsonl`: one record per example with the raw response, parsed JSON, technical
  status, input length and generation attempts
- `run_info.json`: model revision, generation settings, system prompt and selected example IDs
- `environment.txt`: installed package versions

| Folder | Model | Examples | Run by |
|---|---|---:|---|
| `qwen7b_short_*` | Qwen2.5-7B-Instruct, native 32k context | 258 per prompt | Dilek Barutcu |
| `qwen7b_long_*` | Qwen2.5-7B-Instruct, YaRN ×4, 131k context | 42 per prompt | Dilek Barutcu |
| `llama8b_long_*` | Llama-3.1-8B-Instruct, native 131k context | 300 per prompt | Oğuz Kağan Yarım |
| `granite8b_long_*` | Granite-3.1-8B-Instruct, native 131k context | 300 per prompt | Oğuz Kağan Yarım |
| `llama70b_long_*` | Llama-3.1-70B-Instruct, native 131k context | 300 per prompt | Oğuz Kağan Yarım |

All runs used greedy decoding and seed 42 on the Snellius supercomputer. The technical status
of every response matches the `run_status` column of the label files in
[`../evaluation/analysis/data/`](../evaluation/analysis/data/).

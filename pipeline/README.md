# Contract conflict evaluation

This project tests how Qwen2.5-7B-Instruct answers questions when two contract sources agree or disagree. The main question is whether it recognizes a conflict that affects the answer without unnecessarily refusing to answer when a disagreement is irrelevant.

We compare baseline and explained prompts on the same examples. This is an inference experiment; no training or fine-tuning is performed.

## Current status

All four v9 runs were completed and their outputs checked on 18 October 2026. There are 600 response records, with no missing or duplicate example IDs within the planned runs. The dataset fingerprints and source hashes match the saved references. Both prompts use the same example selection and resolved model revision.

| Prompt | Route | Records | `ok` | `truncated` | `format_error` |
|---|---|---:|---:|---:|---:|
| Baseline | Short | 258 | 251 | 6 | 1 |
| Baseline | Long | 42 | 37 | 5 | 0 |
| Explained | Short | 258 | 250 | 8 | 0 |
| Explained | Long | 42 | 37 | 4 | 1 |
| **Total** | | **600** | **575** | **23** | **2** |

These are technical outcomes, not behavioral scores. An `ok` response can still give an incorrect answer or ignore a conflict.

The review package is ready. A separate AI-assisted pilot has examined 12 responses from one contract: eight have proposed categories, and four remain pending because of formatting-only evidence differences. These proposals have not been entered as final human ratings in the workbook. Full behavioral evaluation and precision, recall and F1 results are still pending.

## Dataset

`data/dataset.json` contains 300 examples from 50 contracts, with 60 examples per contributor. Each contract has two questions tested under three conditions, giving six examples per contract and 100 question groups overall.

| Condition | Meaning | Examples per prompt |
|---|---|---:|
| `agreement` | Both sources support the same answer. | 100 |
| `relevant_conflict` | The sources support different answers to the question. | 100 |
| `irrelevant_conflict` | The sources differ, but the difference does not affect the answer. | 100 |

Each record includes the question, full A/B source texts, condition, reference answers, expected behavior and contract metadata. Owner codes M, B, E, O and D support selection and tracking. Source paths may be null; the runner reads the embedded texts.

During consolidation, contract selection was corrected, 180 missing metadata fields were filled, and experiment-revealing headers were removed from four modified FTE records (EX-015 to EX-018). The Dilek subset uses the corrected ten-contract selection and revised questions. Synthetic dates and notice periods were checked during preparation, but the combined dataset has not undergone independent semantic validation.

There is no inferred train/test split. Pilot examples are included in the final runs. Examples from the same contract are related observations and should not be treated as independent samples.

## Prompts and responses

Both prompts share the basic task and output requirements. The explained prompt adds guidance to answer despite irrelevant differences and to report both source-specific answers when the disagreement affects the question.

The model receives only the system prompt, Source A, Source B and the question. Reference answers, condition labels, expected behavior, owner codes and example IDs are withheld. Document text is treated as evidence, not as instructions. Each example starts with fresh messages and no history from previous examples.

The requested output is:

```json
{
  "answer": "...",
  "evidence_A": "...",
  "evidence_B": "..."
}
```

Evidence fields request short exact quotations. Empty evidence is allowed when the source provides no supporting quotation. JSON schema constraints are applied during generation with `lm-format-enforcer`. Constraints improved the initial formatting problem but did not eliminate all failures: two final responses still have `format_error` status.

Raw responses and generation attempts are retained. Evaluation must not rewrite answers or silently repair failed outputs.

## Runtime and context length

The runs used Snellius sessions with one NVIDIA A100-SXM4-40GB GPU, 18 CPUs and 64 GB host memory.

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Resolved revision | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Precision | `bfloat16` |
| Decoding | Greedy |
| Seed | 42, reset for each example |
| Output budgets | 512 tokens initially; 1,024 on retry |
| Python | 3.11.3 |
| PyTorch | 2.14.1+cu130 |
| Transformers | 4.57.6 |
| lm-format-enforcer | 0.11.3 |

If generation does not end with an EOS token, the runner retries from the original input with the larger budget. If the retry also fails to end, the result is `truncated`. Some responses repeated sentences until the budget was exhausted; this remains a recorded failure, not a fully resolved issue.

The original context budget accepted 258 examples after reserving room for the reply. The remaining 42 examples belonged to seven contracts. The longest input was about 74,000 tokens.

| Route | Examples per prompt | Runner | Context |
|---|---:|---|---|
| Short | 258 | `src/run_model_control.py` | Native, no YaRN; 32,768 tokens |
| Long | 42 | `src/run_model_long.py` | YaRN factor 4; 131,072 tokens |

Both runners use SDPA with Flash Attention forced and `logits_to_keep=1`. Full source texts are retained. The longest example completed under both prompts with about 28 GiB of peak allocated GPU memory.

Using the long-context settings on the short Driven pilot increased truncation in baseline responses. A native-context control completed all six baseline examples. The final experiment therefore uses separate routes. Each example follows the same route under both prompts. Report short and long results separately as well as together; pooled results describe this routed system.

## Setup and execution

In the existing Snellius installation:

```bash
cd /home/scur0499/contract_conflict_json_project
source snellius/activate_env.sh
```

The activation script loads the Python module and activates the environment. Do not reinstall dependencies in an already working environment. For a new installation, load the required Python module, create a virtual environment and install `requirements.txt`.

Validation and tests do not need a GPU:

```bash
python src/validate_data.py
python -m unittest discover -s tests -v
```

All 14 tests passed after the long-context update. They check data structure and software behavior, not the quality of model answers.

Before inference, check GPU access:

```bash
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

### Commands used for v9

These commands document the completed experiment. Reviewing saved results does not require rerunning them. They require the final Snellius files, including the control runner, both v9 subsets and the model revision lock.

```bash
python src/run_model_control.py --data data/run_short_v9.json --all --prompt baseline --out outputs/final_v9_short_baseline &&
python src/run_model_long.py --data data/run_long_v9.json --all --prompt baseline --out outputs/final_v9_long_baseline &&
python src/run_model_control.py --data data/run_short_v9.json --all --prompt explained --out outputs/final_v9_short_explained &&
python src/run_model_long.py --data data/run_long_v9.json --all --prompt explained --out outputs/final_v9_long_explained
```

The subsets are disjoint and cover all 300 examples. The long subset contains the examples that failed the original context preflight under either prompt.

To resume after a session ends, rerun the same command with the same output folder. Recorded examples are skipped, including technical failures. If a JSON line was only partly written, preserve the file and inspect that line before resuming. Never run two processes against the same output folder. Keep old pilots and any retry experiments separate from the final results.

Each run saves `responses.jsonl`, `run_info.json`, `reference_snapshot.json` and `environment.txt`. Together these preserve the outputs, attempts, settings, selected IDs, model revision, reference answers, source hashes and environment details.

## Llama and Granite runs

Llama-3.1-8B-Instruct, Granite-3.1-8B-Instruct and Llama-3.1-70B-Instruct were run with
`src/run_model_long_native.py`. It uses the same dataset, prompts, message format and output
schema as the Qwen runs, but a single route with the models' native 131k-token context; RoPE
scaling (YaRN) is explicitly refused. Settings are in `config/experiment_long_native.*.json`
and match the Qwen long-context settings (checked by `tests/test_long_native.py`).

```bash
# 8B models: one A100 (40 GB)
sbatch snellius/run_long_native.sh --config config/experiment_long_native.llama8b.json \
    --all --prompt baseline --out outputs/llama8b_long_baseline
# 70B model: four H100 GPUs, weights cached on scratch
sbatch snellius/run_70b.sh --config config/experiment_long_native.llama70b.json \
    --all --prompt baseline --out outputs/llama70b_long_baseline
```

| Model | Prompt | `ok` | `truncated` | `format_error` | `out_of_memory` |
|---|---|---:|---:|---:|---:|
| Llama-3.1-8B | Baseline | 249 | 48 | 3 | 0 |
| Llama-3.1-8B | Explained | 240 | 51 | 9 | 0 |
| Granite-3.1-8B | Baseline | 239 | 49 | 0 | 12 |
| Granite-3.1-8B | Explained | 236 | 57 | 0 | 7 |
| Llama-3.1-70B | Baseline | 268 | 32 | 0 | 0 |
| Llama-3.1-70B | Explained | 261 | 37 | 2 | 0 |

## Evaluation

The three conditions describe the input. The following four categories describe model behavior.

| Category | Decision rule |
|---|---|
| `expected_correct` | Gives a supported answer and meets the expected behavior, with accurate evidence. For relevant conflict, reports both source-specific answers and their disagreement without unjustified preference. |
| `unnecessary_abstention` | Declines to answer when the sources support the required response. |
| `incorrect_answer` | Gives a substantively wrong answer, omits the requested fact or selects one side of a relevant conflict without justification. |
| `other_error` | Has another failure, such as fabricated evidence or an unusable output format. |

Assign one category per assessable response. When issues overlap, use this priority: incorrect answer, unnecessary abstention, other error, then expected correct only if no issue remains. Reporting both conflicting values and explaining that no unique answer is supported is not automatically unnecessary abstention.

A failed literal quotation check does not by itself prove fabricated evidence. Formatting-only differences require a consistent rule; four pilot responses are pending on this point.

### Review package

The package contains:

- `evaluation_review.xlsx`: all 600 records, category and notes fields, and summary formulas.
- `START_HERE.html` and `examples/`: full sources, reference answers and unchanged model responses.
- `integrity_report.json`: checks against the received data and saved run references.
- `raw/`: the original experiment archive.

Unzip the entire package before opening source links. In the workbook, 577 rows are eligible for review: 575 `ok` and two `format_error` responses. The 23 truncated outputs are retained as `technical_only` without behavioral labels. Percentages remain pending until the eligible rows in each group are rated.

AI-assisted pilot labels are proposals with reasons, stored separately. They are not an independent human evaluation. Any report using these labels must state how they were produced and how much human verification was performed.

### Planned metrics

The main analysis compares the four behavior counts and rates across conditions and prompts. Technical failures will be shown separately. Expected-correct responses divided by all planned examples will also be reported, so exclusions do not inflate overall success. Agreement and irrelevant-conflict responses will be paired by contract and question.

Precision, recall and F1 will measure **recognition of answer-relevant conflict**, not word overlap with reference answers:

- Gold positive: `relevant_conflict`.
- Gold negative: `agreement` or `irrelevant_conflict`.
- Predicted positive: the response recognizes that the sources support incompatible answers to the question.

A generic mention of different sources or an unrelated disagreement is not sufficient. Recognizing a conflict and handling it appropriately are separate judgments: a response can recognize the conflict and still choose one source without justification.

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2TP / (2TP + FP + FN)
```

Use an undefined value when a denominator is zero. State coverage and exclusions. Missing annotations and truncated outputs must not silently become negative predictions. Full recognition annotation and metric computation are still pending; they are not implemented in the workbook or existing summary script.

### Existing per-run CSV workflow

The older scripts remain available:

```bash
python src/prepare_review.py --run outputs/final_v9_short_baseline
# Complete review.csv before running the summary.
python src/summarize_results.py --run outputs/final_v9_short_baseline
```

In `review.csv`, set one category flag to 1 and the other three to 0, then set `review_complete` to `yes`. The generator does not overwrite an existing review file. The summary requires completed ratings and excludes technical failures from its semantic denominator.

The consolidated workbook uses a single category selection instead of four flags. It is not directly interchangeable with this CSV schema. The existing summary script does not compute conflict-detection precision, recall or F1.

## Files and reproducibility

| File | Purpose |
|---|---|
| `data/dataset.json` | Full dataset with embedded A/B texts. |
| `config/experiment.json` | Native-context model and generation settings. |
| `config/experiment_long.json` | Long-context and YaRN settings. |
| `config/model_revision_lock.json` | Resolved revision in the final Snellius project. |
| `prompts/` | Baseline and explained instructions. |
| `src/common.py` | Loading, selection, prompt construction and parsing. |
| `src/output_format.py` | JSON schema constraints. |
| `src/run_model.py` | Original runner, dry runs and preflight checks. |
| `src/run_model_control.py` | Final short-input runner, included in the received archive. |
| `src/run_model_long.py`, `src/long_context.py` | Long-input generation and configuration checks. |
| `src/run_longest.py` | Longest-input feasibility check. |
| `src/prepare_review.py`, `src/summarize_results.py` | Per-run CSV review and summaries. |
| `tests/` | Pipeline, output-format and long-context tests. |

The completed Snellius experiment was received as `evaluation_v9_20261018_101458.tar.gz`.

```text
SHA-256: 25dbaafb065851a636abe66e6c15e5d7013e72ff2bae1f3a9824d4cdb73dd81b
```

In the local workspace, the extracted archive is under `../outputs/evaluation_v9_received/`, the review package under `../outputs/evaluation_v9_review/`, and pilot annotations under `../outputs/evaluation_v9_annotation/`.

The local project folder predates the final Snellius changes. Its missing control runner, v9 data subsets and revision lock are available in the extracted archive. Use the archived files to reproduce v9 rather than assuming this local folder is a complete copy of the final runtime.

## Remaining work and limitations

1. Settle the formatting-only evidence rule and finish response annotations.
2. Check ambiguous cases and a control sample through human review.
3. Calculate behavior rates and conflict-recognition metrics with explicit denominators.
4. Report prompt comparisons, route-specific results and paired agreement/irrelevant behavior.
5. Prepare final tables, figures and example responses.

The dataset is not an independently validated benchmark. Reference answers still require semantic checks. Question wording varies across contributors, pilot observations influenced the execution setup, and examples within a contract are dependent. These limitations should remain explicit in the final report.

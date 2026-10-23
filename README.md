# Relevant vs. Irrelevant Conflicts in Contract Question Answering

Research project for **2AMM20 Research Topics in Data Mining** (TU Eindhoven, 2026–2027),
Understanding Large Language Models track, Group 8.

When an LLM answers a question over two versions of a contract, not every difference between
the versions matters. A change to the governing law should not stop the model from answering
a question about the renewal term. A change to the renewal term itself should be flagged.
We test whether open-weight LLMs can tell these two situations apart.

> **Research question.** Can language models distinguish question-relevant from
> question-irrelevant conflicts across two versions of a contract, and answer or flag the
> disagreement accordingly?

## Benchmark

We built a controlled benchmark from 50 contracts of the
[Contract Understanding Atticus Dataset (CUAD) v1](https://www.atticusprojectai.org/cuad).
For each contract, Source A is the original text. We wrote two questions (Q1, Q2) and created
three versions of Source B:

| Version | Content |
|---|---|
| B0 | Exact copy of A |
| B1 | Copy of A where only the answer to Q1 is changed |
| B2 | Copy of A where only the answer to Q2 is changed |

Pairing each version with each question gives three conditions:

| Sources | Question | Condition | Expected behaviour |
|---|---|---|---|
| A + B0 | Q1 or Q2 | Agreement | Give the answer |
| A + B1 | Q1 | Relevant conflict | Report both answers and the disagreement |
| A + B1 | Q2 | Irrelevant conflict | Give the answer, ignore the unrelated change |

(and symmetrically for B2). This gives 6 examples per contract, **300 examples** in total
(100 per condition), covering nine clause categories such as notice periods, renewal terms,
governing law and warranty duration.

## Experiments

- **Models:** Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Granite-3.1-8B-Instruct,
  Llama-3.1-70B-Instruct
- **Prompts:** a baseline prompt and an *explained* prompt that tells the model how to handle
  relevant and irrelevant differences
  ([prompts](pipeline/prompts/))
- **Setup:** bfloat16, greedy decoding, seed 42, JSON output enforced with
  `lm-format-enforcer`; run on the Snellius supercomputer (A100 / H100)
- **Scale:** 4 models × 2 prompts × 300 examples = **2,400 responses**
- **Evaluation:** each response gets a behaviour label (expected correct / incorrect answer /
  unnecessary abstention / other error) and a binary conflict-detection label;
  95% cluster-bootstrap confidence intervals over contracts, exact McNemar tests with Holm
  correction, and a worst-case sensitivity analysis for failed responses

## Key findings

Expected-correct responses (%) on the strict common set of examples that every run could
answer, with 95% bootstrap CIs for relevant conflicts:

| Model | Agreement | Irrelevant conflict | Relevant conflict (baseline → explained) |
|---|---:|---:|---|
| Qwen2.5-7B | 98–100 | 96 | 5.6 [0, 14] → 11.1 [3, 22] |
| Llama-3.1-8B | 100 | 100 | 5.6 [0, 14] → 0.0 |
| Granite-3.1-8B | 100 | 100 | 5.6 [0, 14] → 13.9 [3, 25] |
| **Llama-3.1-70B** | 100 | 100 | **69.4 [53, 86] → 83.3 [69, 95]** |

- All models handle agreement and irrelevant conflicts almost perfectly, but the 7–8B models
  recognise at most 17% of the conflicts that change the answer.
- Scale matters: Llama-3.1-70B handles most relevant conflicts.
- The explained prompt only helps the 70B model (McNemar, Holm-adjusted p = 0.002).
- When Qwen and Granite miss a conflict, they follow the second source in over 90% of cases.

The full analysis is in the project report.

## Repository structure

```
data/
  generation/           dataset construction: LLM agent and a tested deterministic generator
  quality/              manual and LLM-based quality checks of the generated examples
pipeline/               inference for all four models (configs, prompts, Snellius jobs, tests)
  data/dataset.json     the final 300-example benchmark
evaluation/
  judging/              labelling protocol and method
  analysis/             labels, metrics, figures (main analysis)
  verification/         independent recomputation of all reported numbers
outputs/                raw model responses and run metadata per model and prompt
```

| Path | Content |
|---|---|
| [`data/generation/creating_data_agent.py`](data/generation/creating_data_agent.py) | LLM agent that selects questions and creates the modified contract versions |
| [`data/generation/berkay_generator/`](data/generation/berkay_generator/) | Deterministic, tested generator for one contributor's ten contracts |
| [`data/quality/evaluating_data_quality.ipynb`](data/quality/evaluating_data_quality.ipynb) | Structural, manual and LLM-based dataset checks |
| [`pipeline/`](pipeline/) | Inference for Qwen (native and YaRN routes) and for Llama/Granite (native long context) — see the [pipeline README](pipeline/README.md) |
| [`evaluation/`](evaluation/) | Judging protocol, labels, metric computation, figures and an independent verification — see the [evaluation README](evaluation/README.md) |
| [`outputs/`](outputs/) | Raw responses (`responses.jsonl`), run settings (`run_info.json`) and software environment for every run |

## Reproducing

```sh
# pipeline tests (no GPU or model download needed)
cd pipeline && python3 -m unittest discover -s tests

# dataset generator tests
python3 -I -B data/generation/berkay_generator/build_berkay_dataset.py --self-test

# recompute all reported numbers and figures from the label files
cd evaluation/analysis && pip install -r requirements.txt && python score_metrics.py && python analyze_results.py
```

Model inference needs a GPU; the [pipeline README](pipeline/README.md) lists the Snellius
setup and the exact commands for every run.

## Data and licence

CUAD is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The modified
contract versions are synthetic edits made for this study and are not real legal documents.

## Team

Berkay Aktaş, Dilek Barutcu, Maria Magdalini Kantara, Oğuz Kağan Yarım, Engin Halil Yedirmez
(TU Eindhoven). Supervised by the 2AMM20 ULLM track.

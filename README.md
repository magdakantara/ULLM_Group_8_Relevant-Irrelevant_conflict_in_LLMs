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
  ([prompts](contract_conflict_project/prompts/))
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

| Path | Content |
|---|---|
| [`data/creating_data_agent.py`](data/creating_data_agent.py) | LLM agent that selects questions and creates the modified contract versions |
| [`data/berkay_generator/`](data/berkay_generator/) | Deterministic, tested generator for one contributor's ten contracts |
| [`data/evaluating_data_quality.ipynb`](data/evaluating_data_quality.ipynb) | Manual and LLM-based dataset quality checks |
| [`contract_conflict_project/`](contract_conflict_project/) | Inference pipeline for Qwen, prompts, configs, tests and Snellius scripts ([README](contract_conflict_project/README.md)) |
| [`contract_conflict_differentmodels/`](contract_conflict_differentmodels/) | Native long-context runs for Llama and Granite |
| [`evaluation/`](evaluation/) | Judging protocol and an independent verification of the main results |

## Reproducing

Model inference needs a GPU; see the
[pipeline README](contract_conflict_project/README.md) for the Snellius setup and the exact
commands. The dataset generator tests and the verification script run on a laptop with the
Python standard library (plus `openpyxl` for the generator):

```sh
python3 -I -B data/berkay_generator/build_berkay_dataset.py --self-test
python3 -I evaluation/verification/analysis.py LABEL_DIR dataset.json EXTRA_DIR OUT_DIR
```

## Data and licence

CUAD is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The modified
contract versions are synthetic edits made for this study and are not real legal documents.

## Team

Berkay Aktaş, Dilek Barutcu, Maria Magdalini Kantara, Oğuz Kağan Yarım, Engin Halil Yedirmez
(TU Eindhoven). Supervised by the 2AMM20 ULLM track.

# Evaluation

## Judging

[`judging/JUDGING_PROTOCOL.md`](judging/JUDGING_PROTOCOL.md) defines the behaviour labels,
the conflict-detection rule and how ambiguous cases were decided.
[`judging/METHODS.md`](judging/METHODS.md) describes the judging pass for the Qwen responses
(575 assessable responses, 25 technical failures). The labels are AI-assisted and were not
fully verified by humans.

Two decisions matter most for reading the numbers:

- A response counts as detecting a conflict only if it states that the sources disagree on
  the asked fact. Mentioning an unrelated difference, or listing both values without
  attributing them to the sources, does not count.
- Responses that failed technically (truncated, invalid JSON, out of memory) get no behaviour
  label and are reported separately.

## Verification

[`verification/analysis.py`](verification/analysis.py) recomputes the main results from the
label files with the Python standard library only, independently of the main analysis code:

- expected-correct and detection rates per model, prompt and condition, on all assessable
  responses, on the strict common set and under a worst-case assumption
- 95% confidence intervals: cluster bootstrap over contracts (10,000 resamples, seed 2026) and
  Wilson intervals
- exact McNemar tests (baseline vs. explained prompt), over all conditions and on relevant
  conflicts only, with Holm correction
- false-positive counts, source choice for missed conflicts, and Qwen results per context route

It runs twice, with the original labels and with five corrected Llama-3.1-70B labels (answers
that explicitly reported both source values but had been labelled as missed). Its results match
the main analysis exactly.

```sh
python3 -I verification/analysis.py LABEL_DIR dataset.json EXTRA_DIR OUT_DIR
```

`EXTRA_DIR` must contain `likely_label_errors.csv`, `task7_source_choice.csv` and
`qwen_route_map.csv`.

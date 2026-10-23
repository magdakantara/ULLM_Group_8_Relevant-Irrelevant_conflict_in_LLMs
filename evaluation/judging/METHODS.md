# Judging methodology and disclosure

## Evaluation procedure

The judging stage evaluated 600 responses from the supplied Qwen2.5-7B-Instruct experiment: two prompting strategies (`baseline` and `explained`) applied to 300 examples from 50 contracts. The examples comprised 100 agreement, 100 irrelevant-conflict, and 100 relevant-conflict cases. The generation model and experiment design are reported as supplied by the project authors; generation was not rerun.

Before annotation, the CSV was compared against the original HTML examples, including questions, reference answers, expected behavior, model answers, model-generated evidence, response status, route and review ID. The checks found no mismatches. Original Source A texts were consistent within each contract's six examples, the same A/B versions were used for each paired question, and B0 matched A after whitespace normalization. Relevant source changes and question-target passages were inspected for semantic adjudication. This was not a comprehensive legal audit of every provision in every contract.

One AI evaluator, operating interactively in Codex, reviewed all 575 responses with technical status `ok`. The actual response text was assessed against the supplied evaluation definitions, reference answers, and the question-relevant clauses in the original Source A/B texts. The evaluator knew the benchmark condition and references; the pass was not blinded. No second independent evaluator or completed human verification is claimed. The 23 truncated and two format-error responses were preserved as technical failures, with no semantic labels inferred from them.

Each assessable response received one behavior label, an independently determined binary conflict-detection value, and a concise rationale. Recognizing a disagreement could coexist with an incorrect answer, particularly when a response selected one source arbitrarily. Differences unrelated to the current question were not counted as relevant conflict detection. Model-generated evidence fields were not treated as authoritative source text.

Decisions were explicitly recorded per response in an appendable JSONL ledger during the review. After a final consistency pass, the saved decisions were assembled into `review_judged.csv`; only the three annotation columns were populated. Mechanical validation checked completeness, allowed values, technical-failure separation, source correspondence and preservation of original fields. Mechanical validation does not establish that every semantic judgment is correct.

## Human verification

The supplied protocol requires human verification of ambiguous and unusual cases and a stratified sample of ordinary predictions. This has been prepared, **not completed**. The review table contains 58 targeted responses, including all positive detection decisions, all substantive errors in non-relevant-conflict conditions, and selected boundary cases. An additional 60 ordinary responses were selected deterministically, ten from each condition × prompt stratum after excluding targeted cases. All 118 entries are marked pending.

Human reviewers can confirm or override decisions while preserving the AI ledger and original responses. Until that verification is performed, the labels should be described as completed AI candidate annotations rather than a human-validated gold standard. No final experiment metrics are included in this package.

## Reproducibility and AI disclosure

The evaluation was conducted through a Codex chat, not a separately executed evaluator API pipeline. The exact serving/API model identifier and sampling parameters were not exposed and are recorded as unavailable. A model identifier or temperature setting has not been inferred. This limits independent replication of the semantic judging process and should remain disclosed in any report.

The original inputs, saved decisions, source references, validation code and hashes are preserved. The offline `judge.py` script deterministically reconstructs the completed CSV without invoking a model. It reproduces the recorded annotations, not the evaluator's reasoning process. No API keys or API-specific configuration are needed for this replay. A future API evaluator would be a separate implementation requiring a verified provider and model identifier.

AI assistance was used to inspect the files, judge the responses, prepare the rationales, and generate the validation scripts and documentation. The project authors remain responsible for actual human verification and any required course-specific AI-use declaration.

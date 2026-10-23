# Judgment definitions and adjudication notes

## Scope

This protocol records how the supplied definitions were applied to existing responses. It introduces no new behavior labels and does not change the benchmark conditions, reference answers, sources, questions or model responses.

## Required labels

| Label | Application |
| --- | --- |
| `expected_correct` | Correct supported answer in agreement/irrelevant conflict; in relevant conflict, appropriately reports the answer-relevant disagreement without choosing one source as the single authoritative answer |
| `unnecessary_abstention` | Refuses or abstains because of a supposed relevant conflict when consistent information is sufficient to answer |
| `incorrect_answer` | Substantively wrong answer, missed relevant conflict, false agreement, or arbitrary selection of one conflicting source |
| `other_error` | Semantically malformed/unusable response that cannot meaningfully be assessed under the other categories |

`status=ok` indicates technical generation success only. Statuses `truncated` and `format_error` remain technical failures, with all annotation fields blank. An awkward or unfinished quotation within an `ok` response does not itself make an otherwise assessable answer unusable.

## Independent conflict decision

`conflict_detected=yes` means that the response recognizes incompatible information affecting the current question. Reporting both incompatible answer values can demonstrate recognition without the literal word “conflict.” Merely discussing an unrelated difference does not. The benchmark condition does not determine this annotation.

A response can be `incorrect_answer` and `yes`: it may recognize the conflict and then pick a source arbitrarily or attribute the competing values to the wrong sources. Source B is not inherently newer or more authoritative. A difference appearing only in model-generated evidence is insufficient when the answer itself unambiguously asserts agreement or gives one answer without recognizing the competing value.

## Reading and evidence rules

1. Read the whole assessable answer, not only its first sentence or a keyword match.
2. Check the requested fact and relevant clauses in original `Full source texts / Source A` and `Source B`.
3. Use reference answers as guides, checking semantic equivalence rather than exact string matching.
4. Treat `evidence_A` and `evidence_B` as part of the model's output, never as authoritative substitutes for the sources. A clear source-attribution reversal can be substantive.
5. Separate answer errors from unsupported but immaterial citation/wording asides. Penalize additions that contradict the requested contractual outcome. This boundary warrants human review.
6. Read a later qualification in context. A clear A/B comparison may qualify an initially single-valued sentence; explicitly choosing one source and merely appending a caveat does not necessarily retract that choice.
7. Write the rationale for the observed answer, independently of the condition's expected behavior.

## Important boundary cases for human review

| Review IDs | Current decision and reason to review |
| --- | --- |
| Both prompts, `EX-032` and `EX-034` | `incorrect_answer/no`: Florida court jurisdiction is quoted instead of governing law; matching the state name is insufficient under this reading |
| Both prompts, `EX-051` | `incorrect_answer/yes`: both incompatible laws are reported, but evidence fields reverse A/B attribution; review the attribution boundary |
| `explained::EX-126` | `expected_correct/yes`: the initial 45-day answer is qualified by explicit A=15/B=45 statements; no source is declared authoritative |
| `explained::DIL-09-Allied-B1-Q1` | `incorrect_answer/yes`: explicitly selects B, then notes both dates; the final caveat is interpreted as not retracting the arbitrary selection |
| `explained::DIL-04-Nettaxi-B1-Q1` | `incorrect_answer/yes`: recognizes the dates but repeatedly selects A as the answer |
| `explained::EX-158`, `baseline::EX-161`, `baseline::EX-179` | `incorrect_answer/no`: correct period followed by a substantive contradiction about renewal or its stopping condition |
| Both prompts, `EX-050` and `EX-052` | `expected_correct/no`: a ten-year term is accepted as a sufficient expiry answer without requiring a calendar date |
| Both prompts, `OGUZ-02-B2-Q1` | `expected_correct/no`: common renewal duration is correct; a B-specific notice period is treated as outside the question's requested duration |

Additional scope/citation cases and all positive-detection cases are included in `human_review.csv`. In particular, correct state-law answers that omit the CISG exclusion, unsupported section references, and the distinction between governing law and arbitration-law wording are flagged where relevant. No human agreement is assumed.

## Audit artifacts

`annotations.jsonl` records every decision and its HTML filename. `source_spans.jsonl` provides automatically located target-passage windows around the answer-changing edits, mapped to each source version. Offsets are zero-based, end-exclusive Python character offsets in the HTML-unescaped, stripped source `<pre>` contents. These excerpts are checked verbatim; their selection is a navigation aid and does not certify that they contain every relevant clause. `source_changes.json` records the inspected relevant A/B differences.

`changes_from_previous.csv` lists label/detection changes only. All 575 rationales were reassessed; unchanged labels are not evidence of skipped review. The superseded CSV is archived outside the final package to avoid accidental use.

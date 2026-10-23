import pandas as pd
import numpy as np
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint
import os
import re

os.makedirs("results", exist_ok=True)
os.makedirs("results/extra", exist_ok=True)


# LOAD LABELS

qwen = pd.read_csv("data/qwen7b_labels.csv")
llama8 = pd.read_csv("data/llama8b_labels.csv")
granite = pd.read_csv("data/granite8b_labels.csv")
llama70 = pd.read_csv("data/llama70b_labels.csv")

qwen["model"] = "Qwen 7B"
llama8["model"] = "Llama 8B"
granite["model"] = "Granite 8B"
llama70["model"] = "Llama 70B"

original_models = {
    "Qwen 7B": qwen.copy(),
    "Llama 8B": llama8.copy(),
    "Granite 8B": granite.copy(),
    "Llama 70B": llama70.copy()
}

prompts = ["baseline", "explained"]

conditions = [
    "agreement",
    "irrelevant_conflict",
    "relevant_conflict"
]


# FIND EXTRA DATA

possible_extra_folders = [
    "extra_data",
    "engin-handoff",
    "data/extra_data",
    "data/engin-handoff"
]

extra_folder = None

for folder in possible_extra_folders:
    if os.path.exists(
        os.path.join(folder, "likely_label_errors.csv")
    ):
        extra_folder = folder
        break

if extra_folder is None:
    raise FileNotFoundError(
        "Could not find the extra handoff files. "
        "Put them in a folder called extra_data."
    )

print("\nUsing extra data from:", extra_folder)


# CREATE TRACEABLE LABEL OVERRIDES

likely_errors = pd.read_csv(
    os.path.join(
        extra_folder,
        "likely_label_errors.csv"
    )
)

model_name_map = {
    "qwen7b": "Qwen 7B",
    "llama8b": "Llama 8B",
    "granite8b": "Granite 8B",
    "llama70b": "Llama 70B"
}

overrides = []

for _, row in likely_errors.iterrows():

    model = model_name_map.get(
        row["model"],
        row["model"]
    )

    data = original_models[model]

    match = data[
        (data["example_id"] == row["example_id"]) &
        (data["prompt_type"] == row["prompt_type"])
    ]

    if len(match) != 1:
        print(
            "Warning: could not uniquely find",
            model,
            row["example_id"],
            row["prompt_type"]
        )
        continue

    old_conflict = match.iloc[0]["conflict_detected"]
    old_behavior = match.iloc[0]["behavior_label"]

    overrides.append({
        "model": model,
        "example_id": row["example_id"],
        "prompt_type": row["prompt_type"],
        "field": "conflict_detected",
        "old": old_conflict,
        "new": "yes",
        "reason":
            "Explicitly states the conflicting Source A and Source B values"
    })

    if old_behavior != "expected_correct":
        overrides.append({
            "model": model,
            "example_id": row["example_id"],
            "prompt_type": row["prompt_type"],
            "field": "behavior_label",
            "old": old_behavior,
            "new": "expected_correct",
            "reason":
                "Correctly reports both source-specific conflicting answers"
        })


overrides_df = pd.DataFrame(overrides)

overrides_df.to_csv(
    "results/extra/label_overrides.csv",
    index=False
)


# APPLY OVERRIDES IN MEMORY ONLY

models = {
    model: data.copy()
    for model, data in original_models.items()
}

for _, row in overrides_df.iterrows():

    data = models[row["model"]]

    mask = (
        (data["example_id"] == row["example_id"]) &
        (data["prompt_type"] == row["prompt_type"])
    )

    data.loc[
        mask,
        row["field"]
    ] = row["new"]


# METRIC FUNCTIONS

def conflict_metrics(data):

    actual = (
        data["case_label"] ==
        "relevant_conflict"
    )

    predicted = (
        data["conflict_detected"] ==
        "yes"
    )

    tp = (actual & predicted).sum()
    fp = (~actual & predicted).sum()
    fn = (actual & ~predicted).sum()
    tn = (~actual & ~predicted).sum()

    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = np.nan

    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = np.nan

    if (
        not np.isnan(precision) and
        not np.isnan(recall) and
        precision + recall > 0
    ):
        f1 = (
            2 * precision * recall /
            (precision + recall)
        )
    else:
        f1 = np.nan

    return (
        tp, fp, fn, tn,
        precision, recall, f1
    )


def percent(value):

    if pd.isna(value):
        return np.nan

    return value * 100


# CONTRACT ID FOR CLUSTER BOOTSTRAP

def get_contract_id(example_id):

    example_id = str(example_id)

    match = re.fullmatch(
        r"EX-(\d+)",
        example_id
    )

    if match:

        number = int(match.group(1))

        contract_number = (
            (number - 1) // 6
        ) + 1

        return f"EX-CONTRACT-{contract_number:02d}"

    contract_id = re.sub(
        r"-B[012]-Q[12]$",
        "",
        example_id
    )

    return contract_id


for model, data in models.items():

    data["contract_id"] = (
        data["example_id"]
        .apply(get_contract_id)
    )


for model, data in original_models.items():

    data["contract_id"] = (
        data["example_id"]
        .apply(get_contract_id)
    )


# TECHNICAL RESULTS

technical_results = []

for model, data in models.items():

    for prompt in prompts:

        part = data[
            data["prompt_type"] == prompt
        ]

        total = len(part)

        ok = (
            part["run_status"] ==
            "ok"
        ).sum()

        technical_results.append({
            "model": model,
            "prompt": prompt,
            "total": total,
            "ok": ok,
            "failures": total - ok,
            "assessable_percent":
                ok / total * 100
        })


technical_df = pd.DataFrame(
    technical_results
)

technical_df.to_csv(
    "results/technical_status.csv",
    index=False
)


# ALL ASSESSABLE RESPONSES

all_results = []

for model, data in models.items():

    for prompt in prompts:

        part = data[
            (data["prompt_type"] == prompt) &
            (data["run_status"] == "ok")
        ]

        correct = (
            part["behavior_label"] ==
            "expected_correct"
        ).sum()

        accuracy = (
            correct /
            len(part) *
            100
        )

        (
            tp, fp, fn, tn,
            precision, recall, f1
        ) = conflict_metrics(part)

        all_results.append({
            "model": model,
            "prompt": prompt,
            "assessed": len(part),
            "correct": correct,
            "accuracy": accuracy,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision": percent(precision),
            "recall": percent(recall),
            "f1": percent(f1)
        })


all_results_df = pd.DataFrame(
    all_results
)

all_results_df.to_csv(
    "results/all_results.csv",
    index=False
)


# STRICT COMMON SET

common_ids = None

for model, data in models.items():

    ok = data[
        data["run_status"] == "ok"
    ]

    counts = (
        ok.groupby("example_id")
        ["prompt_type"]
        .nunique()
    )

    ids = set(
        counts[counts == 2].index
    )

    if common_ids is None:
        common_ids = ids
    else:
        common_ids = (
            common_ids.intersection(ids)
        )


common_ids = sorted(common_ids)

print(
    "\nStrict common examples:",
    len(common_ids)
)


common_results = []
common_condition_results = []

for model, data in models.items():

    for prompt in prompts:

        part = data[
            (data["example_id"].isin(common_ids)) &
            (data["prompt_type"] == prompt) &
            (data["run_status"] == "ok")
        ]

        correct = (
            part["behavior_label"] ==
            "expected_correct"
        ).sum()

        accuracy = (
            correct /
            len(part) *
            100
        )

        (
            tp, fp, fn, tn,
            precision, recall, f1
        ) = conflict_metrics(part)

        common_results.append({
            "model": model,
            "prompt": prompt,
            "n": len(part),
            "correct": correct,
            "accuracy": accuracy,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision": percent(precision),
            "recall": percent(recall),
            "f1": percent(f1)
        })

        for condition in conditions:

            condition_data = part[
                part["case_label"] ==
                condition
            ]

            condition_correct = (
                condition_data[
                    "behavior_label"
                ] ==
                "expected_correct"
            ).sum()

            if len(condition_data) > 0:

                condition_accuracy = (
                    condition_correct /
                    len(condition_data) *
                    100
                )

            else:
                condition_accuracy = np.nan

            common_condition_results.append({
                "model": model,
                "prompt": prompt,
                "condition": condition,
                "n": len(condition_data),
                "correct":
                    condition_correct,
                "accuracy":
                    condition_accuracy
            })


common_df = pd.DataFrame(
    common_results
)

common_df.to_csv(
    "results/common_results.csv",
    index=False
)


condition_df = pd.DataFrame(
    common_condition_results
)

condition_df.to_csv(
    "results/common_by_condition.csv",
    index=False
)


reference = models["Qwen 7B"][
    (
        models["Qwen 7B"]["example_id"]
        .isin(common_ids)
    ) &
    (
        models["Qwen 7B"]["prompt_type"]
        == "baseline"
    )
]


distribution = (
    reference["case_label"]
    .value_counts()
)

print("\nCommon set distribution:")
print(distribution)


# PROMPT-SPECIFIC COMMON SETS

prompt_common_results = []

prompt_common_ids = {}

for prompt in prompts:

    prompt_ids = None

    for model, data in models.items():

        ids = set(
            data[
                (data["prompt_type"] == prompt) &
                (data["run_status"] == "ok")
            ]["example_id"]
        )

        if prompt_ids is None:
            prompt_ids = ids
        else:
            prompt_ids = (
                prompt_ids.intersection(ids)
            )

    prompt_common_ids[prompt] = prompt_ids

    print(
        f"\n{prompt} common examples:",
        len(prompt_ids)
    )

    for model, data in models.items():

        part = data[
            (
                data["example_id"]
                .isin(prompt_ids)
            ) &
            (
                data["prompt_type"]
                == prompt
            ) &
            (
                data["run_status"]
                == "ok"
            )
        ]

        correct = (
            part["behavior_label"] ==
            "expected_correct"
        ).sum()

        accuracy = (
            correct /
            len(part) *
            100
        )

        (
            tp, fp, fn, tn,
            precision, recall, f1
        ) = conflict_metrics(part)

        prompt_common_results.append({
            "model": model,
            "prompt": prompt,
            "n": len(part),
            "correct": correct,
            "accuracy": accuracy,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision": percent(precision),
            "recall": percent(recall),
            "f1": percent(f1)
        })


prompt_common_df = pd.DataFrame(
    prompt_common_results
)

prompt_common_df.to_csv(
    "results/prompt_common_results.csv",
    index=False
)


# RELEVANT-CONFLICT PERFORMANCE

relevant_results = []

for model, data in models.items():

    for prompt in prompts:

        part = data[
            (data["prompt_type"] == prompt) &
            (data["run_status"] == "ok") &
            (
                data["case_label"] ==
                "relevant_conflict"
            )
        ]

        correct = (
            part["behavior_label"] ==
            "expected_correct"
        ).sum()

        detected = (
            part["conflict_detected"] ==
            "yes"
        ).sum()

        relevant_results.append({
            "model": model,
            "prompt": prompt,
            "n": len(part),
            "correct": correct,
            "accuracy":
                correct / len(part) * 100,
            "detected": detected,
            "detection_recall":
                detected / len(part) * 100
        })


relevant_df = pd.DataFrame(
    relevant_results
)

relevant_df.to_csv(
    "results/relevant_conflict_results.csv",
    index=False
)


# POOLED MCNEMAR

mcnemar_results = []

for model, data in models.items():

    baseline = data[
        (data["prompt_type"] == "baseline") &
        (data["run_status"] == "ok")
    ][
        ["example_id", "behavior_label"]
    ]

    explained = data[
        (data["prompt_type"] == "explained") &
        (data["run_status"] == "ok")
    ][
        ["example_id", "behavior_label"]
    ]

    baseline = baseline.rename(
        columns={
            "behavior_label":
            "baseline_label"
        }
    )

    explained = explained.rename(
        columns={
            "behavior_label":
            "explained_label"
        }
    )

    paired = baseline.merge(
        explained,
        on="example_id"
    )

    baseline_correct = (
        paired["baseline_label"] ==
        "expected_correct"
    )

    explained_correct = (
        paired["explained_label"] ==
        "expected_correct"
    )

    improved = (
        ~baseline_correct &
        explained_correct
    ).sum()

    worsened = (
        baseline_correct &
        ~explained_correct
    ).sum()

    changed = improved + worsened

    if changed > 0:

        p_value = binomtest(
            improved,
            changed,
            p=0.5,
            alternative="two-sided"
        ).pvalue

    else:
        p_value = 1.0

    mcnemar_results.append({
        "model": model,
        "paired_examples":
            len(paired),
        "improved": improved,
        "worsened": worsened,
        "p_value": p_value
    })


mcnemar_df = pd.DataFrame(
    mcnemar_results
)

mcnemar_df.to_csv(
    "results/mcnemar_results.csv",
    index=False
)


# BEFORE / AFTER OVERRIDE COMPARISON

def simple_results(model_dict, version):

    rows = []

    for model, data in model_dict.items():

        for prompt in prompts:

            part = data[
                (data["prompt_type"] == prompt) &
                (data["run_status"] == "ok")
            ]

            relevant = part[
                part["case_label"] ==
                "relevant_conflict"
            ]

            correct = (
                part["behavior_label"] ==
                "expected_correct"
            ).sum()

            relevant_correct = (
                relevant["behavior_label"] ==
                "expected_correct"
            ).sum()

            detected = (
                relevant["conflict_detected"] ==
                "yes"
            ).sum()

            rows.append({
                "version": version,
                "model": model,
                "prompt": prompt,
                "assessed": len(part),
                "correct": correct,
                "accuracy":
                    correct / len(part) * 100,
                "relevant_n":
                    len(relevant),
                "relevant_correct":
                    relevant_correct,
                "relevant_accuracy":
                    relevant_correct /
                    len(relevant) * 100,
                "relevant_detected":
                    detected,
                "detection_recall":
                    detected /
                    len(relevant) * 100
            })

    return rows


before_after = []

before_after += simple_results(
    original_models,
    "before_overrides"
)

before_after += simple_results(
    models,
    "after_overrides"
)

before_after_df = pd.DataFrame(
    before_after
)

before_after_df.to_csv(
    "results/extra/metrics_before_after.csv",
    index=False
)


# WILSON CONFIDENCE INTERVAL

def wilson_interval(successes, total):

    if total == 0:
        return np.nan, np.nan

    low, high = proportion_confint(
        successes,
        total,
        alpha=0.05,
        method="wilson"
    )

    return low * 100, high * 100


# CLUSTER BOOTSTRAP OVER CONTRACTS
# 10,000 resamples, fixed seed

BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 42


def cluster_bootstrap_rate(
    data,
    success_column,
    success_value
):

    if len(data) == 0:
        return np.nan, np.nan

    # Calculate contract-level statistics only once

    temp = data[
        ["contract_id", success_column]
    ].copy()

    temp["success"] = (
        temp[success_column] ==
        success_value
    ).astype(int)

    contract_stats = (
        temp
        .groupby("contract_id")
        .agg(
            successes=("success", "sum"),
            total=("success", "size")
        )
        .reset_index()
    )

    if len(contract_stats) == 0:
        return np.nan, np.nan

    successes = (
        contract_stats["successes"]
        .to_numpy()
    )

    totals = (
        contract_stats["total"]
        .to_numpy()
    )

    n_contracts = len(contract_stats)

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    # Resample whole contracts with replacement

    sampled_indices = rng.integers(
        0,
        n_contracts,
        size=(
            BOOTSTRAP_SAMPLES,
            n_contracts
        )
    )

    bootstrap_successes = (
        successes[sampled_indices]
        .sum(axis=1)
    )

    bootstrap_totals = (
        totals[sampled_indices]
        .sum(axis=1)
    )

    bootstrap_rates = (
        bootstrap_successes /
        bootstrap_totals *
        100
    )

    low = np.percentile(
        bootstrap_rates,
        2.5
    )

    high = np.percentile(
        bootstrap_rates,
        97.5
    )

    return low, high


# CONFIDENCE INTERVAL TABLE

ci_rows = []


def add_ci_row(
    model,
    prompt,
    analysis,
    condition,
    data,
    success_column,
    success_value
):

    total = len(data)

    successes = (
        data[success_column] ==
        success_value
    ).sum()

    if total > 0:
        rate = successes / total * 100
    else:
        rate = np.nan

    wilson_low, wilson_high = (
        wilson_interval(
            successes,
            total
        )
    )

    boot_low, boot_high = (
        cluster_bootstrap_rate(
            data,
            success_column,
            success_value
        )
    )

    ci_rows.append({
        "model": model,
        "prompt": prompt,
        "analysis": analysis,
        "condition": condition,
        "n": total,
        "successes": successes,
        "rate": rate,
        "bootstrap_low_95":
            boot_low,
        "bootstrap_high_95":
            boot_high,
        "wilson_low_95":
            wilson_low,
        "wilson_high_95":
            wilson_high
    })


print("\nCalculating confidence intervals...")


for model, data in models.items():

    for prompt in prompts:

        assessable = data[
            (data["prompt_type"] == prompt) &
            (data["run_status"] == "ok")
        ]

        # Overall assessable accuracy

        add_ci_row(
            model,
            prompt,
            "all_assessable",
            "overall",
            assessable,
            "behavior_label",
            "expected_correct"
        )

        # Per condition

        for condition in conditions:

            part = assessable[
                assessable["case_label"] ==
                condition
            ]

            add_ci_row(
                model,
                prompt,
                "all_assessable",
                condition,
                part,
                "behavior_label",
                "expected_correct"
            )

        # Relevant-conflict detection recall

        relevant = assessable[
            assessable["case_label"] ==
            "relevant_conflict"
        ]

        add_ci_row(
            model,
            prompt,
            "all_assessable",
            "relevant_conflict_detection",
            relevant,
            "conflict_detected",
            "yes"
        )

        # Strict common

        strict = data[
            (
                data["example_id"]
                .isin(common_ids)
            ) &
            (
                data["prompt_type"] ==
                prompt
            ) &
            (
                data["run_status"] ==
                "ok"
            )
        ]

        add_ci_row(
            model,
            prompt,
            "strict_common",
            "overall",
            strict,
            "behavior_label",
            "expected_correct"
        )

        for condition in conditions:

            part = strict[
                strict["case_label"] ==
                condition
            ]

            add_ci_row(
                model,
                prompt,
                "strict_common",
                condition,
                part,
                "behavior_label",
                "expected_correct"
            )

        strict_relevant = strict[
            strict["case_label"] ==
            "relevant_conflict"
        ]

        add_ci_row(
            model,
            prompt,
            "strict_common",
            "relevant_conflict_detection",
            strict_relevant,
            "conflict_detected",
            "yes"
        )

        # Prompt-specific common

        prompt_ids = (
            prompt_common_ids[prompt]
        )

        prompt_part = data[
            (
                data["example_id"]
                .isin(prompt_ids)
            ) &
            (
                data["prompt_type"] ==
                prompt
            ) &
            (
                data["run_status"] ==
                "ok"
            )
        ]

        add_ci_row(
            model,
            prompt,
            "prompt_specific_common",
            "overall",
            prompt_part,
            "behavior_label",
            "expected_correct"
        )

        prompt_relevant = prompt_part[
            prompt_part["case_label"] ==
            "relevant_conflict"
        ]

        add_ci_row(
            model,
            prompt,
            "prompt_specific_common",
            "relevant_conflict",
            prompt_relevant,
            "behavior_label",
            "expected_correct"
        )

        add_ci_row(
            model,
            prompt,
            "prompt_specific_common",
            "relevant_conflict_detection",
            prompt_relevant,
            "conflict_detected",
            "yes"
        )


ci_df = pd.DataFrame(ci_rows)

ci_df.to_csv(
    "results/extra/confidence_intervals.csv",
    index=False
)

print("Confidence intervals finished.")


# MCNEMAR POOLED + RELEVANT ONLY
# WITH HOLM CORRECTION

def mcnemar_for_data(
    data,
    relevant_only=False
):

    if relevant_only:

        data = data[
            data["case_label"] ==
            "relevant_conflict"
        ]

    baseline = data[
        (data["prompt_type"] == "baseline") &
        (data["run_status"] == "ok")
    ][
        ["example_id", "behavior_label"]
    ]

    explained = data[
        (data["prompt_type"] == "explained") &
        (data["run_status"] == "ok")
    ][
        ["example_id", "behavior_label"]
    ]

    baseline = baseline.rename(
        columns={
            "behavior_label":
            "baseline_label"
        }
    )

    explained = explained.rename(
        columns={
            "behavior_label":
            "explained_label"
        }
    )

    paired = baseline.merge(
        explained,
        on="example_id"
    )

    baseline_correct = (
        paired["baseline_label"] ==
        "expected_correct"
    )

    explained_correct = (
        paired["explained_label"] ==
        "expected_correct"
    )

    improved = (
        ~baseline_correct &
        explained_correct
    ).sum()

    worsened = (
        baseline_correct &
        ~explained_correct
    ).sum()

    changed = improved + worsened

    if changed > 0:

        p_value = binomtest(
            improved,
            changed,
            p=0.5,
            alternative="two-sided"
        ).pvalue

    else:
        p_value = 1.0

    return (
        len(paired),
        improved,
        worsened,
        p_value
    )


def holm_correction(p_values):

    p_values = np.array(
        p_values,
        dtype=float
    )

    n = len(p_values)

    order = np.argsort(p_values)

    adjusted = np.empty(n)

    running_max = 0

    for rank, index in enumerate(order):

        multiplier = n - rank

        value = (
            p_values[index] *
            multiplier
        )

        value = min(
            value,
            1.0
        )

        running_max = max(
            running_max,
            value
        )

        adjusted[index] = (
            running_max
        )

    return adjusted


mcnemar_extra_rows = []

for scope in [
    "pooled",
    "relevant_conflict"
]:

    temp_rows = []

    for model, data in models.items():

        (
            paired,
            improved,
            worsened,
            p_value
        ) = mcnemar_for_data(
            data,
            relevant_only=(
                scope ==
                "relevant_conflict"
            )
        )

        temp_rows.append({
            "scope": scope,
            "model": model,
            "paired_examples":
                paired,
            "improved":
                improved,
            "worsened":
                worsened,
            "p_value":
                p_value
        })

    adjusted = holm_correction(
        [
            row["p_value"]
            for row in temp_rows
        ]
    )

    for row, corrected_p in zip(
        temp_rows,
        adjusted
    ):

        row["p_holm"] = corrected_p

        row["significant_0_05"] = (
            corrected_p < 0.05
        )

        mcnemar_extra_rows.append(
            row
        )


mcnemar_extra_df = pd.DataFrame(
    mcnemar_extra_rows
)

mcnemar_extra_df.to_csv(
    "results/extra/mcnemar_pooled_and_relevant.csv",
    index=False
)


# WORST-CASE SENSITIVITY
# Non-ok = incorrect and not detected

worst_case_rows = []

for model, data in models.items():

    for prompt in prompts:

        part = data[
            data["prompt_type"] ==
            prompt
        ]

        correct = (
            (
                part["run_status"] ==
                "ok"
            ) &
            (
                part["behavior_label"] ==
                "expected_correct"
            )
        ).sum()

        overall_accuracy = (
            correct /
            len(part) *
            100
        )

        relevant = part[
            part["case_label"] ==
            "relevant_conflict"
        ]

        detected = (
            (
                relevant["run_status"] ==
                "ok"
            ) &
            (
                relevant[
                    "conflict_detected"
                ] ==
                "yes"
            )
        ).sum()

        relevant_correct = (
            (
                relevant["run_status"] ==
                "ok"
            ) &
            (
                relevant[
                    "behavior_label"
                ] ==
                "expected_correct"
            )
        ).sum()

        worst_case_rows.append({
            "model": model,
            "prompt": prompt,
            "n": len(part),
            "correct": correct,
            "worst_case_accuracy":
                overall_accuracy,
            "relevant_n":
                len(relevant),
            "relevant_correct":
                relevant_correct,
            "worst_case_relevant_accuracy":
                relevant_correct /
                len(relevant) * 100,
            "relevant_detected":
                detected,
            "worst_case_detection_recall":
                detected /
                len(relevant) * 100
        })


worst_case_df = pd.DataFrame(
    worst_case_rows
)

worst_case_df["accuracy_rank"] = (
    worst_case_df
    .groupby("prompt")[
        "worst_case_accuracy"
    ]
    .rank(
        ascending=False,
        method="min"
    )
)

worst_case_df.to_csv(
    "results/extra/worst_case_sensitivity.csv",
    index=False
)


# DETECTED CONFLICT BUT STILL INCORRECT

detected_incorrect_rows = []

for model, data in models.items():

    for prompt in prompts:

        relevant = data[
            (data["prompt_type"] == prompt) &
            (data["run_status"] == "ok") &
            (
                data["case_label"] ==
                "relevant_conflict"
            )
        ]

        detected = relevant[
            relevant["conflict_detected"] ==
            "yes"
        ]

        detected_but_incorrect = detected[
            detected["behavior_label"] ==
            "incorrect_answer"
        ]

        if len(detected) > 0:

            percent_incorrect = (
                len(detected_but_incorrect) /
                len(detected) *
                100
            )

        else:
            percent_incorrect = np.nan

        detected_incorrect_rows.append({
            "model": model,
            "prompt": prompt,
            "relevant_n":
                len(relevant),
            "detected":
                len(detected),
            "detected_but_incorrect":
                len(detected_but_incorrect),
            "percent_of_detected_incorrect":
                percent_incorrect
        })


detected_incorrect_df = pd.DataFrame(
    detected_incorrect_rows
)

detected_incorrect_df.to_csv(
    "results/extra/detected_but_incorrect.csv",
    index=False
)


# MISSED RELEVANT CONFLICT SOURCE CHOICE

task7 = pd.read_csv(
    os.path.join(
        extra_folder,
        "task7_source_choice.csv"
    )
)

task7["model"] = (
    task7["model"]
    .map(model_name_map)
    .fillna(task7["model"])
)

choice_map = {
    "A": "Source A",
    "B": "Source B",
    "neither": "neither",
    "unclear": "unclear",
    "both_unattributed":
        "both_unattributed",
    "both_attributed":
        "both_attributed"
}

task7["final_choice"] = (
    task7["final_choice"]
    .map(choice_map)
    .fillna(task7["final_choice"])
)


task7_summary = (
    task7
    .groupby(
        [
            "model",
            "prompt_type",
            "final_choice"
        ]
    )
    .size()
    .reset_index(name="count")
)


task7_totals = (
    task7_summary
    .groupby(
        ["model", "prompt_type"]
    )["count"]
    .transform("sum")
)


task7_summary["percent"] = (
    task7_summary["count"] /
    task7_totals *
    100
)


task7_summary.to_csv(
    "results/extra/task7_source_choice_summary.csv",
    index=False
)


# SOURCE A / SOURCE B POSITION ANALYSIS

position_rows = []

for (
    model,
    prompt
), group in task7.groupby(
    ["model", "prompt_type"]
):

    source_a = (
        group["final_choice"] ==
        "Source A"
    ).sum()

    source_b = (
        group["final_choice"] ==
        "Source B"
    ).sum()

    other = (
        ~group["final_choice"].isin(
            ["Source A", "Source B"]
        )
    ).sum()

    a_or_b = source_a + source_b

    if a_or_b > 0:

        source_b_percent = (
            source_b /
            a_or_b *
            100
        )

    else:
        source_b_percent = np.nan

    position_rows.append({
        "model": model,
        "prompt": prompt,
        "total_missed":
            len(group),
        "source_A":
            source_a,
        "source_B":
            source_b,
        "other_or_unclear":
            other,
        "source_B_percent_among_A_or_B":
            source_b_percent
    })


position_df = pd.DataFrame(
    position_rows
)

position_df.to_csv(
    "results/extra/source_position_bias.csv",
    index=False
)


# QWEN ROUTE: SHORT 32K VS LONG YARN

route_map = pd.read_csv(
    os.path.join(
        extra_folder,
        "qwen_route_map.csv"
    )
)


qwen_corrected = models[
    "Qwen 7B"
].merge(
    route_map,
    on="example_id",
    how="left"
)


qwen_route_rows = []


for prompt in prompts:

    for route in ["short", "long"]:

        part = qwen_corrected[
            (
                qwen_corrected[
                    "prompt_type"
                ] ==
                prompt
            ) &
            (
                qwen_corrected["route"]
                == route
            )
        ]

        total = len(part)

        ok = (
            part["run_status"] ==
            "ok"
        ).sum()

        assessable = part[
            part["run_status"] ==
            "ok"
        ]

        correct = (
            assessable[
                "behavior_label"
            ] ==
            "expected_correct"
        ).sum()

        if len(assessable) > 0:

            accuracy = (
                correct /
                len(assessable) *
                100
            )

        else:
            accuracy = np.nan

        relevant = assessable[
            assessable["case_label"] ==
            "relevant_conflict"
        ]

        detected = (
            relevant[
                "conflict_detected"
            ] ==
            "yes"
        ).sum()

        relevant_correct = (
            relevant[
                "behavior_label"
            ] ==
            "expected_correct"
        ).sum()

        if len(relevant) > 0:

            relevant_accuracy = (
                relevant_correct /
                len(relevant) *
                100
            )

            detection_recall = (
                detected /
                len(relevant) *
                100
            )

        else:

            relevant_accuracy = np.nan
            detection_recall = np.nan

        qwen_route_rows.append({
            "prompt": prompt,
            "route": route,
            "total": total,
            "ok": ok,
            "failures":
                total - ok,
            "assessable_percent":
                ok / total * 100
                if total > 0
                else np.nan,
            "assessed":
                len(assessable),
            "correct":
                correct,
            "accuracy":
                accuracy,
            "relevant_n":
                len(relevant),
            "relevant_correct":
                relevant_correct,
            "relevant_accuracy":
                relevant_accuracy,
            "relevant_detected":
                detected,
            "detection_recall":
                detection_recall
        })


qwen_route_df = pd.DataFrame(
    qwen_route_rows
)

qwen_route_df.to_csv(
    "results/extra/qwen_route_results.csv",
    index=False
)


# BORDERLINE SENSITIVITY FILE

borderline = pd.read_csv(
    os.path.join(
        extra_folder,
        "borderline_both_values.csv"
    )
)

borderline.to_csv(
    "results/extra/borderline_cases_for_sensitivity.csv",
    index=False
)


# FP SUMMARY

fp_rows = []

for model, data in models.items():

    for prompt in prompts:

        part = data[
            (data["prompt_type"] == prompt) &
            (data["run_status"] == "ok")
        ]

        (
            tp, fp, fn, tn,
            precision, recall, f1
        ) = conflict_metrics(part)

        fp_rows.append({
            "model": model,
            "prompt": prompt,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision":
                percent(precision),
            "detection_recall":
                percent(recall),
            "f1":
                percent(f1)
        })


fp_df = pd.DataFrame(fp_rows)

fp_df.to_csv(
    "results/extra/conflict_detection_summary.csv",
    index=False
)


# PRINT RESULTS

print("\nTECHNICAL RESULTS")
print(
    technical_df
    .round(2)
    .to_string(index=False)
)


print("\nCOMMON SET RESULTS")
print(
    common_df
    .round(2)
    .to_string(index=False)
)


print("\nCOMMON SET BY CONDITION")
print(
    condition_df
    .round(2)
    .to_string(index=False)
)


print("\nPROMPT-SPECIFIC COMMON RESULTS")
print(
    prompt_common_df
    .round(2)
    .to_string(index=False)
)


print("\nRELEVANT CONFLICT RESULTS")
print(
    relevant_df
    .round(2)
    .to_string(index=False)
)


print("\nPOOLED MCNEMAR RESULTS")
print(
    mcnemar_df
    .round(4)
    .to_string(index=False)
)


print("\nLABEL OVERRIDES")
print(
    overrides_df
    .to_string(index=False)
)


print("\nBEFORE / AFTER OVERRIDES")
print(
    before_after_df
    .round(2)
    .to_string(index=False)
)


print("\nMCNEMAR - POOLED AND RELEVANT ONLY")
print(
    mcnemar_extra_df
    .round(4)
    .to_string(index=False)
)


print("\nWORST-CASE SENSITIVITY")
print(
    worst_case_df
    .round(2)
    .to_string(index=False)
)


print("\nDETECTED BUT STILL INCORRECT")
print(
    detected_incorrect_df
    .round(2)
    .to_string(index=False)
)


print("\nMISSED CONFLICT SOURCE CHOICE")
print(
    position_df
    .round(2)
    .to_string(index=False)
)


print("\nQWEN ROUTE RESULTS")
print(
    qwen_route_df
    .round(2)
    .to_string(index=False)
)


print("\nFP SUMMARY")
print(
    fp_df
    .round(2)
    .to_string(index=False)
)


print(
    "\nExtra results saved in results/extra/"
)
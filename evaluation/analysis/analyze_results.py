import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# PATHS

results_dir = Path("results")
extra_dir = results_dir / "extra"
figures_dir = results_dir / "figures"

figures_dir.mkdir(exist_ok=True)


# LOAD DATA

common_conditions = pd.read_csv(
    results_dir / "common_by_condition.csv"
)

confidence_intervals = pd.read_csv(
    extra_dir / "confidence_intervals.csv"
)

source_choice = pd.read_csv(
    extra_dir / "task7_source_choice_summary.csv"
)


# SETTINGS

model_order = [
    "Qwen 7B",
    "Llama 8B",
    "Granite 8B",
    "Llama 70B"
]

prompt_order = [
    "baseline",
    "explained"
]


# FIGURE 1
# RELEVANT-CONFLICT EXPECTED-CORRECT RATE
# STRICT COMMON SET + 95% CLUSTER BOOTSTRAP CI

performance = common_conditions[
    common_conditions["condition"] ==
    "relevant_conflict"
].copy()


ci = confidence_intervals[
    (
        confidence_intervals["analysis"] ==
        "strict_common"
    ) &
    (
        confidence_intervals["condition"] ==
        "relevant_conflict"
    )
].copy()


figure1 = performance.merge(
    ci[
        [
            "model",
            "prompt",
            "bootstrap_low_95",
            "bootstrap_high_95"
        ]
    ],
    on=["model", "prompt"],
    how="left"
)


figure1["model"] = pd.Categorical(
    figure1["model"],
    categories=model_order,
    ordered=True
)

figure1 = figure1.sort_values(
    ["model", "prompt"]
)


baseline = (
    figure1[
        figure1["prompt"] == "baseline"
    ]
    .set_index("model")
    .reindex(model_order)
)

explained = (
    figure1[
        figure1["prompt"] == "explained"
    ]
    .set_index("model")
    .reindex(model_order)
)


x = np.arange(len(model_order))
width = 0.34


fig, ax = plt.subplots(
    figsize=(3.3, 2.7)
)


baseline_values = (
    baseline["accuracy"]
    .to_numpy()
)

explained_values = (
    explained["accuracy"]
    .to_numpy()
)


baseline_errors = np.vstack(
    [
        baseline_values -
        baseline["bootstrap_low_95"].to_numpy(),

        baseline["bootstrap_high_95"].to_numpy() -
        baseline_values
    ]
)


explained_errors = np.vstack(
    [
        explained_values -
        explained["bootstrap_low_95"].to_numpy(),

        explained["bootstrap_high_95"].to_numpy() -
        explained_values
    ]
)


bars1 = ax.bar(
    x - width / 2,
    baseline_values,
    width,
    label="Baseline",
    yerr=baseline_errors,
    capsize=2.5,
    color="white",
    edgecolor="black",
    linewidth=0.8,
    hatch="//"
)


bars2 = ax.bar(
    x + width / 2,
    explained_values,
    width,
    label="Explained",
    yerr=explained_errors,
    capsize=2.5,
    color="0.65",
    edgecolor="black",
    linewidth=0.8,
    hatch=".."
)


ax.set_ylabel(
    "Expected-correct rate (%)",
    fontsize=8
)

ax.set_xlabel(
    "Model",
    fontsize=8
)

ax.set_xticks(x)

ax.set_xticklabels(
    [
        "Qwen\n7B",
        "Llama\n8B",
        "Granite\n8B",
        "Llama\n70B"
    ],
    fontsize=7
)

ax.tick_params(
    axis="y",
    labelsize=7
)

ax.set_ylim(
    0,
    105
)

ax.legend(
    fontsize=7,
    frameon=False,
    ncol=2,
    loc="upper left"
)


ax.bar_label(
    bars1,
    fmt="%.1f",
    padding=2,
    fontsize=6
)

ax.bar_label(
    bars2,
    fmt="%.1f",
    padding=2,
    fontsize=6
)


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


fig.tight_layout()

fig.savefig(
    figures_dir /
    "relevant_conflict_strict_common.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.close(fig)


# FIGURE 2
# MISSED RELEVANT CONFLICT:
# SOURCE A / SOURCE B / BOTH
# 100% STACKED HORIZONTAL BAR

source_choice = source_choice.copy()


# Combine both categories into one "Both" category

source_choice["choice_group"] = (
    source_choice["final_choice"]
    .replace(
        {
            "Source A": "A",
            "Source B": "B",
            "A": "A",
            "B": "B",
            "both_attributed": "Both",
            "both_unattributed": "Both"
        }
    )
)


# The requested plot is A / B / both.
# Exclude neither / unclear from the displayed percentages.

source_choice = source_choice[
    source_choice["choice_group"].isin(
        ["A", "B", "Both"]
    )
]


source_summary = (
    source_choice
    .groupby(
        [
            "model",
            "prompt_type",
            "choice_group"
        ],
        as_index=False
    )["count"]
    .sum()
)


source_table = source_summary.pivot_table(
    index=[
        "model",
        "prompt_type"
    ],
    columns="choice_group",
    values="count",
    fill_value=0
)


for column in ["A", "B", "Both"]:

    if column not in source_table.columns:
        source_table[column] = 0


source_table = source_table[
    ["A", "B", "Both"]
]


source_percent = (
    source_table
    .div(
        source_table.sum(axis=1),
        axis=0
    )
    * 100
)


rows = []

for model in model_order:

    for prompt in prompt_order:

        key = (
            model,
            prompt
        )

        if key in source_percent.index:

            rows.append({
                "label":
                    f"{model} - "
                    f"{prompt.capitalize()}",
                "A":
                    source_percent.loc[
                        key,
                        "A"
                    ],
                "B":
                    source_percent.loc[
                        key,
                        "B"
                    ],
                "Both":
                    source_percent.loc[
                        key,
                        "Both"
                    ]
            })


plot_data = pd.DataFrame(rows)


fig, ax = plt.subplots(
    figsize=(3.3, 3.2)
)


y = np.arange(
    len(plot_data)
)


left = np.zeros(
    len(plot_data)
)


styles = {
    "A": {
        "color": "white",
        "hatch": "//"
    },
    "B": {
        "color": "0.60",
        "hatch": ".."
    },
    "Both": {
        "color": "0.85",
        "hatch": "xx"
    }
}


for category in [
    "A",
    "B",
    "Both"
]:

    values = (
        plot_data[category]
        .to_numpy()
    )

    ax.barh(
        y,
        values,
        left=left,
        label=category,
        color=styles[category]["color"],
        edgecolor="black",
        linewidth=0.6,
        hatch=styles[category]["hatch"]
    )

    for i, value in enumerate(values):

        if value >= 8:

            ax.text(
                left[i] + value / 2,
                i,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=6
            )

    left += values


ax.set_yticks(y)

ax.set_yticklabels(
    plot_data["label"],
    fontsize=6.5
)

ax.set_xlabel(
    "Share of missed relevant conflicts (%)",
    fontsize=8
)

ax.tick_params(
    axis="x",
    labelsize=7
)

ax.set_xlim(
    0,
    100
)

ax.invert_yaxis()


ax.legend(
    title="Answer followed",
    fontsize=6.5,
    title_fontsize=6.5,
    frameon=False,
    ncol=3,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01)
)


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


fig.tight_layout()

fig.savefig(
    figures_dir /
    "missed_conflict_source_choice.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.close(fig)



# PRINT TABLES FOR REPORT

print(
    "\nFIGURE 1 DATA:"
)

print(
    figure1[
        [
            "model",
            "prompt",
            "n",
            "correct",
            "accuracy",
            "bootstrap_low_95",
            "bootstrap_high_95"
        ]
    ]
    .round(2)
    .to_string(index=False)
)


print(
    "\nFIGURE 2 DATA:"
)

print(
    plot_data
    .round(2)
    .to_string(index=False)
)


print(
    "\nSaved:"
)

print(
    figures_dir /
    "relevant_conflict_strict_common.pdf"
)

print(
    figures_dir /
    "missed_conflict_source_choice.pdf"
)
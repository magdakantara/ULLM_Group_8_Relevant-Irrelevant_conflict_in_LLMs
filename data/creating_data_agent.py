# ==========================================================
# CONTROLLED CONTRACT-CONFLICT DATASET GENERATOR
# Creates 6 cases per accepted contract:
# B0/Q1, B0/Q2, B1/Q1, B1/Q2, B2/Q1, B2/Q2
# ==========================================================

import json
import re
from pathlib import Path
from collections import Counter

import pandas as pd
from openai import OpenAI


# 0. SETTINGS

client = OpenAI(api_key="here we put out api key")

CUAD_FILE = Path(
    r"C:\Users\mgdka\PycharmProjects\PythonProject\data\CUADv1.json"
)

OUTPUT_FILE = Path(
)

MODEL_NAME = "gpt-5"

# we split 10 contract examples per group member for easier handling when validatiing
TARGET_CONTRACTS_PER_OWNER = {
    "Magda": 10,
    "Berkay": 10,
    "Engin": 10,
    "Oguz": 17, # agreed to review more contracts that migth get droped later
    "Dilek": 10,
}
# names split so each person reviews a part of the data for the evlauation part
REVIEWERS = {
    "Magda": "Berkay",
    "Berkay": "Engin",
    "Engin": "Oguz",
    "Oguz": "Dilek",
    "Dilek": "Magda",
}
# specific topics the agent will chose from
SHORTLIST = {
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Termination For Convenience",
    "Cap On Liability",
    "Warranty Duration",
}


# 1. DEFINE EACH PERSON'S CONTRACT POOL

ASSIGNMENTS = {
    "Magda": {
        "Affiliate Agreement": re.compile(r"\bAFFILIATE AGREEMENT\b", re.I),
        "Endorsement Agreement": re.compile(r"\bENDORSEMENT AGREEMENT\b", re.I),
        "Joint Venture Agreement": re.compile(
            r"\bJOINT VENTURE (?:AGREEMENT|CONTRACT)\b", re.I
        ),
        "Strategic Alliance Agreement": re.compile(
            r"\bSTRATEGIC ALLIANCE AGREEMENT\b", re.I
        ),
        "Transportation Agreement": re.compile(
            r"\bTRANSPORTATION(?: SERVICES?)? (?:AGREEMENT|CONTRACT)\b", re.I
        ),
    },

    "Berkay": {
        "Franchise Agreement": re.compile(r"\bFRANCHISE AGREEMENT\b", re.I),
        "Maintenance Agreement": re.compile(r"\bMAINTENANCE AGREEMENT\b", re.I),
        "Manufacturing Agreement": re.compile(r"\bMANUFACTURING AGREEMENT\b", re.I),
        "Outsourcing Agreement": re.compile(r"\bOUTSOURCING AGREEMENT\b", re.I),
        "Supply Agreement": re.compile(r"\bSUPPLY AGREEMENT\b", re.I),
    },

    "Engin": {
        "Development Agreement": re.compile(r"\bDEVELOPMENT AGREEMENT\b", re.I),
        "Hosting Agreement": re.compile(r"\bHOSTING AGREEMENT\b", re.I),
        "IP Agreement": re.compile(
            r"\b(?:INTELLECTUAL PROPERTY|IP) AGREEMENT\b", re.I
        ),
        "License Agreement": re.compile(r"\bLICENSE AGREEMENT\b", re.I),
        "Non-Compete/No-Solicit/Non-Disparagement Agreement": re.compile(
            r"\b(?:NON[- ]COMPETE|NON[- ]SOLICIT|NON[- ]DISPARAGEMENT)", re.I
        ),
    },

    "Oguz": {
        "Agency Agreement": re.compile(r"\bAGENCY AGREEMENT\b", re.I),
        "Distributor Agreement": re.compile(r"\bDISTRIBUTOR AGREEMENT\b", re.I),
        "Marketing Agreement": re.compile(r"\bMARKETING AGREEMENT\b", re.I),
        "Reseller Agreement": re.compile(r"\bRESELLER AGREEMENT\b", re.I),
        "Service Agreement": re.compile(r"\bSERVICE(?:S)? AGREEMENT\b", re.I),
    },

    "Dilek": {
        "Collaboration/Cooperation Agreement": re.compile(
            r"\b(?:COLLABORATION|COOPERATION) AGREEMENT\b", re.I
        ),
        "Co-Branding Agreement": re.compile(
            r"\bCO[- ]BRANDING AGREEMENT\b", re.I
        ),
        "Consulting Agreement": re.compile(r"\bCONSULTING AGREEMENT\b", re.I),
        "Promotion Agreement": re.compile(r"\bPROMOTION AGREEMENT\b", re.I),
        "Sponsorship Agreement": re.compile(r"\bSPONSORSHIP AGREEMENT\b", re.I),
    },
}


# 2. LOAD CUAD

with CUAD_FILE.open("r", encoding="utf-8") as file:
    full_data = json.load(file)

contracts = full_data["data"]

print("Total CUAD contracts:", len(contracts))


# 3. HELPERS

def get_clause_category(question):
    """Extract CUAD's labelled clause category from a QA question."""
    match = re.search(r'related to "([^"]+)"', question)
    return match.group(1) if match else None


def get_contract_type(title, patterns):
    """Return the assigned contract type based on its CUAD title."""
    return next(
        (name for name, pattern in patterns.items() if pattern.search(title)),
        None
    )


def make_example_id(owner, contract_number, version, question_id):
    """Create globally unique IDs."""
    return f"{owner.upper()}-{contract_number:02d}-{version}-{question_id}"


def nonempty(value):
    return isinstance(value, str) and value.strip()


# 4. FIND CANDIDATE CONTRACTS FOR EACH PERSON

candidates_by_owner = {}

for owner, type_patterns in ASSIGNMENTS.items():

    candidate_rows = []

    for contract in contracts:
        title = contract["title"]

        contract_type = get_contract_type(title, type_patterns)

        if contract_type is None:
            continue

        usable_categories = []

        for paragraph in contract["paragraphs"]:
            for qa in paragraph["qas"]:
                category = get_clause_category(qa["question"])

                if (
                    category in SHORTLIST
                    and not qa["is_impossible"]
                    and qa["answers"]
                    and qa["answers"][0]["text"].strip()
                ):
                    usable_categories.append(category)

        distinct_categories = sorted(set(usable_categories))

        full_context = "\n\n".join(
            paragraph["context"] for paragraph in contract["paragraphs"]
        )

        candidate_rows.append({
            "original_contract_id": title,
            "contract_type": contract_type,
            "context": full_context,
            "usable_clause_categories": distinct_categories,
            "number_of_usable_categories": len(distinct_categories),
            "candidate_for_final_set": len(distinct_categories) >= 2,
        })

    candidates_df = pd.DataFrame(candidate_rows).sort_values(
        [
            "candidate_for_final_set",
            "number_of_usable_categories",
            "contract_type",
        ],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    candidates_df.to_csv(
        f"{owner.lower()}_contract_candidates.csv",
        index=False,
    )

    good_contracts = candidates_df.loc[
        candidates_df["candidate_for_final_set"]
    ].to_dict("records")

    candidates_by_owner[owner] = good_contracts

    print(
        f"{owner}: found {len(candidates_df)} contracts; "
        f"{len(good_contracts)} have 2+ usable categories."
    )


# 5. REQUIRE STRUCTURED AGENT OUTPUT

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "clause_category": {
            "type": "string",
            "enum": sorted(SHORTLIST),
        },
        "question": {"type": "string"},
        "answer_source_a": {"type": "string"},
        "answer_source_b": {"type": "string"},
    },
    "required": [
        "clause_category",
        "question",
        "answer_source_a",
        "answer_source_b",
    ],
    "additionalProperties": False,
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "skip_contract": {"type": "boolean"},
        "skip_reason": {"type": ["string", "null"]},

        "question_q1": {
            "anyOf": [QUESTION_SCHEMA, {"type": "null"}]
        },
        "question_q2": {
            "anyOf": [QUESTION_SCHEMA, {"type": "null"}]
        },

        "source_b1": {"type": ["string", "null"]},
        "source_b2": {"type": ["string", "null"]},

        "modification_summary_b1": {"type": ["string", "null"]},
        "modification_summary_b2": {"type": ["string", "null"]},
    },
    "required": [
        "skip_contract",
        "skip_reason",
        "question_q1",
        "question_q2",
        "source_b1",
        "source_b2",
        "modification_summary_b1",
        "modification_summary_b2",
    ],
    "additionalProperties": False,
}


# 6. AGENT INSTRUCTIONS

INSTRUCTIONS = """
You construct controlled two-source legal question-answering examples.

You receive one original contract, Source A.

Use only these clause categories:
- Effective Date
- Expiration Date
- Renewal Term
- Notice Period To Terminate Renewal
- Governing Law
- Termination For Convenience
- Cap On Liability
- Warranty Duration

Choose exactly two independent questions:
- Q1 and Q2 must use two different allowed clause categories.
- Both questions must be explicitly answerable from Source A.
- For each question, return the exact clause category used.

Create two complete versions of Source B:
- B1 changes only the information answering Q1.
- B2 changes only the information answering Q2.

Requirements:
- B1 must change Q1's answer but leave Q2's answer unchanged.
- B2 must change Q2's answer but leave Q1's answer unchanged.
- Make the minimum legal edit needed.
- Preserve all unrelated text exactly.
- If the changed information appears repeatedly in a way that would create
  an internal contradiction, update only the truly repeated reference needed
  for consistency.
- Do not add headers, labels, explanations, contract IDs, or text outside
  the original contract to Source B.
- Do not call either source newer, authoritative, or preferred.
- Answers must be concise and directly supported by the source text.

Skip the contract if you cannot safely create two independent questions and
two controlled edits.

If skipped:
- set skip_contract to true;
- explain why in skip_reason;
- return null for all other fields.

If suitable:
- set skip_contract to false;
- set skip_reason to null;
- return Q1, Q2, complete B1 and B2 texts, and modification summaries.

Return only valid JSON matching the required schema.
"""


# 7. LOAD A CHECKPOINT, IF ONE EXISTS

if OUTPUT_FILE.exists():
    with OUTPUT_FILE.open("r", encoding="utf-8") as file:
        final_rows = json.load(file)

    print(f"Loaded checkpoint with {len(final_rows)} examples.")
else:
    final_rows = []


# 8. GENERATE DATA FOR EACH PERSON
for owner, candidate_contracts in candidates_by_owner.items():

    target = TARGET_CONTRACTS_PER_OWNER[owner]

    existing_contract_ids = {
        row["original_contract_id"]
        for row in final_rows
        if row["owner"] == owner
    }

    accepted_contracts = len(existing_contract_ids)

    print(
        f"\n--- {owner}: {accepted_contracts}/{target} "
        f"contracts already accepted ---"
    )

    if accepted_contracts >= target:
        continue

    for contract in candidate_contracts:

        if accepted_contracts >= target:
            break

        # Avoid regenerating an accepted contract when resuming.
        if contract["original_contract_id"] in existing_contract_ids:
            continue

        response = client.responses.create(
            model=MODEL_NAME,
            instructions=INSTRUCTIONS,
            input=f"""
CONTRACT ID:
{contract["original_contract_id"]}

CONTRACT TYPE:
{contract["contract_type"]}

AVAILABLE ANNOTATED CATEGORIES:
{", ".join(contract["usable_clause_categories"])}

SOURCE A — ORIGINAL CONTRACT:
{contract["context"]}
""",
            text={
                "format": {
                    "type": "json_schema",
                    "name": "controlled_contract_examples",
                    "strict": True,
                    "schema": OUTPUT_SCHEMA,
                }
            },
            store=False,
        )

        result = json.loads(response.output_text)

        if result["skip_contract"]:
            print(
                f"Skipped {owner}: {contract['original_contract_id']}\n"
                f"Reason: {result['skip_reason']}"
            )
            continue

        q1 = result["question_q1"]
        q2 = result["question_q2"]

        source_a = contract["context"]
        source_b1 = result["source_b1"]
        source_b2 = result["source_b2"]

        # Basic structural validation before saving.
        valid = (
            nonempty(source_b1)
            and nonempty(source_b2)
            and source_b1 != source_a
            and source_b2 != source_a
            and q1["clause_category"] != q2["clause_category"]
            and q1["answer_source_a"].strip() != q1["answer_source_b"].strip()
            and q2["answer_source_a"].strip() != q2["answer_source_b"].strip()
        )

        if not valid:
            print(
                f"Rejected invalid generated result for "
                f"{contract['original_contract_id']}"
            )
            continue

        contract_number = accepted_contracts + 1

        common = {
            "owner": owner,
            "original_contract_id": contract["original_contract_id"],
            "contract_type": contract["contract_type"],
            "reviewer": REVIEWERS[owner],
            "review_status": "pending",
            "source_A_path": None,
            "source_B_path": None,
        }

        six_rows = [

            # B0: agreement for both questions
            {
                **common,
                "example_id": make_example_id(
                    owner, contract_number, "B0", "Q1"
                ),
                "clause_category": q1["clause_category"],
                "question": q1["question"],
                "source_A": source_a,
                "source_B": source_a,
                "exactly_what_changed": "No change: Source B is an exact copy of Source A.",
                "case_label": "agreement",
                "answer_supported_by_A": q1["answer_source_a"],
                "answer_supported_by_B": q1["answer_source_a"],
                "expected_behaviour": "Answer correctly.",
            },
            {
                **common,
                "example_id": make_example_id(
                    owner, contract_number, "B0", "Q2"
                ),
                "clause_category": q2["clause_category"],
                "question": q2["question"],
                "source_A": source_a,
                "source_B": source_a,
                "exactly_what_changed": "No change: Source B is an exact copy of Source A.",
                "case_label": "agreement",
                "answer_supported_by_A": q2["answer_source_a"],
                "answer_supported_by_B": q2["answer_source_a"],
                "expected_behaviour": "Answer correctly.",
            },

            # B1: changes Q1 only
            {
                **common,
                "example_id": make_example_id(
                    owner, contract_number, "B1", "Q1"
                ),
                "clause_category": q1["clause_category"],
                "question": q1["question"],
                "source_A": source_a,
                "source_B": source_b1,
                "exactly_what_changed": result["modification_summary_b1"],
                "case_label": "relevant_conflict",
                "answer_supported_by_A": q1["answer_source_a"],
                "answer_supported_by_B": q1["answer_source_b"],
                "expected_behaviour": (
                    "Identify the disagreement; do not arbitrarily select one answer."
                ),
            },
            {
                **common,
                "example_id": make_example_id(
                    owner, contract_number, "B1", "Q2"
                ),
                "clause_category": q2["clause_category"],
                "question": q2["question"],
                "source_A": source_a,
                "source_B": source_b1,
                "exactly_what_changed": result["modification_summary_b1"],
                "case_label": "irrelevant_conflict",
                "answer_supported_by_A": q2["answer_source_a"],
                "answer_supported_by_B": q2["answer_source_a"],
                "expected_behaviour": (
                    "Answer correctly despite the unrelated conflict."
                ),
            },

            # B2: changes Q2 only
            {
                **common,
                "example_id": make_example_id(
                    owner, contract_number, "B2", "Q1"
                ),
                "clause_category": q1["clause_category"],
                "question": q1["question"],
                "source_A": source_a,
                "source_B": source_b2,
                "exactly_what_changed": result["modification_summary_b2"],
                "case_label": "irrelevant_conflict",
                "answer_supported_by_A": q1["answer_source_a"],
                "answer_supported_by_B": q1["answer_source_a"],
                "expected_behaviour": (
                    "Answer correctly despite the unrelated conflict."
                ),
            },
            {
                **common,
                "example_id": make_example_id(
                    owner, contract_number, "B2", "Q2"
                ),
                "clause_category": q2["clause_category"],
                "question": q2["question"],
                "source_A": source_a,
                "source_B": source_b2,
                "exactly_what_changed": result["modification_summary_b2"],
                "case_label": "relevant_conflict",
                "answer_supported_by_A": q2["answer_source_a"],
                "answer_supported_by_B": q2["answer_source_b"],
                "expected_behaviour": (
                    "Identify the disagreement; do not arbitrarily select one answer."
                ),
            },
        ]

        final_rows.extend(six_rows)
        accepted_contracts += 1
        existing_contract_ids.add(contract["original_contract_id"])

        # Save after every accepted contract.
        with OUTPUT_FILE.open("w", encoding="utf-8") as file:
            json.dump(final_rows, file, ensure_ascii=False, indent=2)

        print(
            f"Accepted {owner}: {accepted_contracts}/{target} — "
            f"{contract['original_contract_id']}"
        )


# 9. FINAL VALIDATION

print("\n================ FINAL CHECK ================\n")
print("Total examples:", len(final_rows))

print("\nExamples per owner:")
print(Counter(row["owner"] for row in final_rows))

print("\nExamples per case:")
print(Counter(row["case_label"] for row in final_rows))

print("\nEmpty Source A:")
print(sum(not row["source_A"].strip() for row in final_rows))

print("\nEmpty Source B:")
print(sum(not row["source_B"].strip() for row in final_rows))

all_ids = [row["example_id"] for row in final_rows]
duplicate_ids = [
    example_id
    for example_id, count in Counter(all_ids).items()
    if count > 1
]

print("\nDuplicate IDs:", duplicate_ids)

assert not duplicate_ids, "Duplicate example IDs found."
assert all(row["source_A"].strip() for row in final_rows)
assert all(row["source_B"].strip() for row in final_rows)

print("\nSaved final dataset to:")
print(OUTPUT_FILE)

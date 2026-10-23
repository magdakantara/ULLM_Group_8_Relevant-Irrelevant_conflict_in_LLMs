# Contract-version generator (one contributor's ten contracts)

Builds the A / B0 / B1 / B2 contract versions and the 60 examples for ten CUAD contracts
(Franchise, Maintenance, Manufacturing, Outsourcing and Supply agreements; two of each).
These examples are part of the final 300-example dataset.

The questions, answers and edits are fixed in [`curation.json`](curation.json). The script
applies each edit at an exact character offset of the original CUAD text and checks that:

- B0 is byte-for-byte identical to A
- B1 and B2 differ from A only at the intended span
- each recorded answer and its evidence span exist in the source text
- every contract yields two agreement, two relevant-conflict and two irrelevant-conflict cases

It produces the full text files, a JSONL file and an Excel sheet in the shared team format.
No network access or model inference happens at build time.

```sh
# 43 synthetic positive and negative tests, no CUAD file needed
python3 -I -B build_berkay_dataset.py --self-test

# build the package from CUADv1.json
python3 -I -B build_berkay_dataset.py --source CUADv1.json --curation curation.json \
    --output OUTPUT_DIR --zip OUTPUT.zip
```

Tested with Python 3.13; requires `openpyxl`.

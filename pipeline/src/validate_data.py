"""Validate the shared JSON without loading a model or changing the dataset."""
import argparse
import json
from common import DEFAULT_DATASET, load_dataset, validate_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(DEFAULT_DATASET))
    args = parser.parse_args()
    print(json.dumps(validate_dataset(load_dataset(args.data)), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

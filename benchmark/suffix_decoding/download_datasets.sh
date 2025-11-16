#!/bin/bash
# Download datasets for suffix decoding benchmarks
# This script downloads Specbench and Blazedit datasets

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"

echo "Creating data directory: ${DATA_DIR}"
mkdir -p "${DATA_DIR}"

# Download Specbench dataset
echo "Downloading Specbench dataset..."
SPECBENCH_URL="https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl"
curl -L "${SPECBENCH_URL}" -o "${DATA_DIR}/specbench_question.jsonl"
echo "Specbench dataset downloaded to ${DATA_DIR}/specbench_question.jsonl"

# Download Blazedit dataset using Python (HuggingFace datasets library)
echo "Downloading Blazedit dataset..."
python3 << 'EOF'
import os
import json
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: Please install datasets library: pip install datasets")
    exit(1)

script_dir = Path(__file__).parent.absolute() if hasattr(Path(__file__), 'parent') else Path.cwd()
data_dir = script_dir / "data"
data_dir.mkdir(exist_ok=True)

# Download 5k char version
print("Loading Blazedit (5k char version)...")
dataset = load_dataset("vdaita/edit_5k_char", split="train")

# Convert to JSONL format compatible with our benchmark
blazedit_file = data_dir / "blazedit_5k.jsonl"
with open(blazedit_file, "w", encoding="utf-8") as f:
    for idx, item in enumerate(dataset):
        # Extract prompt from the item
        # The dataset has 'input' and 'output' fields
        prompt = item.get('input', '') or item.get('prompt', '') or str(item)
        record = {
            "question_id": idx,
            "category": "code_editing",
            "turns": [prompt],
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Blazedit dataset downloaded to {blazedit_file}")

# Also download 10k char version
print("Loading Blazedit (10k char version)...")
dataset_10k = load_dataset("vdaita/edit_10k_char", split="train")

blazedit_10k_file = data_dir / "blazedit_10k.jsonl"
with open(blazedit_10k_file, "w", encoding="utf-8") as f:
    for idx, item in enumerate(dataset_10k):
        prompt = item.get('input', '') or item.get('prompt', '') or str(item)
        record = {
            "question_id": idx,
            "category": "code_editing",
            "turns": [prompt],
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Blazedit 10k dataset downloaded to {blazedit_10k_file}")
EOF

echo ""
echo "✓ All datasets downloaded successfully!"
echo "  - Specbench: ${DATA_DIR}/specbench_question.jsonl"
echo "  - Blazedit (5k): ${DATA_DIR}/blazedit_5k.jsonl"
echo "  - Blazedit (10k): ${DATA_DIR}/blazedit_10k.jsonl"

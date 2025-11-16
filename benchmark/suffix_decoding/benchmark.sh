#!/bin/bash
# Comprehensive benchmark script for suffix decoding in SGLang
# This script replicates the vLLM PR #25784 benchmark methodology
#
# Usage:
#   ./benchmark.sh [MODEL_NAME] [PORT]
#
# Example:
#   ./benchmark.sh meta-llama/Llama-3.1-8B-Instruct 30000

set -e

# Configuration
MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${2:-30000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
RESULTS_DIR="${SCRIPT_DIR}/results"
SERVER_URL="http://127.0.0.1:${PORT}"

# Benchmark parameters matching vLLM PR
SPEC_LENS=(5 12 32)
CONCURRENCIES=(1 4 16 64)
SPECBENCH_MAX_TOKENS=256
BLAZEDIT_MAX_TOKENS=1024

echo "=========================================="
echo "SGLang Suffix Decoding Benchmark"
echo "=========================================="
echo "Model: ${MODEL}"
echo "Port: ${PORT}"
echo "Results directory: ${RESULTS_DIR}"
echo ""

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Check if datasets exist
if [ ! -f "${DATA_DIR}/specbench_question.jsonl" ]; then
    echo "ERROR: Specbench dataset not found. Please run ./download_datasets.sh first"
    exit 1
fi

if [ ! -f "${DATA_DIR}/blazedit_5k.jsonl" ]; then
    echo "ERROR: Blazedit dataset not found. Please run ./download_datasets.sh first"
    exit 1
fi

# Function to wait for server to be ready
wait_for_server() {
    echo "Waiting for server at ${SERVER_URL} to be ready..."
    for i in {1..60}; do
        if curl -s "${SERVER_URL}/health" > /dev/null 2>&1 || \
           curl -s "${SERVER_URL}/get_server_info" > /dev/null 2>&1; then
            echo "Server is ready!"
            sleep 5  # Give it a few more seconds to fully initialize
            return 0
        fi
        echo "  Attempt $i/60: Server not ready yet, waiting..."
        sleep 2
    done
    echo "ERROR: Server did not become ready in time"
    return 1
}

# Function to kill server
kill_server() {
    echo "Stopping any existing SGLang servers on port ${PORT}..."
    pkill -f "python.*sglang.*launch_server.*port.*${PORT}" || true
    pkill -f "python.*sglang.launch_server.*${PORT}" || true
    sleep 3
}

# Function to run benchmark for a specific configuration
run_benchmark() {
    local method=$1
    local spec_len=$2
    local dataset=$3
    local max_tokens=$4
    local extra_args=$5
    local result_file=$6

    echo ""
    echo "Running: method=${method}, spec_len=${spec_len}, dataset=${dataset}, max_tokens=${max_tokens}"

    python3 "${SCRIPT_DIR}/../../scripts/specbench_client.py" \
        --dataset "${dataset}" \
        --server "${SERVER_URL}" \
        --max-new-tokens "${max_tokens}" \
        --concurrencies "${CONCURRENCIES[@]}" \
        --method "${method}" \
        --spec-len "${spec_len}" \
        --output-table "${result_file}"

    echo "Results saved to ${result_file}"
}

# Function to launch server with specific configuration
launch_server() {
    local method=$1
    local spec_len=$2
    local enable_cache=$3

    kill_server

    echo ""
    echo "=========================================="
    echo "Launching server: ${method} (spec_len=${spec_len}, cache=${enable_cache})"
    echo "=========================================="

    local cmd="python -m sglang.launch_server --model ${MODEL} --port ${PORT}"

    case "${method}" in
        "suffix_w_cache")
            cmd="${cmd} --speculative-algorithm SUFFIX"
            cmd="${cmd} --speculative-suffix-max-tree-depth ${spec_len}"
            # Enable prefix caching by default
            ;;
        "suffix_wo_cache")
            cmd="${cmd} --speculative-algorithm SUFFIX"
            cmd="${cmd} --speculative-suffix-max-tree-depth ${spec_len}"
            cmd="${cmd} --disable-radix-cache"
            ;;
        "ngram_5_5")
            cmd="${cmd} --speculative-algorithm NGRAM"
            cmd="${cmd} --speculative-ngram-min-match-window-size 5"
            cmd="${cmd} --speculative-ngram-max-match-window-size 5"
            cmd="${cmd} --speculative-num-draft-tokens ${spec_len}"
            ;;
        "ngram_3_5")
            cmd="${cmd} --speculative-algorithm NGRAM"
            cmd="${cmd} --speculative-ngram-min-match-window-size 3"
            cmd="${cmd} --speculative-ngram-max-match-window-size 5"
            cmd="${cmd} --speculative-num-draft-tokens ${spec_len}"
            ;;
        *)
            echo "ERROR: Unknown method ${method}"
            exit 1
            ;;
    esac

    echo "Command: ${cmd}"

    # Launch server in background
    nohup ${cmd} > "${RESULTS_DIR}/server_${method}_${spec_len}.log" 2>&1 &
    local server_pid=$!
    echo "Server PID: ${server_pid}"

    # Wait for server to be ready
    if ! wait_for_server; then
        echo "ERROR: Server failed to start. Check logs at ${RESULTS_DIR}/server_${method}_${spec_len}.log"
        kill_server
        exit 1
    fi
}

echo ""
echo "=========================================="
echo "BENCHMARK 1: SPECBENCH (max_tokens=${SPECBENCH_MAX_TOKENS})"
echo "=========================================="

# Specbench - Suffix with cache
for spec_len in "${SPEC_LENS[@]}"; do
    launch_server "suffix_w_cache" "${spec_len}" "true"
    run_benchmark \
        "suffix (w/ cache)" \
        "${spec_len}" \
        "${DATA_DIR}/specbench_question.jsonl" \
        "${SPECBENCH_MAX_TOKENS}" \
        "" \
        "${RESULTS_DIR}/specbench_suffix_cache_${spec_len}.csv"
done

# Specbench - Suffix without cache
for spec_len in "${SPEC_LENS[@]}"; do
    launch_server "suffix_wo_cache" "${spec_len}" "false"
    run_benchmark \
        "suffix (w/o cache)" \
        "${spec_len}" \
        "${DATA_DIR}/specbench_question.jsonl" \
        "${SPECBENCH_MAX_TOKENS}" \
        "" \
        "${RESULTS_DIR}/specbench_suffix_nocache_${spec_len}.csv"
done

# Specbench - NGRAM [5, 5]
for spec_len in "${SPEC_LENS[@]}"; do
    launch_server "ngram_5_5" "${spec_len}" "true"
    run_benchmark \
        "ngram [5, 5]" \
        "${spec_len}" \
        "${DATA_DIR}/specbench_question.jsonl" \
        "${SPECBENCH_MAX_TOKENS}" \
        "" \
        "${RESULTS_DIR}/specbench_ngram_5_5_${spec_len}.csv"
done

# Specbench - NGRAM [3, 5]
for spec_len in "${SPEC_LENS[@]}"; do
    launch_server "ngram_3_5" "${spec_len}" "true"
    run_benchmark \
        "ngram [3, 5]" \
        "${spec_len}" \
        "${DATA_DIR}/specbench_question.jsonl" \
        "${SPECBENCH_MAX_TOKENS}" \
        "" \
        "${RESULTS_DIR}/specbench_ngram_3_5_${spec_len}.csv"
done

echo ""
echo "=========================================="
echo "BENCHMARK 2: BLAZEDIT (max_tokens=${BLAZEDIT_MAX_TOKENS})"
echo "=========================================="

# Blazedit - Suffix with cache
for spec_len in "${SPEC_LENS[@]}"; do
    launch_server "suffix_w_cache" "${spec_len}" "true"
    run_benchmark \
        "suffix (w/ cache)" \
        "${spec_len}" \
        "${DATA_DIR}/blazedit_5k.jsonl" \
        "${BLAZEDIT_MAX_TOKENS}" \
        "" \
        "${RESULTS_DIR}/blazedit_suffix_cache_${spec_len}.csv"
done

# Blazedit - Suffix without cache
for spec_len in "${SPEC_LENS[@]}"; do
    launch_server "suffix_wo_cache" "${spec_len}" "false"
    run_benchmark \
        "suffix (w/o cache)" \
        "${spec_len}" \
        "${DATA_DIR}/blazedit_5k.jsonl" \
        "${BLAZEDIT_MAX_TOKENS}" \
        "" \
        "${RESULTS_DIR}/blazedit_suffix_nocache_${spec_len}.csv"
done

# Blazedit - NGRAM [5, 5]
for spec_len in "${SPEC_LENS[@]}"; do
    launch_server "ngram_5_5" "${spec_len}" "true"
    run_benchmark \
        "ngram [5, 5]" \
        "${spec_len}" \
        "${DATA_DIR}/blazedit_5k.jsonl" \
        "${BLAZEDIT_MAX_TOKENS}" \
        "" \
        "${RESULTS_DIR}/blazedit_ngram_5_5_${spec_len}.csv"
done

# Blazedit - NGRAM [3, 5]
for spec_len in "${SPEC_LENS[@]}"; do
    launch_server "ngram_3_5" "${spec_len}" "true"
    run_benchmark \
        "ngram [3, 5]" \
        "${spec_len}" \
        "${DATA_DIR}/blazedit_5k.jsonl" \
        "${BLAZEDIT_MAX_TOKENS}" \
        "" \
        "${RESULTS_DIR}/blazedit_ngram_3_5_${spec_len}.csv"
done

# Cleanup
kill_server

echo ""
echo "=========================================="
echo "BENCHMARK COMPLETE!"
echo "=========================================="
echo "Results saved to: ${RESULTS_DIR}"
echo ""
echo "To generate formatted tables, run:"
echo "  python3 parse_results.py"

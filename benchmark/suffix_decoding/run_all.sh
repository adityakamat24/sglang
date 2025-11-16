#!/bin/bash
###############################################################################
# Master Automation Script for Suffix Decoding Benchmarks
#
# This script runs the complete benchmark suite end-to-end:
# 1. Checks dependencies
# 2. Downloads datasets
# 3. Runs all benchmark configurations
# 4. Generates formatted results
# 5. Creates summary report
#
# Usage:
#   ./run_all.sh [MODEL_NAME] [PORT]
#
# Example:
#   ./run_all.sh meta-llama/Llama-3.1-8B-Instruct 30000
#
###############################################################################

set -e  # Exit on error

# Configuration
MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${2:-30000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/run_all.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "${LOG_FILE}"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "${LOG_FILE}"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "${LOG_FILE}"
}

# Print header
print_header() {
    echo "###############################################################################"
    echo "#                                                                             #"
    echo "#           SGLang Suffix Decoding Benchmark - Full Automation               #"
    echo "#                                                                             #"
    echo "###############################################################################"
    echo ""
    echo "Model: ${MODEL}"
    echo "Port: ${PORT}"
    echo "Working Directory: ${SCRIPT_DIR}"
    echo "Log File: ${LOG_FILE}"
    echo "Start Time: $(date)"
    echo ""
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "python3 not found. Please install Python 3.8+"
        exit 1
    fi
    log_success "Python 3 found: $(python3 --version)"

    # Check pip packages
    local missing_packages=()

    if ! python3 -c "import datasets" 2>/dev/null; then
        missing_packages+=("datasets")
    fi

    if ! python3 -c "import requests" 2>/dev/null; then
        missing_packages+=("requests")
    fi

    if ! python3 -c "import arctic_inference" 2>/dev/null; then
        log_warning "arctic-inference not found. Installing..."
        pip install arctic-inference || {
            log_error "Failed to install arctic-inference"
            exit 1
        }
    fi

    if [ ${#missing_packages[@]} -ne 0 ]; then
        log_warning "Missing packages: ${missing_packages[*]}"
        log_info "Installing missing packages..."
        pip install "${missing_packages[@]}" || {
            log_error "Failed to install required packages"
            exit 1
        }
    fi

    log_success "All dependencies satisfied"

    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        log_info "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "${LOG_FILE}"
    else
        log_warning "nvidia-smi not found. Cannot verify GPU availability"
    fi

    # Check if SGLang is installed
    if ! python3 -c "import sglang" 2>/dev/null; then
        log_error "SGLang not installed. Please install SGLang first."
        exit 1
    fi
    log_success "SGLang found"
}

# Make scripts executable
setup_scripts() {
    log_info "Setting up scripts..."
    chmod +x "${SCRIPT_DIR}/download_datasets.sh" 2>/dev/null || true
    chmod +x "${SCRIPT_DIR}/benchmark.sh" 2>/dev/null || true
    chmod +x "${SCRIPT_DIR}/parse_results.py" 2>/dev/null || true
    log_success "Scripts are executable"
}

# Download datasets
download_datasets() {
    log_info "=========================================="
    log_info "STEP 1: Downloading Datasets"
    log_info "=========================================="

    if [ -f "${SCRIPT_DIR}/data/specbench_question.jsonl" ] && \
       [ -f "${SCRIPT_DIR}/data/blazedit_5k.jsonl" ]; then
        log_warning "Datasets already exist. Skipping download."
        log_info "To re-download, delete the data/ directory first."
    else
        log_info "Running download_datasets.sh..."
        cd "${SCRIPT_DIR}"
        bash "${SCRIPT_DIR}/download_datasets.sh" 2>&1 | tee -a "${LOG_FILE}"

        if [ $? -eq 0 ]; then
            log_success "Datasets downloaded successfully"
        else
            log_error "Dataset download failed"
            exit 1
        fi
    fi
}

# Run benchmarks
run_benchmarks() {
    log_info "=========================================="
    log_info "STEP 2: Running Benchmarks"
    log_info "=========================================="
    log_info "This may take several hours depending on your hardware..."
    log_info "You can monitor progress in: ${LOG_FILE}"
    echo ""

    cd "${SCRIPT_DIR}"
    bash "${SCRIPT_DIR}/benchmark.sh" "${MODEL}" "${PORT}" 2>&1 | tee -a "${LOG_FILE}"

    if [ $? -eq 0 ]; then
        log_success "Benchmarks completed successfully"
    else
        log_error "Benchmark execution failed"
        log_info "Check the logs for details: ${LOG_FILE}"
        exit 1
    fi
}

# Generate results
generate_results() {
    log_info "=========================================="
    log_info "STEP 3: Generating Results Report"
    log_info "=========================================="

    cd "${SCRIPT_DIR}"
    python3 "${SCRIPT_DIR}/parse_results.py" \
        --results-dir "${SCRIPT_DIR}/results" \
        --output "RESULTS.md" 2>&1 | tee -a "${LOG_FILE}"

    if [ $? -eq 0 ]; then
        log_success "Results report generated"
    else
        log_error "Failed to generate results report"
        exit 1
    fi
}

# Create summary
create_summary() {
    log_info "=========================================="
    log_info "Creating Final Summary"
    log_info "=========================================="

    local summary_file="${SCRIPT_DIR}/SUMMARY.txt"

    cat > "${summary_file}" << EOF
###############################################################################
#                 Suffix Decoding Benchmark - Execution Summary               #
###############################################################################

Execution Date: $(date)
Model: ${MODEL}
Port: ${PORT}

Hardware Information:
$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "GPU information not available")

Python Version: $(python3 --version)
SGLang Version: $(python3 -c "import sglang; print(sglang.__version__)" 2>/dev/null || echo "Unknown")

###############################################################################
#                               File Locations                                #
###############################################################################

Results Directory: ${SCRIPT_DIR}/results/
Formatted Results: ${SCRIPT_DIR}/results/RESULTS.md
Full Log: ${LOG_FILE}
Summary: ${summary_file}

###############################################################################
#                              Dataset Information                            #
###############################################################################

Specbench: $(wc -l < "${SCRIPT_DIR}/data/specbench_question.jsonl" 2>/dev/null || echo "0") questions
Blazedit 5k: $(wc -l < "${SCRIPT_DIR}/data/blazedit_5k.jsonl" 2>/dev/null || echo "0") examples

###############################################################################
#                            Benchmark Configurations                         #
###############################################################################

Methods Tested:
  - suffix (w/ cache)
  - suffix (w/o cache)
  - ngram [5, 5]
  - ngram [3, 5]

Speculation Lengths: 5, 12, 32
Concurrency Levels: 1, 4, 16, 64

Benchmarks:
  - Specbench (max_tokens=256)
  - Blazedit (max_tokens=1024)

Total Configurations: 96
  (4 methods × 3 spec_lens × 4 concurrencies × 2 benchmarks)

###############################################################################
#                                 Results Files                               #
###############################################################################

CSV Files Generated:
$(ls -1 "${SCRIPT_DIR}/results/"*.csv 2>/dev/null | wc -l || echo "0") files

Server Logs:
$(ls -1 "${SCRIPT_DIR}/results/"server_*.log 2>/dev/null | wc -l || echo "0") files

###############################################################################
#                              Next Steps                                     #
###############################################################################

1. View formatted results:
   cat ${SCRIPT_DIR}/results/RESULTS.md

2. Download results to local machine:
   scp -r user@server:${SCRIPT_DIR}/results/ ./local_results/

3. View specific CSV results:
   ls -lh ${SCRIPT_DIR}/results/*.csv

4. Check server logs if needed:
   ls -lh ${SCRIPT_DIR}/results/server_*.log

###############################################################################

Execution completed at: $(date)

EOF

    log_success "Summary created: ${summary_file}"
}

# Main execution
main() {
    # Initialize log file
    echo "=== Benchmark Execution Started at $(date) ===" > "${LOG_FILE}"

    print_header | tee -a "${LOG_FILE}"

    # Step 0: Setup
    check_dependencies
    setup_scripts

    # Step 1: Download datasets
    download_datasets

    # Step 2: Run benchmarks
    run_benchmarks

    # Step 3: Generate results
    generate_results

    # Step 4: Create summary
    create_summary

    # Final message
    echo ""
    log_success "=========================================="
    log_success "ALL BENCHMARKS COMPLETED SUCCESSFULLY!"
    log_success "=========================================="
    echo ""
    log_info "Results Location:"
    log_info "  - Formatted Report: ${SCRIPT_DIR}/results/RESULTS.md"
    log_info "  - Summary: ${SCRIPT_DIR}/SUMMARY.txt"
    log_info "  - Full Log: ${LOG_FILE}"
    log_info "  - CSV Files: ${SCRIPT_DIR}/results/*.csv"
    echo ""
    log_info "To view results:"
    log_info "  cat ${SCRIPT_DIR}/results/RESULTS.md"
    echo ""
    log_info "To download all results:"
    log_info "  scp -r <user>@<server>:${SCRIPT_DIR}/results/ ./local_results/"
    echo ""

    # Display results preview
    if [ -f "${SCRIPT_DIR}/results/RESULTS.md" ]; then
        echo ""
        log_info "=========================================="
        log_info "Results Preview (first 50 lines):"
        log_info "=========================================="
        head -n 50 "${SCRIPT_DIR}/results/RESULTS.md"
        echo ""
        log_info "... (see full results in ${SCRIPT_DIR}/results/RESULTS.md)"
    fi
}

# Trap errors
trap 'log_error "Script failed at line $LINENO. Check ${LOG_FILE} for details."; exit 1' ERR

# Run main function
main "$@"

log_info "Script completed at: $(date)"
echo "=== Benchmark Execution Completed at $(date) ===" >> "${LOG_FILE}"

#!/usr/bin/env bash
# Run all INT8 GEMM benchmarks on Intel Xeon Gold 6544Y
# Usage: bash run_all.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda base (MKL is in base env)
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate base 2>/dev/null || true

echo "============================================="
echo "  Intel Xeon Gold 6544Y INT8 GEMM Benchmark"
echo "============================================="

# Step 1: Quick GEMM benchmark (TOPS only, ~5 min)
echo ""
echo ">>> Step 1: GEMM Benchmark (TOPS, no power)"
python3 benchmark_int8_gemm.py --warmup 50 --iterations 200

# Step 2: Decode sequence benchmark (~3 min)
echo ""
echo ">>> Step 2: Decode Sequence Benchmark"
python3 benchmark_int8_gemm.py --warmup 50 --iterations 200 --decode-steps "512,1024,1536"

# Step 3: Energy efficiency (TOPS/W + decode energy)
# This takes ~20 min with 30s per config (12 configs + 9 decode sequences)
echo ""
echo ">>> Step 3: Energy Efficiency Test (TOPS/W)"
python3 energy_efficiency_test.py --measure-sec 30 --decode-steps "512,1024,1536"

echo ""
echo ">>> All benchmarks complete. Results in results/"
ls -la results/

#!/usr/bin/env bash
# Fluxo reproduzivel para uma maquina NVIDIA, PARCODE ou Google Colab.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v nvidia-smi >/dev/null || { echo "nvidia-smi nao encontrado" >&2; exit 1; }
command -v nvcc >/dev/null || { echo "nvcc nao encontrado: instale/ative o CUDA Toolkit" >&2; exit 1; }
nvidia-smi
nvcc --version

if [[ ! -f data/covtype.data ]]; then
    mkdir -p data
    curl -fL https://kdd.ics.uci.edu/databases/covertype/covtype.data.gz \
        -o data/covtype.data.gz
    gzip -dkf data/covtype.data.gz
fi
[[ $(wc -l < data/covtype.data) == 581012 ]] || {
    echo "Dataset invalido: esperado 581012 linhas" >&2; exit 1;
}

make clean
make all test
make cuda test_cuda CUDA_ARCH="${CUDA_ARCH:-sm_75}"
scripts/benchmark_cuda.sh data/covtype.data 64 50 42 "${REPS:-3}"

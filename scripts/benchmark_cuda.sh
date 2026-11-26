#!/usr/bin/env bash
# Benchmark controlado entre sequencial e CUDA na mesma maquina.
# Uso: scripts/benchmark_cuda.sh [dataset] [K] [iters] [seed] [repeticoes]
# BENCH_SKIP_BUILD=1 reutiliza binarios ja compilados. BENCH_OUT escolhe o
# arquivo; por padrao cada experimento recebe timestamp e nunca sobrescreve outro.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=${1:-data/covtype.data}
K=${2:-64}
ITERS=${3:-50}
SEED=${4:-42}
REPS=${5:-3}
OUT=${BENCH_OUT:-results/benchmark_cuda_$(date +%Y%m%d-%H%M%S).txt}
NVCC_CMD=${NVCC:-nvcc}

[[ -f $DATA ]] || { echo "Dataset nao encontrado: $DATA" >&2; exit 1; }
if [[ ${BENCH_SKIP_BUILD:-0} != 1 ]]; then make -s seq cuda; fi
[[ -x bin/kmeans_seq && -x bin/kmeans_cuda ]] || {
    echo "Binarios ausentes; execute make seq cuda" >&2; exit 1;
}
mkdir -p "$(dirname "$OUT")"
[[ ! -e $OUT ]] || { echo "Saida ja existe, escolha BENCH_OUT diferente: $OUT" >&2; exit 1; }

median() {
    printf '%s\n' "$@" | sort -g | awk '
        {v[NR]=$1}
        END {if (NR%2) print v[(NR+1)/2]; else print (v[NR/2]+v[NR/2+1])/2}'
}

within_tolerance() {
    awk -v got="$1" -v expected="$2" 'BEGIN {
        if (expected==0) exit !(got==0)
        d=(got>expected?got-expected:expected-got)/(expected<0?-expected:expected)
        exit !(d<=0.01)
    }'
}

validate_all() {
    local expected=$1 mode=$2 value
    shift 2
    for value in "$@"; do
        within_tolerance "$value" "$expected" || {
            echo "Falha: inercia $mode divergente: $value (referencia $expected)" >&2
            return 1
        }
    done
}

bench() {
    local times=() reals=() inertias=() output
    for run in $(seq "$REPS"); do
        echo "Executando $1 ($run/$REPS)..." >&2
        output=$( { /usr/bin/time -p "$@" "$DATA" "$K" "$ITERS" "$SEED"; } 2>&1 )
        grep -q "^iters=$ITERS$" <<<"$output" || {
            echo "Execucao falhou: $*" >&2; echo "$output" >&2; exit 1;
        }
        times+=("$(awk -F= '/^time_s=/{print $2}' <<<"$output")")
        reals+=("$(awk '/^real /{print $2}' <<<"$output")")
        inertias+=("$(awk -F= '/^inertia=/{print $2}' <<<"$output")")
    done
    TIME=$(median "${times[@]}")
    REAL=$(median "${reals[@]}")
    INERTIA=$(median "${inertias[@]}")
    ALL_TIMES="${times[*]}"
    ALL_REALS="${reals[*]}"
    ALL_INERTIAS="${inertias[*]}"
}

if command -v "$NVCC_CMD" >/dev/null 2>&1; then
    NVCC_VERSION=$("$NVCC_CMD" --version | awk '/release/{print; exit}')
else
    NVCC_VERSION="indisponivel no PATH durante o benchmark"
fi
ARCH_DESCRIPTION=${CUDA_ARCH:-"nao informada (binario preexistente ou padrao do nvcc)"}

{
    echo "# Benchmark K-means: sequencial vs CUDA"
    echo "data:       $(date --iso-8601=seconds)"
    echo "cpu:        $(lscpu | sed -n 's/^Model name: *//p')"
    echo "nucleos:    $(lscpu -p=core | grep -v '^#' | sort -u | wc -l) fisicos, $(nproc --all) logicos"
    echo "ram:        $(free -h | awk '/^Mem:/{print $2}')"
    echo "so:         $(uname -srmo)"
    echo "gcc:        $(gcc --version | head -1)"
    echo "nvcc:       $NVCC_VERSION"
    echo "gpu:        $(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader)"
    echo "driver:     $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
    echo "flags:      gcc -O3 -std=c11 | nvcc -O3 -std=c++14; CUDA_ARCH=$ARCH_DESCRIPTION"
    echo "cuda:       256 threads/bloco; distancia e acumuladores double; reducao parcial em shared memory"
    echo "params:     dataset=$DATA K=$K iters=$ITERS seed=$SEED repeticoes=$REPS (mediana)"
    echo
} | tee "$OUT"

bench ./bin/kmeans_seq
SEQ_TIME=$TIME; SEQ_INERTIA=$INERTIA
read -r -a SEQ_INERTIAS <<<"$ALL_INERTIAS"
validate_all "$SEQ_INERTIA" sequencial "${SEQ_INERTIAS[@]}" || exit 1
printf '%-6s %10s %10s %8s %18s %11s  %s\n' \
    modo time_s real_s speedup inercia dif_inercia "tempos_individuais" | tee -a "$OUT"
printf '%-6s %10.3f %10.2f %8s %18s %11s  %s\n' \
    seq "$TIME" "$REAL" - "$INERTIA" - "$ALL_TIMES" | tee -a "$OUT"
echo "# real_s individuais seq: $ALL_REALS" | tee -a "$OUT"

bench ./bin/kmeans_cuda
CUDA_TIME=$TIME; CUDA_INERTIA=$INERTIA
SPEEDUP=$(awk -v seq="$SEQ_TIME" -v cuda="$CUDA_TIME" 'BEGIN{printf "%.2f",seq/cuda}')
DIFF=$(awk -v seq="$SEQ_INERTIA" -v cuda="$CUDA_INERTIA" 'BEGIN{
    d=(seq==0?(cuda==0?0:1):(seq>cuda?seq-cuda:cuda-seq)/seq); printf "%.2e",d
}')
printf '%-6s %10.3f %10.2f %8s %18s %11s  %s\n' \
    cuda "$TIME" "$REAL" "$SPEEDUP" "$INERTIA" "$DIFF" "$ALL_TIMES" | tee -a "$OUT"
echo "# real_s individuais cuda: $ALL_REALS" | tee -a "$OUT"

read -r -a CUDA_INERTIAS <<<"$ALL_INERTIAS"
validate_all "$SEQ_INERTIA" CUDA "${CUDA_INERTIAS[@]}" || exit 1

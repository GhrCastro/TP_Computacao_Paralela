#!/usr/bin/env bash
# Benchmark sequencial vs OpenMP com 1,2,4,8,16,32 threads.
# Uso: scripts/benchmark.sh [dataset] [K] [iters] [seed] [repeticoes]
# Saida: tabela no terminal e em results/benchmark_omp.txt.
#   time_s  = tempo de kmeans_run (impresso pelo binario), mediana das repeticoes
#   real_s  = processo inteiro (/usr/bin/time -p), mediana; inclui leitura do CSV
#   speedup = T_seq / T_omp(T); eficiencia = speedup / T
#   dif_inercia = |inercia_omp - inercia_seq| / inercia_seq (tolerancia: 1%)
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=${1:-data/covtype.data}
K=${2:-64}
ITERS=${3:-50}
SEED=${4:-42}
REPS=${5:-3}
THREADS="1 2 4 8 16 32"
OUT=results/benchmark_omp.txt

[ -f "$DATA" ] || { echo "Dataset nao encontrado: $DATA (ver docs/BASELINE.md)" >&2; exit 1; }
make -s seq omp
mkdir -p results

# Mediana de uma lista de numeros; para N par usa a media dos dois centrais.
median() {
    printf '%s\n' "$@" | sort -g | awk '
        {v[NR]=$1}
        END {if (NR%2) print v[(NR+1)/2]; else print (v[NR/2]+v[NR/2+1])/2}'
}

# Executa um binario REPS vezes; define TIME, REAL, INERTIA e ALL (tempos individuais).
bench() {
    local times=() reals=() inertias=() out actual_threads
    for _ in $(seq "$REPS"); do
        out=$( { /usr/bin/time -p "$@" "$DATA" "$K" "$ITERS" "$SEED"; } 2>&1 )
        grep -q "^iters=$ITERS$" <<<"$out" || { echo "Execucao falhou: $*" >&2; echo "$out" >&2; exit 1; }
        times+=("$(awk -F= '/^time_s=/{print $2}' <<<"$out")")
        reals+=("$(awk '/^real /{print $2}' <<<"$out")")
        inertias+=("$(awk -F= '/^inertia=/{print $2}' <<<"$out")")
        if [[ $1 == *kmeans_omp ]]; then
            actual_threads=$(awk -F= '/^omp_threads=/{print $2}' <<<"$out")
            [[ $actual_threads == "${OMP_NUM_THREADS:-}" ]] || {
                echo "OpenMP usou $actual_threads threads; esperado ${OMP_NUM_THREADS:-nao definido}" >&2
                exit 1
            }
        fi
    done
    TIME=$(median "${times[@]}")
    REAL=$(median "${reals[@]}")
    INERTIA=$(median "${inertias[@]}")
    ALL="${times[*]}"
    ALL_INERTIAS="${inertias[*]}"
}

{
    echo "# Benchmark K-means: sequencial vs OpenMP"
    echo "data:     $(date -u +%Y-%m-%dT%H:%MZ)"
    echo "cpu:      $(lscpu | sed -n 's/^Model name: *//p')"
    echo "nucleos:  $(lscpu -p=core | grep -v '^#' | sort -u | wc -l) fisicos, $(nproc --all) logicos"
    echo "so:       $(uname -sr)"
    echo "gcc:      $(gcc --version | head -1)"
    echo "flags:    -O3 -std=c11 (seq) | -O3 -std=c11 -fopenmp (omp), schedule(static)"
    echo "params:   dataset=$DATA K=$K iters=$ITERS seed=$SEED repeticoes=$REPS (mediana)"
    echo
} | tee "$OUT"

bench ./bin/kmeans_seq
SEQ_TIME=$TIME SEQ_INERTIA=$INERTIA
for value in $ALL_INERTIAS; do
    awk -v got="$value" -v expected="$SEQ_INERTIA" 'BEGIN {
        d=(got>expected?got-expected:expected-got)/expected; exit !(d<=0.01)
    }' || { echo "Falha: inercia sequencial inconsistente: $value" >&2; exit 1; }
done
{
    printf '%-8s %9s %9s %8s %10s %18s %11s  %s\n' \
        modo time_s real_s speedup eficiencia inercia dif_inercia "tempos_individuais"
    printf '%-8s %9.3f %9.2f %8s %10s %18s %11s  %s\n' \
        seq "$TIME" "$REAL" - - "$INERTIA" - "$ALL"
} | tee -a "$OUT"

STATUS=0
for T in $THREADS; do
    OMP_NUM_THREADS=$T bench ./bin/kmeans_omp
    for value in $ALL_INERTIAS; do
        awk -v got="$value" -v expected="$SEQ_INERTIA" 'BEGIN {
            d=(got>expected?got-expected:expected-got)/expected; exit !(d<=0.01)
        }' || { echo "Falha: inercia omp-$T divergente: $value" >&2; STATUS=1; }
    done
    row=$(awk -v t="$T" -v ts="$TIME" -v r="$REAL" -v seq="$SEQ_TIME" \
              -v inr="$INERTIA" -v sinr="$SEQ_INERTIA" -v all="$ALL" 'BEGIN{
        d=(inr>sinr?inr-sinr:sinr-inr)/sinr; s=seq/ts
        printf "omp-%-4s %9.3f %9.2f %8.2f %10.2f %18s %11.2e  %s%s",
               t, ts, r, s, s/t, inr, d, all, (d<=0.01?"":"  FALHA: dif>1%")}')
    echo "$row" | tee -a "$OUT"
    if [[ $row == *FALHA* ]]; then STATUS=1; fi
done
exit $STATUS

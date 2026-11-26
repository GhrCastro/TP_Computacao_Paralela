#!/usr/bin/env bash
# Roteiro automatizado para demonstrar o TP no WSL.
# Uso: source scripts/apresentacao_wsl.sh
# Opcoes: DEMO_ITERS=3 (padrao) e DEMO_THREADS="1 2 4 8 16 32".

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    echo "Este roteiro deve ser carregado com: source scripts/apresentacao_wsl.sh" >&2
    exit 2
fi

tp_kmeans_demo_wsl() (
    set -Eeuo pipefail

    local script_source script_dir project_dir tool_dir demo_tmp
    local demo_iters demo_threads gpu_cc cuda_arch
    local nvcc_bin cuda_runtime_root cuda_nvcc_root cuda_apt_root
    local cyan green yellow red reset
    local CDPATH=

    # CDPATH pode fazer `cd` imprimir o destino e corromper command substitution.
    # Resolva a partir de um caminho absoluto, silencie cd e use diretorio fisico.
    script_source=${BASH_SOURCE[0]}
    if [[ $script_source != /* ]]; then script_source="$PWD/$script_source"; fi
    script_dir=$(cd -P -- "$(dirname -- "$script_source")" >/dev/null && pwd -P)
    project_dir=$(cd -P -- "$script_dir/.." >/dev/null && pwd -P)
    [[ -n $project_dir && -f $project_dir/Makefile ]] || {
        printf '[ERRO] Nao foi possivel localizar a raiz do projeto a partir de %s\n' "$script_source" >&2
        return 1
    }
    tool_dir="$project_dir/.wsl-cuda"
    demo_iters=${DEMO_ITERS:-3}
    demo_threads=${DEMO_THREADS:-"1 2 4 8 16 32"}
    demo_tmp=$(mktemp -d /tmp/tp-kmeans-demo-XXXXXX)
    trap 'rm -r -- "$demo_tmp"' EXIT

    if [[ -t 1 ]]; then
        cyan=$'\033[1;36m'; green=$'\033[1;32m'; yellow=$'\033[1;33m'
        red=$'\033[1;31m'; reset=$'\033[0m'
    else
        cyan= green= yellow= red= reset=
    fi

    section() { printf '\n%s== %s ==%s\n' "$cyan" "$1" "$reset"; }
    ok() { printf '%s[OK]%s %s\n' "$green" "$reset" "$1"; }
    warn() { printf '%s[AVISO]%s %s\n' "$yellow" "$reset" "$1"; }
    fail() { printf '%s[ERRO]%s %s\n' "$red" "$reset" "$1" >&2; return 1; }

    valid_positive_int() { [[ $1 =~ ^[1-9][0-9]*$ ]]; }
    valid_positive_int "$demo_iters" || fail "DEMO_ITERS deve ser inteiro positivo"
    for thread_count in $demo_threads; do
        valid_positive_int "$thread_count" || fail "DEMO_THREADS contem valor invalido: $thread_count"
    done

    setup_local_cuda() {
        local venv_dir deb_dir deb_file
        section "Preparando CUDA Toolkit local (primeira execucao)"
        command -v python3 >/dev/null || fail "python3 nao encontrado"
        command -v apt-get >/dev/null || fail "apt-get nao encontrado"
        command -v dpkg-deb >/dev/null || fail "dpkg-deb nao encontrado"

        venv_dir="$tool_dir/venv"
        deb_dir="$tool_dir/debs"
        cuda_apt_root="$tool_dir/apt-root"
        mkdir -p "$deb_dir" "$cuda_apt_root"

        if [[ ! -x $venv_dir/bin/python ]]; then
            python3 -m venv "$venv_dir"
        fi
        "$venv_dir/bin/pip" install --disable-pip-version-check --quiet \
            nvidia-cuda-nvcc-cu12==12.4.131 \
            nvidia-cuda-runtime-cu12==12.4.127

        deb_file=$(find "$deb_dir" -maxdepth 1 -type f -name 'nvidia-cuda-toolkit_*.deb' -print -quit)
        if [[ -z $deb_file ]]; then
            (cd "$deb_dir" && apt-get download nvidia-cuda-toolkit)
            deb_file=$(find "$deb_dir" -maxdepth 1 -type f -name 'nvidia-cuda-toolkit_*.deb' -print -quit)
        fi
        [[ -n $deb_file ]] || fail "Nao foi possivel baixar o pacote nvidia-cuda-toolkit"
        if [[ ! -x $cuda_apt_root/usr/lib/nvidia-cuda-toolkit/bin/nvcc ]]; then
            dpkg-deb -x "$deb_file" "$cuda_apt_root"
        fi
        ok "Toolkit local preparado em .wsl-cuda (sem sudo)"
    }

    configure_cuda() {
        local local_venv
        if command -v nvcc >/dev/null 2>&1; then
            nvcc_bin=$(command -v nvcc)
            cuda_runtime_root=""
            cuda_nvcc_root=""
            cuda_apt_root=""
            ok "NVCC do sistema: $nvcc_bin"
            return 0
        fi

        local_venv="$tool_dir/venv"
        cuda_apt_root="$tool_dir/apt-root"
        nvcc_bin="$cuda_apt_root/usr/lib/nvidia-cuda-toolkit/bin/nvcc"
        if [[ ! -x $nvcc_bin ]]; then
            setup_local_cuda
        fi
        cuda_runtime_root=$(find "$local_venv/lib" -type d -path '*/site-packages/nvidia/cuda_runtime' -print -quit)
        cuda_nvcc_root=$(find "$local_venv/lib" -type d -path '*/site-packages/nvidia/cuda_nvcc' -print -quit)
        [[ -n $cuda_runtime_root && -n $cuda_nvcc_root ]] || fail "Pacotes CUDA locais incompletos"

        if [[ ! -e $cuda_runtime_root/lib/libcudart.so ]]; then
            ln -s libcudart.so.12 "$cuda_runtime_root/lib/libcudart.so"
        fi
        PATH="$cuda_apt_root/usr/bin:$cuda_apt_root/usr/lib/nvidia-cuda-toolkit/bin:$cuda_nvcc_root/bin:$PATH"
        export PATH
        ok "NVCC local: $nvcc_bin"
    }

    cuda_make() {
        if [[ -n $cuda_runtime_root ]]; then
            make "$@" CUDA_ARCH="$cuda_arch" NVCC="$nvcc_bin" \
                CUDA_HOME="$cuda_runtime_root" \
                CUDA_CPPFLAGS="-I$cuda_runtime_root/include -I$cuda_nvcc_root/include"
        else
            make "$@" CUDA_ARCH="$cuda_arch" NVCC="$nvcc_bin"
        fi
    }

    inertia_matches() {
        awk -v got="$1" -v expected="$2" 'BEGIN {
            if (expected==0) exit !(got==0)
            d=(got>expected?got-expected:expected-got)/expected
            exit !(d<=0.01)
        }'
    }

    run_and_save() {
        local label=$1 output=$2
        shift 2
        printf '\n%s-- %s --%s\n' "$yellow" "$label" "$reset"
        "$@" | tee "$output"
    }

    cd "$project_dir"
    section "TP K-means: demonstracao automatica no WSL"
    printf 'Projeto: %s\nIteracoes da demo: %s\nThreads OpenMP: %s\n' \
        "$project_dir" "$demo_iters" "$demo_threads"

    section "Diagnostico do ambiente"
    uname -a
    lscpu | sed -n 's/^Model name: */CPU: /p; s/^CPU(s): */CPUs logicas: /p; s/^Core(s) per socket: */Nucleos por socket: /p'
    free -h | sed -n '1,2p'
    gcc --version | head -1
    make --version | head -1
    command -v nvidia-smi >/dev/null || fail "nvidia-smi nao encontrado; GPU indisponivel no WSL"
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader

    gpu_cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
    [[ $gpu_cc =~ ^[0-9]+$ ]] || fail "Compute capability invalida: $gpu_cc"
    cuda_arch="sm_$gpu_cc"
    printf 'CUDA_ARCH detectada: %s\n' "$cuda_arch"
    configure_cuda
    "$nvcc_bin" --version | tail -4

    section "Dataset UCI Covertype"
    if [[ ! -f data/covtype.data ]]; then
        warn "Dataset ausente; baixando da fonte oficial"
        mkdir -p data
        curl -fL https://kdd.ics.uci.edu/databases/covertype/covtype.data.gz \
            -o data/covtype.data.gz
        gzip -dkf data/covtype.data.gz
    fi
    [[ $(wc -l < data/covtype.data) -eq 581012 ]] || fail "Dataset deve ter 581012 linhas"
    ok "data/covtype.data: 581012 linhas"

    section "Compilacao e testes"
    make clean
    make all
    make test
    cuda_make cuda
    cuda_make test_cuda
    ok "CPU, OpenMP e CUDA compilados e testados"

    section "Execucoes ao vivo no dataset completo"
    run_and_save "Sequencial" "$demo_tmp/seq.txt" \
        ./bin/kmeans_seq data/covtype.data 64 "$demo_iters" 42

    for thread_count in $demo_threads; do
        run_and_save "OpenMP: $thread_count thread(s)" "$demo_tmp/omp-$thread_count.txt" \
            env OMP_DYNAMIC=FALSE OMP_NUM_THREADS="$thread_count" \
            ./bin/kmeans_omp data/covtype.data 64 "$demo_iters" 42
    done

    run_and_save "CUDA" "$demo_tmp/cuda.txt" \
        ./bin/kmeans_cuda data/covtype.data 64 "$demo_iters" 42

    section "Resumo das execucoes ao vivo"
    printf '%-12s %8s %14s %12s\n' "modo" "threads" "time_s" "inercia"
    printf '%-12s %8s %14s %12s\n' "------------" "--------" "--------------" "------------"

    local seq_inertia time_value inertia_value actual_threads
    seq_inertia=$(awk -F= '/^inertia=/{print $2}' "$demo_tmp/seq.txt")
    time_value=$(awk -F= '/^time_s=/{print $2}' "$demo_tmp/seq.txt")
    printf '%-12s %8s %14.6f %12.6f\n' seq - "$time_value" "$seq_inertia"

    for thread_count in $demo_threads; do
        time_value=$(awk -F= '/^time_s=/{print $2}' "$demo_tmp/omp-$thread_count.txt")
        inertia_value=$(awk -F= '/^inertia=/{print $2}' "$demo_tmp/omp-$thread_count.txt")
        actual_threads=$(awk -F= '/^omp_threads=/{print $2}' "$demo_tmp/omp-$thread_count.txt")
        [[ $actual_threads == "$thread_count" ]] || fail "OpenMP pediu $thread_count e usou $actual_threads"
        inertia_matches "$inertia_value" "$seq_inertia" || fail "Inercia OpenMP-$thread_count divergente"
        printf '%-12s %8s %14.6f %12.6f\n' omp "$actual_threads" "$time_value" "$inertia_value"
    done

    time_value=$(awk -F= '/^time_s=/{print $2}' "$demo_tmp/cuda.txt")
    inertia_value=$(awk -F= '/^inertia=/{print $2}' "$demo_tmp/cuda.txt")
    inertia_matches "$inertia_value" "$seq_inertia" || fail "Inercia CUDA divergente"
    printf '%-12s %8s %14.6f %12.6f\n' cuda GPU "$time_value" "$inertia_value"
    ok "Todas as inercias da demonstracao estao dentro da tolerancia de 1%"

    section "Resultados oficiais preservados (K=64, 50 iteracoes, seed=42)"
    sed -n '1,120p' results/benchmark_omp.txt
    printf '\n'
    sed -n '1,120p' results/benchmark_cuda.txt

    section "Conclusao"
    ok "Demonstracao concluida com sucesso"
    printf 'Para repetir: source scripts/apresentacao_wsl.sh\n'
    printf 'Para usar outras iteracoes: DEMO_ITERS=1 source scripts/apresentacao_wsl.sh\n'
    printf 'Para limitar threads: DEMO_THREADS="1 2 4 8" source scripts/apresentacao_wsl.sh\n'
)

# Nao coloque esta chamada em `if`, `&&` ou `||`: esses contextos desativam o
# errexit dentro da funcao Bash e permitiriam continuar depois de uma falha.
tp_kmeans_demo_wsl "$@"
tp_demo_status=$?
unset -f tp_kmeans_demo_wsl
return "$tp_demo_status"

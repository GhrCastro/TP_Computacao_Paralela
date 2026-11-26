# Baseline sequencial — Felipe

## Dados reais
```sh
mkdir -p data bin
curl -fL https://kdd.ics.uci.edu/databases/covertype/covtype.data.gz -o data/covtype.data.gz
gzip -dk data/covtype.data.gz
wc -l data/covtype.data
```
Esperado: 581012 linhas, 54 atributos utilizados. Fonte: UCI Covertype.
Nao versionar os dados nem executaveis.

## Compilacao e testes
```sh
gcc -O3 -std=c11 -Wall -Wextra -Iinclude src/main.c src/common.c src/seq.c -lm -o bin/kmeans_seq
gcc -O3 -std=c11 -Wall -Wextra -Iinclude tests/test_seq.c src/common.c src/seq.c -lm -o bin/test_seq
./bin/test_seq
```
O Makefile integrado oferece os mesmos comandos e tambem os alvos CUDA.

## Calibracao
```sh
lscpu
uname -a
gcc --version
/usr/bin/time -p ./bin/kmeans_seq data/covtype.data 64 50 42
```
Foram realizados dois experimentos reais, sem misturar maquinas:

1. Arthur, i5-12400/Fedora 44/GCC 16.2.1: sequencial 54.001 s; OpenMP
   1/2/4/8/16/32 = 54.140/27.073/13.729/11.910/10.008/9.249 s.
2. Gustavo, i7-10750H/WSL2/GCC 11.4 e GTX 1660 Ti/CUDA 11.5:
   sequencial 147.170 s e CUDA 13.068 s (medianas de tres).

Em ambos: Covertype completo, K=64, 50 iteracoes, seed=42 e inercia
120313.163175. Assim, o requisito de sequencial >=10 s esta comprovado.
Detalhes e tempos individuais: `results/benchmark_omp.txt` e
`results/benchmark_cuda.txt`.

Esses dois arquivos sao evidencias historicas e nao devem ser sobrescritos.
Novas execucoes dos scripts recebem nomes com timestamp por padrao; a variavel
`BENCH_OUT` permite definir explicitamente outro destino ainda inexistente.

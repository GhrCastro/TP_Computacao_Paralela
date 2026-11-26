# TP Computacao Paralela — K-means em C, OpenMP e CUDA

Trabalho pratico da PUC Minas (2026/2) com tres implementacoes explicitas do
algoritmo de Lloyd sobre o dataset real UCI Covertype (581.012 registros e 54
atributos): sequencial, OpenMP e CUDA.

## Integrantes

- Felipe Barros Ratton de Almeida — sequencial e infraestrutura compartilhada.
- Arthur Carvalho Rodrigues — OpenMP e benchmarks multicore.
- Gustavo Henrique R. de Castro — CUDA, validacao cruzada e integracao.

## Inicio rapido

```sh
mkdir -p data
curl -fL https://kdd.ics.uci.edu/databases/covertype/covtype.data.gz \
  -o data/covtype.data.gz
gzip -dk data/covtype.data.gz

make all test                 # CPU; `all` nao exige CUDA
make cuda test_cuda           # requer CUDA Toolkit e GPU CC >= 6.0

./bin/kmeans_seq  data/covtype.data 64 50 42
OMP_NUM_THREADS=8 ./bin/kmeans_omp data/covtype.data 64 50 42
./bin/kmeans_cuda data/covtype.data 64 50 42
```

Informe a arquitetura da GPU ao compilar: na GTX 1660 Ti medida foi usado
`make clean cuda CUDA_ARCH=sm_75`; para uma GPU 8.6, use `sm_86`. Sem a
variavel, o NVCC usa seu proprio padrao. `CUDA_HOME` e `CUDA_LIBDIR` podem ser
configurados quando o toolkit estiver fora do caminho usual. A compilacao nao
usa `fast-math`.

## Resultados reais

No experimento CPU de Arthur (i5-12400, Fedora 44), a mediana sequencial foi
54,001 s e OpenMP atingiu 9,249 s com 32 threads (speedup 5,84). No experimento
independente de Gustavo (i7-10750H/WSL2 + GTX 1660 Ti), as medianas foram
147,170 s sequencial e 13,068 s CUDA (speedup 11,26). A inercia foi
120313,163175 em todas as configuracoes. Maquinas diferentes nao foram usadas
na mesma razao de speedup.

O documento de entrega exigido pelo professor e [readme.txt](readme.txt). Veja
tambem [o contrato tecnico](docs/CONTRATO.md),
[os resultados OpenMP](results/benchmark_omp.txt),
[os resultados CUDA](results/benchmark_cuda.txt) e
[o guia de entrevista](docs/ENTREVISTA.md). Para o dia da banca, abra
[APRESENTACAO.html](APRESENTACAO.html) no navegador: ele contem o roteiro de
execucao, comandos copiaveis, saidas esperadas e diagnostico, tudo offline.
No WSL, a demonstracao inteira tambem pode ser executada com um unico comando:
`source scripts/apresentacao_wsl.sh`.

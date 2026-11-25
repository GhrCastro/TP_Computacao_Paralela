# TP Computação Paralela — K-means (sequencial, OpenMP, CUDA)

Projeto de Computação Paralela — PUC Minas, 2026/2.

**Integrantes**
- Felipe Barros Ratton de Almeida — base real, infraestrutura de dados, K-means sequencial.
- Arthur Carvalho Rodrigues — K-means OpenMP, testes multicore e medição CPU.
- Gustavo Henrique R. de Castro — K-means CUDA, integração, validação cruzada e entrega.

## Objetivo
Implementar K-means do zero em C (CPU sequencial e OpenMP) e CUDA C/C++ (GPU), com mesmas entradas, parâmetros e critérios de correção, medindo tempos e speedup. O enunciado exige pelo menos 10 s na versão sequencial no ambiente de teste, medição OpenMP com 1/2/4/8/16/32 threads, comentários de paralelização e documentação de compilação/execução em **readme.txt**.

## Dataset escolhido
[UCI Covertype](https://archive.ics.uci.edu/dataset/31/covertype) (581.012 registros, 54 atributos + rótulo de classe). Baixar o arquivo `covtype.data.gz` da [fonte original](https://kdd.ics.uci.edu/databases/covertype/covtype.data.gz) e descompactar em `data/covtype.data`. Não versionar dados nem executáveis. Os primeiros dez atributos quantitativos serão normalizados uma vez pela rotina compartilhada; descartar o último campo (rótulo).

## Organização
- `include/kmeans.h` — API C compartilhada; ver `docs/CONTRATO.md`.
- `src/common.c` — leitura, normalização, inicialização e funções auxiliares (Felipe).
- `src/main.c` — CLI compartilhada e medição do cálculo (Felipe).
- `src/seq.c` — referência sequencial (Felipe).
- `src/omp.c` — multicore (Arthur).
- `src/cuda.cu` — GPU (Gustavo).
- `scripts/benchmark.sh` — medições repetidas (Arthur).
- `tests/` — testes pequenos e verificação cruzada (trio).
- `readme.txt` — instruções finais de compilação, execução, tempos reais e análise.

As três frentes devem usar **o mesmo contrato** e alterar **arquivos distintos**. Criar PRs a partir de `main`, preferencialmente na ordem: base sequencial → OpenMP → CUDA/integração. Não afirmar speedup sem medições reais, não simular GPU e não colocar números inventados nos comentários.

## Parâmetros iniciais de benchmark
`K=64`, `ITERS=50`, `SEED=42`. Calibrar no equipamento final para respeitar o requisito da versão sequencial com pelo menos 10 s e reexecutar todas as versões sob os mesmos parâmetros.

**Status**: planejamento inicial; implementações e medições ainda pendentes.

# Contrato técnico compartilhado (congelado para os 3 blocos)

Este contrato deve ser respeitado por Felipe (sequencial), Arthur (OpenMP) e Gustavo (CUDA), sem modificar a assinatura pública sem discussão na issue-pai.

## Modelo matemático
K-means de Lloyd: (1) atribuir cada ponto ao centroide de menor **distância euclidiana ao quadrado**, com desempate pelo menor índice de cluster; (2) somar pontos e contar ocorrências por cluster; (3) atualizar centroide como média; (4) repetir exatamente `iters` vezes. Caso um cluster fique vazio, manter seu centroide anterior. Não usar bibliotecas de K-means. A sequência de centroides e os resultados podem divergir levemente por arredondamento de ponto flutuante; verificar também a inércia (SSE).

## Dados
Arquivo original UCI Covertype: 581.012 linhas, 55 campos separados por vírgula, **54 atributos** e um último rótulo que não entra no K-means. Sem cabeçalho. `load_dataset` carrega `float` row-major `points[i * d + j]`, com `d=54`. Calcular mínimo e máximo dos 10 atributos numéricos sobre o dataset inteiro; normalizar estes 10 para [0,1] com `(x-min)/(max-min)` e tratar amplitude zero como 0. As outras 44 colunas binárias não mudam. Esta etapa deve ocorrer *uma vez* em `common.c`, antes de qualquer benchmark.

Atenção: `data/` não vai para Git; baixar a base real pelo link da UCI (não substituir por amostras sintéticas ou duplicações). Dados menores são permitidos apenas nos testes unitários, não no benchmark do enunciado.

## Assinatura comum
`include/kmeans.h` define `Dataset`, `load_dataset`, `free_dataset`, `init_centroids`, `calculate_inertia` e `kmeans_run`. Cada binário inclui **apenas um** dos arquivos `src/seq.c`, `src/omp.c`, `src/cuda.cu`, que implementa `kmeans_run`. `centroids` chega inicializado e é atualizado in-place; `labels` possui N inteiros. O código retorna 0 ao concluir sem erro.

Inicialização: selecionar K índices distintos do dataset de modo reproduzível, com PRNG próprio e seed fixa, dentro de `common.c`. Nunca reinicializar os centroides de maneira diferente entre as versões.

## CLI padrão e build
- `make seq`, `make omp`, `make cuda` (este último requer NVIDIA CUDA Toolkit/GPU real).
- `./bin/kmeans_seq data/covtype.data 64 50 42`
- `./bin/kmeans_omp data/covtype.data 64 50 42`
- `./bin/kmeans_cuda data/covtype.data 64 50 42`
- Argumentos: `<dataset_path> <K> <iters> <seed>`. Falhas de argumentos e de arquivo devem encerrar com erro.
- Compilação CPU com `gcc -O3`; OpenMP com `gcc -O3 -fopenmp`; GPU com `nvcc -O3`. Não medir debug builds.
- Todas imprimem: `mode=...`, `n=...`, `d=...`, `k=...`, `iters=...`, `seed=...`, `time_s=...`, `inertia=...`; opcionalmente exportar centroides para comparação.
- O intervalo `time_s` começa imediatamente antes e termina imediatamente após `kmeans_run`; mede execução do K-means sobre dados já carregados/normalizados. Em CUDA inclui alocação, transferências host/device e sincronização necessários ao resultado. Registrar também `/usr/bin/time` para o processo completo, sem confundir os dois relógios.
- `calculate_inertia` é feito **depois** de encerrar `time_s` para todos os modos (medição padronizada); inércia = soma das menores distâncias quadráticas de todos os pontos aos centroides finais.

## Testes de aceite comuns
1. Dataset pequeno de 2D com dois grupos claramente separados; também K=1, N não múltiplo de 256 e cluster vazio.
2. Toda versão termina sem corrida de dados e sem erro CUDA, centroide NaN ou memória vazada grosseiramente.
3. Mesmo K, seed, iters e dataset; comparar SSE/inércia com tolerância relativa definida (por exemplo, <=1%); diferenças significativas exigem investigação, não ajuste artificial.
4. Ambiente e sistema: CPU, núcleos físicos/lógicos, memória, GPU, compiladores, flags, parâmetros, repetições. Registrar CPU com threads 1/2/4/8/16/32, mesmo se houver oversubscription.
5. Versão sequencial >=10 s comprovados no equipamento de teste. Ajustar K/iters com trabalho real e repetir **todas** as versões.
6. Comentar no topo das implementações tempos reais e parâmetros, marcando valores como *pendentes* até executar. Comentar o porquê de cada trecho paralelo.
7. `readme.txt` final contém: explicação do algoritmo, dataset e fonte, comandos de compilação/execução, hardware, tabela de tempos, speedups, escalabilidade e limitações.

## Responsáveis
Felipe: `include/kmeans.h`, `src/common.c`, `src/main.c`, `src/seq.c` e loader.
Arthur: `src/omp.c`, `scripts/benchmark.sh`, resultados CPU.
Gustavo: `src/cuda.cu`, resultados GPU, validação cruzada, `readme.txt` e integração final.

## Coordenação e Git
Branches recomendadas: `feat/felipe-baseline`, `feat/arthur-openmp`, `feat/gustavo-cuda` (todas criadas da main que contém o contrato). Não modificar arquivos alheios sem avisar. Arthur e Gustavo podem criar stubs locais enquanto aguardam o merge do baseline; após ele entrar na main, rebasear suas branches. Nada de tempos falsos ou dados sintéticos como comprovação final.

TRABALHO PRATICO DE COMPUTACAO PARALELA — K-MEANS
PUC Minas — 2026/2
==================================================

1. INTEGRANTES
--------------
Felipe Barros Ratton de Almeida: versao sequencial e infraestrutura comum.
Arthur Carvalho Rodrigues: versao OpenMP e benchmarks multicore.
Gustavo Henrique R. de Castro: versao CUDA, validacao e integracao final.

2. APLICACAO E ALGORITMO
------------------------
O programa agrupa registros semelhantes sem usar os rotulos de classe. K-means
e um algoritmo de aprendizado nao supervisionado: recebe os pontos e K, mas nao
respostas corretas. Cada grupo e representado pela media de seus pontos, chamada
centroide.

As tres versoes implementam explicitamente o algoritmo de Lloyd:
  1. associar cada ponto ao centroide de menor distancia euclidiana quadratica;
  2. somar os pontos e contar os membros de cada cluster;
  3. substituir cada centroide pela media do cluster;
  4. repetir exatamente I iteracoes.

Nao e necessario calcular raiz quadrada: ela preserva a ordenacao das
distancias. Em empate, a comparacao estrita escolhe o menor indice. Um cluster
vazio conserva seu centroide anterior. A inercia (SSE) e a soma das menores
distancias quadraticas aos centroides finais.

Complexidade temporal: O(I*N*K*D). Memoria comum: O(N*D + K*D + N). CUDA
acrescenta parciais O(B*K*D), onde B e limitado a oito blocos por SM.

Semantica de labels: todas as versoes retornam as associacoes calculadas no
inicio da ultima iteracao, usadas para produzir os centroides finais. Por isso,
em um caso limite, elas podem diferir de uma nova classificacao pelos centroides
ja atualizados. A inercia e sempre recalculada contra os centroides finais. Essa
semantica existente foi preservada de forma consistente nas tres versoes.

3. DATASET REAL E PRE-PROCESSAMENTO
-----------------------------------
UCI Covertype:
https://archive.ics.uci.edu/dataset/31/covertype

O arquivo possui 581.012 registros, 54 atributos e uma ultima coluna de classe.
src/common.c descarta a classe, normaliza Min-Max os primeiros dez atributos
quantitativos usando o dataset inteiro e preserva os outros 44 atributos
binarios. Os pontos ficam como float row-major. Centroides iniciais sao K pontos
distintos escolhidos pelo mesmo SplitMix64/Fisher-Yates parcial e mesma seed em
todas as versoes.

Download (o dataset nao e versionado):

  mkdir -p data
  curl -fL https://kdd.ics.uci.edu/databases/covertype/covtype.data.gz \
    -o data/covtype.data.gz
  gzip -dk data/covtype.data.gz
  wc -l data/covtype.data

O ultimo comando deve imprimir 581012. SHA-256 do .gz usado no experimento:
614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771.

4. IMPLEMENTACOES E PADROES PARALELOS
--------------------------------------
Sequencial (src/seq.c): percorre pontos, centroides e dimensoes, acumula em
double e atualiza centroides em float. E a referencia matematica.

OpenMP (src/omp.c): aplica MAP/Fork-Join ao laco de pontos com
parallel for schedule(static). labels[i] e exclusivo de uma iteracao do laco.
As somas e contagens usam reduction por secoes de array: cada thread recebe
copias privadas e o runtime as combina ao final (REDUCE). Centroides sao apenas
lidos na regiao paralela e atualizados depois da barreira implicita. Isso evita
data races sem atomic/critical no laco quente. schedule(static) e adequado
porque cada ponto executa o mesmo K*D e reduz overhead de escalonamento.

CUDA (src/cuda.cu): pontos, centroides e labels sao alocados uma vez e ficam no
device durante as I iteracoes. Um grid de blocos de 256 threads executa MAP: cada
thread trata um ponto (ou mais, via grid-stride) e procura o centroide. Para
K=64,D=54, cada bloco usa cerca de 28 KiB de shared memory para somas double e
contagens. atomicAdd resolve concorrencia dentro do bloco. Cada bloco grava um
parcial independente na memoria global; kernels separados reduzem contagens e
somas e atualizam K*D coordenadas. Se K*D nao couber em shared memory, existe um
fallback correto com atomicas globais. A GPU deve suportar compute capability
>=6.0 por causa de atomicAdd(double).

As iteracoes nao podem ser executadas paralelamente: a iteracao t+1 depende dos
centroides produzidos por t. A ordem dos kernels no mesmo stream e a
sincronizacao final garantem essa dependencia. O programa verifica erros de
alocacao, copia, lancamento e sincronizacao, copia HtoD antes do laco e somente
centroides/labels DtoH ao final. Distancias e somas continuam em double; nao ha
fast-math nem reducao silenciosa de precisao.

5. COMPILACAO E TESTES
----------------------
Requisitos CPU: GCC, GNU Make e suporte OpenMP.
Requisitos CUDA: CUDA Toolkit com nvcc, driver NVIDIA e GPU CC >= 6.0.

  make seq
  make omp
  make cuda CUDA_ARCH=sm_75    # GTX 1660 Ti do experimento
  make cuda CUDA_ARCH=sm_86    # exemplo para GPU com CC 8.6
  make test                    # loader + sequencial + OpenMP
  make test_cuda               # executa de fato na GPU
  make clean

make all compila seq+omp e deliberadamente nao exige CUDA, permitindo uso em
maquinas CPU-only. Flags: GCC -O3 -std=c11; OpenMP acrescenta -fopenmp; NVCC
-O3 -std=c++14 e -arch quando CUDA_ARCH e informado. CUDA_HOME, CUDA_LIBDIR,
CUDA_CPPFLAGS e CUDA_LDFLAGS permitem toolkits fora do caminho usual. Os C
compartilhados sao compilados por GCC; por padrao g++ vincula esses objetos e o
objeto do NVCC com libcudart dinamica, opcao compativel com o toolkit isolado do
experimento. Uma instalacao completa tambem pode usar
`CUDA_LINKER=nvcc CUDA_LDLIBS=-lm`.

Testes CUDA cobrem: dois grupos separados, K=1, cluster vazio, empate, N menor
que um bloco, N=257, parametros invalidos, inicializacao deterministica,
centroides finitos, labels validos e comparacao com uma referencia CPU local.
Um caso K=64,D=1024 exige cerca de 513 KiB por bloco e exercita explicitamente
o fallback de acumulacao global, inclusive com inercia zero.
Os testes foram compilados por nvcc 11.5 e executados na GTX 1660 Ti: OK. O
compute-sanitizer 11.5 do pacote isolado foi tentado, mas nao conseguiu injetar
no processo sob este WSL/driver (`Target application terminated before first
instrumented API call`). Portanto nao se afirma uma aprovacao do sanitizador;
essa verificacao deve ser repetida em uma instalacao completa compativel.

6. EXECUCAO
-----------
Parametros: <arquivo> <K> <iteracoes> <seed>.

  ./bin/kmeans_seq data/covtype.data 64 50 42
  OMP_NUM_THREADS=8 ./bin/kmeans_omp data/covtype.data 64 50 42
  ./bin/kmeans_cuda data/covtype.data 64 50 42

Saida: mode, n, d, k, iters, seed, time_s e inertia. OpenMP tambem imprime o
numero efetivo de threads. time_s mede somente kmeans_run; no CUDA inclui
cudaMalloc, HtoD, kernels, sincronizacao e DtoH. Leitura/normalizacao e SSE final
ficam fora para todas as versoes. Tempo total:

  /usr/bin/time -p ./bin/kmeans_cuda data/covtype.data 64 50 42

Benchmarks reproduziveis (cada execucao cria um `.txt` com timestamp; use
BENCH_OUT para escolher outro arquivo sem sobrescrever resultados existentes):

  scripts/benchmark.sh data/covtype.data 64 50 42 3
  scripts/benchmark_cuda.sh data/covtype.data 64 50 42 3

O benchmark OpenMP compara `OMP_NUM_THREADS` solicitado com
`omp_get_num_threads()` capturado dentro da regiao `parallel` que contem o
`for` real. Ja
`omp_get_max_threads()` fora da regiao informa apenas o limite da proxima
equipe e nao comprova quantas threads executaram o algoritmo. Todas as
repeticoes, e nao apenas a mediana, tem a inercia validada; se a referencia for
zero, somente zero e aceito.

7. RESULTADOS EXPERIMENTAIS REAIS
---------------------------------
Experimento A — Arthur, 2026-11-25:
Intel Core i5-12400, 6 nucleos/12 threads, 32 GB, Fedora 44,
GCC 16.2.1. Covertype completo, K=64, I=50, seed=42; mediana de tres.

  Versao       time_s   Speedup  Eficiencia
  Sequencial    54.001     1.00       -
  OpenMP 1      54.140     1.00      1.00
  OpenMP 2      27.073     1.99      1.00
  OpenMP 4      13.729     3.93      0.98
  OpenMP 8      11.910     4.53      0.57
  OpenMP 16     10.008     5.40      0.34
  OpenMP 32      9.249     5.84      0.18

Experimento B — Gustavo, 2026-11-25 (maquina diferente):
Intel Core i7-10750H, 6 nucleos/12 threads, 15 GiB, WSL2;
GTX 1660 Ti 6 GiB, CC 7.5, driver 621.24; GCC 11.4, CUDA Toolkit 11.5
extraido sem privilegios. Mesmos dados/parametros; mediana de tres.

  Versao       time_s   real_s  Speedup   tempos time_s individuais
  Sequencial   147.170   156.35    1.00    147.588 147.170 146.845
  CUDA          13.068    23.52   11.26     11.013  13.068  14.529

Inercia em todas as configuracoes dos dois experimentos: 120313.163175.
Diferenca relativa CUDA versus sequencial: 0.00e+00. O speedup CUDA usa somente
o sequencial do experimento B; os 54.001 s do experimento A nao entram nessa
divisao. Evidencias integrais estao em results/benchmark_omp.txt e
results/benchmark_cuda.txt.

8. ANALISE CRITICA E LIMITACOES
-------------------------------
OpenMP escala quase linear ate quatro threads. Aos seis nucleos fisicos, a parte
paralela se aproxima do limite do hardware; hyper-threading, oversubscription,
reducao de K*D acumuladores, largura de banda e a parte sequencial reduzem a
eficiencia acima disso, conforme a Lei de Amdahl.

CUDA explora milhares de threads e reutiliza centroides pelo cache, mantendo os
dados residentes. A reducao parcial evita atomicas globais para cada coordenada,
produzindo speedup 11.26 mesmo com FP64, que e relativamente fraco na GTX 1660
Ti. A variacao 11.013–14.529 s e compativel com um notebook compartilhando GPU,
temperatura e frequencia; por isso foi usada mediana. O tempo total ainda inclui
cerca de dez segundos de parsing/normalizacao CPU, mas a comparacao principal
time_s segue o contrato e inclui todos os custos CUDA internos.

Limitacoes: K deve caber em N; a implementacao faz exatamente I iteracoes, sem
criterio de convergencia antecipada; a inicializacao aleatoria simples pode
chegar a minimos locais; FP64 e atomicas podem ser lentos em GPUs de consumo;
K*D grande usa o fallback global; e os labels tem a semantica explicada na secao
2. Nao foram testados tamanhos de bloco diferentes, pois o enunciado nao exige.

9. VALIDACAO DE CORRETUDE
-------------------------
O mesmo loader, inicializacao, seed, desempate, cluster vazio, iteracoes e
precisao sao usados. Testes sinteticos conferem valores e bordas contra CPU.
No dataset real, todas as repeticoes CPU, OpenMP e CUDA produziram a mesma
inercia; a tolerancia automatica e 1%, embora a diferenca observada tenha sido
zero. Compilacao isolada nao foi tratada como prova: test_cuda e o benchmark
foram executados em GPU real. Nenhum numero foi simulado ou estimado como
medicao.

Para Google Colab: selecionar Runtime > Change runtime type > GPU, clonar o
repositorio, executar os comandos de download acima, conferir nvidia-smi e
nvcc --version, usar make cuda test_cuda com CUDA_ARCH adequado a GPU e executar
scripts/benchmark_cuda.sh. Registrar tambem um sequencial nessa mesma VM;
resultados de sessoes ou maquinas diferentes nao devem formar um speedup.

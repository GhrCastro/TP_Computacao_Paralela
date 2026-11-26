/* K-means de Lloyd com OpenMP.
 * Compilacao: gcc -O3 -std=c11 -fopenmp. time_s mede so kmeans_run (sem
 * leitura, normalizacao ou SSE final). Gerado por scripts/benchmark.sh
 * (resumo completo em results/benchmark_omp.txt).
 *
 * Tempos reais medidos em 2026-11-25, Covertype completo (n=581012, d=54),
 * K=64, iters=50, seed=42, mediana de 3 execucoes:
 *   Hardware: Intel Core i5-12400 (6 nucleos fisicos, 12 logicos), 32 GB,
 *             Linux 7.3.3 (Fedora 44), GCC 16.2.1.
 *
 *   threads   time_s   speedup   eficiencia
 *   seq       54.001      -          -
 *   1         54.140    1.00       1.00
 *   2         27.073    1.99       1.00
 *   4         13.729    3.93       0.98
 *   8         11.910    4.53       0.57
 *   16        10.008    5.40       0.34
 *   32         9.249    5.84       0.18
 *   Inercia 120313.163175 identica ao sequencial em todas as configuracoes.
 *   Experimento CUDA separado: GTX 1660 Ti, seq=147.170 s, CUDA=13.068 s,
 *   speedup=11.26 (results/benchmark_cuda.txt). Nao misturar as maquinas.
 *   Ate 4 threads o ganho e quase linear. Acima de 6 threads os nucleos
 *   fisicos acabam: 8 threads ja usa hyper-threading e 16/32 sao
 *   oversubscription (mais threads que os 12 logicos), por isso a eficiencia
 *   cai. O pequeno ganho ate 32 threads foi medido, nao garantido.
 *
 * Onde paralelizamos:
 *   Apenas o laco sobre os N pontos (atribuicao + acumulacao). Cada ponto
 *   e independente dos outros dentro de uma iteracao, entao o laco e dividido
 *   em blocos contiguos de pontos, um por thread (schedule(static)).
 *
 * Como evitamos data races:
 *   - labels[i]: cada i e escrito por exatamente uma thread.
 *   - sums/counts: clausula reduction(+: sums[:kd], counts[:k]). O OpenMP da a
 *     cada thread copias privadas zeradas (equivalente a local_sums[T][K*D] e
 *     local_counts[T][K]) e soma essas copias nos vetores originais uma unica
 *     vez, no fim do laco. Nao ha atomic/critical no laco quente, e como cada
 *     thread escreve na sua propria copia nao ha false sharing.
 *   - centroids: so e lido durante o laco paralelo; a atualizacao (K*D
 *     operacoes, desprezivel) acontece depois, fora da regiao paralela.
 *
 * O que NAO paralelizamos:
 *   As iteracoes do K-means: a iteracao t+1 depende dos centroides de t.
 *
 * Resultado vs sequencial: distancias, desempate (menor indice) e cluster
 * vazio sao identicos ao seq.c. Unica diferenca: a ordem em que as somas
 * parciais em double sao combinadas, o que pode mudar os ultimos bits dos
 * centroides (diferenca de inercia esperada << 1%).
 */
#include "kmeans.h"
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

int kmeans_run(const Dataset *data,int k,int iters,float *centroids,int *labels) {
    if (!data || !data->points || !centroids || !labels || !data->n ||
        data->d<=0 || k<=0 || iters<=0 || (size_t)k>data->n ||
        (size_t)k>SIZE_MAX/(size_t)data->d/sizeof(double)) return 1;
    const size_t n=data->n, kd=(size_t)k*data->d;
    const int d=data->d;
    const float *points=data->points;
    double *sums=calloc(kd,sizeof *sums);
    size_t *counts=calloc((size_t)k,sizeof *counts);
    if (!sums || !counts) { free(sums); free(counts); return 1; }
    for(int t=0;t<iters;++t) {
        memset(sums,0,kd*sizeof *sums);
        memset(counts,0,(size_t)k*sizeof *counts);
        /* Pontos divididos entre threads; somas/contagens reduzidas no fim. */
        #pragma omp parallel for schedule(static) reduction(+: sums[:kd], counts[:k])
        for(size_t i=0;i<n;++i) {
            const float *p=points+i*d;
            double best=INFINITY;
            int label=0;
            for(int c=0;c<k;++c) {
                double dist=0;
                for(int j=0;j<d;++j) {
                    double delta=(double)p[j]-centroids[(size_t)c*d+j];
                    dist+=delta*delta;
                }
                /* Comparacao estrita preserva menor indice no empate. */
                if(dist<best) { best=dist; label=c; }
            }
            labels[i]=label;
            ++counts[label];
            for(int j=0;j<d;++j) sums[(size_t)label*d+j]+=p[j];
        }
        /* Sequencial: K*D operacoes. Clusters vazios mantem o centroide. */
        for(int c=0;c<k;++c)
            if(counts[c])
                for(int j=0;j<d;++j)
                    centroids[(size_t)c*d+j]=(float)(sums[(size_t)c*d+j]/counts[c]);
    }
    free(sums); free(counts); return 0;
}

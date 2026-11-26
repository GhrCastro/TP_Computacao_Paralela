/* K-means sequencial de Lloyd.
 * Tempos CPU reais do experimento de Arthur (2026-11-25, i5-12400 6C/12T,
 * 32 GB, Fedora 44, GCC 16.2.1, Covertype, K=64, 50 iteracoes, seed=42,
 * mediana de 3; time_s nao inclui leitura, normalizacao nem SSE final):
 *   seq=54.001 s; OpenMP 1=54.140, 2=27.073, 4=13.729, 8=11.910,
 *   16=10.008, 32=9.249 s. Inercia: 120313.163175 em todas.
 * Uma medicao independente CPU/CUDA consta em results/benchmark_cuda.txt.
 * Compilacao sequencial: gcc -O3.
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
    size_t kd=(size_t)k*data->d;
    double *sums=calloc(kd,sizeof *sums);
    size_t *counts=calloc((size_t)k,sizeof *counts);
    if (!sums || !counts) { free(sums); free(counts); return 1; }
    for(int t=0;t<iters;++t) {
        memset(sums,0,kd*sizeof *sums);
        memset(counts,0,(size_t)k*sizeof *counts);
        for(size_t i=0;i<data->n;++i) {
            const float *p=data->points+i*data->d;
            double best=INFINITY;
            int label=0;
            for(int c=0;c<k;++c) {
                double dist=0;
                for(int j=0;j<data->d;++j) {
                    double delta=(double)p[j]-centroids[(size_t)c*data->d+j];
                    dist+=delta*delta;
                }
                /* Comparacao estrita preserva menor indice no empate. */
                if(dist<best) { best=dist; label=c; }
            }
            labels[i]=label;
            ++counts[label];
            for(int j=0;j<data->d;++j) sums[(size_t)label*data->d+j]+=p[j];
        }
        /* Clusters vazios mantem o centroide anterior. */
        for(int c=0;c<k;++c)
            if(counts[c])
                for(int j=0;j<data->d;++j)
                    centroids[(size_t)c*data->d+j]=(float)(sums[(size_t)c*data->d+j]/counts[c]);
    }
    free(sums); free(counts); return 0;
}

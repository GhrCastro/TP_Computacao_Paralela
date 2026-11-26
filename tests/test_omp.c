/* Testes do K-means OpenMP (src/omp.c).
 * 1) Mesmos casos de tests/test_seq.c, repetidos com 1..32 threads.
 * 2) Dataset sintetico (N=1001, nao multiplo de nenhum numero de threads):
 *    labels identicos e inercia com diferenca relativa <= 1e-6 entre 1 e T
 *    threads. Compilar: gcc -O3 -fopenmp -Iinclude tests/test_omp.c
 *    src/common.c src/omp.c -lm
 */
#include "kmeans.h"
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void small_cases(void) {
    float p[]={0,0,0,2,10,10,10,12};
    Dataset d={4,2,p};
    int labels[4];
    /* Dois grupos separados. */
    float c[]={0,0,10,10};
    assert(kmeans_run(&d,2,2,c,labels)==0);
    assert(c[0]==0 && c[1]==1 && c[2]==10 && c[3]==11);
    assert(labels[0]==0 && labels[1]==0 && labels[2]==1 && labels[3]==1);
    assert(fabs(calculate_inertia(&d,c,2)-4)<1e-9);
    /* K=1: centroide vira a media de todos. */
    assert(kmeans_run(&d,1,1,c,labels)==0);
    assert(c[0]==5 && c[1]==6);
    /* Cluster vazio mantem o centroide anterior. */
    float empty[]={0,0,100,100};
    assert(kmeans_run(&d,2,1,empty,labels)==0);
    assert(empty[2]==100 && empty[3]==100);
    /* Empate: ponto (1,0) equidistante vai para o menor indice. */
    float tiep[]={0,0,2,0,1,0}, tiec[]={0,0,2,0};
    Dataset tie={3,2,tiep}; int tl[3];
    assert(kmeans_run(&tie,2,1,tiec,tl)==0 && tl[2]==0);
    /* Parametros invalidos. */
    assert(kmeans_run(&d,5,1,c,labels)!=0);
    assert(kmeans_run(&d,2,0,c,labels)!=0);
}

/* Roda K-means no dataset sintetico com T threads. */
static double run_with(int threads,const Dataset *data,int k,float *cent,int *labels) {
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    assert(init_centroids(data,k,42,cent)==0);
    assert(kmeans_run(data,k,10,cent,labels)==0);
    assert(kmeans_omp_last_thread_count()==threads);
    for(size_t i=0;i<(size_t)k*data->d;++i) assert(isfinite(cent[i]));
    return calculate_inertia(data,cent,k);
}

int main(void) {
    const int threads[]={1,2,4,8,16,32};
    for(int t=0;t<6;++t) { omp_set_num_threads(threads[t]); small_cases(); }

    /* PRNG simples (LCG) so para gerar dados de teste. */
    enum { N=1001, D=5, K=7 };
    float *pts=malloc(sizeof(float)*N*D);
    unsigned s=12345;
    for(int i=0;i<N*D;++i) { s=s*1103515245u+12345u; pts[i]=(float)(s>>8)/(1<<24); }
    Dataset data={N,D,pts};
    float c1[K*D], ct[K*D];
    int l1[N], lt[N];
    double ref=run_with(1,&data,K,c1,l1);
    for(int t=1;t<6;++t) {
        double sse=run_with(threads[t],&data,K,ct,lt);
        assert(fabs(sse-ref)/ref<=1e-6);
        assert(!memcmp(l1,lt,sizeof l1));
    }
    free(pts);
    puts("Testes OpenMP OK (1,2,4,8,16,32 threads)");
    return 0;
}

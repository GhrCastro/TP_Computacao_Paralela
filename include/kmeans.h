#ifndef KMEANS_H
#define KMEANS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t n;       /* numero de pontos */
    int d;          /* dimensoes por ponto (Covertype: 54) */
    float *points;  /* layout row-major: points[i*d + j] */
} Dataset;

/* Felipe: dados reais CSV, normalizacao comum e memoria */
int load_dataset(const char *path, Dataset *out);
void free_dataset(Dataset *data);

/* Felipe: seleciona k pontos distintos com PRNG reproduzivel */
int init_centroids(const Dataset *data, int k, unsigned seed, float *centroids);

/* Felipe: soma das distancias quadraticas aos centroides mais proximos */
double calculate_inertia(const Dataset *data, const float *centroids, int k);

/* Implementada individualmente em seq.c, omp.c OU cuda.cu:
   centroids ja inicializados, atualizacao in-place;
   labels possui data->n elementos; 0 sucesso, !=0 erro. */
int kmeans_run(const Dataset *data, int k, int iters,
               float *centroids, int *labels);

#ifdef _OPENMP
/* Numero real de threads da ultima regiao paralela executada por omp.c. */
int kmeans_omp_last_thread_count(void);
#endif

#ifdef __cplusplus
}
#endif

#endif /* KMEANS_H */

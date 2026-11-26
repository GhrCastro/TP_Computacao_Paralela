/* Testes de corretude do backend CUDA. Executar somente em GPU NVIDIA real. */
#include "kmeans.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int reference(const Dataset *data, int k, int iters, float *centroids,
                     int *labels) {
    const size_t kd = (size_t)k * data->d;
    double *sums = (double *)calloc(kd, sizeof *sums);
    size_t *counts = (size_t *)calloc((size_t)k, sizeof *counts);
    if (!sums || !counts) { free(sums); free(counts); return 1; }
    for (int t = 0; t < iters; ++t) {
        memset(sums, 0, kd * sizeof *sums);
        memset(counts, 0, (size_t)k * sizeof *counts);
        for (size_t i = 0; i < data->n; ++i) {
            double best = INFINITY;
            int label = 0;
            for (int c = 0; c < k; ++c) {
                double distance = 0.0;
                for (int j = 0; j < data->d; ++j) {
                    double delta = (double)data->points[i * data->d + j] -
                                   centroids[(size_t)c * data->d + j];
                    distance += delta * delta;
                }
                if (distance < best) { best = distance; label = c; }
            }
            labels[i] = label;
            ++counts[label];
            for (int j = 0; j < data->d; ++j)
                sums[(size_t)label * data->d + j] +=
                    data->points[i * data->d + j];
        }
        for (int c = 0; c < k; ++c)
            if (counts[c])
                for (int j = 0; j < data->d; ++j)
                    centroids[(size_t)c * data->d + j] =
                        (float)(sums[(size_t)c * data->d + j] / counts[c]);
    }
    free(sums); free(counts); return 0;
}

static void compare_case(const Dataset *data, int k, int iters,
                         const float *initial) {
    const size_t kd = (size_t)k * data->d;
    float *cpu = (float *)malloc(kd * sizeof *cpu);
    float *gpu = (float *)malloc(kd * sizeof *gpu);
    int *cpu_labels = (int *)malloc(data->n * sizeof *cpu_labels);
    int *gpu_labels = (int *)malloc(data->n * sizeof *gpu_labels);
    assert(cpu && gpu && cpu_labels && gpu_labels);
    memcpy(cpu, initial, kd * sizeof *cpu);
    memcpy(gpu, initial, kd * sizeof *gpu);
    assert(reference(data, k, iters, cpu, cpu_labels) == 0);
    assert(kmeans_run(data, k, iters, gpu, gpu_labels) == 0);
    for (size_t q = 0; q < kd; ++q) {
        assert(isfinite(gpu[q]));
        assert(fabs((double)gpu[q] - cpu[q]) <= 1e-5);
    }
    for (size_t i = 0; i < data->n; ++i) {
        assert(gpu_labels[i] >= 0 && gpu_labels[i] < k);
        assert(gpu_labels[i] == cpu_labels[i]);
    }
    const double cpu_sse = calculate_inertia(data, cpu, k);
    const double gpu_sse = calculate_inertia(data, gpu, k);
    assert(isfinite(gpu_sse));
    assert(cpu_sse == 0.0 ? gpu_sse == 0.0
                          : fabs(gpu_sse - cpu_sse) / cpu_sse <= 1e-6);
    free(gpu_labels); free(cpu_labels); free(gpu); free(cpu);
}

int main(void) {
    /* N menor que um bloco; dois grupos, K=1 e cluster vazio. */
    float points[] = {0,0, 0,2, 10,10, 10,12};
    Dataset small = {4, 2, points};
    float separated[] = {0,0, 10,10};
    compare_case(&small, 2, 2, separated);
    float one[] = {0,0};
    compare_case(&small, 1, 1, one);
    float empty[] = {0,0, 100,100};
    compare_case(&small, 2, 1, empty);

    /* Empate deve escolher o menor indice. */
    float tie_points[] = {0,0, 2,0, 1,0};
    float tie_centroids[] = {0,0, 2,0};
    Dataset tie = {3, 2, tie_points};
    compare_case(&tie, 2, 1, tie_centroids);
    int tie_labels[3];
    assert(kmeans_run(&tie, 2, 1, tie_centroids, tie_labels) == 0);
    assert(tie_labels[2] == 0);

    /* N=257 cobre o ultimo bloco parcial (N nao multiplo de 256). */
    enum { N = 257, D = 5, K = 7 };
    float *many = (float *)malloc((size_t)N * D * sizeof *many);
    float initial[K * D], initial_again[K * D];
    int labels[N];
    assert(many);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < D; ++j)
            many[(size_t)i * D + j] = (float)(i % K) * 10.0f + j * 0.01f;
    Dataset nonmultiple = {N, D, many};
    assert(init_centroids(&nonmultiple, K, 42, initial) == 0);
    assert(init_centroids(&nonmultiple, K, 42, initial_again) == 0);
    assert(memcmp(initial, initial_again, sizeof initial) == 0);
    compare_case(&nonmultiple, K, 5, initial);

    /* Parametros invalidos devem falhar antes de acessar a GPU. */
    assert(kmeans_run(NULL, 1, 1, initial, labels) != 0);
    assert(kmeans_run(&nonmultiple, 0, 1, initial, labels) != 0);
    assert(kmeans_run(&nonmultiple, K, 0, initial, labels) != 0);
    assert(kmeans_run(&nonmultiple, N + 1, 1, initial, labels) != 0);
    free(many);
    puts("Testes CUDA OK");
    return 0;
}

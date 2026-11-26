/* K-means de Lloyd em CUDA.
 *
 * Tempo CUDA real: PENDENTE. O driver enxerga uma GTX 1660 Ti (CC 7.5), mas
 * o CUDA Toolkit/nvcc nao esta instalado no ambiente desta implementacao.
 * Nao foi inventada medicao. Consultar results/benchmark_cuda.txt.
 *
 * Paralelizacao:
 *   - MAP: uma thread associa cada ponto ao centroide mais proximo;
 *   - REDUCE: cada bloco acumula somas/contagens em memoria compartilhada e
 *     grava um parcial; kernels posteriores combinam os parciais;
 *   - UPDATE: uma thread por coordenada atualiza os centroides nao vazios.
 * Pontos, centroides e labels permanecem no device em todas as iteracoes.
 * As iteracoes nao sao paralelas entre si: t+1 depende dos centroides de t.
 * Distancias e acumuladores usam double, como a referencia sequencial.
 */
#include "kmeans.h"
#include <cuda_runtime.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

enum { THREADS_PER_BLOCK = 256 };

/* Caminho principal: atomicas ficam restritas ao bloco. Cada bloco produz um
 * vetor parcial independente, portanto nao ha corrida entre blocos. */
__global__ static void assign_partial_kernel(
    const float *points, size_t n, int d, int k, const float *centroids,
    int *labels, double *partial_sums, unsigned long long *partial_counts) {
    extern __shared__ unsigned char shared_raw[];
    const size_t kd = (size_t)k * d;
    double *sums = reinterpret_cast<double *>(shared_raw);
    unsigned long long *counts =
        reinterpret_cast<unsigned long long *>(sums + kd);

    for (size_t q = threadIdx.x; q < kd; q += blockDim.x) sums[q] = 0.0;
    for (int c = threadIdx.x; c < k; c += blockDim.x) counts[c] = 0;
    __syncthreads();

    const size_t first = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = first; i < n; i += stride) {
        const float *point = points + i * (size_t)d;
        double best = INFINITY;
        int label = 0;
        for (int c = 0; c < k; ++c) {
            double distance = 0.0;
            const float *centroid = centroids + (size_t)c * d;
            for (int j = 0; j < d; ++j) {
                const double delta = (double)point[j] - centroid[j];
                distance += delta * delta;
            }
            /* Comparacao estrita preserva o menor indice em caso de empate. */
            if (distance < best) {
                best = distance;
                label = c;
            }
        }
        labels[i] = label;
        atomicAdd(counts + label, 1ULL);
        for (int j = 0; j < d; ++j)
            atomicAdd(sums + (size_t)label * d + j, (double)point[j]);
    }
    __syncthreads();

    double *block_sums = partial_sums + (size_t)blockIdx.x * kd;
    unsigned long long *block_counts =
        partial_counts + (size_t)blockIdx.x * k;
    for (size_t q = threadIdx.x; q < kd; q += blockDim.x)
        block_sums[q] = sums[q];
    for (int c = threadIdx.x; c < k; c += blockDim.x)
        block_counts[c] = counts[c];
}

/* Fallback para K*D grande demais para a shared memory do dispositivo. */
__global__ static void assign_global_kernel(
    const float *points, size_t n, int d, int k, const float *centroids,
    int *labels, double *sums, unsigned long long *counts) {
    const size_t first = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = first; i < n; i += stride) {
        const float *point = points + i * (size_t)d;
        double best = INFINITY;
        int label = 0;
        for (int c = 0; c < k; ++c) {
            double distance = 0.0;
            const float *centroid = centroids + (size_t)c * d;
            for (int j = 0; j < d; ++j) {
                const double delta = (double)point[j] - centroid[j];
                distance += delta * delta;
            }
            if (distance < best) {
                best = distance;
                label = c;
            }
        }
        labels[i] = label;
        atomicAdd(counts + label, 1ULL);
        for (int j = 0; j < d; ++j)
            atomicAdd(sums + (size_t)label * d + j, (double)point[j]);
    }
}

__global__ static void reduce_counts_kernel(
    const unsigned long long *partials, int blocks, int k,
    unsigned long long *totals) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) return;
    unsigned long long total = 0;
    for (int b = 0; b < blocks; ++b) total += partials[(size_t)b * k + c];
    totals[c] = total;
}

__global__ static void reduce_update_kernel(
    const double *partials, int blocks, int d, size_t kd,
    const unsigned long long *counts, float *centroids) {
    const size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= kd) return;
    double sum = 0.0;
    for (int b = 0; b < blocks; ++b) sum += partials[(size_t)b * kd + q];
    const unsigned long long count = counts[q / (size_t)d];
    /* Cluster vazio: nao escreve, preservando exatamente o centroide anterior. */
    if (count) centroids[q] = (float)(sum / (double)count);
}

__global__ static void update_kernel(
    const double *sums, const unsigned long long *counts, int d, size_t kd,
    float *centroids) {
    const size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= kd) return;
    const unsigned long long count = counts[q / (size_t)d];
    if (count) centroids[q] = (float)(sums[q] / (double)count);
}

extern "C" int kmeans_run(const Dataset *data, int k, int iters,
                           float *centroids, int *labels) {
    float *d_points = NULL, *d_centroids = NULL;
    int *d_labels = NULL;
    double *d_sums = NULL;
    unsigned long long *d_counts = NULL;
    cudaDeviceProp prop;
    cudaError_t error;
    size_t n, kd, points_bytes, centroids_bytes, labels_bytes;
    size_t sums_bytes, counts_bytes, shared_bytes;
    size_t requested_blocks, partial_sum_items, partial_count_items;
    size_t update_blocks_size, partial_counts_bytes;
    int device = 0, blocks = 0, update_blocks = 0, count_blocks = 0;
    int use_partials = 0, rc = 1;

    if (!data || !data->points || !centroids || !labels || !data->n ||
        data->d <= 0 || k <= 0 || iters <= 0 || (size_t)k > data->n)
        return 1;
    n = data->n;
    if ((size_t)data->d > SIZE_MAX / n ||
        n * (size_t)data->d > SIZE_MAX / sizeof(float) ||
        (size_t)k > SIZE_MAX / (size_t)data->d ||
        (size_t)k * (size_t)data->d > SIZE_MAX / sizeof(double) ||
        n > SIZE_MAX / sizeof(int))
        return 1;
    kd = (size_t)k * data->d;
    points_bytes = n * (size_t)data->d * sizeof(float);
    centroids_bytes = kd * sizeof(float);
    labels_bytes = n * sizeof(int);
    sums_bytes = kd * sizeof(double);
    counts_bytes = (size_t)k * sizeof(unsigned long long);
    if (sums_bytes > SIZE_MAX - counts_bytes) return 1;
    shared_bytes = sums_bytes + counts_bytes;

#define CUDA_TRY(call) do {                                                     \
    error = (call);                                                             \
    if (error != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA: %s falhou: %s\n", #call,                      \
                cudaGetErrorString(error));                                     \
        goto cleanup;                                                           \
    }                                                                           \
} while (0)
#define CUDA_LAUNCH_CHECK() CUDA_TRY(cudaGetLastError())

    CUDA_TRY(cudaGetDevice(&device));
    CUDA_TRY(cudaGetDeviceProperties(&prop, device));
    if (prop.major < 6) {
        fprintf(stderr, "CUDA: compute capability >= 6.0 e necessaria para "
                        "atomicAdd(double); detectada %d.%d\n", prop.major,
                prop.minor);
        goto cleanup;
    }
    requested_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    /* O grid-stride loop permite limitar parciais e ainda cobrir todo N. */
    if (requested_blocks > (size_t)prop.multiProcessorCount * 8)
        requested_blocks = (size_t)prop.multiProcessorCount * 8;
    if (requested_blocks > (size_t)prop.maxGridSize[0])
        requested_blocks = (size_t)prop.maxGridSize[0];
    if (!requested_blocks || requested_blocks > INT_MAX) goto cleanup;
    blocks = (int)requested_blocks;
    if (kd > SIZE_MAX - (THREADS_PER_BLOCK - 1)) goto cleanup;
    update_blocks_size = (kd + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (!update_blocks_size || update_blocks_size > INT_MAX ||
        update_blocks_size > (size_t)prop.maxGridSize[0])
        goto cleanup;
    update_blocks = (int)update_blocks_size;
    count_blocks = (k + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    CUDA_TRY(cudaMalloc((void **)&d_points, points_bytes));
    CUDA_TRY(cudaMalloc((void **)&d_centroids, centroids_bytes));
    CUDA_TRY(cudaMalloc((void **)&d_labels, labels_bytes));
    CUDA_TRY(cudaMemcpy(d_points, data->points, points_bytes,
                        cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(d_centroids, centroids, centroids_bytes,
                        cudaMemcpyHostToDevice));

    use_partials = shared_bytes <= (size_t)prop.sharedMemPerBlock;
    if (use_partials) {
        if ((size_t)blocks > SIZE_MAX / kd ||
            (partial_sum_items = (size_t)blocks * kd) > SIZE_MAX / sizeof(double) ||
            (size_t)blocks > SIZE_MAX / (size_t)k ||
            (partial_count_items = (size_t)blocks * k) >
                SIZE_MAX / sizeof(unsigned long long) ||
            partial_count_items * sizeof(unsigned long long) >
                SIZE_MAX - counts_bytes)
            goto cleanup;
        partial_counts_bytes =
            partial_count_items * sizeof(unsigned long long) + counts_bytes;
        CUDA_TRY(cudaMalloc((void **)&d_sums,
                            partial_sum_items * sizeof(double)));
        CUDA_TRY(cudaMalloc((void **)&d_counts, partial_counts_bytes));
    } else {
        CUDA_TRY(cudaMalloc((void **)&d_sums, sums_bytes));
        CUDA_TRY(cudaMalloc((void **)&d_counts, counts_bytes));
    }

    for (int iteration = 0; iteration < iters; ++iteration) {
        if (use_partials) {
            unsigned long long *totals = d_counts + (size_t)blocks * k;
            assign_partial_kernel<<<blocks, THREADS_PER_BLOCK, shared_bytes>>>(
                d_points, n, data->d, k, d_centroids, d_labels, d_sums,
                d_counts);
            CUDA_LAUNCH_CHECK();
            reduce_counts_kernel<<<count_blocks, THREADS_PER_BLOCK>>>(
                d_counts, blocks, k, totals);
            CUDA_LAUNCH_CHECK();
            reduce_update_kernel<<<update_blocks, THREADS_PER_BLOCK>>>(
                d_sums, blocks, data->d, kd, totals, d_centroids);
            CUDA_LAUNCH_CHECK();
        } else {
            CUDA_TRY(cudaMemset(d_sums, 0, sums_bytes));
            CUDA_TRY(cudaMemset(d_counts, 0, counts_bytes));
            assign_global_kernel<<<blocks, THREADS_PER_BLOCK>>>(
                d_points, n, data->d, k, d_centroids, d_labels, d_sums,
                d_counts);
            CUDA_LAUNCH_CHECK();
            update_kernel<<<update_blocks, THREADS_PER_BLOCK>>>(
                d_sums, d_counts, data->d, kd, d_centroids);
            CUDA_LAUNCH_CHECK();
        }
    }
    CUDA_TRY(cudaDeviceSynchronize());
    CUDA_TRY(cudaMemcpy(centroids, d_centroids, centroids_bytes,
                        cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(labels, d_labels, labels_bytes, cudaMemcpyDeviceToHost));
    rc = 0;

cleanup:
    cudaFree(d_counts);
    cudaFree(d_sums);
    cudaFree(d_labels);
    cudaFree(d_centroids);
    cudaFree(d_points);
    return rc;
#undef CUDA_LAUNCH_CHECK
#undef CUDA_TRY
}

#define _POSIX_C_SOURCE 200809L
#include "kmeans.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <ctype.h>

void free_dataset(Dataset *data) {
    if (data) { free(data->points); *data = (Dataset){0}; }
}

int load_dataset(const char *path, Dataset *out) {
    if (!path || !out) return 1;
    *out = (Dataset){0};
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return 1; }
    char *line = NULL;
    size_t len = 0, cap = 0, n = 0;
    float *points = NULL, lo[10], hi[10];
    ssize_t bytes;
    while ((bytes = getline(&line, &len, f)) >= 0) {
        if (memchr(line, 0, (size_t)bytes)) goto fail;
        float row[54];
        char *p = line;
        for (int j = 0; j < 55; ++j) {
            char *end;
            errno = 0;
            float x = strtof(p, &end);
            if (end == p || errno || !isfinite(x)) goto fail;
            if (j >= 10 && j < 54 && x != 0 && x != 1) goto fail;
            if (j == 54 && (x < 1 || x > 7 || floorf(x) != x)) goto fail;
            if (j < 54) row[j] = x;
            p = end;
            if (j < 54) { if (*p != ',') goto fail; ++p; }
            else { while (isspace((unsigned char)*p)) ++p; if (*p) goto fail; }
        }
        if (n == cap) {
            size_t next = cap ? cap + cap/2 : 4096;
            if (next <= cap || next > SIZE_MAX / 54 / sizeof(float)) goto fail;
            float *tmp = realloc(points, next * 54 * sizeof(float));
            if (!tmp) goto fail;
            points = tmp; cap = next;
        }
        memcpy(points + n*54, row, sizeof row);
        for (int j=0;j<10;++j) {
            if (!n || row[j]<lo[j]) lo[j]=row[j];
            if (!n || row[j]>hi[j]) hi[j]=row[j];
        }
        ++n;
    }
    if (ferror(f) || !feof(f) || !n) goto fail;
    for (size_t i=0;i<n;++i)
        for (int j=0;j<10;++j) {
            double range=(double)hi[j]-lo[j];
            points[i*54+j]=range ? (float)(((double)points[i*54+j]-lo[j])/range) : 0;
        }
    free(line); fclose(f);
    *out=(Dataset){n,54,points};
    return 0;
fail:
    fprintf(stderr,"%s: erro de leitura, CSV ou memoria na linha %zu\n",path,n+1);
    free(points); free(line); fclose(f); return 1;
}

/* SplitMix64: estado local, independente da libc. */
static uint64_t next_random(uint64_t *state) {
    uint64_t z=(*state+=UINT64_C(0x9e3779b97f4a7c15));
    z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);
    return z^(z>>31);
}
int init_centroids(const Dataset *data,int k,unsigned seed,float *centroids) {
    if (!data || !data->points || !centroids || data->d<=0 || k<=0 ||
        (size_t)k>data->n || data->n>SIZE_MAX/sizeof(size_t)) return 1;
    size_t *idx=malloc(data->n*sizeof *idx);
    if (!idx) return 1;
    for(size_t i=0;i<data->n;++i) idx[i]=i;
    uint64_t state=seed;
    /* Fisher-Yates parcial: indices distintos, sem reposicao. */
    for(int c=0;c<k;++c) {
        size_t i=(size_t)c, j=i+next_random(&state)%(data->n-i);
        size_t tmp=idx[i]; idx[i]=idx[j]; idx[j]=tmp;
        memcpy(centroids+i*data->d,data->points+idx[i]*data->d,
               (size_t)data->d*sizeof(float));
    }
    free(idx); return 0;
}
double calculate_inertia(const Dataset *data,const float *centroids,int k) {
    if (!data || !data->points || !centroids || k<=0 || data->d<=0) return NAN;
    double total=0;
    for(size_t i=0;i<data->n;++i) {
        double best=INFINITY;
        for(int c=0;c<k;++c) {
            double dist=0;
            for(int j=0;j<data->d;++j) {
                double delta=(double)data->points[i*data->d+j]-centroids[(size_t)c*data->d+j];
                dist+=delta*delta;
            }
            if(dist<best) best=dist;
        }
        total+=best;
    }
    return total;
}

#define _POSIX_C_SOURCE 200809L
#include "kmeans.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <string.h>

static int parse(const char *s,unsigned long max,unsigned long *out) {
    if (!s || !*s) return 1;
    for(const char *p=s;*p;++p) if(*p<'0'||*p>'9') return 1;
    char *end; errno=0;
    unsigned long v=strtoul(s,&end,10);
    if(errno||*end||v>max) return 1;
    *out=v; return 0;
}
int main(int argc,char **argv) {
    unsigned long k,iters,seed;
    if(argc!=5 || parse(argv[2],INT_MAX,&k) || !k ||
       parse(argv[3],INT_MAX,&iters) || !iters || parse(argv[4],UINT_MAX,&seed)) {
        fprintf(stderr,"Uso: %s <dataset_path> <K> <iters> <seed>\n",argv[0]); return 1;
    }
    Dataset data={0};
    if(load_dataset(argv[1],&data)) return 1;
    int rc=1;
    float *centroids=NULL; int *labels=NULL;
    if(k>data.n || k>SIZE_MAX/(size_t)data.d/sizeof(float) ||
       data.n>SIZE_MAX/sizeof(int)) goto done;
    centroids=malloc(k*(size_t)data.d*sizeof(float));
    labels=malloc(data.n*sizeof(int));
    if(!centroids||!labels||init_centroids(&data,(int)k,(unsigned)seed,centroids)) goto done;
    struct timespec a,b;
    if(clock_gettime(CLOCK_MONOTONIC,&a)) goto done;
    int status=kmeans_run(&data,(int)k,(int)iters,centroids,labels);
    if(clock_gettime(CLOCK_MONOTONIC,&b)||status) goto done;
    double elapsed=(b.tv_sec-a.tv_sec)+(b.tv_nsec-a.tv_nsec)/1e9;
    double inertia=calculate_inertia(&data,centroids,(int)k);
    if(!isfinite(inertia)) goto done;
    const char *name=strrchr(argv[0],'/');
    name=name?name+1:argv[0];
    const char *mode=!strcmp(name,"kmeans_omp")?"omp":!strcmp(name,"kmeans_cuda")?"cuda":"seq";
    printf("mode=%s\nn=%zu\nd=%d\nk=%lu\niters=%lu\nseed=%lu\ntime_s=%.9f\ninertia=%.12g\n",
           mode,data.n,data.d,k,iters,seed,elapsed,inertia);
    rc=0;
done:
    if(rc) fprintf(stderr,"Falha na execucao do K-means\n");
    free(centroids); free(labels); free_dataset(&data); return rc;
}

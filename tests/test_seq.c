#include "kmeans.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
int main(void) {
    float p[]={0,0,0,2,10,10,10,12};
    Dataset d={4,2,p};
    int labels[4];
    float c[]={0,0,10,10};
    assert(kmeans_run(&d,2,2,c,labels)==0);
    assert(c[0]==0 && c[1]==1 && c[2]==10 && c[3]==11);
    assert(fabs(calculate_inertia(&d,c,2)-4)<1e-9);
    assert(kmeans_run(&d,1,1,c,labels)==0);
    assert(c[0]==5 && c[1]==6);
    float empty[]={0,0,100,100};
    assert(kmeans_run(&d,2,1,empty,labels)==0);
    assert(empty[2]==100 && empty[3]==100);
    float a[8],b[8];
    assert(init_centroids(&d,4,42,a)==0);
    assert(init_centroids(&d,4,42,b)==0);
    assert(!memcmp(a,b,sizeof a));
    for(int i=0;i<4;++i) for(int j=0;j<i;++j)
        assert(a[2*i]!=a[2*j] || a[2*i+1]!=a[2*j+1]);
    float tiep[]={0,0,2,0,1,0}, tiec[]={0,0,2,0};
    Dataset tie={3,2,tiep}; int tl[3];
    assert(kmeans_run(&tie,2,1,tiec,tl)==0 && tl[2]==0);
    puts("Testes sequenciais OK");
    return 0;
}

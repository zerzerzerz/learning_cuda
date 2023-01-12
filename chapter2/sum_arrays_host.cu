#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdio.h>

void sum_arrays_on_host(float *a, float *b, float *c, const int N){
    for(int i=0; i<N; ++i){
        c[i] = a[i] + b[i];
    }
}

void init_data(float *a, const int N){
    for(int i=0; i<N; ++i){
        a[i] = (float)i;
    }
}

int main(int argc, char **argv){
    int N = 4;
    int size = sizeof(float) * N;
    float *h_a, *h_b, *h_c;
    h_a = (float*) malloc(size);
    h_b = (float*) malloc(size);
    h_c = (float*) malloc(size);

    init_data(h_a, N);
    init_data(h_b, N);

    sum_arrays_on_host(h_a, h_b, h_c, N);

    for(int i=0; i<N; ++i){
        printf("%.4f\n", h_c[i]);
    }

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
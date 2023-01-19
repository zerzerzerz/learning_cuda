#include "utils.cuh"

double get_time(){
    struct timeval current_time;
    gettimeofday(&current_time, NULL);
    return ((double)current_time.tv_sec + (double)current_time.tv_usec * 1e-6);
}

int sum(int *data, int size){
    // sum an 1d array in host
    // return the sum of array
    int s = 0;
    for(int i=0; i<size; ++i){
        s += data[i];
    }
    return s;
}


void init(int *data, int size){
    for(int i=0; i<size; ++i){
        data[i] = rand() % 10;
    }
    return;
}


int is_same(int *a, int *b, int size){
    // check whether two arrays are the same
    for(int i=0; i<size; ++i){
        if(a[i] != b[i]) return 0;
    }
    return 1;
}


void add_host(int *a, int *b, int *c, int size){
    for(int i=0; i<size; ++i){
        c[i] = a[i] + b[i];
    }
}


__global__ void add_device(int *a, int *b, int *c, int size){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int id = bid * blockDim.x + tid;

    if(id < size){
        c[id] = a[id] + b[id];
    }

    return;
}
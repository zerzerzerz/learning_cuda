#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

const int N = 1<<20;


__global__ void kernel_1(){
    double sum = 0.0;
    for(int i=0; i<N; ++i){
        sum += tan(0.1);
    }
}

__global__ void kernel_2(){
    double sum = 0.0;
    for(int i=0; i<N; ++i){
        sum += tan(0.1);
    }
}

__global__ void kernel_3(){
    double sum = 0.0;
    for(int i=0; i<N; ++i){
        sum += tan(0.1);
    }
}

__global__ void kernel_4(){
    double sum = 0.0;
    for(int i=0; i<N; ++i){
        sum += tan(0.1);
    }
}

int main(){
    const size_t block = 1;
    const size_t grid = 1;
    const int num_stream = 4;
    cudaStream_t *streams = (cudaStream_t*)malloc(num_stream * sizeof(cudaStream_t));
    float elapsed_time;

    for(int i=0; i<num_stream; ++i){
        cudaStreamCreate(&streams[i]);
    }

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);


    cudaEventRecord(start);
    for(int i=0; i<num_stream; ++i){
        kernel_1<<<grid, block, 0, streams[i]>>>();
        kernel_2<<<grid, block, 0, streams[i]>>>();
        kernel_3<<<grid, block, 0, streams[i]>>>();
        kernel_4<<<grid, block, 0, streams[i]>>>();
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);


    cudaEventElapsedTime(&elapsed_time, start, end);
    printf("Elapsed time = %.6f\n", elapsed_time);


    free(streams);
    return 0;
}
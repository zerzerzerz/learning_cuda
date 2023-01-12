#include <stdio.h>
#include <cuda_runtime.h>

__global__ void nested_hello_world (int size, int depth){
    int tid = threadIdx.x;
    printf("Recuision = %d, Hello world from thread %d, block %d, blockDim.x %d\n", depth, tid, blockIdx.x, blockDim.x);

    if (size == 1) return;
    int n_threads = size >> 1;
    if (tid == 0 && n_threads > 0){
        nested_hello_world <<<1, n_threads>>> (n_threads, ++depth);
        printf("Nested execution depth: %d\n", depth);
    }
    return;
}


int main(int argc, char **argv){
    int size = 8;
    int depth = 0;
    // printf("Hello, world from CPU\n");
    if(argc == 1){
        nested_hello_world<<<1,size>>>(size, depth);
    }
    else{
        nested_hello_world<<<atoi(argv[1]),size>>>(size, depth);
    }
    cudaDeviceSynchronize();
    return 0;
}
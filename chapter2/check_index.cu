#include <cuda_runtime.h>
#include <stdio.h>

__global__ void check_index(void){
    printf("threadIdx(%d,%d,%d), blockIdx(%d,%d,%d), blockDim(%d,%d,%d), gridDim(%d,%d,%d)\n",
        threadIdx.x, threadIdx.y, threadIdx.z,
        blockIdx.x, blockIdx.y, blockIdx.z,
        blockDim.x, blockDim.y, blockDim.z,
        gridDim.x, gridDim.y, gridDim.z
    );
    return;
}


int main(){
    int n_elements = 32;
    dim3 block(4);
    dim3 grid((n_elements + block.x - 1) / block.x);

    printf("block is (%d,%d,%d)\n", block.x, block.y, block.z);
    printf("grid is (%d,%d,%d)\n", grid.x, grid.y, grid.z);
    check_index<<<grid, block>>> ();

    cudaDeviceReset();
}
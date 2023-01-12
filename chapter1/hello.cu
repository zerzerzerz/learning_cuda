#include <stdio.h>

__global__ void hello_from_gpu(void){
    printf("Hello, world from GPU!\n");
    return;
}

int main(void){
    printf("Hello, world from CPU!\n");
    hello_from_gpu<<<1,10>>>();
    cudaDeviceReset();
    return 0;
}
#include <stdio.h>
#include <cuda_runtime.h>

__device__ int a_d;

int main(){
    int *var, *var_d;
    size_t size = 4;
    size_t bytes = sizeof(int) * size;
    cudaHostAlloc(&var, bytes, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&var_d, var, 0);
    printf("Unified Virtual Address\n");
    printf("[CPU] %p\n", var);
    printf("[GPU] %p\n", var_d);

    int a_h = 1;
    int* a_p;
    cudaGetSymbolAddress((void**)&a_p, a_d);
    printf("Symbol Address\n");
    printf("[CPU] %p\n", &a_h);
    printf("[GPU] %p\n", a_p);
    printf("[Error] Use & in host to get address of device variable is %p\n", &a_d);


    cudaFreeHost(var);
    cudaDeviceReset();
    return 0;
}
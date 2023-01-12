#include <stdio.h>
#include <cuda_runtime.h>

int main(){
    cudaDeviceProp device_prop;
    int dev = 0;
    cudaGetDeviceProperties(&device_prop, dev);
    printf("%s\n", device_prop.name);
    return 0;
}
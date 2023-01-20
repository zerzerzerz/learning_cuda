#include "../utils/utils.cuh"



int main(){
    int *a, *b, *c, *d;
    auto size = 1<<20;
    int byte = size * sizeof(int);
    dim3 block(256);
    dim3 grid((size-1+block.x)/block.x);
    double s,e;

    cudaHostAlloc(&a, byte, cudaHostAllocMapped);
    cudaHostAlloc(&b, byte, cudaHostAllocMapped);
    cudaHostAlloc(&c, byte, cudaHostAllocMapped);
    cudaHostAlloc(&d, byte, cudaHostAllocMapped);

    init(a, size);
    init(b, size);

    s = get_time();
    add_device<<<grid, block>>>(a,b,c,size);
    cudaDeviceSynchronize();
    e = get_time();
    printf("[GPU] time = %.6f\n", e - s);


    s = get_time();
    add_host(a,b,d,size);
    e = get_time();
    printf("[CPU] time = %.6f\n", e - s);


    int flag = is_same(c,d,size);
    printf("Flag = %d\n", flag);

    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    cudaFreeHost(d);
    cudaDeviceReset();

    return 0;



}
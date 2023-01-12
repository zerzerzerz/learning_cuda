#include <stdio.h>

#define check_error(call) \
{\
    const cudaError_t res = call; \
    if (res != cudaSuccess){ \
        printf("Error: %s:%d, ", __FILE__, __LINE__);\
        printf("code: %d, reason: %s\n", res, cudaGetErrorString(res)); \
    }\
}

__global__ void sum_device(float *a, float *b, float *c, const int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("threadIdx.x = %d\n", i);
    c[i] = a[i] + b[i];
    // printf("threadIdx.x = %d is finished\n", i);
}

void init_data(float *a, const int N){
    for(int i=0; i<N; ++i){
        a[i] = (float)i;
    }
    return;
}

int main(){
    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c;

    int n_elements = 32;
    int n_bytes = n_elements * sizeof(float);

    h_a = (float*)malloc(n_bytes);
    h_b = (float*)malloc(n_bytes);
    h_c = (float*)malloc(n_bytes);

    cudaMalloc(&d_a, n_bytes);
    cudaMalloc(&d_b, n_bytes);
    check_error(cudaMalloc(&d_c, n_bytes));

    init_data(h_a, n_elements);
    init_data(h_b, n_elements);

    cudaMemcpy(d_a, h_a, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n_bytes, cudaMemcpyHostToDevice);

    dim3 block(n_elements);
    dim3 grid((n_elements + block.x - 1) / block.x);

    sum_device<<<grid, block>>>(d_a, d_b, d_c, n_elements);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c, d_c, n_bytes, cudaMemcpyDeviceToHost);

    for(int i=0; i<n_elements; ++i){
        printf("%f ", h_c[i]);
    }
    printf("\n");

    free(h_a);
    free(h_b);
    free(h_c);

    // cudaFree(d_a);
    // cudaFree(d_b);
    // cudaFree(d_c);

    cudaDeviceReset();
    return 0;
}
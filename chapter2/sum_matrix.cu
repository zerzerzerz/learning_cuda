#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

double cpu_second(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

void init(float *a, const int N){
    for(int i=0; i<N; ++i){
        a[i] = i;
    }
}


// nx means row_size
// ny means col_size
// x is horizontal coordinate
// y is vertical coordinate
void sum_host(float *a, float *b, float *c, int nx, int ny){
    float *ia, *ib, *ic;
    ia = a;
    ib = b;
    ic = c;

    for(int iy=0; iy<ny; ++iy){
        for(int ix=0; ix<nx; ++ix){
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
    // printf("sum on cpu finished\n");
}




__global__ void sum_device(float *a, float *b, float *c, int nx, int ny){
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy*nx + ix;

    if(ix < nx && iy < ny){
        c[idx] = a[idx] + b[idx];
    }
}


int check_result(float *cpu, float* gpu, int nx, int ny){
    float eps = 1e-6;
    int nxy = nx * ny;
    for(int i=0; i<nxy; ++i){
        float diff = cpu[i] - gpu[i];
        if(diff < 0.0){
            diff = -diff;
        }
        if(diff > eps) return 0;
    }
    return 1;
}

int main(){
    int nx = 1<<10;
    int ny = 1<<10;

    int nxy = nx * ny;
    int n_bytes = nxy * sizeof(float);

    float *h_a, *h_b, *host_ref, *device_ref;

    double start, end;

    h_a = (float*) malloc(n_bytes);
    h_b = (float*) malloc(n_bytes);
    host_ref = (float*) malloc(n_bytes);
    device_ref = (float*) malloc(n_bytes);

    init(h_a, nxy);
    init(h_b, nxy);

    memset(host_ref, 0, n_bytes);
    memset(device_ref, 0, n_bytes);

    start = cpu_second();
    sum_host(h_a, h_b, host_ref, nx, ny);
    end = cpu_second();
    printf("cpu %f sec\n", end - start);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n_bytes);
    cudaMalloc(&d_b, n_bytes);
    cudaMalloc(&d_c, n_bytes);

    cudaMemcpy(d_a, h_a, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n_bytes, cudaMemcpyHostToDevice);

    int dim_x = 32;
    int dim_y = 32;
    dim3 block(dim_x, dim_y);
    dim3 grid((nx-1+block.x)/block.x, (ny-1+block.y)/block.y);
    printf("blockDim(%d,%d,%d)\n", block.x, block.y, block.z);
    printf("gridDim(%d,%d,%d)\n", grid.x, grid.y, grid.z);

    start = cpu_second();
    sum_device<<<grid, block>>>(d_a, d_b, d_c, nx, ny);
    cudaDeviceSynchronize();
    end = cpu_second();
    printf("gpu %f sec\n", end - start);


    cudaMemcpy(device_ref, d_c, n_bytes, cudaMemcpyDeviceToHost);


    // printf("%d\n", check_result(host_ref, device_ref, nx, ny));

    free(h_a);
    free(h_b);
    free(device_ref);
    free(host_ref);

    cudaDeviceReset();

    return 0;

}
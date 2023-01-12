#include "../utils/utils.cuh"

__global__ void sum_device_recursive(int *g_idata, int* g_odata, int isize){
    int *idata = g_idata + blockIdx.x * blockDim.x;
    int *odata = g_odata + blockIdx.x;

    int tid = threadIdx.x;
    if(isize == 2 && tid == 0){
        odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    int istride = isize >> 1;
    if(istride > 1 && tid + istride < isize){
        idata[tid] += idata[tid + istride];
    }
    __syncthreads();


    if(tid == 0){
        sum_device_recursive<<<1, istride>>> (idata, odata, istride);
        cudaDeviceSynchronize();
    }

    __syncthreads();
}


__global__ void sum_device_recursive_no_sync(int *g_idata, int* g_odata, int isize){
    int *idata = g_idata + blockIdx.x * blockDim.x;
    int *odata = g_odata + blockIdx.x;

    int tid = threadIdx.x;
    if(isize == 2 && tid == 0){
        // printf("odata is %p\n", odata);
        odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    int istride = isize >> 1;
    if(istride > 1 && tid + istride < isize){
        idata[tid] += idata[tid + istride];
    }


    if(tid == 0){
        sum_device_recursive_no_sync<<<1, istride>>> (idata, odata, istride);
    }

}


__global__ void sum_device_recursive_2(int *g_idata, int *g_odata, int istride, const int idim){
    int *idata = g_idata + blockIdx.x * idim;
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if(istride == 1 && tid == 0){
        g_odata[bid] = idata[0] + idata[1];
        return;
    }

    idata[tid] += idata[tid + istride];

    if (bid == 0 && tid == 0){
        sum_device_recursive_2<<<gridDim.x, istride/2>>>(g_idata, g_odata, istride/2, idim);
    }

}


int main(){
    int size = (int)1 << 20;
    int n_bytes = size * sizeof(int);
    dim3 block(512);
    dim3 grid((size - 1 + block.x) / block.x);


    int *data = (int*) malloc(n_bytes);
    init(data, size);
    double s = get_time();
    int sum_host_ref = sum(data, size);
    printf("[CPU] sum = %d, time = %.6f sec(s)\n", sum_host_ref, get_time() - s);



    int *data_device, *res_device, *res_host_from_device;
    cudaMalloc(&data_device, n_bytes);
    cudaMemcpy(data_device, data, n_bytes, cudaMemcpyHostToDevice);

    cudaMalloc(&res_device, grid.x * sizeof(int));
    cudaMemset(res_device, 0, grid.x * sizeof(int));

    res_host_from_device = (int*) malloc(grid.x * sizeof(int));

    s = get_time();

    // sum_device_recursive<<<grid, block>>>(data_device, res_device, block.x);
    // sum_device_recursive_no_sync<<<grid, block>>>(data_device, res_device, block.x);
    sum_device_recursive_2<<<grid, block.x/2>>>(data_device, res_device, block.x/2, block.x);
    
    cudaDeviceSynchronize();

    double e = get_time();
    cudaMemcpy(res_host_from_device, res_device, grid.x * sizeof(int), cudaMemcpyDeviceToHost);

    int sum_gpu_ref = 0;
    for(int i=0; i < grid.x; ++i){
        sum_gpu_ref += res_host_from_device[i];
    }

    printf("[GPU] sum = %d, time = %.6f sec(s)\n", sum_gpu_ref, e - s);


    free(data);
    free(res_host_from_device);
    cudaDeviceReset();
    return 0;
}
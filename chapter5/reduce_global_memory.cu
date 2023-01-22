#include "../utils/utils.cuh"
typedef void (*Kernel) (int*, int*, size_t);
const size_t BLOCK_SIZE = 256;


__global__ void reduce_interleaved(int *g_idata, int *g_odata, size_t num_elem){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int id = bid*blockDim.x + tid;
    int *idata = g_idata + bid*blockDim.x;

    if(id >= num_elem) return;
    for(int stride = blockDim.x/2; stride>0; stride/=2){
        if(tid < stride){
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0) g_odata[bid] = idata[0];
}


__global__ void reduce_global_memory(int *g_idata, int *g_odata, size_t num_elem){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int id = bid*blockDim.x + tid;
    int *idata = g_idata + bid*blockDim.x;

    if(id >= num_elem) return;

    if(blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid+512];
    __syncthreads();

    if(blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid+256];
    __syncthreads();

    if(blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid+128];
    __syncthreads();

    if(blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid+64];
    __syncthreads();

    // for(int stride=blockDim.x/2; stride>32; stride/=2){
    //     if(tid < stride){
    //         idata[tid] += idata[tid+stride];
    //     }
    //     __syncthreads();
    // }

    if(tid < 32){
        volatile int* vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if(tid == 0) g_odata[bid] = idata[0];

}


__global__ void reduce_global_memory_shared_memory(int *g_idata, int *g_odata, size_t num_elem){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int id = bid*blockDim.x + tid;
    if(id >= num_elem) return;

    int *idata_g = g_idata + bid*blockDim.x;
    __shared__ int idata[BLOCK_SIZE];

    idata[tid] = idata_g[tid];
    __syncthreads();

    if(blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid+512];
    __syncthreads();

    if(blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid+256];
    __syncthreads();

    if(blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid+128];
    __syncthreads();

    if(blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid+64];
    __syncthreads();


    if(tid < 32){
        volatile int* vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if(tid == 0) g_odata[bid] = idata[0];
}


__global__ void reduce_global_memory_shared_memory_unroll4(int *g_idata, int *g_odata, size_t num_elem){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int id = 4*bid*blockDim.x + tid;
    __shared__ int idata[BLOCK_SIZE];

    int tmp = 0;
    if(id + 3*blockDim.x < num_elem){
        int a1 = g_idata[id];
        int a2 = g_idata[id+blockDim.x];
        int a3 = g_idata[id+2*blockDim.x];
        int a4 = g_idata[id+3*blockDim.x];
        tmp = a1+a2+a3+a4;
    }

    idata[tid] = tmp;
    __syncthreads();
    


    if(blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid+512];
    __syncthreads();

    if(blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid+256];
    __syncthreads();

    if(blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid+128];
    __syncthreads();

    if(blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid+64];
    __syncthreads();


    if(tid < 32){
        volatile int* vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if(tid == 0) g_odata[bid] = idata[0];
}


__global__ void reduce_global_memory_shared_memory_unroll8(int *g_idata, int *g_odata, size_t num_elem){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int id = 8*bid*blockDim.x + tid;
    __shared__ int idata[BLOCK_SIZE];

    int tmp = 0;
    if(id + 7*blockDim.x < num_elem){
        int a1 = g_idata[id];
        int a2 = g_idata[id+1*blockDim.x];
        int a3 = g_idata[id+2*blockDim.x];
        int a4 = g_idata[id+3*blockDim.x];
        int a5 = g_idata[id+4*blockDim.x];
        int a6 = g_idata[id+5*blockDim.x];
        int a7 = g_idata[id+6*blockDim.x];
        int a8 = g_idata[id+7*blockDim.x];
        tmp = a1+a2+a3+a4+a5+a6+a7+a8;
    }

    idata[tid] = tmp;
    __syncthreads();
    


    if(blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid+512];
    __syncthreads();

    if(blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid+256];
    __syncthreads();

    if(blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid+128];
    __syncthreads();

    if(blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid+64];
    __syncthreads();


    if(tid < 32){
        volatile int* vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if(tid == 0) g_odata[bid] = idata[0];
}


void run(Kernel k, size_t num_elem, size_t num_block, dim3 block, int* data, int* data_original, int *block_sum, int res_cpu, const char* label){
    size_t num_byte = num_elem * sizeof(int);
    double s,e;
    int res_gpu;
    cudaMemcpy(data, data_original, num_byte, cudaMemcpyDefault);
    cudaMemset(block_sum, 0, num_block*sizeof(int));
    s = get_time();
    k<<<num_block, block>>>(data, block_sum, num_elem);
    cudaDeviceSynchronize();
    e = get_time();
    res_gpu = sum(block_sum, num_block);
    printf("[GPU] %-16s, res = %d, time = %f, same = %d\n", label, res_gpu, e-s, res_cpu==res_gpu);
    return;
}


int main(){
    size_t num_elem = 1<<24;
    size_t num_byte = num_elem * sizeof(int);
    double s,e;
    int res_cpu;
    dim3 block(BLOCK_SIZE);
    size_t num_block = my_div(num_elem, block.x);


    int *data_original, *data_h, *data_d, *data_block_sum_d;
    cudaMallocManaged(&data_original, num_byte);
    cudaMallocManaged(&data_h, num_byte);
    cudaMallocManaged(&data_d, num_byte);
    cudaMallocManaged(&data_block_sum_d, num_block * sizeof(int));
    init(data_original, num_elem);


    cudaMemcpy(data_h, data_original, num_byte, cudaMemcpyDefault);
    s = get_time();
    res_cpu = sum(data_h, num_elem);
    e = get_time();
    printf("[CPU] %-16s, res = %d, time = %f, same = %d\n", "Normal", res_cpu, e-s, 1);


    run(reduce_interleaved, num_elem, num_block, block, data_d, data_original, data_block_sum_d, res_cpu, "Interleave");
    run(reduce_global_memory, num_elem, num_block, block, data_d, data_original, data_block_sum_d, res_cpu, "GMemory");
    run(reduce_global_memory_shared_memory, num_elem, num_block, block, data_d, data_original, data_block_sum_d, res_cpu, "SMemory");
    run(reduce_global_memory_shared_memory_unroll4, num_elem, num_block/4, block, data_d, data_original, data_block_sum_d, res_cpu, "SMemoryUnroll4");
    run(reduce_global_memory_shared_memory_unroll8, num_elem, num_block/8, block, data_d, data_original, data_block_sum_d, res_cpu, "SMemoryUnroll8");


    cudaFree(data_original);
    cudaFree(data_h);
    cudaFree(data_d);
    cudaFree(data_block_sum_d);
    cudaDeviceReset();
    return 0;
}
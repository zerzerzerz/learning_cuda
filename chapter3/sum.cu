#include "../utils/utils.cuh"

int recursive_reduce(int *data, int num_elem){
    if(num_elem == 1) return data[0];

    int stride = num_elem / 2;
    for(int i=0; i<stride; ++i){
        data[i] += data[i+stride];
    }
    return recursive_reduce(data, stride);
}


__global__ void reduce_neighbored(int* data, int* output, int n){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int id = bid * blockDim.x + tid;
    if (id >= n) return;

    int* g_data = bid * blockDim.x + data;
    for(int stride = 1; stride <= blockDim.x / 2; stride*=2){
        if(tid % (2*stride) == 0){
            g_data[tid] += g_data[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0) output[bid] = g_data[0];
}


__global__ void recude_neighbored_less(int* data, int* output, int n){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    auto g_data = bid * blockDim.x + data;

    for(int stride = 1; stride <= blockDim.x / 2; stride*=2){
        int start = tid * 2 * stride;
        if(start < blockDim.x){
            g_data[start] += g_data[start + stride];
        }
        __syncthreads();
    }

    if (tid == 0) output[bid] = g_data[0];
}


__global__ void reduce_interleaved(int *data, int* output, int n){
    auto tid = threadIdx.x;
    auto bid = blockIdx.x;
    auto id = bid * blockDim.x + tid;
    if (id >= n) return;
    auto g_data = data + bid*blockDim.x;
    for(auto stride = blockDim.x/2; stride>=1; stride/=2){
        if(tid < stride){
            g_data[tid] += g_data[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0) output[bid] = g_data[0];

}


__global__ void reduce_interleaved_unroll8(int *data, int* output, int n){
    auto tid = threadIdx.x;
    auto bid = blockIdx.x;
    auto id = 8*bid*blockDim.x + tid;
    auto g_data = data + 8*bid*blockDim.x;
    if (id >= n) return;

    if (id + 7*blockDim.x < n){
        data[id] += data[id + 1*blockDim.x];
        data[id] += data[id + 2*blockDim.x];
        data[id] += data[id + 3*blockDim.x];
        data[id] += data[id + 4*blockDim.x];
        data[id] += data[id + 5*blockDim.x];
        data[id] += data[id + 6*blockDim.x];
        data[id] += data[id + 7*blockDim.x];
    }
    __syncthreads();

    for(auto stride = blockDim.x/2; stride>=1; stride/=2){
        if(tid < stride){
            g_data[tid] += g_data[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0) output[bid] = g_data[0];

}



__global__ void reduce_interleaved_unrollwarp(int *data, int* output, int n){
    auto tid = threadIdx.x;
    auto bid = blockIdx.x;
    auto id = 8*bid*blockDim.x + tid;
    auto g_data = data + 8*bid*blockDim.x;
    if (id >= n) return;

    if (id + 7*blockDim.x < n){
        data[id] += data[id + 1*blockDim.x];
        data[id] += data[id + 2*blockDim.x];
        data[id] += data[id + 3*blockDim.x];
        data[id] += data[id + 4*blockDim.x];
        data[id] += data[id + 5*blockDim.x];
        data[id] += data[id + 6*blockDim.x];
        data[id] += data[id + 7*blockDim.x];
    }
    __syncthreads();

    for(auto stride = blockDim.x/2; stride>32; stride/=2){
        if(tid < stride){
            g_data[tid] += g_data[tid + stride];
        }
        __syncthreads();
    }

    if(tid < 32){
        volatile auto vmem = g_data;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if(tid == 0) output[bid] = g_data[0];

}




int main(int argc, char** argv){
    // declare settings
    int *data, *original_data;
    int num_elem = 1<<20;
    size_t bytes = num_elem * sizeof(int);
    int ans = 0, ans_ori = 0;
    double s,e;
    int *data_d, *output_d, *output_h;
    dim3 block(512);
    dim3 grid((num_elem - 1 + block.x) / block.x);

    
    // allocate memory on host
    original_data = (int*)malloc(bytes);
    data = (int*)malloc(bytes);
    output_h = (int*)malloc(grid.x * sizeof(int));
    init(original_data, num_elem);


    // allocate memory on device
    cudaMalloc(&data_d, bytes);
    cudaMalloc(&output_d, grid.x * sizeof(int));


    // normal sum on host
    memcpy(data, original_data, bytes);
    s = get_time();
    ans_ori = sum(data, num_elem);
    e = get_time();
    printf("[CPU][%-25s] ans = %d, time = %f\n", "Normal" ,ans_ori, e-s);


    // recursive reduce on host
    // memcpy(data, original_data, bytes);
    // s = get_time();
    // ans = recursive_reduce(data, num_elem);
    // e = get_time();
    // printf("[CPU][%-25s] ans = %d, time = %f, same = %d\n", "Recursive", ans, e-s, ans == ans_ori);

    
    // reduce neighboured on device
    cudaMemcpy(data_d, original_data, bytes, cudaMemcpyHostToDevice);
    cudaMemset(output_d, 0, grid.x * sizeof(int));
    s = get_time();
    reduce_neighbored<<<grid, block>>>(data_d, output_d, num_elem);
    cudaMemcpy(output_h, output_d, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    e = get_time();
    ans = 0;
    for(int i=0; i<grid.x; ++i){
        ans += output_h[i];
    }
    printf("[GPU][%-25s] ans = %d, time = %f, same = %d\n", "Neighboured", ans, e-s, ans == ans_ori);


    // reduce neighboured less on device
    cudaMemcpy(data_d, original_data, bytes, cudaMemcpyHostToDevice);
    cudaMemset(output_d, 0, grid.x * sizeof(int));
    s = get_time();
    recude_neighbored_less<<<grid, block>>>(data_d, output_d, num_elem);
    cudaMemcpy(output_h, output_d, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    e = get_time();
    ans = 0;
    for(int i=0; i<grid.x; ++i){
        ans += output_h[i];
    }
    printf("[GPU][%-25s] ans = %d, time = %f, same = %d\n", "Neighboured Less", ans, e-s, ans == ans_ori);


    // reduce_interleaved on device
    cudaMemcpy(data_d, original_data, bytes, cudaMemcpyHostToDevice);
    cudaMemset(output_d, 0, grid.x * sizeof(int));
    s = get_time();
    reduce_interleaved<<<grid, block>>>(data_d, output_d, num_elem);
    cudaMemcpy(output_h, output_d, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    e = get_time();
    ans = 0;
    for(int i=0; i<grid.x; ++i){
        ans += output_h[i];
    }
    printf("[GPU][%-25s] ans = %d, time = %f, same = %d\n", "Interleaved", ans, e-s, ans == ans_ori);


    // reduce_interleaved + unroll8 on device
    cudaMemcpy(data_d, original_data, bytes, cudaMemcpyHostToDevice);
    cudaMemset(output_d, 0, grid.x * sizeof(int));
    s = get_time();
    reduce_interleaved_unroll8<<<grid.x/8, block>>>(data_d, output_d, num_elem);
    cudaMemcpy(output_h, output_d, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    e = get_time();
    ans = 0;
    for(int i=0; i<grid.x; ++i){
        ans += output_h[i];
    }
    printf("[GPU][%-25s] ans = %d, time = %f, same = %d\n", "Interleaved-Unroll8", ans, e-s, ans == ans_ori);


    // reduce_interleaved + unroll8 on device
    cudaMemcpy(data_d, original_data, bytes, cudaMemcpyHostToDevice);
    cudaMemset(output_d, 0, grid.x * sizeof(int));
    s = get_time();
    reduce_interleaved_unrollwarp<<<grid.x/8, block>>>(data_d, output_d, num_elem);
    cudaMemcpy(output_h, output_d, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    e = get_time();
    ans = 0;
    for(int i=0; i<grid.x; ++i){
        ans += output_h[i];
    }
    printf("[GPU][%-25s] ans = %d, time = %f, same = %d\n", "Interleaved-UnrollWarp", ans, e-s, ans == ans_ori);




    free(original_data);
    free(data);
    free(output_h);
    cudaFree(data_d);
    cudaFree(output_d);
    return 0;
}
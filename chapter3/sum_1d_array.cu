#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

void init(int *a, int size){
    for(int i=0; i<size; ++i){
        a[i] = rand() % 10;
    }
    return;
}

// normal sum
int sum_1d_array(int *a, int size){
    int sum = 0;
    for(int i=0; i<size; ++i){
        sum += a[i];
    }
    return sum;
}


int interleaved_sum_1d_array(int *a, int size){
    if(size == 1) return a[0];
    
    const int stride = size / 2;
    for(int i=0; i<stride; ++i){
        a[i] += a[i+stride];
    }

    return interleaved_sum_1d_array(a, stride);
}


int neighboured_sum_1d_array(int *a, int size){
    for(int s=1; s<size; s*=2){ 
        for(int i=0; i+s < size; i+=s){
            a[i] += a[i+s];
        }
    }
    return a[0];
}

/*
@param: data_in is the array to be sum
@param: data_out is the array which saves the block sum
*/
__global__ void reduce_neighbour(int *data_in, int *data_out, unsigned int n){
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    if(idx >= n) return;
    
    int *g_data = data_in + blockIdx.x * blockDim.x;

    for(int stride=1; stride < blockDim.x; stride*=2){
        if(tid % (2*stride) == 0){
            g_data[tid] += g_data[tid+stride];
        }
        __syncthreads();
    }

    if(tid == 0){
        data_out[blockIdx.x] = g_data[0];
    }
}


__global__ void reduce_neighbour_less_divergence(int *data_in, int *data_out, unsigned int n){
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    if(idx >= n) return;
    
    int *g_data = data_in + blockIdx.x * blockDim.x;

    for(int stride=1; stride < blockDim.x; stride*=2){
        int loc = 2 * stride * tid;
        if(loc < blockDim.x){
            g_data[loc] += g_data[loc+stride];
        }
        __syncthreads();
    }

    if(tid == 0){
        data_out[blockIdx.x] = g_data[0];
    }
}





__global__ void reduce_interleaved(int *data_in, int *data_out, unsigned int n){
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;
    if(idx >= n) return;

    int* g_data = blockDim.x * blockIdx.x + data_in;

    for(int stride=blockDim.x/2; stride>0; stride/=2){
        if(tid < stride){
            g_data[tid] += g_data[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0){
        data_out[blockIdx.x] = g_data[0];
    }
}


int main(){
    int size = (int)1<<24;
    int bytes = size * sizeof(int);
    int *a = (int*)malloc(bytes);
    init(a, size);
    int sum = sum_1d_array(a, size);
    printf("Sum normal is %d\n", sum);
    // printf("Sum interleaved is %d\n", interleaved_sum_1d_array(a, size));


    // the size of thread block is the same as size of data block
    int dim = 512;
    dim3 block(dim);
    dim3 grid(
        (size + block.x - 1) /  block.x
    );


    int *data_in, *data_out, *data_out_cpu;
    data_out_cpu = (int*) malloc(grid.x * sizeof(int));
    cudaMalloc(&data_in, bytes);
    cudaMalloc(&data_out, grid.x * sizeof(int));
    cudaMemset(data_out, 0, grid.x * sizeof(int));
    cudaMemcpy(data_in, a, bytes, cudaMemcpyHostToDevice);

    // reduce_neighbour<<<grid, block>>>(data_in, data_out, size);
    // reduce_neighbour_less_divergence<<<grid, block>>>(data_in, data_out, size);
    reduce_interleaved<<<grid, block>>>(data_in, data_out, size);
    
    
    cudaMemcpy(data_out_cpu, data_out, grid.x * sizeof(int), cudaMemcpyDeviceToHost);

    int sum_1 = 0;
    for(int i=0; i<grid.x; ++i){
        sum_1 += data_out_cpu[i];
    }

    printf("Sum gpu is %d\n", sum_1);
    int same = (sum == sum_1);
    printf("Same = %d\n", same);

    free(a);
    free(data_out_cpu);
    cudaDeviceReset();

    return 0;
}
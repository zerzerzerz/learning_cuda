#include "../utils/utils.cuh"

const int block_size_x = 16;
const int block_size_y = 32;
const int padding = 1;
const int unrolling = 4;

typedef void (*Kernel)(const int* idata, int* odata, const int n_row, const int n_col);

void transpose_cpu(const int* idata, int* odata, const int n_row, const int n_col){
    for(int r=0; r<n_row; ++r){
        for(int c=0; c<n_col; ++c){
            int src = r*n_col + c;
            int dst = c*n_row + r;
            odata[dst] = idata[src];
        }
    }
}


__global__ void transpose_row(const int* idata, int* odata, const int n_row, const int n_col){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < n_col && y < n_row){
        int src = y*n_col + x;
        int dst = x*n_row + y;
        odata[dst] = idata[src];
    }
}


__global__ void transpose_col(const int* idata, int* odata, const int n_row, const int n_col){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = y * n_col + x;
    if(x < n_col && y < n_row){
        int dst = tid;
        y = tid / n_row;
        x = tid % n_row;
        int src = x * n_col + y;
        odata[dst] = idata[src];
    }
}


__global__ void transpose_smem(const int* idata, int* odata, const int n_row, const int n_col){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = y * n_col + x;
    __shared__ int smem[block_size_y][block_size_x];

    if(x < n_col && y < n_row){
        smem[threadIdx.y][threadIdx.x] = idata[tid];
    }

    __syncthreads();

    if(x < n_col && y < n_row){
        int bid = threadIdx.y * blockDim.x + threadIdx.x;
        int row = bid / blockDim.y;
        int col = bid % blockDim.y;
        x = blockDim.y * blockIdx.y + col;
        y = blockDim.x * blockIdx.x + row;
        int dst = y * n_row + x;
        odata[dst] = smem[col][row];
    }
}


__global__ void transpose_smem_padding(const int* idata, int* odata, const int n_row, const int n_col){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = y * n_col + x;
    __shared__ int smem[block_size_y][block_size_x+padding];

    if(x < n_col && y < n_row){
        smem[threadIdx.y][threadIdx.x] = idata[tid];
    }

    __syncthreads();

    if(x < n_col && y < n_row){
        int bid = threadIdx.y * blockDim.x + threadIdx.x;
        int row = bid / blockDim.y;
        int col = bid % blockDim.y;
        x = blockDim.y * blockIdx.y + col;
        y = blockDim.x * blockIdx.x + row;
        int dst = y * n_row + x;
        odata[dst] = smem[col][row];
    }
}


__global__ void transpose_col_unroll(const int* idata, int* odata, const int n_row, const int n_col){
    int x = (unrolling * blockIdx.x * blockDim.x) + threadIdx.x - blockDim.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    #pragma unroll 4
    for(int i=0; i<unrolling; ++i){
        x += blockDim.x;
        if(x < n_col && y < n_row){
            int tid = y * n_col + x;
            int dst = tid;
            int y_T = tid / n_row;
            int x_T = tid % n_row;
            int src = x_T * n_col + y_T;
            odata[dst] = idata[src];
        }
    }
    
}


__global__ void transpose_col_unroll_smem(const int* idata, int* odata, const int n_row, const int n_col){

    __shared__ int smem[block_size_y][block_size_x*unrolling+padding];
    #pragma unroll 4
    for(int i=0; i<unrolling; ++i){
        int x = unrolling * blockIdx.x * blockDim.x + threadIdx.x + i*blockDim.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int src = y*n_col + x;
        if(x<n_col && y<n_row){
            smem[threadIdx.y][threadIdx.x + i*blockDim.x] = idata[src];
        }
    }
    __syncthreads();

    #pragma unroll 4
    for(int i=0; i<unrolling; ++i){
        int x = unrolling * blockIdx.x * blockDim.x + threadIdx.x + i*blockDim.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if(x<n_col && y<n_row){
            int bid = threadIdx.x + threadIdx.y * blockDim.x;
            int row = bid / block_size_y;
            int col = bid % block_size_y;
            x = blockIdx.y * blockDim.y + col;
            y = unrolling * blockIdx.x * blockDim.x + row + i*blockDim.x;
            int dst = y * n_row + x;
            odata[dst] = smem[col][row+i*blockDim.x];
        }
    }
}


void run(Kernel k, const int* idata, int* odata, const int n_row, const int n_col, const int grid_size_x, const int grid_size_y, const int cpu, const int* odata_cpu, const char* label){
    cudaMemset(odata, 0, n_row * n_col * sizeof(int));
    double s,e;
    dim3 block(block_size_x, block_size_y);
    dim3 grid(grid_size_x, grid_size_y);

    s = get_time();
    if(cpu){
        k(idata, odata, n_row, n_col);
    }
    else{
        k<<<grid,block>>>(idata, odata, n_row, n_col);
    }
    CHECK(cudaDeviceSynchronize());
    e = get_time();
    int flag = (cpu)? 1: is_same(odata, odata_cpu, n_row*n_col);
    printf("[%-20s] time=%.6f flag=%d\n", label, e-s, flag);
}



int main(){
    int n_row = 1024;
    int n_col = 2048;
    
    int n_elem = n_row * n_col;
    int n_byte = n_elem * sizeof(int);

    int *idata, *odata_cpu, *odata_gpu;
    CHECK(cudaMallocManaged(&idata, n_byte));
    CHECK(cudaMallocManaged(&odata_cpu, n_byte));
    CHECK(cudaMallocManaged(&odata_gpu, n_byte));
    init(idata, n_elem);

    int grid_size_x, grid_size_y;
    grid_size_x = my_div(n_col, block_size_x);
    grid_size_y = my_div(n_row, block_size_y);
    printf("block  = (%d, %d)\n", block_size_x, block_size_y);
    printf("grid   = (%d, %d)\n", grid_size_x, grid_size_y);
    printf("shape  = (%d, %d)\n", n_col, n_row);

    run(transpose_cpu, idata, odata_cpu, n_row, n_col, grid_size_x, grid_size_y, 1, NULL, "CPU");
    run(transpose_row, idata, odata_gpu, n_row, n_col, grid_size_x, grid_size_y, 0, odata_cpu, "WarmUp");
    run(transpose_row, idata, odata_gpu, n_row, n_col, grid_size_x, grid_size_y, 0, odata_cpu, "GPU_row");
    run(transpose_col, idata, odata_gpu, n_row, n_col, grid_size_x, grid_size_y, 0, odata_cpu, "GPU_col");
    run(transpose_smem, idata, odata_gpu, n_row, n_col, grid_size_x, grid_size_y, 0, odata_cpu, "GPU_smem");
    run(transpose_smem, idata, odata_gpu, n_row, n_col, grid_size_x, grid_size_y, 0, odata_cpu, "GPU_smem_padding");
    
    grid_size_x = my_div(n_col, block_size_x*unrolling);
    run(transpose_col_unroll, idata, odata_gpu, n_row, n_col, grid_size_x, grid_size_y, 0, odata_cpu, "GPU_col_unroll");
    run(transpose_col_unroll_smem, idata, odata_gpu, n_row, n_col, grid_size_x, grid_size_y, 0, odata_cpu, "GPU_col_unroll_smem");


    cudaFree(idata);
    cudaFree(odata_cpu);
    cudaFree(odata_gpu);
    return 0;
}
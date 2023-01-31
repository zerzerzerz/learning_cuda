#include "../utils/utils.cuh"

const int nx = 32;
const int ny = 128;
const int block_x = 8;
const int block_y = 32;
typedef void (*Kernel)(const int* idata, int *odata, const int nx, const int ny);


void transpose_cpu(const int* idata, int* odata, const int nx, const int ny){
    for(int y=0; y<ny; ++y){
        for(int x=0; x<nx; ++x){
            int index1 = y*nx + x;
            int index2 = x*ny + y;
            odata[index2] = idata[index1];
        }
    }
    return;
}


// 让write写转置之后的矩阵的(x,y)位置，对应的linear index = y*ny + x，因为转置之后一行ny个元素
// 对应于原来的矩阵中(y,x)位置，linear index = x*nx + y
// 所以重点落在怎么求转置矩阵的(x,y)
// 不仅仅行数列数交换，block也要做转置
__global__ void transpose_gpu(const int* idata, int* odata, const int nx, const int ny){
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int tgt_index = y*ny + x;
    int src_index = x*nx + y;
    if(x < ny && y < nx){
        odata[tgt_index] = idata[src_index];
    }
}

__global__ void transpose_gpu_smem(const int* idata, int* odata, const int nx, const int ny){
    __shared__ int smem[block_y][block_x];
    int x,y;
    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < nx && y < ny){
        smem[threadIdx.y][threadIdx.x] = idata[y*nx + x];
        __syncthreads();
        int bid = threadIdx.y * blockDim.x + threadIdx.x;

        // coordinate in transposed block
        int row_T = bid / blockDim.y;
        int col_T = bid % blockDim.y;
        x = blockIdx.y * blockDim.y + col_T;
        y = blockIdx.x * blockDim.x + row_T;

        odata[y*ny+x] = smem[col_T][row_T];
    }


}


__global__ void transpose_gpu_row(const int* idata, int* odata, const int nx, const int ny){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int src = y*nx + x;
    int dst = x*ny + y;
    if(x<nx && y<ny){
        odata[dst] = idata[src];
    }
}


__global__ void transpose_gpu_unroll4(const int* idata, int* odata, const int nx, const int ny){
    int x = 4 * blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int tgt_index = y*ny + x;
    int src_index = x*nx + y;
    if(x+3*blockDim.y < ny && y < nx){
        odata[tgt_index] = idata[src_index];
        odata[tgt_index +   blockDim.y] = idata[src_index + nx*blockDim.y];
        odata[tgt_index + 2*blockDim.y] = idata[src_index + 2*nx*blockDim.y];
        odata[tgt_index + 3*blockDim.y] = idata[src_index + 3*nx*blockDim.y];
    }
}


void run(Kernel k, const int* idata, int* odata, const int nx, const int ny, const dim3 block, const dim3 grid, int* res_cpu, const char* label , const int cpu){
    double s,e;
    cudaMemset(odata, 0, sizeof(int) * nx * ny);
    s = get_time();
    if(cpu){
        k(idata, odata, nx, ny);
        CHECK(cudaDeviceSynchronize());
    }
    else{
        k<<<grid,block>>>(idata, odata, nx, ny);
        CHECK(cudaDeviceSynchronize());
    }
    e = get_time();
    int flag = (cpu)? 1: is_same(odata, res_cpu, nx*ny);
    printf("[%-20s], time=%f, flag=%d\n", label, e-s, flag);
    return;
}   


int main(){
    int *idata, *odata_cpu, *odata_gpu;
    size_t num_elem = nx * ny;
    size_t num_byte = num_elem * sizeof(int);

    dim3 block(block_x, block_y);
    dim3 grid(my_div(nx, block_x), my_div(ny, block_y));

    cudaMallocManaged(&idata, num_byte);
    cudaMallocManaged(&odata_cpu, num_byte);
    cudaMallocManaged(&odata_gpu, num_byte);

    init(idata, num_elem);
    printf("nx=%d, ny=%d\n", nx, ny);
    run(transpose_cpu, idata, odata_cpu, nx, ny, block, grid, NULL, "CPU", 1);
    run(transpose_gpu_row, idata, odata_gpu, nx, ny, block, grid, odata_cpu, "GPU_warmup", 0);
    run(transpose_gpu_row, idata, odata_gpu, nx, ny, block, grid, odata_cpu, "GPU_row", 0);
    run(transpose_gpu, idata, odata_gpu, nx, ny, block, grid, odata_cpu, "GPU_col", 0);
    run(transpose_gpu_smem, idata, odata_gpu, nx, ny, block, grid, odata_cpu, "GPU_col_smem", 0);
    
    grid.y /= 4;
    run(transpose_gpu_unroll4, idata, odata_gpu, nx, ny, block, grid, odata_cpu, "GPU_col_unroll4", 0);


    cudaFree(idata);
    cudaFree(odata_cpu);
    cudaFree(odata_gpu);
    return 0;
}
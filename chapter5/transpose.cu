#include "../utils/utils.cuh"
const int block_size_x = 32;
const int block_size_y = 16;

void transpose_cpu(const int* idata, int *odata, const int nx, const int ny){
    int i_index, o_index;
    for(int y=0; y<ny; ++y){
        for(int x=0; x<nx; ++x){
            i_index = y*nx + x;
            o_index = x*ny + y;
            odata[o_index] = idata[i_index];
        }
    }
    return;
}


__global__ void transpose_gpu(const int *idata, int *odata, const int nx, const int ny){
    // smem的shape和original block shape一样
    __shared__ int smem[block_size_y][block_size_x];

    // 求出某个element在全局的linear index
    int ix_ori = blockIdx.x * blockDim.x + threadIdx.x;
    int iy_ori = blockIdx.y * blockDim.y + threadIdx.y;
    int ti = iy_ori*nx + ix_ori;

    // 这个线程在块内的linear index
    int tid = threadIdx.y*blockDim.x + threadIdx.x;

    // 这个线程负责的转置之后的哪个块内位置
    int irow = tid / blockDim.y;
    int icol = tid % blockDim.y;

    // 矩阵转置之后block也要随着转置
    int new_block_index_x = blockIdx.y;
    int new_block_index_y = blockIdx.x;
    int new_block_size_x = blockDim.y;
    int new_block_size_y = blockDim.x;
    
    // 计算转置之后矩阵的2D坐标 + linear index
    int ix = new_block_index_x * new_block_size_x + icol;
    int iy = new_block_index_y * new_block_size_y + irow;
    int to = iy*ny + ix;

    if(ix < ny && iy < nx){
        smem[threadIdx.y][threadIdx.x] = idata[ti];
        __syncthreads();

        odata[to] = smem[icol][irow];
    }
}


int main(){
    int *data_original, *res_cpu, *res_gpu;
    int nx = 1<<12;
    int ny = 1<<13;
    dim3 block(block_size_x,block_size_y);
    dim3 grid(my_div(nx, block.x), my_div(ny, block.y));


    int num_elem = nx * ny;
    int num_byte = num_elem * sizeof(int);
    double s,e;


    cudaMallocManaged(&data_original, num_byte);
    cudaMallocManaged(&res_cpu, num_byte);
    cudaMallocManaged(&res_gpu, num_byte);
    init(data_original, num_elem);


    s = get_time();
    transpose_cpu(data_original, res_cpu, nx, ny);
    e = get_time();
    printf("[CPU] time = %.6f\n", e-s);


    printf("grid<<<%d,%d>>>\n", grid.x, grid.y);
    printf("block<<<%d,%d>>>\n", block.x, block.y);
    s = get_time();
    transpose_gpu<<<grid, block>>>(data_original, res_gpu, nx, ny);
    cudaDeviceSynchronize();
    e = get_time();


    printf("[GPU] time = %.6f, same = %d\n", e-s, is_same(res_cpu, res_gpu, num_elem));


    cudaFree(data_original);
    cudaFree(res_cpu);
    cudaFree(res_gpu);
    return 0;
}
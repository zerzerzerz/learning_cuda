#include "../utils/utils.cuh"

int main(int argc, char** argv){
    int *a, *b, *res_cpu, *res_gpu, *res2_gpu;
    int row = 512;
    int num_elem = row * row;
    int bytes = num_elem * sizeof(int);
    int block_size_x = 16, block_size_y = 16;


    if(argc > 2){
        block_size_x = atoi(argv[1]);
        block_size_y = atoi(argv[2]);
    }

    dim3 block(block_size_x, block_size_y);
    dim3 grid(my_div(row, block.x), my_div(row, block.y));
    double s,e;
    int flag;


    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&res_cpu, bytes);
    cudaMallocManaged(&res_gpu, bytes);
    cudaMallocManaged(&res2_gpu, bytes);


    init(a, num_elem);
    init(b, num_elem);
    cudaMemset(res_cpu, 0, bytes);
    cudaMemset(res_gpu, 0, bytes);


    s = get_time();
    mm_host(a,b,res_cpu,row,row,row);
    e = get_time();
    printf("[CPU] time = %.6f\n", e-s);


    s = get_time();
    mm_device<<<grid,block>>>(a,b,res_gpu,row,row,row);
    cudaDeviceSynchronize();
    e = get_time();
    flag = is_same(res_cpu, res_gpu, num_elem);
    printf("[GPU] time = %.6f, same = %d\n", e-s, flag);


    s = get_time();
    mm2_device<<<grid,block>>>(a,b,res2_gpu,row,row,row);
    cudaDeviceSynchronize();
    e = get_time();
    flag = is_same(res_cpu, res2_gpu, num_elem);
    printf("[GPU] time = %.6f, same = %d\n", e-s, flag);




    cudaFree(a);
    cudaFree(b);
    cudaFree(res_cpu);
    cudaFree(res_gpu);
    cudaFree(res2_gpu);
    return 0;
}
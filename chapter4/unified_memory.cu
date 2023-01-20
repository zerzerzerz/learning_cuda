#include "../utils/utils.cuh"

int main(int argc, char** argv){
    int *a, *b, *res_cpu, *res_gpu;
    int num_elem = 1<<20;
    int bytes = num_elem * sizeof(int);
    int block_size = 512;
    if(argc > 1){
        block_size = atoi(argv[1]);
    }

    dim3 block(block_size);
    dim3 grid(my_div(num_elem, block.x));
    double s,e;


    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&res_cpu, bytes);
    cudaMallocManaged(&res_gpu, bytes);


    init(a, num_elem);
    init(b, num_elem);
    cudaMemset(res_cpu, 0, bytes);
    cudaMemset(res_gpu, 0, bytes);


    s = get_time();
    add_host(a,b,res_cpu,num_elem);
    e = get_time();
    printf("[CPU] time = %.6f\n", e-s);


    s = get_time();
    add_device<<<grid,block>>>(a,b,res_gpu,num_elem);
    cudaDeviceSynchronize();
    e = get_time();
    printf("[GPU] time = %.6f\n", e-s);


    auto flag = is_same(res_cpu, res_gpu, num_elem);
    printf("Same flag = %d\n", flag);


    cudaFree(a);
    cudaFree(b);
    cudaFree(res_cpu);
    cudaFree(res_gpu);
    return 0;
}
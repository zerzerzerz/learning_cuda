#include "../utils/utils.cuh"

// 先声明shared memory 数组，但是动态声明，后面再指定size
extern __shared__ int b[];

__global__ void display(int *a, int size){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int id = bid*blockDim.x + tid;
    if (id < size){
        printf("Bid = %2d, tid = %2d, a[%2d] = %2d\n", bid, tid, id, a[id]);
        printf("Bid = %2d, tid = %2d, b[%2d] = %2d\n", bid, tid, id, b[id]);
    }
}


int main(){
    int *a;
    size_t size = 1<<4;
    size_t bytes = size * sizeof(int);
    dim3 block(4);
    dim3 grid(my_div(size, block.x));

    cudaMallocManaged(&a, bytes);

    // <<<>>>里面传递第三个参数，标识shared memory数组的大小
    display<<<grid, block, size>>>(a, size);




    cudaFree(a);
    cudaDeviceReset();
    return 0;
}
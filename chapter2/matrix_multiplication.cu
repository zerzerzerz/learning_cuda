#include "../utils/utils.cuh"

void display(int *A, int row_A, int col_A){
    for(int i=0; i<row_A; ++i){
        for(int j=0; j<col_A; ++j){
            printf("%d ", A[i*col_A+j]);
        }
        printf("\n");
    }
}


int main(){
    int row_A = 512;
    int col_A = 512;
    int row_B = col_A;
    int col_B = 512;
    int row_C = row_A;
    int col_C = col_B;

    int n_A = row_A * col_A;
    int n_B = row_B * col_B;
    int n_C = row_C * col_C;

    int bytes_A = n_A * sizeof(int);
    int bytes_B = n_B * sizeof(int);
    int bytes_C = n_C * sizeof(int);

    int dim1 = 32;
    int dim2 = 32;

    int *A = (int*) malloc(bytes_A);
    int *B = (int*) malloc(bytes_B);
    int *C = (int*) malloc(bytes_C);
    int *C_cpu_from_gpu = (int*) malloc(bytes_C);

    init(A, n_A);
    init(B, n_B);

    double s,e;
    s = get_time();
    mm_host(A,B,C,row_A,col_A,col_B);
    e = get_time();
    printf("cpu: %f sec\n", e-s);

    dim3 block(dim1, dim2);
    dim3 grid((row_C+block.x-1) / block.x, (col_C+block.y-1) / block.y);
    // printf("blockDim=(%d,%d,%d)\n", block.x, block.y, block.z);
    // printf("gridDim=(%d,%d,%d)\n", grid.x, grid.y, grid.z);


    int *A_gpu, *B_gpu, *C_gpu;
    cudaMalloc(&A_gpu, bytes_A);
    cudaMalloc(&B_gpu, bytes_B);
    cudaMalloc(&C_gpu, bytes_C);

    cudaMemcpy(A_gpu, A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, bytes_B, cudaMemcpyHostToDevice);

    s = get_time();
    mm_device<<<grid, block>>>(A_gpu,B_gpu,C_gpu,row_A,col_A,col_B);
    cudaDeviceSynchronize();
    e = get_time();
    printf("gpu: %f sec\n", e-s);

    CHECK(cudaMemcpy(C_cpu_from_gpu, C_gpu, bytes_C, cudaMemcpyDeviceToHost));
    
    printf("%d\n", is_same(C, C_cpu_from_gpu, n_C));

    // display(C, row_C, col_C);
    // printf("\n");
    // display(C_cpu_from_gpu, row_C, col_C);

    free(A);
    free(B);
    free(C);
    free(C_cpu_from_gpu);
    cudaDeviceReset();

    return 0;
}

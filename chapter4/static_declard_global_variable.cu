#include "../utils/utils.cuh"

__device__ float data_d;

__global__ void display_and_modify(){
    printf("[GPU] data_d = %f\n", data_d);
    data_d += 1.0;
    return;
}


int main(){
    float data_h = 3.14;
    printf("[CPU] data_h = %f\n", data_h);
    cudaMemcpyToSymbol(data_d, &data_h, sizeof(data_h));

    display_and_modify<<<1,1>>>();
    // cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(&data_h, data_d, sizeof(data_h));
    printf("[CPU] data_h = %f\n", data_h);

    cudaDeviceReset();
    return 0;

}
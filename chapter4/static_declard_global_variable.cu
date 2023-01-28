#include "../utils/utils.cuh"
const int size = 4;
__device__ float data_d[size];

__global__ void display_and_modify(){
    printf("[GPU] data_d = %f\n", data_d[0]);
    data_d[0] += 1.0;
    return;
}


int main(){
    float data_h[size] = {1,1,1,1};
    printf("[CPU] data_h = %f\n", data_h[0]);
    cudaMemcpyToSymbol(data_d, &data_h, size * sizeof(float));

    display_and_modify<<<1,1>>>();
    // cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(&data_h, data_d, size * sizeof(float));
    printf("[CPU] data_h = %f\n", data_h[0]);

    cudaDeviceReset();
    return 0;

}
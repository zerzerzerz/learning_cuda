#include <stdio.h>
#include <string.h>
#include "../utils/utils.cuh"

struct Info{
    int data;
    char* text;
};

__global__ void kernel(Info* info){
    printf("[GPU] info->data = %d\n", info->data);
    printf("[GPU] info->text = %p\n", info->text);
    printf("[GPU] info->text = %s\n", info->text);
}



void launch(Info* info){
    Info* info_d;
    char* text_d;
    int text_len = strlen(info->text);

    // info_d 是host variable，但是存储的地址，指向device memory
    cudaMalloc(&info_d, sizeof(Info));
    cudaMalloc(&text_d, text_len);

    // info是地址，指向host memory；info_d是地址，指向device memory，所以使用cudaMemcpyHostToDevice
    cudaMemcpy(info_d, info, sizeof(Info), cudaMemcpyHostToDevice);
    cudaMemcpy(text_d, info->text, text_len, cudaMemcpyHostToDevice);

    // 仅仅完成前面两步还不够，这个时候info->text和info_d->text完全一样，都指向host memory
    // 需要将text_d的值（text_d存储的地址）赋给info_d->text
    // text_d存储的是device memory address，但本身是一个host variable
    // info_d->text已经位于device memory上了
    // 因此direction选择cudaMemcpyHostToDevice
    cudaMemcpy(&(info_d->text), &text_d, sizeof(char*), cudaMemcpyHostToDevice);

    printf("String's location in device is %p\n", text_d);

    kernel<<<1,1>>>(info_d);
    CHECK(cudaDeviceSynchronize());

    cudaFree(text_d);
    cudaFree(info_d);

    return;

}



int main(){
    Info* info = (Info*)malloc(sizeof(Info));
    info->data = 1;
    const char* text = "Hello, world!";
    info->text = (char*)malloc(strlen(text)+1);
    strcpy(info->text, text);

    printf("[CPU] info->data = %d\n", info->data);
    printf("[CPU] info->text = %p\n", info->text);
    printf("[CPU] text's address = %p\n", text);


    launch(info);
    free(info->text);
    free(info);


    // Unified Memory
    Info* info_um = NULL;
    cudaMallocManaged(&info_um, sizeof(Info));
    cudaMallocManaged(&(info_um->text), strlen(text)+1);
    info_um->data = 1;
    strcpy(info_um->text, text);
    kernel<<<1,1>>>(info_um);
    CHECK(cudaDeviceSynchronize());
    cudaFree(info_um->text);
    cudaFree(info_um);
    


    return 0;
}
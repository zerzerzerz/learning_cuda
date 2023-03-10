#ifndef header_utils_time
#define header_utils_time

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

double get_time();
int sum(int*, int);
void init(int* data, const int n_elem);
int is_same(const int*,const int*, int num_elem);
void add_host(int*, int*, int*, int);
__global__ void add_device(int*, int*, int*, int);
int my_div(int, int);
void mm_host(int*, int*, int*, int, int, int);
void display_2d(const int*, const int nrow, const int ncol);
__global__ void mm_device(int*, int*, int*, int, int, int);
__global__ void mm2_device(int*, int*, int*, int, int, int);
void check_device(const int device);

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}


#endif
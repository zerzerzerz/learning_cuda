#include "utils.cuh"

double get_time(){
    struct timeval current_time;
    gettimeofday(&current_time, NULL);
    return ((double)current_time.tv_sec + (double)current_time.tv_usec * 1e-6);
}

int sum(int *data, int size){
    // sum an 1d array in host
    // return the sum of array
    int s = 0;
    for(int i=0; i<size; ++i){
        s += data[i];
    }
    return s;
}


void init(int *data, int size){
    for(int i=0; i<size; ++i){
        data[i] = (rand() % 10) + 1;
        // data[i] = i;
    }
    return;
}


int is_same(int *a, int *b, int size){
    // check whether two arrays are the same
    for(int i=0; i<size; ++i){
        if(a[i] != b[i]) return 0;
    }
    return 1;
}


void add_host(int *a, int *b, int *c, int size){
    for(int i=0; i<size; ++i){
        c[i] = a[i] + b[i];
    }
}


int my_div(int x, int y){
    return (x+y-1) / y;
}



// mm means matrix multiplication
void mm_host(int *A, int *B, int *C, int row_A, int col_A, int col_B){
    int sum, a, b;
    // int row_B = col_A;
    // int row_C = row_A;
    int col_C = col_B;

    for(int i=0; i<row_A; ++i){
        for(int j=0; j<col_B; ++j){
            sum = 0;
            for(int k=0; k<col_A; ++k){
                a = A[i*col_A + k];
                b = B[k*col_B + j];
                sum += a*b;
            }
            C[i*col_C + j] = sum;
        }
    }
}


__global__ void mm2_device(int *A, int *B, int *C, int row_A, int col_A, int col_B){
    // interpret x as outer index
    // slower
    // int row_B = col_A;
    int row_C = row_A;
    int col_C = col_B;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < row_C && j < col_C){
        int sum = 0;
        int a = 0;
        int b = 0;
        for(int k=0; k<col_A; ++k){
            a = A[i*col_A+k];
            b = B[k*col_B+j];
            sum += a*b;
        }
        C[i*col_C+j] = sum;
    }
}


__global__ void mm_device(int *A, int *B, int *C, int row_A, int col_A, int col_B){
    // interpret x as inner index
    // faster
    // int row_B = col_A;
    int row_C = row_A;
    int col_C = col_B;

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < row_C && j < col_C){
        int sum = 0;
        int a = 0;
        int b = 0;
        for(int k=0; k<col_A; ++k){
            a = A[i*col_A+k];
            b = B[k*col_B+j];
            sum += a*b;
        }
        C[i*col_C+j] = sum;
    }
}


__global__ void add_device(int *a, int *b, int *c, int size){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int id = bid * blockDim.x + tid;

    if(id < size){
        c[id] = a[id] + b[id];
    }

    return;
}


void display_2d(const int* a, int nrow, int ncol){
    for(int i=0; i<nrow; ++i){
        for(int j=0; j<ncol; ++j){
            printf("%d ", a[i*ncol+j]);
        }
        printf("\n");
    }
    printf("\n");
}
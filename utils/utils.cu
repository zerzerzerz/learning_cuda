#include "utils.cuh"

double get_time(){
    struct timeval current_time;
    gettimeofday(&current_time, NULL);
    return ((double)current_time.tv_sec + (double)current_time.tv_usec * 1e-6);
}

int sum(int *data, int size){
    int s = 0;
    for(int i=0; i<size; ++i){
        s += data[i];
    }
    return s;
}

void init(int *data, int size){
    for(int i=0; i<size; ++i){
        data[i] = rand() % 10;
        // data[i] = 1;
    }
    return;
}
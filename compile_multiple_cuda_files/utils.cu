#include "utils.cuh"

double get_time(){
    struct timeval current_time;
    gettimeofday(&current_time, NULL);
    return ((double)current_time.tv_sec + (double)current_time.tv_usec * 1e-6);
}
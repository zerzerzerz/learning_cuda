#ifndef header_utils_time
#define header_utils_time

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

double get_time();
int sum(int*, int);
void init(int*, int);

#endif
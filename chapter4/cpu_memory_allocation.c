#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(){
    int *a;
    int size = 4;
    a = (int*)malloc(size * sizeof(int));
    memset(a, -1, size * sizeof(int));
    for(int i=0; i<size; ++i){
        printf("a[%d] = %d\n", i, a[i]);
    }
    return 0;
}
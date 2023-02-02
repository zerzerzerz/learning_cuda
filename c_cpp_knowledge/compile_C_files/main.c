#include <stdio.h>
#include "utils.h"

int main(){
    int a = 1, b = 2;
    int c = add(a,b);
    printf("%d + %d = %d\n", a, b, c);
    return 0;
}
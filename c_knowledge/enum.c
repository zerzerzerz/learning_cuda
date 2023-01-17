#include <stdio.h>


// how to define enum
enum STATE {
    SUCCESS,
    WARNING,
    FAIL,
    UNDEFINED
};

int main(){
    int a = 1;

    // state   a enum variable
    // declare a enum variavle
    enum STATE s = SUCCESS;

    if(a == WARNING){
        printf("This is warning\n");
    }
    
    if(s == SUCCESS){
        printf("This is success\n");
    }

    return SUCCESS;
}
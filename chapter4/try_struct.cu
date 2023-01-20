#include <stdio.h>
#include <string.h>

typedef unsigned long long ull;

struct Student{
    char name [64];
    ull id;
};

int main(){
    Student s;
    strcpy(s.name, "Ruizhe Zhong");
    s.id = 519021910025;
    printf("name = %s, id = %lld\n", s.name, s.id);
    return 0;
}
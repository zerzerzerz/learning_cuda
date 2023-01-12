# Compile C Files
## Preprocessing
- 预处理/预编译
- 将所有的`#include`以及宏定义替换为其真正的内容
```bash
gcc -E main.c -o main.i
```
或者是
```bash
cpp main.c -o main.s
```
- `-E`表示让编译器在预处理之后就退出
- `-o`指定文件的输出命
- `-I`指定头文件目录，因为我在include头文件的时候加上文件夹了，所以就没在这里指定目录
- 预处理之后的结果依然是文本文件

## Compilation
- 编译
- 将预处理之后的程序转换为特定的汇编代码(assembly code)
```bash
gcc -S main.c -o main.s
```
- `-S`表示让编译器在编译之后停止，生成汇编代码，也是文本文件


## Assemble
- 汇编
- 从汇编代码生成机器代码(machine code)，生成的文件叫做**目标文件**，是二进制文件，一般为`.o`格式
- 每个文件都生成一个目标文件
```bash
as main.s -o main.o
```
或者
```bash
gcc -c main.s -o main.o
```
- `-c`表示完成汇编后停止
- 也可以直接从`.c`生成`.o`

## Linking
- 链接
- 将多个目标文件以及需要的库链接起来，形成最终的可执行文件

## Summary
- 所以编译的时候，输入gcc的文件不包含头文件，在预编译阶段就已经替换掉了
- [参考资料](https://www.cnblogs.com/carpenterlee/p/5994681.html)
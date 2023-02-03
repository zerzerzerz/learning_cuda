# README
- 看了一本新书
- 中文版的
- cuda编程基础与实践
- ![https://github.com/MAhaitao999/CUDA_Programming](https://github.com/MAhaitao999/CUDA_Programming)
- 这本书比较简单，就做一些知识点总结

# 知识点总结
- 在定义cuda函数的时候，返回值类型和函数执行空间标识符可以交换顺序，也就是说`void __global__ kernel();`和`__global__ void kernel();`都是可以的
- `__global__`修饰的函数称为核函数，host调用，device执行，不能有返回值
- `__host__`主机函数，主机调用，主机执行，一般可以省略
- `__device__`设备函数，被核函数或者设备函数调用，在设备上执行，设备函数可以有返回值
- `__host__`和`__device__`可以一起使用，其他的修饰符的组合不能一起用
- 静态全局内存变量
  - 在任何函数外进行生命，可以声明单个变量，也可以声明数组
  - 静态体现在，编译时就能确定数组的大小
  - 全局体现在，任何kernel中都可以直接访问到，但是host上不行
  - 如果想在host和device之间传递静态全局内存变量，需要使用`cudaMemcpyToSymbol`和`cudaMemcpyFromSymbol`两个函数，其中symbol其实就是静态全局内存变量的形参名字

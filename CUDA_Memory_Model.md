# CUDA Memory Model
- 物理实现
  - cpu和gpu的memory都是用DRAM实现的
  - cache用的是SRAM
  - 外存一般用磁盘、flash drive、固态硬盘
## CUDA内存模型
- 可以按照是否可编程来进行分类，也就是对于一块存储，程序员是否可以显式决定，什么数据放在那里
- 在cpu中，L1 L2 cache都是不可编程的，但是在gpu中，提供了更多的可编程memory，比如
  - register
  - shared memory
  - local memory
  - constant memory
  - texture memory
  - global memory
- 每个thread都有自己的local memory和register
- shared memory为block所有，并且block内的全部thread都可以使用
- constant和texture都是read only，同时对全部的thread可见

### Register
- 寄存器是最快的内存空间，kernel中声明的自动变量（也就是没有任何其他的修饰符）都存放在寄存器中
- array也有可能放在寄存器中，但是索引这个array的index是constant并且编译时确定
- 寄存器溢出的时候，变量会存放到local memory

### Local Memory
- 在kernel中，寄存器放不下的变量就放到local memory中
- local memory其实和global memory有着相同的物理地址，所以local memory和global memory一样，都是高延迟、低带宽

### Shared Memory
- on-chip memory，所以比local和global快很多
- 在kernel中进行声明
- 整个block内的thread都可见
- L1 cache和shared memory共享一定的空间，可以在运行时设置二者的分配比例
- 对应的修饰符是`__shared__`

### Constant Memory
- 必须在kernel外部进行声明
- 当warp内的32个线程访问同一个地址的时候，比较适合使用constant memory
- 配有read-only constant cache
- kernel只能读取其中的值，因此需要在host上进行初始化
- 相关的函数是`cudaMemcpyToSymbol`和`cudaMemcpyFromSymbol`

### Texture Memory
- 略

### Global Memory
- 通常是使用`cudaMalloc`动态分配
- 也可以静态分配，使用修饰符`__device__`


## GPU Caches
- cache和cpu一样，都是不可编程的
- L1 cache，每个SM都有1个
- L2 cache，全部的SM共享1个
- Read-only constant cache，每个SM都有1个
- Read-only texture cache，每个SM都有1个
- GPU load是走cache的，但是store不走cache

## Summary
- cuda声明变量


|Qualifier|Variable Name|Memory|Scopy|Lifespan|
|-|-|-|-|-|
||`float var`|register|thread|thread|
||`float var [100]`|local memory|thread|thread|
|`__shared__`|标量或者数组|shared memory|block|block|
|`__device__`|标量或者数组|global memory|global|application|
|`__constant__`|标量或者数组|constant memory|global|application|
- device memory的性质

|Memory|On/Off Chip|Cached|Access|Scope|Lifespan|
|-|-|-|-|-|-|
|register|on||RW|thread|thread|
|local|off|Yes|RW|thread|thread|
|shared|on|Yes|RW|block|block|
|global|off|Yes|RW|ALL threads|Application|
|constant|off|Yes|RW|ALL threads|Application|
|texture|off|Yes|RW|ALL threads|Application|


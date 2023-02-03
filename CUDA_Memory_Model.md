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
```c
// 这里相当于直接声明了一个device上的变量
// 当然也可以声明数组
__device__ int symbol;
int var_h = 1;

// 要注意传递的参数没有&，可以理解为传递了reference
// 更主要的是，在host上不能通过&来获取device variable的地址
cudaMemcpyToSymbol(symbol, &var_h, sizeof(int));
cudaMemcpyFromSymbol(&var_h, symbol, sizeof(int));

// 如果真的想获得symbol的地址，需要用下面这个函数
// 之后就可以使用cudaMemcpy来传递了
int* pointer;
cudaGetSymbolAddress(&pointer, symbol);
```



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

### Pinned Memory
- host memory默认是可分页的，可能存储在硬盘上（虚拟内存）
- 对于gpu来说，从可分页的host memory复制数据不安全，因此cuda driver首先创建临时的pinned host memory，将src host data先复制到这里，然后再从pinned memory复制到device memory上
- 可以使用`cudaMallocHost`直接创建host pinned memory，切记使用`cudaFreeHost`来释放


### Zero-Copy Memory
- host和device都可以直接访问这块内存
- device memory不足的时候可以用一下
- 使用`cudaHostAlloc`来创建，这个函数有个flag需要设置
  - `cudaHostAllocDefault` 在host上创建普通的pinned memory
  - `cudaHostAllocPortable` 可以在多个device上使用
  - `cudaHostAllocWriteCombined` 适合host写，device读
  - `cudaHostAllocMapped` 这个就是zero-copy memory，在host上创建一块内存，然后映射到device address space上
- 必须使用`cudaFreeHost`进行释放
- 很慢，因为每次内存事务都需要走PCIe bus
- 传给kernel的时候，不能直接把host上的指针传进去，需要使用`cudaHostGetDevicePointer`拿到映射到device memory之后的地址再传


### Unified Virtual Addressing (UVA)
- host memory和device memory共享同一个虚拟地址空间
- 好处就是，创建的zero-copy host memory，host pointer可以直接传递给kernel/device，不需要先把host pointer传给device pointer了


### Unified Memory
- UM创建一个受管理的内存池(create a pool of managed memory)，每次从这个pool中分配的内存，对host和device都是accessible，有着相同的address/pointer
- UM在统一的内存空间中，自动在host和device中间迁移数据
- 静态声明的话，比如`__device__ __managed__ int var;` 必须全局声明，host和device都可以获取
- 使用`cudaMallocManaged`动态分配
- 申请超量显存
  - 要求GPU架构不低于Pascal架构


### Glonal Memory Read
- 有三条路径
  - L1/L2 cache
  - constant cache
  - read-only cache
- L1/L2 cache
  - L1 cache 128 bytes cache line
  - L2 cache 32 bytes cache line
- Read-Only Cache
  - 读取的时候使用`__lgd`函数
  - 从帕斯卡架构是`sm_60`
  - 或者传递参数的时候直接加上`const __restrict__`


### Global Memory Write
- 没有L1 cache参与，只有L2 cache，访问粒度是32 bytes
- 一次可以做1/2/4哥segment


## CUDA Runtime API
- `cudaError_t cudaMalloc(void **devPtr, size_t count);` 分配global memory
- `cudaError_t cudaMemset(void *devPtr, int value, size_t count);` 填充
- `cudaError_t cudaFree(void *devPtr);` 释放显存
- `cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);` 在host和device之间传递数据
- `__host__​cudaError_t cudaMemcpyToSymbol ( const void* symbol, const void* src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice );` 把host上的变量/数组copy到symbol，要注意symbol其实是个引用，这里不能用&，因为不能在host上用&来获取device上变量的地址
- `__host__​cudaError_t cudaMemcpyFromSymbol ( void* dst, const T& symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost );` 类似的，从device上的symbol复制到host
- `__host__​cudaError_t cudaGetSymbolAddress ( void** devPtr, const void* symbol );` 获取一个symbol的地址，然后就可以用`cudaMemcpy`的函数来操作了
- `cudaError_t cudaMallocHost(void **devPtr, size_t count);` 在host上创建pinned memory，仍然需要copy到device memory上才能使用，需要使用`cudaFreeHost`来释放host pinned memory
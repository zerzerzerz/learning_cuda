# Shared Memory and Constant Memory
- GPU内存
  - on-borad memory (比如global memory，high latency)
  - on-chip memory (比如shared memory， low latency)
- Shared Memory
  - SMEM
  - 每个SM都有
  - 一个block内的全部thread都可以访问，是在当前block内共享的
- SMEM分配
  - 修饰符`__shared__`
  - local/global
    - 既可以在kernel内声明，只有当前kernel可见
    - 也可以在kernel外面声明，所有的kernel均可见
  - static/dynamic
    - `extern __shared__ int tile [];` 动态声明，既可以在kernel内部声明，也可以在kernel外面声明
    - 如果动态声明，kernel调用的时候，shared memory占用多少个字节需要在`<<<>>>`中作为第三个参数指定
- Memory Banks
  - SMEM划分为32个相同大小的module，称为memory bank，这32个memory bank可以被一个warp里面的32个线程同时访问
  - parallel（每个thread访问一个bank）
  - serial（若干个thread访问同一个bank，且在该bank里面访问的地址不同）
  - broadcast（全部thread访问同一个bank里面的同一个地址）
- Byte Mapping
  - Fermi架构，连续的4个字节映射到一个bank，每个bank的带宽是32bits / 2 clk cycles
  - Kepler架构，连续的4/8个字节映射到一个bank，每个bank的带宽是64bits / 1 clk cycle


## Synchronization
- Barrier
  - 全部的线程在这个地方都会等待其他线程到达这里
  - `__syncthreads();`可以让一个block内的全部线程同步
  - 如果在if中调用，需要确保当前块内全部线程都能走到这个地方
- Memory fence
  - All calling threads stall until all modifications to memory are visible to all other calling threads
  - Memory fence functions ensure that **any memory write before the fence is visible to other threads** after the fence
  - 这里的memory指的是global memory和shared memory
  - 有`__threadfence_block(); __threadfence(); __threadfence_system();`分别可以实现block内、grid内、device和host之间
- Volatile Qualifier
  - 在global或者shared memory中声明变量的时候，如果加上`volatile`修饰符，就可以阻止编译器优化它，也就是不会将其存放在register或者local memory中
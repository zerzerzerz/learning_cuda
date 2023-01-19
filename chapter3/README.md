# CUDA Execution Model
- 整理书的第三章
## SM
### Overview
- SM = stream multiprocessors,流式多处理器
- 一块GPU通常有很多SM
- 当kernel grid启动的时候，thread block就分配到可用的SM上，一个block只能分配到一个SM上，一个SM可以驻留很多block。直到这个block完成，它都一直停留在这个SM上
### SM的组成
- CUDA cores
- shared memory
- L1 cache
- Register file
- Load/Store Unit (LD/ST)
- Special Function Units
- Warp Scheduler & dispatcher
### CUDA管理线程的架构
- SIMT(Single Instruction Multiple Threads)
- **管理线程的时候，32个thread为一组，称为warp，在任意时刻，同一个warp里面的thread都执行相同的instruction**
- 每个thread都有自己的instruction address counter和register state
- SIMT有点类似于SIMD，但是同一个warp里面的thread是支持异步执行的，每个thread都有自己的指令地址计数器、寄存器状态、执行路径

|Logical|Hardware|
|-|-|
|thread|CUDA core|
|thread block|SM|
|grid|device|

- 在SM中，memory按照block进行划分，register按照thread进行划分，因此同一个block之间的不同thread可以进行协作和沟通
- 当一个warp空转的时候，SM可以直接切换到另一个available warp进行执行，这个切换几乎没有overhead，因为memory & register等等资源都已经分配好了

### Fermi GPU Architecture
- host interface
  - 通过PCI express连接cpu和gpu
- GigaThread
  - global scheduler，将block分配到SM warp scheduler
- Load/Store Units
  - 负责计算src和dst地址
- SFU
  - 执行一些内置函数，比如sin，cos，sqrt，插值
- Warp scheduler
  - GigaThread将block分配到warp scheduler之后，warp scheduler负责将这个block内的thread划分为warp
- Instruction dispatcher
  - 将指令分配给warp，对于一个warp，它里面的线程执行相同的指令
- Configurable memory
  - Fermi架构的一个特性就是提供了64KB的片上内存，划分为shared memory和L1 cache
  - 具体怎么划分，cuda提供了runtime api去指定多少内存划分为shared memory，多少划分为L1 cache


## Warp Execution
- 一个warp包含连续的32个线程，这些线程按照SIMT的方式运行，也就是同一时刻它们都执行相同的指令
- block可以被设置为1d，2d，3d的tensor，但这些都是逻辑上的，实际上从硬件角度来看，都是1d的tensor，`blockIdx.x`是最内层的坐标，`blockIdx.z`是最外层的坐标

## Warp Divergence
- 同一个warp里面的线程没有执行相同的if分支
- 尽可能去优化代码来保证同一个warp里面的thread都执行相同的if分支

## Resource Partition
- 对于warp来说，execution context包含下面这些
  - program counter
  - register
  - shared memory
- 这个context一直保存在on-chip直到这个warp生命周期结束，因此warp的切换很快，没有什么开销
- register保存在register file上，每个SM都有一组32bit的register，存储在register file上
- 一个SM通常可以驻留很多block，但如果1个block都放不下，这个kernel就fail
- 对于一个block而言，如果它需要的资源都已经分配好了，那么他就是active block
- active block包含的warp称为active warp，active warp又可以继续分为下面三类
  - selected（其实就是正在执行的warp）
  - stalled（熄火的，停滞的，也就是没准备好执行的）
  - eligible（有资格的，符合条件的，各种资源+指令都OK，随时都可以执行）
- 如果某个warp进入了stalled状态，warp scheduler立即找一个eligible warp去替代它

## Latency Hiding
- Latency
  - 一条指令从开始执行到完成执行，所需要的时钟周期数量
  - 为了将latency隐藏，在每个时钟周期，需要准备足够的eligible warp
- 指令分类
  - arithmetic instruction（10-20 clk cycle）
  - memory instruction（400-800 clk cycle）
- Little's Law
  - 用来估计所需的eligible warp
  - $number\_ of\_ required \_ warps = latency \times throughput$
- 对于算术指令，throughput（吞吐量）理解为每个时钟周期内正在执行的warp的数量
- 对于memory指令，吞吐量理解为内存速率也就是带宽bandwidth

## Synchronization
- System-level
  - 等待host和device的work全部完成
- Block-level
  - 等待一个block内的thread执行到相同的checkpoint



## nvprof查看metric
- 因为运行nvprof需要sudo权限，所以就简单列出常见的metric
- `branch_efficiency` $\frac{\#branch - \#diverged \ branch}{\#branch}$
- `achieved_occupancy` 查看active warp在全部warp中的比重
- `gld_throughput` 查看global load throughput，是个速率，memory read速率（g表示global，ld表示load）
- `gld_efficiency` ratio of requested global load throughput to required global load throughput
- `stall_sync`查看空转的warps

## 数组规模
- 如果数组的大小不是2的次幂，也没有关系，因为设置block的时候，可以设置block的规模是2的次幂
- 这样就能保证每个block处理相对简单的情况
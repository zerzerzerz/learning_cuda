# CUDA Memory Model
- 按照内存是否可编程，可以分为两种
  - programmable 可以显式控制数据放在那里
  - non-programmable 不能显式控制控制数据放在那里，依赖于自动控制
- cuda内存模型
  - register
  - shared memory
  - local memory
  - constant memory
  - texture memory
  - global memory

## Register
- gpu里面最快的，但是也是最小的
- kernel里面的自动变量一般放在寄存器里面，就是declare变量的时候，没有任何的修饰符
- kernel中声明的数组也可以放在寄存器中，前提是用来reference数据的indices，必须是常量，并且编译时可以确定, `Arrays declared in a kernel may also be stored in registers, but only if the indices used to reference the array are constant and can be determined at compile time.`
- 寄存器变量是每个thread私有的，生命周期和kernel一样
- scope是thread，lifetime也是thread
- 如果寄存器不够，就spill over（溢出）到local memory

## Local Memory
- 寄存器放不下，就溢出到local memory
- 数组索引编译时不确定，就放在这里`Local arrays referenced with indices whose values cannot be determined at compile-time.`
- 数组或者结构体太大，寄存器放不下，就放到这里
- 虽然名字叫做local memory，但是和global memory在相同的物理位置，并且也是高延迟，低带宽(high latency, low bandwidth)

## Shared Memory
- `__shared__`
- on-chip，和global和local比起来，高带宽低延迟
- 和L1 cache很像，但是可编程
- 同一个block内的线程都可以访问
- 访问shared需要线程间同步`void __syncthreads();`
- 可以设置L1 cache和shared memory之间的分配关系

## Constant Memory
- read-only
- 只能在host初始化，然后copy到device上
- 比较适合存储数学公式里面的系数和常量

## Texture Memory
- read-only
- 也是一种global memory，通过一种read-only cache进行访问，这种cache支持hardware filtering，可以在read阶段进行浮点数插值
- texture memory针对2D data做了优化，比较适合访问2D data

## Global Memory
- 最大、最慢、最普遍
- 可以static声明（`__device__`）也可以dynamic声明（`cudaMalloc`）

## GPU Caches
- L1
- L2
- Read-only constant
- Read-only texture
- cache都是不可编程的，memory load可以被cached，但是memory store不能被cached
- L2 cache只有1个，被SM共享，其他cache都是每个SM一个
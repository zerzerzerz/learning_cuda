# Streams and Concurrency
- 之前探讨的并行都是kernel级别的，比如循环展开、调整blocksize等等
- 下面来探讨grid级别的并行，也就是许多不同的kernel同时执行

## CUDA Operation
- 通常可以被分类为同步和异步的
- synchronous operation指的是，调用的时候会block host，直到完成
- asynchronous operation指的是，调用之后，立刻将control返回给host

## CUDA Stream
> - A CUDA stream refers to a sequence of asynchronous CUDA operations which execute on a device in the order issued by the host code
> - 也就是将一系列的cuda异步操作放在一起，就构成了一个stream，stream里面的操作按照host code指定的顺序来
- 可以将stream理解为一个queue，里面的元素是异步的cuda运算，这些算子按照顺序依次执行
- 上面提到的异步指的是，kernel和host是异步的，一个kernel被调用之后，控制权立马交还给host
- 同一个stream中的operation有着严格的执行顺序，但是不同stream之间没啥要求
- stream中OP的执行顺序就是按照host code上指定的来的
### Types of CUDA Streams
- null stream (default stream)
  - 调用kernel和进行数据传递的时候，如果没有指定stream，就用的这个默认的stream
  - synchronous stream，大部分(most operations)这里面的操作都会阻塞host的执行，**但是kernel launch不会，kernel launch还是异步的，不会阻塞host**
- non-null stream
  - 也就是asynchronous stream，里面的operations都不会阻塞host的执行(ALL operations)
  - 又可以继续分为下面两类
    - blocking stream
      - blocking stream是异步的，但是会等前面的null stream
      - 同时null stream也会等前面的blocking stream
    - non-blocking stream
      - 谁也不等
- `cudaError_t cudaStreamSynchronize(cudaStream_t stream);`是让某个stream和host进行同步，让host强制等待stream完成
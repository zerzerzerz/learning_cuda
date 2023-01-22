# Shared Memory and Constant Memory
- GPU内存
  - on-borad memory (比如global memory)
  - on-chip memory (比如shared memory)
- Shared Memory
  - 每个SM都有
- SMEM划分为32个bank
  - parallel（每个thread访问一个bank）
  - serial（若干个thread访问同一个bank且在该bank里面访问的地址不同）
  - broadcast（全部thread访问同一个bank里面的同一个地址）
# NVCC Compilation
- cuda的编译器驱动（compiler driver）就是nvcc
- 首先将code分离为host code和device code
  - host code支持完整的C++代码，但是device只支持部分的C++
- nvcc首先将device code编译为PTX伪汇编代码(assembly)，再将PTX编译为二进制的cubin目标代码
- 编译PTX
  - 通过`-arch=compute_XY`指定虚拟架构计算能力，来确定代码中能够使用的cuda功能
- 编译cubin代码
  - 通过`-code=sm_XY`来指定真实架构的计算能力
  - 编译生成的二进制cubin文件只能在`X.Z`的GPU上运行，其中`Z >= Y`
  - 真实架构的计算能力必须大于等于虚拟架构的计算能力
- fat binary
```bash
nvcc src.cu \
-gencode arch=compute_35,code=sm_35 \
-gencode arch=compute_50,code=sm_50 \
-gencode arch=compute_60,code=sm_60 \
-gencode arch=compute_70,code=sm_70 \
-o tgt.cubin
```
- 这样编译出来的二进制文件包含4个二进制文件，编译出来的总文件称为fat binary，在不同GPU中可以自动选择二进制版本，但是会增加文件大小和编译时间
- JIT(Just-In-Time)
  - 可以在运行可执行文件的时候，从其中保留的PTX代码中，临时编译出一个cubin文件
  - 如果想在可执行文件中保留/嵌入PTX代码，需要这样指定，两个虚拟架构必须完全一致
  - 可以在将来更先进架构的GPU上，运行可执行文件的时候，加载并编译嵌入在其中的PTX文件
```bash
-gencode arch=compute_XY,code=compute_XY
```
- 如果是`-arch=sm_XY`，等价于`-gencode arch=compute_XY, code=sm_XY -gencode arch=compute_XY, code=compute_XY`，看来是同时嵌入了PTX代码
utils = ../utils/utils.cu
CUDA_VISIBLE_DEVICES=1

transpose: transpose.cu $(utils)
	export CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES); \
	nvcc $^ -o $@.exe; \
	./$@.exe

reduce_global_memory: reduce_global_memory.cu $(utils)
	nvcc $^ -o $@.exe
	./$@.exe

try: try.cu $(utils)
	nvcc $^ -o $@.exe
	./$@.exe

.PHONY: clean

clean:
	if ls *.exe > /dev/null 2>&1; then rm *.exe; fi
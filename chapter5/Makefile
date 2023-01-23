utils = ../utils/utils.cu

transpose: transpose.cu $(utils)
	nvcc $^ -o $@.exe
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
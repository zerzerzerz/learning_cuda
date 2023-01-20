utils = ../utils/utils.cu


unified_memory2: unified_memory2.cu $(utils)
	nvcc $^ -o $@.exe
	./$@.exe 16 16

unified_memory: unified_memory.cu $(utils)
	nvcc $^ -o $@.exe
	./$@.exe 512


uva: uva.cu $(utils)
	nvcc $^ -o $@.exe
	./$@.exe

cpu_memory_allocation: cpu_memory_allocation.c
	$(CC) $< -o $@.exe
	./$@.exe

static_declard_global_variable: static_declard_global_variable.cu $(utils)
	nvcc $^ -o $@.exe
	./$@.exe

.PHONY: clean

clean:
	if [ -f *.exe ]; then rm *.exe; fi
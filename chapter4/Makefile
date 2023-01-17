utils = ../utils/utils.cu

static_declard_global_variable: static_declard_global_variable.cu $(utils)
	nvcc $^ -o $@.exe
	./$@.exe

.PHONY: clean

clean:
	if [ -f *.exe ]; then rm *.exe; fi
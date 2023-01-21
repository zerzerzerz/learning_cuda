utils=../utils/utils.cu

sum: sum.cu $(utils)
	nvcc $^ -o $@.exe
	./$@.exe

sum_1d_array_dynamic_parallelism:
	nvcc sum_1d_array_dynamic_parallelism.cu $(utils) -arch=sm_35 -rdc=true -lcudadevrt -o sum_1d_array_dynamic_parallelism.exe  
	./sum_1d_array_dynamic_parallelism.exe

dynamic_parallelism:
	nvcc dynamic_parallelism.cu -arch=sm_35 -rdc=true -lcudadevrt -o dynamic_parallelism.exe  
	./dynamic_parallelism.exe

sum_1d_array:
	nvcc sum_1d_array.cu -o sum_1d_array.exe
	./sum_1d_array.exe

check:
	nvcc check.cu -o check.exe
	./check.exe

tmp: tmp.c time.o
	$(CC) -c tmp.c -o $@.o
	$(CC) -o $@.exe $@.o time.o
	./$@.exe

time.o: ../utils/time.c ../utils/time.h
	$(CC) -c $< -o $@


clean:
	if ls *.exe > /dev/null 2>&1; then rm *.exe; fi
utils=../utils/utils.cu

matrix_multiplication:
	nvcc matrix_multiplication.cu $(utils) -o matrix_multiplication.exe
	./matrix_multiplication.exe

sum_matrix:
	nvcc sum_matrix.cu -o sum_matrix.exe
	./sum_matrix.exe

sum_device:
	nvcc sum_device.cu -o sum_device.exe
	./sum_device.exe

sum_host:
	nvcc sum_arrays_host.cu -o sum_host.exe

check_index:
	nvcc check_index.cu -o check_index.exe
	./check_index.exe

tmp:
	nvcc tmp.cu -o tmp.exe
	./tmp.exe

clean:
	if [ -e *.exe ]; then rm *.exe; fi
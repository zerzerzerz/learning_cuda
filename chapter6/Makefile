test: test.cu
	nvcc $^ -o $@.exe
	./$@.exe

.PHONY: clean
clean:
	if ls *.exe > /dev/null 2>&1; then rm *.exe; fi
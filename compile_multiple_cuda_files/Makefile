main.exe: main.cu utils.cu
	nvcc $^ -o $@ 

.PHONY: clean
clean:
	rm *.exe
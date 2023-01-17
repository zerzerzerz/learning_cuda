main.exe: main.cu utils.cu
	nvcc $^ -o $@ 

.PHONY: clean
clean:
	if [ -e *.exe ]; then rm *.exe; fi
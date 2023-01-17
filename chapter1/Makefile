.PHONY: clean
target:
	nvcc hello.cu -arch sm_61 -o hello.exe

clean:
	if [ -e *.exe ]; then rm *.exe; fi
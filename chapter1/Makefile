.PHONY: clean
target:
	nvcc hello.cu -arch sm_61 -o hello.exe

clean:
	if ls *.exe > /dev/null 2>&1; then rm *.exe; fi
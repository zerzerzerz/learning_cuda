.PHONY: clean
target:
	nvcc hello.cu -o hello.exe
	./hello.exe

clean:
	if ls *.exe > /dev/null 2>&1; then rm *.exe; fi
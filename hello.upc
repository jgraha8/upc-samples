#include <upc_relaxed.h>
#include <stdio.h>

int main(int argc_, char *argv_[]) {
	printf("Hello from thread %i/%i\n", MYTHREAD, THREADS);
	upc_barrier;
	return 0;
}



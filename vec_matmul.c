#include <upc_relaxed.h>

#define N THREADS

shared [N] int a[N][N];
shared int b[N], c[N];

int main(void)
{
	int i,j;

	upc_forall( i=0; i<N; i++; i) {
		printf("%d: %d\n", MYTHREAD, i);
		c[i] = 0;
		for( j=0; j<N; j++ ) {
			c[i] += a[i][j]*b[j];
		}
	}

	for( i=0; i<N; i++ ) {
		if( i == MYTHREAD ) {
			printf("&a[0][0] = %p\n", &a[MYTHREAD][0]);
			printf("c[%d] = %d\n", i, c[i]);
		}
		upc_barrier;
	}
	
	return 0;
}

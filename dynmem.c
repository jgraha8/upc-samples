#include <upc_relaxed.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>

typedef struct {
	struct timeval stop, start;	
} uclock_t;

void uclock_start( uclock_t *this_ )
{
	gettimeofday(&this_->start, NULL);
}

void uclock_stop( uclock_t *this_ )
{
	gettimeofday(&this_->stop, NULL);
}

double uclock_time( const uclock_t *this_ )
{
	const double c = pow(10.0,-6.0);
	return ((double)(this_->stop.tv_sec + c*this_->stop.tv_usec -
			 this_->start.tv_sec - c*this_->start.tv_usec));
}


#define N (256*256)
#define CYCLES 1000
int main(void)
{

	uclock_t clock;
	
	shared[N] int *ptr=NULL;
	shared[N] int *ptr_chk=NULL;	
	int *pl=NULL, *pl_chk=NULL;
	int i,n;

	const int offset = MYTHREAD*N;	

	ptr = (shared[N] int *)upc_all_alloc(THREADS,N*sizeof(int));
	ptr_chk = (shared[N] int *)upc_all_alloc(THREADS,N*sizeof(int));
	upc_barrier;

	// Set local pointer
	pl = (int *)&ptr[MYTHREAD*N];
	pl_chk = (int *)&ptr_chk[MYTHREAD*N];	

	printf("%d: threadof, phaseof, addrfield = %d, %d, %p\n", MYTHREAD, upc_threadof(ptr),
	       upc_phaseof(ptr), upc_addrfield(ptr));
	upc_barrier;

	uclock_start(&clock);
	for( n=0; n<CYCLES; n++ ) {
		upc_forall( i=0; i<N*THREADS; i++; &ptr[i]) {
			ptr[i] = i;
		}
	}
	upc_barrier;
	uclock_stop(&clock);	

	if( MYTHREAD == 0 )
		printf("forall parallel time %15.7e\n", uclock_time(&clock));

	upc_barrier;
	uclock_start(&clock);

	for( n=0; n<CYCLES; n++ ) {
		for( i=0; i<N; i++) {
			pl_chk[i] = offset+i;
		}
	}
	upc_barrier;
	uclock_stop(&clock);	

	if( MYTHREAD == 0 )
		printf("local parallel time %15.7e\n", uclock_time(&clock));

	// Check pointer values
	for( i=0; i<N; i++ )
		assert( ptr[i + offset] == pl_chk[i] );
	
	upc_barrier;
	uclock_start(&clock);
	if( MYTHREAD == THREADS-1 ) {
		for( n=0; n<CYCLES; n++ ) {
			for( i=0; i<N*THREADS; i++ ) {
				ptr[i] = 0;
			}
		}
	}
	upc_barrier;
	uclock_stop(&clock);
	if( MYTHREAD == 0 )
		printf("single time %15.7e\n", uclock_time(&clock));
		

	// Have tr = THREAD-1 set the ptr values
	if( MYTHREAD == 0 ) {
		pl[0] = -2;
	}
	upc_barrier;

	if( MYTHREAD == 0 ) {
		printf("ptr[0] = %d\n", ptr[0]);
	}
	upc_barrier;
	
	upc_all_free(ptr);
	
	return 0;
}

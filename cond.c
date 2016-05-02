#include <upc_relaxed.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdbool.h>

#define NX 512
#define NY 512

#define DX 0.05
#define DY 0.05
#define NSTEPS 100000

#define ALPHA 1.0

#define CFL 0.95

#define NWORKSP 3
#define OUTPUT_FREQ 5.0

#define TOL 1.0e-6

static bool buffer_shared = false;

typedef struct {
	struct timeval stop, start;
	double time;
} uclock_t;

double uclock_time( const uclock_t *this_ )
{
	const double c = pow(10.0,-6.0);
	return ((double)(this_->stop.tv_sec + c*this_->stop.tv_usec -
			 this_->start.tv_sec - c*this_->start.tv_usec));
}

void uclock_start( uclock_t *this_ )
{
	gettimeofday(&this_->start, NULL);
}

void uclock_stop( uclock_t *this_ )
{
	gettimeofday(&this_->stop, NULL);
	this_->time = uclock_time(this_);
	
}


void uclock_split_stop( uclock_t *this_ )
{
	gettimeofday(&this_->stop, NULL);
	this_->time += uclock_time(this_);
	
}

typedef struct {
	int dims[2];
	size_t size;
	shared[] double *sh_v;
	double *v;
	double *__v_nbr; // Buffer for holding neighbor values
} field_t;

typedef struct {
	int th_id, th_size;
	int dims[2]; // Global dims
	size_t size; // Global size
	shared[2] int (*sh_offset)[2]; // offset for local field block
	int (*offset)[2];
	shared field_t *sh_field;
	field_t *field;
} pfield_t;


void field_ctor( field_t *this_, int *dims_ )
{
	memcpy(this_->dims, dims_, sizeof(this_->dims));
	this_->size = dims_[0]*dims_[1];
	this_->sh_v = (shared[] double *)upc_alloc(this_->size*sizeof(*this_->v));
	this_->v = (double *)&this_->sh_v[0];
	this_->__v_nbr = (double *)malloc(dims_[1]*sizeof(double));
}

void field_dtor( field_t *this_ )
{
	free(this_->__v_nbr);
	this_->v = NULL;
	upc_free(this_->sh_v);
	this_->size = 0;
	memset(this_->dims, 0, sizeof(this_->dims));
}

inline void field_copy( field_t *this_, const field_t *f_ )
{
	assert(this_->size == f_->size );
	memcpy( this_->v, f_->v, this_->size*sizeof(*this_->v));
}

inline size_t field_eval_index(const field_t *this_, int i_, int j_ )
{
	return (size_t)i_*this_->dims[1] + j_;
}

void pfield_ctor( pfield_t *this_, int th_id_, int th_size_, int *dims_ )
{
	shared[2] int (*sh_dims_th)[2] = (shared[2] int (*)[2])upc_all_alloc(th_size_,sizeof(int [2]));
	int i;

	// Set the thread info and global parameters
	this_->th_id   = th_id_;
	this_->th_size = th_size_;
	memcpy(this_->dims, dims_, sizeof(this_->dims));
	this_->size = dims_[0]*dims_[1]; 

	// Set the thread dimensions in the shared vector
	sh_dims_th[th_id_][0] = dims_[0]/th_size_ + (1 ? th_id_ < dims_[0]%th_size_ : 0); // Remaining dimensions are thread cyclic
	sh_dims_th[th_id_][1] = dims_[1];

	this_->sh_offset = (shared[2] int (*)[2])upc_all_alloc(th_size_,sizeof(int [2]));
	this_->offset = (int (*)[2])(this_->sh_offset + th_id_);
	memset( this_->offset, 0, sizeof(int [2]));

	upc_barrier;

	for( i=1; i<th_size_; i++ ) {	
		if( i == th_id_ ) {
			this_->sh_offset[i][0] = this_->sh_offset[i-1][0] + sh_dims_th[i-1][0];
		}
		upc_barrier;
	}

	this_->sh_field = (shared field_t *)upc_all_alloc(th_size_,sizeof(field_t));
	this_->field = (field_t *)(this_->sh_field + th_id_);
	
	// Determine the local thread info
	field_ctor(this_->field, (int *)sh_dims_th[th_id_]);

	upc_all_free(sh_dims_th);
	upc_barrier;
}

void pfield_dtor( pfield_t *this_ )
{
	field_dtor(this_->field);
	upc_all_free(this_->sh_field);
	this_->size = 0;
	memset(this_->dims, 0, sizeof(this_->dims));
	this_->th_size = 0;
	this_->th_id = 0;

	upc_barrier;	
}

void pfield_set( pfield_t *this_, double v_ )
{
	field_t *f = this_->field;
	const double *v_end = f->v + f->size;
	double *v = f->v;
	while( v < v_end ) *(v++) = v_;

	upc_barrier;	
}

inline void pfield_copy( pfield_t *this_, const pfield_t *f_ )
{
	field_copy(this_->field, f_->field );
}

// Returns the local block vector index for a given thread reference
inline size_t pfield_local_index_th(const pfield_t *this_, int th_id_, int i_, int j_ )
{
	return (size_t)i_*this_->sh_field[th_id_].dims[1] + j_;
}

inline void pfield_eval_ij_th(const pfield_t *this_, int th_id_, int i_th_, int j_th_, int *i_, int *j_ )
{
	*i_ = i_th_ + this_->sh_offset[th_id_][0];
	*j_ = j_th_;
}

double field_get_max( const field_t *this_ )
{
	double v_max = -1.0e32;
	double *v     = this_->v;
	double *v_end = v + this_->size;

	while( v < v_end ) {
		if( v_max < *v ) v_max = *v;
		v++;
	}
	return v_max;
}

double field_get_min( const field_t *this_ )
{
	double v_min = 1.0e32;
	double *v     = this_->v;
	double *v_end = v + this_->size;

	while( v < v_end ) {
		if( *v < v_min ) v_min = *v;
		v++;
	}
	return v_min;
}

shared double sh_val; // Shared value with affinity thread=0
upc_lock_t *lock = NULL;

void init_locks()
{
	lock = upc_all_lock_alloc();
}

void fini_locks()
{
	if( MYTHREAD == 0 ) {
		if( lock ) upc_lock_free(lock);
	}
}
double allreduce_max( double v_, int th_id_ )
{
	assert( lock );

	if( th_id_ == 0 ) {
		sh_val = -1.0e32;
	}
	upc_lock( lock );
	if( sh_val < v_ ) {
		sh_val = v_;
	}
	upc_unlock( lock );

	upc_barrier;
	
	return sh_val;
}

double allreduce_min( double v_, int th_id_ )
{
	assert( lock );

	if( th_id_ == 0 ) {
		sh_val = 1.0e32;
	}
	upc_lock( lock );
	if( v_ < sh_val ) {
		sh_val = v_;
	}
	upc_unlock( lock );

	upc_barrier;
	
	return sh_val;
}

inline double pfield_get_max( const pfield_t *this_ )
{
	double v_max = field_get_max(this_->field );
	v_max = allreduce_max( v_max, this_->th_id );
	return v_max;
}

inline double pfield_get_min( const pfield_t *this_ )
{
	double v_min = field_get_min(this_->field );
	v_min = allreduce_min( v_min, this_->th_id );
	return v_min;
}
void pfield_eval_ddx( pfield_t *this_, pfield_t *ddx_ )
{
	assert( this_->dims[0] == ddx_->dims[0] &&
		this_->dims[1] == ddx_->dims[1] );

	const field_t *f=this_->field;
        double *dv = ddx_->field->v;
	
	double *v_im1, *v_ip1;

	int th_id_nbr;
	const shared field_t *f_nbr;
	const shared[] double *sh_v_im1, *sh_v_ip1;
	
	int i,j,idx;

	// Interior, non-boundary (physical and inter-thread) points
	for(i=1; i<f->dims[0]-1; i++ ) {
		v_im1 = f->v + field_eval_index(f,i-1,0);
		v_ip1 = f->v + field_eval_index(f,i+1,0);
		idx = field_eval_index(f,i,0);
		for( j=0; j<f->dims[1]; j++ ) {
			dv[idx++] = 0.5*(v_ip1[j] - v_im1[j]) / DX;
		}
	}

	// Left inter-thread boundary
	if( 0 < this_->th_id ) {

		i = 0;
		v_ip1 = f->v + field_eval_index(f,i+1,0);
		v_im1 = f->__v_nbr;

		th_id_nbr = this_->th_id - 1;
		f_nbr = this_->sh_field + th_id_nbr;
		sh_v_im1 = f_nbr->sh_v + pfield_local_index_th( this_, th_id_nbr, f_nbr->dims[0]-1, 0);

		idx = field_eval_index(f,i,0);
		if( buffer_shared ) {
			upc_memget( v_im1, sh_v_im1, f->dims[1]*sizeof(double) );		

			for( j=0; j<f->dims[1]; j++ ) {	
				dv[idx++] = 0.5*(v_ip1[j] - v_im1[j]) / DX;
			}
		} else {
			for( j=0; j<f->dims[1]; j++ ) {	
				dv[idx++] = 0.5*(v_ip1[j] - sh_v_im1[j]) / DX;
			}
		}
		
	}
	
	// Right inter-thread boundary
	if( this_->th_id < this_->th_size-1 ) {

		i = f->dims[0]-1;
		v_ip1 = f->__v_nbr;
		v_im1 = f->v + field_eval_index(f,i-1,0); // Local vector

		th_id_nbr = this_->th_id+1;
		f_nbr     = this_->sh_field + th_id_nbr; // Neighbor thread
		sh_v_ip1  = f_nbr->sh_v + pfield_local_index_th( this_, th_id_nbr, 0, 0 );

		idx = field_eval_index(f,i,0);

		if( buffer_shared ) {
			upc_memget( v_ip1, sh_v_ip1, f->dims[1]*sizeof(double) );		

			for( j=0; j<f->dims[1]; j++ ) {
				dv[idx++] = 0.5*(v_ip1[j] - v_im1[j]) / DX;
			}
		} else {
			for( j=0; j<f->dims[1]; j++ ) {
				dv[idx++] = 0.5*(sh_v_ip1[j] - v_im1[j]) / DX;
			}
		}
	}
	
	// Left physical boundary
	if( this_->th_id == 0 ) {
		i = 0;
		v_ip1 = f->v + field_eval_index(f,i+1,0);		
		idx = field_eval_index(f,i,0);
		for( j=0; j<f->dims[1]; j++ ) {
			dv[idx] = ( v_ip1[j] - f->v[idx] ) / DX;
			idx++;
		}
	}

	// Right phyical boundary
	if( this_->th_id ==  this_->th_size-1 ) {
		i = f->dims[0]-1;
		v_im1 = f->v + field_eval_index(f,i-1,0);
		
		idx = field_eval_index(f,i,0);
		for( j=0; j<f->dims[1]; j++ ) {
			dv[idx] = ( f->v[idx] - v_im1[j] ) / DX;
			idx++;
		}
	}

	upc_barrier;
	
}

void pfield_eval_ddy( const pfield_t *this_, pfield_t *ddy_ )
{
	assert( this_->dims[0] == ddy_->dims[0] &&
		this_->dims[1] == ddy_->dims[1] );

	const field_t *f=this_->field;
	const double *v = f->v;
        double *dv = ddy_->field->v;
	
	int i,j,idx;

	for( i=0; i<f->dims[0]; i++ ) {
		// j=0
		j=0;
		idx = field_eval_index(f,i,j);
		dv[idx] = ( v[idx+1] - v[idx] ) / DY;

		for( j=1; j<f->dims[1]-1; j++ ) {
			idx++;
			dv[idx] = 0.5*( v[idx+1] - v[idx-1] ) / DY;
		}

		j=f->dims[1]-1;
		idx = field_eval_index(f,i,j);
		dv[idx] = ( v[idx] - v[idx-1] ) / DY;		
	}

	upc_barrier;
}

void pfield_eval_lap( pfield_t *this_, int nws_, pfield_t *ws_, pfield_t *lap_ )
{
	size_t i;
	assert( nws_ >= 3 );
	pfield_eval_ddx( this_, ws_ );
	pfield_eval_ddx( ws_, ws_+1 );

	pfield_eval_ddy( this_, ws_ );
	pfield_eval_ddy( ws_, ws_+2 );

	double *d2x = ws_[1].field->v;
	double *d2y = ws_[2].field->v;	
	double *L   = lap_->field->v;

	for( i=0; i<lap_->field->size; i++ )
		L[i] = d2x[i] + d2y[i];

	upc_barrier;
	     
}

void pfield_mul_scalar( pfield_t *this_, double s_ )
{
	size_t size = this_->field->size;
	double *v = this_->field->v;

	size_t i;
	for( i = 0; i<size; i++ )
		v[i] *= s_;

	upc_barrier;
}

void pfield_add( pfield_t *this_, const pfield_t *g_ )
{
	size_t size = this_->field->size;
	double *v = this_->field->v;
	double *v_g = g_->field->v;

	size_t i;
	for( i = 0; i<size; i++ )
		v[i] += v_g[i];

	upc_barrier;
}

void pfield_write_file( const pfield_t *this_, const char *fname_ )
{
	shared field_t *sh_f;
	shared[] double *sh_v;
	double *v;

	int i,j,idx, n;
	int i_g, j_g;

	FILE *fid = fopen(fname_,"w");
	
	for( n=0; n<this_->th_size; n++ ) {
		sh_f      = this_->sh_field + n;
		sh_v      = sh_f->sh_v;
		for( i=0; i<sh_f->dims[0]; i++ ) {
			pfield_eval_ij_th( this_, n, i, 0, &i_g, &j_g );			  
			idx = pfield_local_index_th( this_, n, i, 0 );
			for( j=0; j<sh_f->dims[1]; j++ ) {
				fprintf(fid,"%15.7e %15.7e %15.7e\n", i_g*DX, (j_g++)*DY, sh_v[idx++]);
			}
			fprintf(fid,"\n");
		}
	}
	fclose(fid);
}

void set_bc( pfield_t *T_ )
{
	field_t *f = T_->field;
	double *v = f->v;
	int i,j,ii;
	
	if( T_->th_id == 0 ) {
		i=0;
		ii = field_eval_index(f,i,0);
		for( j=0; j<f->dims[1]; j++ ) {
			v[ii++] = 0.0;
		}
	}
	
	if( T_->th_id == T_->th_size - 1 ) {
		i=f->dims[0]-1;
		ii = field_eval_index(f,i,0);
		for( j=0; j<f->dims[1]; j++ ) {
			v[ii++] = 0.0;
		}
	}

	for( i=0; i<f->dims[0]; i++ ) {
		j=0;
		ii = field_eval_index(f,i,j);
		v[ii] = 0.0;

		j=f->dims[1]-1;
		//printf("ii = %d\n", ii);		
		ii = field_eval_index(f,i,j);
		v[ii] = 1.0;
	}

	upc_barrier;
}

double compute_Linf_norm( const pfield_t *f_, const pfield_t *g_ )
{
	assert( f_->field->size == g_->field->size );

	const size_t N = f_->field->size;
	const double *f_v = f_->field->v;
	const double *g_v = g_->field->v;

	size_t i;
	double dv;
	double dv_max = -1.0e32;
	
	// Find the max difference between the two fields
	for( i=0; i<N; i++ ) {
		dv = fabs( f_v[i] - g_v[i] );
		if( dv_max < dv ) dv_max = dv;
	}

	return allreduce_max( dv_max, f_->th_id );
}

int main(int argc_, char *argv_[])
{

	pfield_t T[2], D;
	pfield_t ws[3];
	uclock_t clock;
	shared[] uclock_t *sh_elapsed_clock = (shared[] uclock_t *)upc_all_alloc( 1, sizeof(uclock_t) );
	double T_max, T_min;
	double err=1.0;
	
	const double dt = pow( ( DX < DY ? DX : DY ), 2.0 ) * CFL / ALPHA;

	int i,n;
	int dims[] = { NX, NY };

	n=1;
	while( n < argc_ ) {
		if( strncmp(argv_[n], "-b", 2) == 0 ) {
			buffer_shared = true;
			if( MYTHREAD == 0 )
				printf("Buffering shared\n");
		} else {
			printf("unknown argument: %s\n", argv_[n]);
		}
		n++;	
	}

	clock.time = 0.0;
	if( MYTHREAD == 0 ) sh_elapsed_clock->time = 0.0;

	init_locks();
	
	pfield_ctor(T, MYTHREAD, THREADS, dims);
	pfield_ctor(T+1, MYTHREAD, THREADS, dims);	
	pfield_ctor(&D, MYTHREAD, THREADS, dims );

	for( n=0; n<NWORKSP; n++ )
		pfield_ctor(ws+n, MYTHREAD, THREADS, dims );
	
	pfield_set(T,0.0);
	set_bc(T);

	if( MYTHREAD == 0 ) 
		pfield_write_file( T, "T-init.dat" );

	n=0;
	err = 1.0;
	
	while( err > TOL ) {

		n++;

		upc_barrier;

		if( MYTHREAD == 0 ) {
			uclock_start((uclock_t *)sh_elapsed_clock);
			uclock_start( &clock );
		}

		pfield_copy( T+1, T );	
		
		pfield_eval_lap( T, NWORKSP, ws, &D);
		pfield_mul_scalar( &D, dt*ALPHA );

		// Perform update
		pfield_add( T, &D );
		set_bc(T);

		err = compute_Linf_norm( T, T+1 );

		upc_barrier;
		if( MYTHREAD == 0 ) {
			uclock_split_stop( &clock );
			uclock_split_stop( (uclock_t *)sh_elapsed_clock );			
		}
		upc_barrier;
		
		if( sh_elapsed_clock->time > OUTPUT_FREQ ) {
			T_max = pfield_get_max( T );
			T_min = pfield_get_min( T );
			if( MYTHREAD == 0 )  {
				sh_elapsed_clock->time = 0.0;
				printf("iteration = %d, exec speed [iter/s] = %15.7e, err/tol = %15.7e, T_max = %15.7e, T_min = %15.7e\n", n, (double)n / clock.time, err / TOL, T_max, T_min);
				fflush(stdout);
			}
		}

	}

	if( MYTHREAD == 0 ) {
		printf("Total wall time = %15.7e s\n", clock.time);
		fflush(stdout);

		pfield_write_file( T, "T.dat");
	}

	upc_all_free( sh_elapsed_clock );
	
	for( n=0; n<NWORKSP; n++ )
		pfield_dtor(ws+n);

	pfield_dtor(&D);
	pfield_dtor(T);
	pfield_dtor(T+1);	

	fini_locks();
	
	upc_global_exit;

	return 0;
}

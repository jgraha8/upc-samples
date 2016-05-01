#include <upc_relaxed.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>

#define NX 32
#define NY 32

#define DX 0.05
#define DY 0.05
#define NSTEPS 1000

#define ALPHA 1.0
#define DT 0.0001

#define NWORKSP 3
#define OUTPUT_FREQ 100
typedef struct {
	int offset[2]; // offset for local field block	
	int dims[2];
	size_t size;
	shared[] double *sh_v;
	double *v;
} field_t;

typedef struct {
	int th_id, th_size;
	int dims[2]; // Global dims
	size_t size; // Global size
	shared field_t *sh_field;
	field_t *field;
} pfield_t;


void field_ctor( field_t *this_, int *dims_, int *offset_ )
{
	memcpy(this_->dims, dims_, sizeof(this_->dims));
	memcpy(this_->offset, offset_, sizeof(this_->offset));
	this_->size = dims_[0]*dims_[1];
	this_->sh_v = (shared[] double *)upc_alloc(this_->size*sizeof(*this_->v));
	this_->v = (double *)&this_->sh_v[0];
}

void field_dtor( field_t *this_ )
{
	this_->v = NULL;
	upc_free(this_->sh_v);
	this_->size = 0;
	memset(this_->dims, 0, sizeof(this_->dims));
}

inline size_t field_eval_index(const field_t *this_, int i_, int j_ )
{
	return (size_t)i_*this_->dims[1] + j_;
}

void pfield_ctor( pfield_t *this_, int th_id_, int th_size_, int *dims_ )
{
	int dims_th[] = { dims_[0]/th_size_ + (1 ? th_id_ < dims_[0]%th_size_ : 0),
			  dims_[1] };
	int offset_th[] = { 0, 0 };
	int i;
	
	this_->th_id   = th_id_;
	this_->th_size = th_size_;
	memcpy(this_->dims, dims_, sizeof(this_->dims));
	this_->size = dims_[0]*dims_[1];

	memset(offset_th, 0, sizeof(offset_th));
	for( i=0; i<this_->th_id; i++ ) {
		offset_th[0] += dims_[0]/th_size_ + (1 ? i < dims_[0]%th_size_ : 0);
	}

	this_->sh_field = (shared field_t *)upc_all_alloc(th_size_,sizeof(field_t));
	this_->field = (field_t *)&this_->sh_field[th_id_];
	
	// Determine the local thread info
	field_ctor(this_->field, dims_th, offset_th);

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

inline size_t pfield_eval_index(const pfield_t *this_, int i_, int j_ )
{
	return (size_t)i_*this_->dims[1] + j_;
}

inline size_t pfield_eval_index_local(const pfield_t *this_, int i_, int j_ )
{
	return (size_t)(i_+this_->field->offset[0])*this_->dims[1] + j_;
}

inline size_t sh_field_eval_index(const shared field_t *this_, int i_, int j_ )
{
	return (size_t)i_*this_->dims[1] + j_;
}

void pfield_eval_ddx( const pfield_t *this_, pfield_t *ddx_ )
{
	assert( this_->dims[0] == ddx_->dims[0] &&
		this_->dims[1] == ddx_->dims[1] );

	const field_t *f=this_->field;

        double *dv = ddx_->field->v;
	
	const double *v_im1, *v_ip1;

	const shared field_t *f_nbr;
	const shared[] double *sh_v_im1, *sh_v_ip1;
	
	int i,j,ii;

	for(i=1; i<f->dims[0]-1; i++ ) {
		v_im1 = f->v + field_eval_index(f,i-1,0);
		v_ip1 = f->v + field_eval_index(f,i+1,0);
		ii = field_eval_index(f,i,0);
		for( j=0; j<f->dims[1]; j++ ) {
			dv[ii++] = 0.5*(v_ip1[j] - v_im1[j]) / DX;
		}
	}


	// Left inter-thread boundary
	if( 0 < this_->th_id ) {
		i = 0;
		v_ip1 = f->v + field_eval_index(f,i+1,0);

		f_nbr = this_->sh_field + this_->th_id - 1;
		sh_v_im1 = f_nbr->sh_v + sh_field_eval_index(f_nbr,f_nbr->dims[0]-1,0);

		ii = field_eval_index(f,i,0);
		for( j=0; j<f->dims[1]; j++ ) {
			dv[ii++] = 0.5*(v_ip1[j] - sh_v_im1[j]) / DX;
		}	
		
	}
	
	// Right inter-thread boundary
	if( this_->th_id < this_->th_size-1 ) {
		
		i = f->dims[0]-1;
		v_im1    = f->v + field_eval_index(f,i-1,0); // Local vector
		
		f_nbr   = this_->sh_field + this_->th_id+1; // Neighbor thread
		sh_v_ip1 = f_nbr->sh_v + sh_field_eval_index(f_nbr,0,0);

		ii = field_eval_index(f,i,0);
		for( j=0; j<f->dims[1]; j++ ) {
			dv[ii++] = 0.5*(sh_v_ip1[j] - v_im1[j]) / DX;
		}	
	}
	
	// Left physical boundary
	if( this_->th_id == 0 ) {
		i = 0;
		v_ip1 = f->v + field_eval_index(f,i+1,0);		
		ii = field_eval_index(f,i,0);
		for( j=0; j<f->dims[1]; j++ ) {
			dv[ii] = ( v_ip1[j] - f->v[ii] ) / DX;
			ii++;
		}
	}

	// Right phyical boundary
	if( this_->th_id ==  this_->th_size-1 ) {
		i = f->dims[0]-1;
		v_im1 = f->v + field_eval_index(f,i-1,0);
		
		ii = field_eval_index(f,i,0);
		for( j=0; j<f->dims[1]; j++ ) {
			dv[ii] = ( f->v[ii] - v_im1[j] ) / DX;
			ii++;
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
	
	int i,j,ii;

	for( i=0; i<f->dims[0]; i++ ) {
		// j=0
		j=0;
		ii = field_eval_index(f,i,j);
		dv[ii] = ( v[ii+1] - v[ii] ) / DY;

		for( j=1; j<f->dims[1]-1; j++ ) {
			ii++;
			dv[ii] = 0.5*( v[ii+1] - v[ii-1] ) / DY;
		}

		j=f->dims[1]-1;
		ii = field_eval_index(f,i,j);
		dv[ii] = ( v[ii] - v[ii-1] ) / DY;		
	}

	upc_barrier;
}

void pfield_eval_lap( const pfield_t *this_, int nws_, pfield_t *ws_, pfield_t *lap_ )
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

	int i,j,ii, n;

	FILE *fid = fopen(fname_,"w");
	
	for( n=0; n<this_->th_size; n++ ) {

		sh_f = this_->sh_field + n;
		sh_v = sh_f->sh_v;

		/* printf("sh_f->size = %zd\n", sh_f->size); */

		/* for( i=0; i<sh_f->size; i++ ) { */
		/* 	fprintf(fid,"%15.7e\n", sh_v[i]); */
		/* } */
		for( i=0; i<sh_f->dims[0]; i++ ) {
			for( j=0; j<sh_f->dims[1]; j++ ) {
				ii = sh_field_eval_index( sh_f, i, j);
				fprintf(fid,"%15.7e %15.7e %15.7e\n", (i+sh_f->offset[0])*DX, j*DY, sh_v[ii]);
			}
			fprintf(fid,"\n");
			fflush(fid);
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

int main(void)
{

	pfield_t T, D;
	pfield_t ws[3];

	int i,n;
	int dims[] = { NX, NY };
	
	pfield_ctor(&T, MYTHREAD, THREADS, dims);
	pfield_ctor(&D, MYTHREAD, THREADS, dims );

	for( n=0; n<NWORKSP; n++ )
		pfield_ctor(ws+n, MYTHREAD, THREADS, dims );
	
	pfield_set(&T,0.0);
	set_bc(&T);

	{
		char fname[64];
		sprintf(fname,"T-init-%d.dat", T.th_id);
		FILE *fid = fopen(fname,"w");
		int i,j,ii;
		ii=0;
		for( i=0; i<T.field->dims[0]; i++ ) {
			for( j=0; j<T.field->dims[1]; j++ ) {
				fprintf(fid,"%15.7e\n", T.field->v[ii]);
				ii++;
			}
		}
		fclose(fid);
	}

	if( MYTHREAD == 0 ) 
		pfield_write_file( &T, "T-init.dat" );

	for( n=0; n<NSTEPS; n++ ) {
		pfield_eval_lap( &T, NWORKSP, ws, &D);
		pfield_mul_scalar( &D, DT*ALPHA );

		// Perform update
		pfield_add( &T, &D );
		set_bc(&T);		

		if( n % 10 == 0 && MYTHREAD == 0 )  {
			printf("Completed %6.3f%%\n", 100.0*(double)n / NSTEPS);
		}
		if( n % OUTPUT_FREQ == 0 ) {
			if( MYTHREAD == 0 ) {
				char fname[64];
				sprintf(fname,"T-%d.dat", n);
				pfield_write_file( &T, fname);
			}
			upc_barrier;
		}


	
	}
	for( n=0; n<NWORKSP; n++ )
		pfield_dtor(ws+n);

	pfield_dtor(&D);
	pfield_dtor(&T);

	upc_global_exit;

	return 0;
}

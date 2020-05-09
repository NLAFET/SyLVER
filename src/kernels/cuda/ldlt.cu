#include "kernels/cuda/ldlt.cuh"

namespace sylver {
namespace spldlt {
namespace cuda {

////////////////////////////////////////////////////////////////////////////
   
extern __shared__ volatile double SharedMemory[];

////////////////////////////////////////////////////////////////////////////
   
/*

The next function selects pivot based on the pending row number ip
and the column number for the largest element in this row.

Three options are considered:

(1) use 1x1 pivot a11 = fs[ip + ld*ip],

(2) use 1x1 pivot a22 = fs[jp + ld*jp],

(3) use 2x2 pivot

 | a_11 a_12 |
 | a_12 a_22 |,

where a12 = fs[ip + ld*jp].

The pivot that has the smallest inverse is selected.

*/

template< typename ELEMENT_TYPE >
__device__ void
dev_select_pivots_at_root( 
    const ELEMENT_TYPE *const fs,
    const int ld, // leading dimension of fs
    int &ip,
    int &jp,
    ELEMENT_TYPE &a11,
    ELEMENT_TYPE &a12,
    ELEMENT_TYPE &a22,
    ELEMENT_TYPE &det
) {
  // select the pivot based on the row's largest element index
  if (ip != jp) { // choose between 1x1 and 2x2 pivots
    a11 = fs[ip + ld*ip];
    a12 = fs[ip + ld*jp];
    a22 = fs[jp + ld*jp];
    det = a11*a22 - a12*a12; // determinant of 2x2 pivot stored in det
    if ( (fabs(a12) + fabs(a11) + fabs(a22))*fabs(a11) > fabs(det) ) {
      if (fabs(a11) > fabs(a22) ) { // choose the best 1x1 alternative
        jp = ip; // select a11
        det = a11; // pivot value stored in det
      }
      else {
        ip = jp; // select a22
        det = a22; // pivot value stored in det
      }
    }
    else if ( (fabs(a12) + fabs(a11) + fabs(a22))*fabs(a22) > fabs(det) ) {
      ip = jp; // select a22
      det = a22; // pivot value stored in det
    }
  }
  else {
    det = fs[ip + ld*ip]; // pivot value stored in det
  }
}

////////////////////////////////////////////////////////////////////////////
   
/*
 LDLT factorization kernel for the root delays block.

 The columns which the above factorization kernels failed to pivot are
 delayed, ie left unchanged, until some other columns in the same node
 are successfully pivoted, after which pivoting of delayed columns is
 attempted again. When a factorization subroutine terminates,
 generally there still may be delayed columns which this subroutine
 cannot possibly pivot, and they are passed on to the parent node in
 the elimination tree.  At the root node, however, this is not
 possible, and a special kernel given below is applied to delayed
 columns, which together with the respective rows now form a square
 block at the lower left corner of the root node matrix.
 
 The main difference between the factorization kernel below and those
 above is that the pivot is sought in the whole matrix because, in the
 above notation, blocks A_u and A_l are no longer present. Since this
 matrix may be too large to fit into shared memory, the kernel below
 works mostly in the global memory (shared memory is only used for
 finding the largest element of a column).
*/
template< typename ELEMENT_TYPE >
__global__ void
cu_square_ldlt(
    const int n, 
    ELEMENT_TYPE *const a, // A on input, L on output
    ELEMENT_TYPE *const f, // L
    ELEMENT_TYPE *const w, // L*D
    ELEMENT_TYPE *const d, // main diag and subdiag of the inverse of D
    const int ld, // leading dimension of a, f, w
    const ELEMENT_TYPE delta, // Pivoting threshold
    const ELEMENT_TYPE eps, // Zero pivot threashold
    int *const ind, // same as in cu_block_fact
    int *const stat // same as in cu_block_fact
) {
  int x, y;
  int col;
  int ip, jp;
  int pivoted, recent;
  ELEMENT_TYPE a11, a12, a22, det;

  volatile ELEMENT_TYPE *work = (volatile ELEMENT_TYPE*)SharedMemory; // work array
  volatile int *const iwork = (volatile int*)&(work[blockDim.x]); // integer work array
  volatile int *const iw = (volatile int*)&(iwork[blockDim.x]); // iw[0]: failure flag, 
                                       // iw[1]: largest col. elem. index
  
  for ( x = threadIdx.x; x < n; x += blockDim.x ) {
    ind[x] = 0; // initialize pivot index/processed columns mask
    for ( y = 0; y < n; y++ )
      f[x + ld*y] = a[x + ld*y]; // copy A to L
  }
  for ( x = threadIdx.x; x < 2*n; x += blockDim.x )
    d[x] = 0.0; // initialize D
  __syncthreads();

  pivoted = 0; // n.o. pivoted cols
  
  for ( int pass = 0; ; pass++ ) { // failed cols are skipped until next pass

    recent = 0; // n.o. cols pivoted during this pass

    for ( col = 0; col < n; ) {

      if ( ind[col] ) {
        col++; // already pivoted, move on
        continue;
      }

      if ( threadIdx.x == 0 )
        iw[0] = 0; // initialize failure flag
      __syncthreads();

      // find the largest element in the pending column
      //
      // first, each thread finds its candidate for the largest one
      a11 = -1.0;
      y = -1;
      for ( x = threadIdx.x; x < n; x += blockDim.x ) {
        if ( ind[x] == 0 ) {
          a12 = fabs(f[x + ld*col]);
          if ( a12 >= a11 ) {
            a11 = a12;
            y = x;
          }
        }
      }
      work[threadIdx.x] = a11; // the largest one for this thread
      iwork[threadIdx.x] = y; // its index
      __syncthreads();

      // now first 8 threads reduce the number of candidates to 8
      if ( threadIdx.x < 8 ) {
        for ( x = threadIdx.x + 8; x < blockDim.x; x += 8 )
          if ( iwork[x] >= 0 && work[x] > work[threadIdx.x] ) {
            work[threadIdx.x] = work[x];
            iwork[threadIdx.x] = iwork[x];
          }
      }    
      __syncthreads();
      // the first thread finds the largest element and its index
      if ( threadIdx.x == 0 ) {
        y = 0;
        for ( x = 1; x < 8 && x < blockDim.x; x++ )
          if ( iwork[x] >= 0 && (iwork[y] < 0 || work[x] > work[y]) )
            y = x;
        iw[1] = iwork[y]; // the largest element index
      }
      __syncthreads();

      // select the pivot based on the largest element index
      ip = col;
      jp = iw[1];

      dev_select_pivots_at_root< ELEMENT_TYPE >
        ( f, ld, ip, jp, a11, a12, a22, det );

      // try to pivot
      if ( ip == jp ) { // 1x1 pivot
        for ( x = threadIdx.x; x < n; x += blockDim.x )
          if ( ind[x] == 0 )
            if ( dev_1x1_pivot_fails< ELEMENT_TYPE >
              ( x, ip, f, w, ld, det, delta, eps ) )
                iw[0] = 1;
      }
      else { // 2x2 pivot
        for ( x = threadIdx.x; x < n; x += blockDim.x )
          if ( ind[x] == 0 )
            if ( dev_2x2_pivot_fails< ELEMENT_TYPE >
              ( x, ip, jp, f, w, ld, a11, a12, a22, det, delta, eps ) )
                iw[0] = 1;
      }
      __syncthreads();
      if ( iw[0] ) { // pivot failed, restore the failed column(s)
        for ( x = threadIdx.x; x < n; x += blockDim.x ) {
          if ( ind[x] )
            continue;
          f[x + ld*ip] = w[x + ld*ip];
          if ( ip != jp )
            f[x + ld*jp] = w[x + ld*jp];
        }
        __syncthreads();
        col++; // move on
        continue;
      }

      if ( threadIdx.x == 0 ) {
        // mark pivoted columns and invert the pivot if possible
        ind[ip] = pivoted + 1;
        if ( ip == jp ) {
          if ( fabs(det) > eps ) // ok to invert
            d[2*pivoted] = 1.0/det;
        }
        else {
          ind[jp] = pivoted + 2;
          if ( fabs(det) > fabs(a11)*fabs(a22)*1.0e-15 &&
               fabs(det) > eps*(fabs(a11) + fabs(a22) + fabs(a12)) ) {
            // ok to invert          
            d[2*pivoted    ] = a22/det;
            d[2*pivoted + 1] = -a12/det;
            d[2*pivoted + 2] = a11/det;
          }
        }
      }
      __syncthreads();

      // update pivot counters
      if ( ip == jp ) {
        pivoted++;
        recent++;
      }
      else {
        pivoted += 2;
        recent += 2;
      }

      // eliminate pivoted columns from non-processed
      if ( ip == jp ) {
        for ( x = threadIdx.x; x < n; x += blockDim.x )
          for ( y = 0; y < n; y++ )
            if ( x != ip && ind[y] == 0 )
              f[x + ld*y] -= f[x + ld*ip] * f[ip + ld*y];
      }
      else {
        for ( x = threadIdx.x; x < n; x += blockDim.x ) {
          for ( y = 0; y < n; y++ ) {
            if ( x != ip && x != jp && ind[y] == 0 ) {
              f[x + ld*y] -= f[x + ld*ip] * f[ip + ld*y] + 
                             f[x + ld*jp] * f[jp + ld*y];
            }
          }
        }
      }
      __syncthreads();

      if ( ip == col ) // this column is pivoted, move on
        col++;

    } // loop across columns
    
    if ( pivoted == n // all done
            || 
         recent == 0 ) // no pivotable columns left
      break;
  } // pass
  
  if ( threadIdx.x == 0 )
    stat[0] = pivoted;
  
  if ( pivoted < n ) // factorization failed
    return;

  // copy L to A
  for ( x = threadIdx.x; x < n; x += blockDim.x )
    for ( y = 0; y < n; y++ )
      a[ind[x] - 1 + ld*(ind[y] - 1)] = f[x + ld*y];

}

////////////////////////////////////////////////////////////////////////////

template<typename T>
void square_ldlt( 
      cudaStream_t *stream, 
      int n, 
      T* a, 
      T* f, 
      T* w,
      T* d,
      int ld,
      T delta, T eps,
      int* index,
      int* stat
      )
{
  int nt = min(n, 256);
  int sm = nt*sizeof(T) + (nt + 2)*sizeof(int);
  cu_square_ldlt< T ><<< 1, nt, sm, *stream >>>
    ( n, a, f, w, d, ld, delta, eps, index, stat );
}

// fp64
template void square_ldlt<double>( 
      cudaStream_t *stream, int n, double* a, double* f, double* w, double* d,
      int ld, double delta, double eps, int* index, int* stat);
   
}}} // End of namespace sylver::spldlt::cuda

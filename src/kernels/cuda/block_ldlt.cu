#include "kernels/cuda/ldlt.cuh"

#define FAVOUR2x2 100
#define CBLOCKS 3
#define MCBLOCKS 8
#define BLOCKS 7
#define MBLOCKS 11
#define BLOCK_SIZE 8

#define MAX_CUDA_BLOCKS 65535

namespace sylver {
namespace spldlt {
namespace cuda {

////////////////////////////////////////////////////////////////////////////
   
__global__ void
cu_block_ldlt_init(
    const int ncols,
    int *const stat,
    int *const ind
) {
  if (threadIdx.x == 0) {
    stat[0] = ncols; // successful pivots
    stat[1] = 0;
  }
  if (threadIdx.x < ncols) ind[threadIdx.x] = ncols + 1;
}
   
////////////////////////////////////////////////////////////////////////////

/*

The next function initializes L and the main diagonal and subdiagonal 
of D**(-1).

L and L*D are stored in two shared memory arrays fs and fds, each
arranged into TILES square tiles of size TILE_SIZE. The kernel for
factorizing just one node uses TILES = 7, and the one for simultaneous
factorization of several nodes uses TILES = 11.

Each CUDA block uses dev_init_fact to load A_d into the first tile of fs 
and up to (TILES - 1)*TILE_SIZE rows of A_u and A_l into the remaining 
TILES - 1 tiles.

The two diagonals of D**(-1) are stored in a shared memory array
of size 2*TILE_SIZE, initialized to 0 by this kernel.

*/
template <
typename ELEMENT_TYPE,
unsigned int TILE_SIZE,
unsigned int TILES
>
__device__ void
dev_init_fact(
    const unsigned int block, // relative CUDA block number
    const int nrows, 
    const int ncols, 
    const int offp,
    const ELEMENT_TYPE *const a, // array of elements of A
    const int lda, // leading dimension of a
    volatile ELEMENT_TYPE *const fs, // initial L factor (shared mem)
    volatile ELEMENT_TYPE *const ds // initial D**(-1) (shared mem)
) {
  const int SIZE_X = TILES * TILE_SIZE;

  int x, y; // position indices

  y = threadIdx.y % TILE_SIZE; // fs & fds column processed by this thread

  if ( threadIdx.y < TILE_SIZE ) {
    for ( int tile = 0; tile < TILES; tile += 2 ) {
      if ( tile ) { // load A_u and A_l's even tiles into shared memory
        x = threadIdx.x + (tile - 1)*TILE_SIZE +
            (TILES - 1)*TILE_SIZE*block; // offdiagonal row index in A
        if ( x >= offp )
          x += ncols; // skip A_d
        fs[threadIdx.x + tile*TILE_SIZE + SIZE_X*threadIdx.y] =
          ( x < nrows && threadIdx.y < ncols ) ? 
            a[x + lda*threadIdx.y] : 0.0;
      }
      else { // load A_d
        fs[threadIdx.x + SIZE_X*threadIdx.y] =
          ( threadIdx.x < ncols && threadIdx.y < ncols ) ?
            a[offp + threadIdx.x + lda*threadIdx.y] : 0.0;
      }
    }
  }
  else {
    // load A_u and A_l's odd tiles into shared memory
    for (int tile = 1; tile < TILES; tile += 2) {
      x = threadIdx.x + (tile - 1)*TILE_SIZE +
        (TILES - 1)*TILE_SIZE*block;
      if (x >= offp)
        x += ncols;
      fs[threadIdx.x + tile*TILE_SIZE + SIZE_X*y] =
        ((x < nrows) && (y < ncols)) ? a[x + lda*y] : 0.0;
    }
  }
  // main diagonal and subdiagonal of D**(-1) set to 0
  if (threadIdx.y < 2)
    ds[2*threadIdx.x + threadIdx.y] = 0.0;

}

/* The next function uploads L, L*D and D to global memory */

template <
typename ELEMENT_TYPE,
unsigned int TILE_SIZE,
unsigned int TILES
>
__device__ void
dev_save_fact(
    const unsigned int block,
    const int nrows,
    const int ncols,
    const int offp, 
    const int my, // save only if my is non-zero
    const volatile ELEMENT_TYPE *const fs, // L (shared mem)
    const volatile ELEMENT_TYPE *const fds, // L*D (shared mem)
    const volatile ELEMENT_TYPE *const ds, // 2 diags of D**(-1) (shared mem)
    ELEMENT_TYPE *const f, // L (global mem)
    const int ldf, // leading dimension of f
    ELEMENT_TYPE *const fd, // L*D (global mem)
    const int ldfd, // leading dimension of fd
    ELEMENT_TYPE *const d // 2 diags of D**(-1) (global mem)
) {
  const int SIZE_X = TILES * TILE_SIZE;

  int x, y; // position indices

  y = threadIdx.y % TILE_SIZE; // fs & fds column processed by this thread

  if ( threadIdx.y < TILE_SIZE ) { // warps 0, 1
    for ( int tile = 0; tile < TILES; tile += 2 ) {
      if ( tile ) { // upload L_u, L_l, L_u*D and L_l*D's even tiles
        x = threadIdx.x + (tile - 1)*TILE_SIZE +
            (TILES - 1)*TILE_SIZE*block;
        if ( x >= offp ) // skip L_d
          x += ncols;
        if ( x < nrows && threadIdx.y < ncols && my ) {
          f[x + ldf*threadIdx.y] =
            fs[threadIdx.x + tile*TILE_SIZE + SIZE_X*threadIdx.y];
          fd[x + ldfd*threadIdx.y] =
            fds[threadIdx.x + tile*TILE_SIZE + SIZE_X*threadIdx.y];
        }
      }
      else if ( block == 0 ) { 
        // upload L_d and L_d*D
        if ( threadIdx.x < ncols && threadIdx.y < ncols && my ) {
          f[offp + threadIdx.x + ldf*threadIdx.y] =
            fs[threadIdx.x + SIZE_X*threadIdx.y];
          fd[offp + threadIdx.x + ldfd*threadIdx.y] =
            fds[threadIdx.x + SIZE_X*threadIdx.y];
        }
        // upload D**(-1)
        if ( threadIdx.x < 2 && threadIdx.y < ncols )
          d[threadIdx.x + 2*threadIdx.y] = ds[threadIdx.x + 2*threadIdx.y];
      }
    } // loop through even tiles ends here
  }
  else { // upload L_u, L_l, L_u*D and L_l*D's odd tiles (warps 2, 3)
    for (int tile = 1; tile < TILES; tile += 2) {
      x = threadIdx.x + (tile - 1)*TILE_SIZE +
        (TILES - 1)*TILE_SIZE*block;
      if (x >= offp) // skip L_d
        x += ncols;
      if ((x < nrows) && (y < ncols) && my) {
        f[x + ldf*y] = fs[threadIdx.x + tile*TILE_SIZE + SIZE_X*y];
        fd[x + ldfd*y] = fds[threadIdx.x + tile*TILE_SIZE + SIZE_X*y];
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////

/* The next function finds the largest element of the first row of A_d */

template <
typename ELEMENT_TYPE,
unsigned int TILE_SIZE,
unsigned int TILES
>
__device__ void
dev_init_max(
    const int ncols,
    const volatile ELEMENT_TYPE *const fs,
    const int mx, // this thread mask
    volatile int *const mask, // pivot index/mask
    volatile bool *const not_max, // "not largest" flag
    volatile int &jps, // the index of the largest element
    volatile int &quit // pivoting failure flag
) {
  const int SIZE_X = TILES*TILE_SIZE;

  if (threadIdx.y == 0) {
    mask[threadIdx.x] = mx; // initialize the pivot index
    not_max[threadIdx.x] = mx; // initialize the "not largest" flag
  }
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    jps = TILE_SIZE; // initialize pivot col jp: cf the case of a tie below
    quit = 0; // initialize failure flag
  }
  __syncthreads();

  // check if the element in the column threadIdx.x
  // of the first row is (one of) the largest one(s)
  if ((threadIdx.x < ncols) && (threadIdx.y < ncols) &&
      (threadIdx.x != threadIdx.y) &&
      (fabs(fs[SIZE_X*threadIdx.x]) < fabs(fs[SIZE_X*threadIdx.y])))
    not_max[threadIdx.x] = 1; // no good: a larger value exists elsewhere
  __syncthreads();

  // select the leftmost among the largest elements of the row
  if ((threadIdx.y == 0) && (not_max[threadIdx.x] == 0))
    atomicMin((int*)&jps, threadIdx.x); // in case of a tie, choose the leftmost
  __syncthreads();
}

////////////////////////////////////////////////////////////////////////////

template< typename ELEMENT_TYPE >
__device__ void
dev_select_pivots( 
    const volatile ELEMENT_TYPE *const fs, 
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
    if ( (fabs(a12) + fabs(a11) + fabs(a22))*fabs(a11) > FAVOUR2x2*fabs(det) ) {
      if ( fabs(a11) > fabs(a22) ) { // choose the best 1x1 alternative
        jp = ip; // select a11
        det = a11; // pivot value stored in det
      }
      else {
        ip = jp; // select a22
        det = a22; // pivot value stored in det
      }
    }
    else if ( (fabs(a12) + fabs(a11) + fabs(a22))*fabs(a22) > FAVOUR2x2*fabs(det) ) {
      ip = jp; // select a22
      det = a22; // pivot value stored in det
    }
  }
  else
    det = fs[ip + ld*ip]; // pivot value stored in det
}

/* The next function eliminates the pivoted column from non-pivoted */

template < 
typename ELEMENT_TYPE, 
unsigned int TILE_SIZE, 
unsigned int TILES // = 7 for a single node and = 11 for many nodes
>
__device__ void
dev_eliminate_1x1(
    int &x, // row for this thread
    const int y, // column for this thread
    const int ip, // pivoted column
    volatile ELEMENT_TYPE *const fs,
    const int ld,
    const ELEMENT_TYPE p // pivot value
) {
  if ( x != ip )
    fs[x + ld*y] -= p * fs[x + ld*ip];
  x += 2*TILE_SIZE; // move to the next tile pair
  fs[x + ld*y] -= p * fs[x + ld*ip];
  if ( TILES == 11 ) { // several nodes case
    x += 2*TILE_SIZE; // move to the next tile pair
    fs[x + ld*y] -= p * fs[x + ld*ip];
    x += 2*TILE_SIZE; // move to the next tile pair
    fs[x + ld*y] -= p * fs[x + ld*ip];
  }
}

/* The next function eliminates the two pivoted columns from non-pivoted */

template< typename ELEMENT_TYPE, 
unsigned int TILE_SIZE, unsigned int TILES >
__device__ void
dev_eliminate_2x2(
    int &x,
    const int y,
    const int ip,
    const int jp,
    volatile ELEMENT_TYPE *const fs,
    const int ld,
    const ELEMENT_TYPE pi,
    const ELEMENT_TYPE pj
) {
  if ( x != ip && x != jp )
    fs[x + ld*y] -= pi * fs[x + ld*ip] + pj * fs[x + ld*jp];
  x += 2*TILE_SIZE; // move to the next tile pair
  fs[x + ld*y] -= pi * fs[x + ld*ip] + pj * fs[x + ld*jp];
  if ( TILES == 11 ) { // several nodes case
    x += 2*TILE_SIZE; // move to the next tile pair
    fs[x + ld*y] -= pi * fs[x + ld*ip] + pj * fs[x + ld*jp];
    x += 2*TILE_SIZE; // move to the next tile pair
    fs[x + ld*y] -= pi * fs[x + ld*ip] + pj * fs[x + ld*jp];
  }
}

/* The next function performs elimination in one tile only */

template< typename ELEMENT_TYPE, unsigned int TILE_SIZE >
inline __device__ void
dev_eliminate(
    int &x,
    const int y,
    const int ip,
    const int jp,
    volatile ELEMENT_TYPE *const fs,
    const int ld,
    const ELEMENT_TYPE pi,
    const ELEMENT_TYPE pj
) {
  x += TILE_SIZE;
  if ( ip == jp )
      fs[x + ld*y] -= pi * fs[x + ld*ip];
  else
      fs[x + ld*y] -= pi * fs[x + ld*ip] + pj * fs[x + ld*jp];
}
   
////////////////////////////////////////////////////////////////////////////
      
/*
Performs the factorization (LDLT).

The outline of the factorization algorithm is as follows.
1. L = A
2. A diagonal block of L of size 1 or 2 is selected
3. A division of the corresponding (one or two) columns of L
   by the selected block (pivoting) is considered and 
   is accepted only if the elements of the resulting 
   columns are not going to be greater than the inverse
   of the "pivoting threshold" delta; otherwise kernel 
   terminates.
4. If not all columns are pivoted, go to 2.

Called by cu_block_ldlt and cu_multiblock_ldlt factorization kernels.

*/
template< typename ELEMENT_TYPE, 
unsigned int TILE_SIZE, unsigned int TILES >
__device__ void
dev_block_ldlt(
    const unsigned int block,
    const int nrows, // number of rows of the factorized matrix
    const int ncols, // number of columns thereof
    const int offp, // number of rows above the pivot block
    ELEMENT_TYPE *const a, // array of elements of A
    const int lda, // leading dimension of a
    ELEMENT_TYPE *const f, // array of elements of the L factor
    const int ldf, // leading dimension of f
    ELEMENT_TYPE *const fd, // array of elements of L*D
    const int ldfd, // leading dimension of fd
    ELEMENT_TYPE *const d, // array for main diagonal and subdiagonal of D
    const ELEMENT_TYPE delta, // pivoting threashold
    const ELEMENT_TYPE eps, // zero pivot threashold
    int *const index, // pivot order index
    int *const stat  // number of successful pivots
) {
  const int SIZE_X = TILES*TILE_SIZE;

  int ip, jp; // pivot row and col indices
  int x, y; // position indices
  int mx, my; // masks
  ELEMENT_TYPE a11, a12, a22, det; // 2x2 pivot data

  __shared__ volatile ELEMENT_TYPE fs[SIZE_X*TILE_SIZE]; // work array for f
  __shared__ volatile ELEMENT_TYPE fds[SIZE_X*TILE_SIZE]; // work array for fd
  __shared__ volatile ELEMENT_TYPE ds[2*TILE_SIZE]; // work array for d
  __shared__ volatile int mask[TILE_SIZE]; // pivot mask/index
  __shared__ volatile bool not_max[TILE_SIZE]; // flag for finding the largest row elm
  
  __shared__ volatile int quit; // failure flag
  __shared__ volatile int jps; // pivot column index
  
  y = threadIdx.y % TILE_SIZE; // fs & fds column processed by this thread

  // load the diagonal and off-diagonal tiles into shared memory
  dev_init_fact< ELEMENT_TYPE, TILE_SIZE, TILES >
    ( block, nrows, ncols, offp, a, lda, fs, ds );

  mx = (threadIdx.x < ncols ? 0 : ncols + 1); // initial pivot index

  // find the largest element in the first row
  dev_init_max< ELEMENT_TYPE, TILE_SIZE, TILES >
    ( ncols, fs, mx, mask, not_max, jps, quit );

  for ( int row = 0, pivoted = 0; row < ncols; ) {

    // select the pivot based on the row's largest element index jps
    ip = row;
    jp = jps;
    dev_select_pivots< ELEMENT_TYPE >
      ( fs, SIZE_X, ip, jp, a11, a12, a22, det );
    __syncthreads();

    if ( threadIdx.y < TILE_SIZE + 4 ) { // the first 3 warps try to pivot

      x = threadIdx.x + TILE_SIZE*threadIdx.y; // fs/fds row to process
      if (  x < SIZE_X && (threadIdx.y || mx == 0 || mx > ncols) ) {
                       // elements of the pivot block that should have been
                       // zeroed by elimination are ignored
        if ( ip == jp ) { // 1x1 pivot
          if ( dev_1x1_pivot_fails< ELEMENT_TYPE >
            ( x, ip, fs, fds, SIZE_X, det, delta, eps ) )
            quit = 1;
        }
        else { // 2x2 pivot
          if ( dev_2x2_pivot_fails< ELEMENT_TYPE >
            ( x, ip, jp, fs, fds, SIZE_X, a11, a12, a22, det, delta, eps ) )
            quit = 1;
        }
      }

    }
    else { // meanwhile, one thread of the fourth warp is inverting the pivot
    
      if ( threadIdx.x == 0 && threadIdx.y == TILE_SIZE + 4 ) {
        mask[ip] = pivoted + 1; // assume pivot is ok for now
        if ( ip == jp ) {
          if ( fabs(det) > eps )
            ds[2*pivoted] = 1.0/det; // ok to invert
        }
        else {
          mask[jp] = pivoted + 2; // assume pivot is ok for now
          if ( fabs(det) > fabs(a11)*fabs(a22)*1.0e-15 &&
               fabs(det) > eps*(fabs(a11) + fabs(a22) + fabs(a12)) ) {          
            ds[2*pivoted    ] = a22/det;
            ds[2*pivoted + 1] = -a12/det;
            ds[2*pivoted + 2] = a11/det;
          }
        }
        if ( atomicMin(&stat[0], ncols) <= pivoted )
          quit = 1; // some other CUDA block failed to pivot this column
      }

    } // warp fork ends here

    __syncthreads();
    if ( quit ) {
      if ( threadIdx.x == 0 && threadIdx.y == 0 ) {
        atomicMin(&stat[0], pivoted); // record the failure in stat[0]
        // column(s) should not be saved - mark as non-processed
        mask[ip] = 0;
        if ( ip != jp )
          mask[jp] = 0;
      }
      __syncthreads();
      break; // done
    }
    
    // update successful pivots count
    if ( ip == jp )
      pivoted++;
    else
      pivoted += 2;

    // find next pivot row to process
    if ( ip == row )
      row++; // move forward only if this row participated in pivoting

    while ( row < ncols && mask[row] )
      row++; // skip processed rows (parts of previous 2x2 pivots)

    // eliminate the recently pivoted column(s) from the rest

    // first row to be processed by this thread
    x = threadIdx.x + (threadIdx.y/TILE_SIZE)*TILE_SIZE;

    mx = mask[threadIdx.x];
    my = mask[y];

    // process the first (TILES - 3) tiles right away;
    // the even tiles are processed by the first two warps,
    // the odd by the other two    
    if ( ip == jp ) {
      a11 = fs[ip + SIZE_X*y];
      if ( my == 0 )
        dev_eliminate_1x1< ELEMENT_TYPE, TILE_SIZE, TILES >
          ( x, y, ip, fs, SIZE_X, a11 );
    }
    else {
      a11 = fs[ip + SIZE_X*y];
      a12 = fs[jp + SIZE_X*y];
      if ( my == 0 )
        dev_eliminate_2x2< ELEMENT_TYPE, TILE_SIZE, TILES >
          ( x, y, ip, jp, fs, SIZE_X, a11, a12 );
    }
    
    // from here on, the first two warps deal with finding the largest element
    // in the next pivot row, while the other two continue elimination
    // in the remaining three tiles

    if ( threadIdx.y < TILE_SIZE ) {
      if ( row < ncols && threadIdx.y == 0 ) {
        not_max[threadIdx.x] = mx; // mask away processed elements
        if ( threadIdx.x == 0 )
          jps = TILE_SIZE; // initialise the largest element column index
      }
    }
    else { // do elimination in the (TILES - 2)-th tile
      if ( my == 0 )
        dev_eliminate< ELEMENT_TYPE, TILE_SIZE >
          ( x, y, ip, jp, fs, SIZE_X, a11, a12 );
    }
    __syncthreads();

    if ( threadIdx.y < TILE_SIZE ) {  
      // mark elements in the pending row that cannot be largest
      if ( row < ncols ) {
        // check the element in column threadIdx.x
        if ( threadIdx.x != threadIdx.y && mx == 0 && my == 0 &&
             fabs(fs[row + SIZE_X*threadIdx.x]) < 
             fabs(fs[row + SIZE_X*threadIdx.y]) )
          not_max[threadIdx.x] = 1; // no good: a larger value exists elsewhere
      }
    }
    else { // do elimination in the (TILES - 1)-th tile
      if ( my == 0 )
        dev_eliminate< ELEMENT_TYPE, TILE_SIZE >
          ( x, y, ip, jp, fs, SIZE_X, a11, a12 );
    }
    __syncthreads();

    if ( threadIdx.y < TILE_SIZE ) {
      // select leftmost largest element in the row
      if ( row < ncols ) {
        if ( threadIdx.y == 0 && not_max[threadIdx.x] == 0 )
          atomicMin((int*)&jps, threadIdx.x); // in case of a tie, choose the leftmost
      }
    }
    else { // do elimination in the (TILES)-th tile
      if ( my == 0 )
        dev_eliminate< ELEMENT_TYPE, TILE_SIZE >
          ( x, y, ip, jp, fs, SIZE_X, a11, a12 );
    }
    __syncthreads();

  } // for loop through pivot rows ends here

  my = mask[y];

  // update successful pivot ordering in index;
  // if this CUDA block failed to pivot the part of column threadIdx.y of A
  // delegated to it, then possible successful pivoting of its other parts 
  // by other blocks is canceled by zeroing index[threadIdx.y];
  // if some other part of this column is unsuccessful, index[threadIdx.y]
  // remains zero
  if ( threadIdx.x == 0 && threadIdx.y < ncols )
    atomicMin(&index[threadIdx.y], my);

  // save L and D factors and LD
  dev_save_fact< ELEMENT_TYPE, TILE_SIZE, TILES >
    ( block, nrows, ncols, offp, my, fs, fds, ds, f, ldf, fd, ldfd, d );
}
   
////////////////////////////////////////////////////////////////////////////
   
template
< 
typename ELEMENT_TYPE,
unsigned int TILE_SIZE,
unsigned int TILES
>
__global__ void
cu_block_ldlt(
    const int nrows, // n.o. rows in A
    const int ncols, // n.o. cols in A (<= TILE_SIZE)
    const int offp,  // n.o. rows in A_u
    ELEMENT_TYPE *const a, // array of A's elements
    const int lda, // leading dimension of a
    ELEMENT_TYPE *const f, // array of L's elements
    const int ldf, // leading dimension of f
    ELEMENT_TYPE *const fd, // array of (L*D)'s elements
    const int ldfd, // leading dimension of fd
    ELEMENT_TYPE *const d, // array of D**(-1)'s diagonal and subdiagonal elements
    const ELEMENT_TYPE delta, // pivoting threshold
    const ELEMENT_TYPE eps, // zero column threshold:
    // the column is zeroed if all elements are <= eps
    int *const index, // pivot index (cf. permutation matrix P)
    int *const stat // n.o. successful pivots
) {
   dev_block_ldlt< ELEMENT_TYPE, TILE_SIZE, TILES >
      ( blockIdx.x, nrows, ncols, offp, a, lda, f, ldf,
         fd, ldfd, d, delta, eps, index, stat );
   return;
}

////////////////////////////////////////////////////////////////////////////

template<typename T>
void block_ldlt(
      cudaStream_t const stream,
      int nrows, int ncols, int p,
      T* a, int lda,
      T* f, int ldf,
      T* fd, int ldfd,
      T* d,
      T delta, T eps,
      int* index, int* stat
      ) {
   
   int nblocks = (nrows - ncols - 1)/(BLOCK_SIZE*(BLOCKS - 1)) + 1;
   cu_block_ldlt_init<<< 1, BLOCK_SIZE, 0, stream >>>( ncols, stat, index );
  
   dim3 threads(BLOCK_SIZE, 2*BLOCK_SIZE);
   cu_block_ldlt
      < T, BLOCK_SIZE, BLOCKS >
      <<< nblocks, threads, 0, stream >>>
      ( nrows, ncols, p, a, lda, f, ldf, fd, ldfd, d, delta, eps, index, stat );
}

// fp64
template void block_ldlt<double>(
      cudaStream_t const stream,
      int nrows, int ncols, int p,
      double* a, int lda,
      double* f, int ldf,
      double* fd, int ldfd,
      double* d,
      double delta, double eps,
      int* index, int* stat);
   
}}} // End of namespace sylver::spldlt::cuda

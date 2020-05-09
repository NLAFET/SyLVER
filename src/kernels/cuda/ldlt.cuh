/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

namespace sylver {
namespace spldlt {
namespace cuda {

////////////////////////////////////////////////////////////////////////////

/* The next function tries to apply 1x1 pivot. */

template< typename ELEMENT_TYPE >
__device__ bool
dev_1x1_pivot_fails(
    const int x,
    const int ip,
    volatile ELEMENT_TYPE *const fs,
    volatile ELEMENT_TYPE *const fds,
    const int ld,
    const ELEMENT_TYPE det,
    const ELEMENT_TYPE delta,
    const ELEMENT_TYPE eps
) {
  // the column of fds is that of fs before the division by pivot
  const ELEMENT_TYPE u = fds[x + ld*ip] = fs[x + ld*ip];
  if ( fabs(det) <= eps ) { // the pivot is considered to be zero
    if ( fabs(u) <= eps ) { // the off-diagonal is considered to be zero
      if ( x == ip )
        fs[x + ld*ip] = 1.0;
      else
        fs[x + ld*ip] = 0.0;
    }
    else {      // non-zero off-diagonal element found ->
      return 1; // this column to be delayed
    }
  }
  else if ( fabs(det) <= delta*fabs(u) ) // pivot too small ->
    return 1; // this column to be delayed
  else
    fs[x + ld*ip] = u/det; // ok to divide
  return 0;
}

/* The next function tries to apply 1x1 pivot. */

template< typename ELEMENT_TYPE >
__device__ bool
dev_2x2_pivot_fails(
    const int x,
    const int ip,
    const int jp,
    volatile ELEMENT_TYPE *const fs,
    volatile ELEMENT_TYPE *const fds,
    const int ld,
    const ELEMENT_TYPE a11,
    const ELEMENT_TYPE a12,
    const ELEMENT_TYPE a22,
    const ELEMENT_TYPE det,
    const ELEMENT_TYPE delta,
    const ELEMENT_TYPE eps
) {
  // the columns of fds is those of fd before division by pivot
  const ELEMENT_TYPE u = fds[x + ld*ip] = fs[x + ld*ip];
  const ELEMENT_TYPE v = fds[x + ld*jp] = fs[x + ld*jp];
  if ( fabs(det) <= fabs(a11)*fabs(a22)*1.0e-15 ||
       // the determinant is smaller than round-off errors ->
       // the pivot is considered to be zero
       fabs(det) <= eps*(fabs(a11) + fabs(a22) + fabs(a12)) 
       // the inverse of the pivot is of the order 1/eps ->
       // the pivot is considered to be zero
    ) {
    if ( max(fabs(u), fabs(v)) <= eps ) { // the off-diagonal is "zero"
      if ( x == ip ) {
        fs[x + ld*ip] = 1.0;
        fs[x + ld*jp] = 0.0;
      }
      else if ( x == jp ) {
        fs[x + ld*ip] = 0.0;
        fs[x + ld*jp] = 1.0;
      }
      else {
        fs[x + ld*ip] = 0.0;
        fs[x + ld*jp] = 0.0;
      }
    }
    else // non-zero off-diagonal element found ->
      return 1; // this column to be delayed
  }
  else if ( fabs(det) <= 
             delta*max(fabs(a22*u - a12*v), fabs(a11*v - a12*u)) )
             // pivot too small ->
    return 1; // this column to be delayed
  else { // ok to divide
    fs[x + ld*ip] = (a22*u - a12*v)/det;
    fs[x + ld*jp] = (a11*v - a12*u)/det;
  }
  return 0;
}
   
}}} // End of namespace sylver::spldlt::cuda

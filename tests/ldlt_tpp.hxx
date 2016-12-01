#pragma once

#include <cmath>
#include <iostream>
#include <limits>

#include "ssids/cpu/kernels/wrappers.hxx"
#include "ssids/cpu/kernels/ldlt_tpp.hxx"

using namespace spral::ssids::cpu;

namespace spldlt {

   // Run tests for the LDLT with threshold partial pivoting
   // factorization kernel (sequential)
   int run_ldlt_tpp_tests();

   template<typename T>
   void solve(int m, int n, const int *perm, const T *l, int ldl, const T *d, const T *b, T *x) {
      for(int i=0; i<m; i++) x[i] = b[perm[i]];
      // Fwd slv
      ldlt_tpp_solve_fwd(m, n, l, ldl, 1, x, m);
      ldlt_tpp_solve_fwd(m-n, m-n, &l[n*(ldl+1)], ldl, 1, &x[n], m);
      // Diag slv
      ldlt_tpp_solve_diag(n, d, x);
      ldlt_tpp_solve_diag(m-n, &d[2*n], &x[n]);
      // Bwd slv
      ldlt_tpp_solve_bwd(m-n, m-n, &l[n*(ldl+1)], ldl, 1, &x[n], m);
      ldlt_tpp_solve_bwd(m, n, l, ldl, 1, x, m);
      // Undo permutation
      T *temp = new T[m];
      for(int i=0; i<m; i++) temp[i] = x[i];
      for(int i=0; i<m; i++) x[perm[i]] = temp[i];
      // Free mem
      delete[] temp;
   }

}

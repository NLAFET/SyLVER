#pragma once

// SpLDLT
#include "SymbolicFront.hxx"
#include "NumericFront.hxx"
#include "kernels/factor_indef.hxx"
// SpLDLT tests
#include "common.hxx"

// STD
#include <vector>
#include <cstdio>
#include <chrono>

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"
#include "ssids/cpu/kernels/ldlt_tpp.hxx"
#include "ssids/cpu/kernels/ldlt_app.hxx"
// SSIDS tests
#include "tests/ssids/kernels/AlignedAllocator.hxx"
#include "tests/ssids/kernels/framework.hxx"

namespace spldlt { namespace tests {

      template<typename T>
      int form_contrib_test(
            T u, T small, bool posdef,
            int m, int n, int from, int to,
            int blksz, 
            int test=0, int seed=0) {
         
      bool failed = false;

      ////////////////////////////////////////
      // Check input param

      ASSERT_TRUE(m >= n);
      ASSERT_TRUE(to >= from);
      ASSERT_TRUE(from >= 0);
      ASSERT_TRUE(to < n);
            
      // Generate test matrix
      int lda = spral::ssids::cpu::align_lda<T>(m);
      T* a = new double[m*lda];

      if (posdef) gen_posdef(m, a, lda);
      else gen_sym_indef(m, a, lda);

      // Generate a RHS based on x=1, b=Ax
      T *b = new T[m];
      gen_rhs(m, a, lda, b);

      T* l = new double[m*lda];
      memcpy(l, a, lda*m*sizeof(T)); // Copy a to l

      ////////////////////////////////////////
      // Setup front

      // Setup pool allocator
      typedef BuddyAllocator<T, std::allocator<T>> PoolAllocator;
      PoolAllocator pool_alloc(lda*n);
      // Symbolic front
      SymbolicFront sfront;
      sfront.nrow = m;
      sfront.ncol = n;

      // Numeric front
      NumericFront<T, PoolAllocator> front(sfront, pool_alloc, blksz);
      front.ndelay_in = 0; // No incoming delayed columns      
      front.ndelay_out = 0;
      // Setup allocator for factors
      typedef spral::test::AlignedAllocator<T> FactorAllocator;
      FactorAllocator allocT;
      // Make lcol m columns wide for debugging
      size_t len = (lda+2)*m; // Includes D
      front.lcol = allocT.allocate(len);
      // Allocate contribution blocks
      front.alloc_contrib_blocks();

      ////////////////////////////////////////
      
      int nelim = 0;
      int* perm = new int[m];
      for(int i=0; i<m; i++) perm[i] = i;
      T *d = &front.lcol[lda*n];
      const int PAGE_SIZE = 8*1024*1024; // 8 MB
      spral::ssids::cpu::Workspace work(PAGE_SIZE);
      
      T *ld = new T[2*m];
      nelim += ldlt_tpp_factor(
            m, n-from, perm, l, lda,
            d, ld, m, true, u, small, 0, l, lda);



      printf("[form_contrib_test] first pass nelim = %d\n", nelim);

      if (nelim > 0) {
         memcpy(front.lcol, l, lda*n*sizeof(T)); // Copy l to front.lcol
         
         if (from>0)
            do_update<T>(m-n, from, &l[n*(lda+1)], &l[n], lda, d);

         // form_contrib(front, work, from, to);
         // add_cb_to_a(front, l, lda);

         // do_update<T>(m-n, to-from, &l[n*(lda+1)], &l[n+lda*(from+1)], lda, &d[2*(from+1)]);
         do_update<T>(m-n, nelim-from, &l[n*(lda+1)], &l[n+lda*(from+1)], lda, &d[2*(from+1)]);

         // if (to<nelim)
            // do_update<T>(m-n, nelim-to, &l[n*(lda+1)], &l[n+lda*(to+1)], lda, &d[2*(to+1)]);

      }
      // do_update<T>(m-n, nelim, &l[n*(lda+1)], &l[n], lda, d);

      nelim += ldlt_tpp_factor(
            m-nelim, m-nelim, &perm[nelim], &l[nelim*(lda+1)], lda,
            &d[2*nelim], ld, m, true, u, small, nelim, &l[nelim], lda);
      delete[] ld;


      printf("[form_contrib_test] nelim = %d\n", nelim);


      ////////////////////////////////////////
      
      // Perform solve
      T *soln = new T[m];
      solve(m, nelim, perm, l, lda, d, b, soln);

      // Check residual
      T bwderr = backward_error(m, a, lda, b, 1, soln, m);
      /*if(debug)*/ printf("bwderr = %le\n", bwderr);
      EXPECT_LE(bwderr, 5e-14) << "(test " << test << " seed " << seed << ")" << std::endl;
            
      ////////////////////////////////////////
      // Cleanup memory
      delete[] a;
      delete[] b;
      delete[] l;
      delete[] perm;

      return failed ? -1 : 0;
   }

   // Run tests for the node factorization
   int run_form_contrib_tests();

}} // namespace spldlt::tests

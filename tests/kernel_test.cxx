#include <cstdio>

#include "tests/ssids/kernels/framework.hxx"
#include "ssids/cpu/kernels/wrappers.hxx"
#include "ldlt_tpp.hxx"

// using namespace spral::ssids::cpu;

int main(void) {
   
   printf("[Tests] Test kernels\n");

   // int m = 128;
   // int lda = m;
   // double *a = new double[m*lda];

   // gen_posdef(m, a, lda);

   printf("[Tests] Test LDLT TPP kernels\n");
   spldlt::run_ldlt_tpp_tests();
}

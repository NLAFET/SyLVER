#include <cstdio>

#include "tests/ssids/kernels/framework.hxx"
#include "ssids/cpu/kernels/wrappers.hxx"

using namespace spral::ssids::cpu;

int main(void) {
   
   printf("Test Kernels\n");

   int m = 128;
   int lda = m;
   double *a = new double[m*lda];

   gen_posdef(m, a, lda);

   
}

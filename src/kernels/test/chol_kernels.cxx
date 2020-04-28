#include <gtest/gtest.h>

#include <random>
#include <cstring>

#include "kernels/factor.hxx"

namespace {

using ValueTypes =
   ::testing::Types<double>;
   // ::testing::Types<float, double>;

template <typename T>
class CholKernels : public ::testing::Test {
protected:
   using value_type = T;
   CholKernels()
      :
      // Square block
      sqr_sym_dd_m(51), sqr_sym_dd_n(51), sqr_sym_dd_lda(56),
      sqr_sym_dd_block(new T[sqr_sym_dd_lda*sqr_sym_dd_n])
   {
      this->generate_sym_dd_block(
            sqr_sym_dd_m, sqr_sym_dd_n, sqr_sym_dd_block.get(), sqr_sym_dd_lda);
   }

   void generate_unsym_block(int m, int n, T *a, int lda) {

      std::default_random_engine generator;
      std::uniform_real_distribution<T> dis(-1.0, 1.0);

      for (int j = 0; j < n; ++j) {
         for (int i = 0; i < m; ++i) {
            a[j*lda+i] = dis(generator);
         }
      }
      
   }

   void generate_sym_block(int m, int n, T *a, int lda) {

      std::default_random_engine generator;
      std::uniform_real_distribution<T> dis(-1.0, 1.0);

      for (int j = 0; j < n; ++j) {
         for (int i = j; i < m; ++i) {
            a[j*lda+i] = dis(generator);
            a[i*lda+j] = a[j*lda+i];
         }
      }
   }

   void generate_sym_dd_block(int m, int n, T *a, int lda) {

      // Generate a symetric block
      generate_sym_block(m, n, a, lda);

      T max_mn = std::max(m,n);
      // Make it diagonally dominant by setting a_jj = max(m,n) +
      // |a_jj|
      for (int j = 0; j < n; ++j) {
         a[j*lda+j] = std::fabs(a[j*lda+j]) + max_mn;
      }
   }

   // FIXME: would certainly be much better to compute a (componentwise)
   // backward error for the factorization
   void check_block_lwr_eq(int m, int n, T *a, int lda, T *b, int ldb) {

      for (int j = 0; j < n; ++j) {
         for (int i = j; i < m; ++i) {
            T err_ij = std::fabs(a[j*lda+i] - b[j*lda+i]);
            if(std::is_same<value_type, double>::value) {
               ASSERT_LT(err_ij, 1e-14);
            }
            else if (std::is_same<value_type, float>::value) {
               ASSERT_LT(err_ij, 1e-6);
            }
            else {
               FAIL();
            }
         }
      }

   }

   // Square block
   int sqr_sym_dd_m, sqr_sym_dd_n, sqr_sym_dd_lda;
   std::unique_ptr<T> sqr_sym_dd_block;

};

TYPED_TEST_CASE(CholKernels, ValueTypes);

TYPED_TEST(CholKernels, CholDiagBlockSqr)
{
   
   using value_type = typename TestFixture::value_type;
   int m = this->sqr_sym_dd_m;
   int n = this->sqr_sym_dd_n;
   value_type *a = this->sqr_sym_dd_block.get();
   int lda = this->sqr_sym_dd_lda;
   
   std::unique_ptr<value_type> block_cpy(
         new value_type[lda*m]);

   std::memcpy(block_cpy.get(), a, lda*n*sizeof(value_type));

   // Factor block copy using LAPACK
   sylver::host_potrf(sylver::FILL_MODE_LWR, n, block_cpy.get(), lda);
   
   // Factor block with SyLVER kernel
   sylver::spldlt::factorize_diag_block(
         m, n, a, lda);

   // Check computed factors below diagonal
   this->check_block_lwr_eq(m, n, a, lda, block_cpy.get(), lda);

}

TYPED_TEST(CholKernels, CholDiagBlockRecN1)
{

   using value_type = typename TestFixture::value_type;
   int m = this->sqr_sym_dd_m;
   value_type *a = this->sqr_sym_dd_block.get();
   int lda = this->sqr_sym_dd_lda;

   // Get copy of test block
   std::unique_ptr<value_type> block_cpy(
         new value_type[lda*m]);

   std::memcpy(block_cpy.get(), a, lda*m*sizeof(value_type));

   // Factor block copy using LAPACK
   sylver::host_potrf(sylver::FILL_MODE_LWR, m, block_cpy.get(), lda);

   // Block width
   int n = 1;

   // Factor block with SyLVER kernel
   sylver::spldlt::factorize_diag_block(
         m, n, a, lda);

   // Check computed factors below diagonal
   this->check_block_lwr_eq(m, n, a, lda, block_cpy.get(), lda);

}

TYPED_TEST(CholKernels, CholDiagBlockRecN23)
{

   using value_type = typename TestFixture::value_type;
   int m = this->sqr_sym_dd_m;
   value_type *a = this->sqr_sym_dd_block.get();
   int lda = this->sqr_sym_dd_lda;

   // Get copy of test block
   std::unique_ptr<value_type> block_cpy(
         new value_type[lda*m]);

   std::memcpy(block_cpy.get(), a, lda*m*sizeof(value_type));

   // Factor block copy using LAPACK
   sylver::host_potrf(sylver::FILL_MODE_LWR, m, block_cpy.get(), lda);

   // Block width
   int n = 23;

   // Factor block with SyLVER kernel
   sylver::spldlt::factorize_diag_block(
         m, n, a, lda);

   // Check computed factors below diagonal
   this->check_block_lwr_eq(m, n, a, lda, block_cpy.get(), lda);

}

}

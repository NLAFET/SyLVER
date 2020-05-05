#include "sylver/kernels/CopyBackup.hxx"

#include <gtest/gtest.h>

#include <random>
#include <vector>

namespace {

using ValueTypes =
   ::testing::Types<double>;
   // ::testing::Types<float, double>;

template <typename T>
class CopyBackup : public ::testing::Test {
protected:
   using value_type = T;

   CopyBackup() { }

   void populate_matrix(int m, int n, T *a, int lda) {
      std::default_random_engine generator;
      std::uniform_real_distribution<T> dis(-1.0, 1.0);

      for (int j = 0; j < n; ++j) {
         for (int i = 0; i < m; ++i) {
            a[j*lda+i] = dis(generator);
         }
      }
   }

   void check_block(
         int m, int n, T *a, int lda, int blksz, int iblk, int jblk,
         T* block, int ldblk) {

      int blkm = std::min(m-iblk*blksz, blksz);
      int blkn = std::min(n-jblk*blksz, blksz);
      
      for (int j = 0; j < blkn; ++j) {
         for (int i = 0; i < blkn; ++i) {
            ASSERT_EQ(a[(jblk*blksz+j)*lda + iblk*blksz+i], block[j*ldblk+i]);
         }
      }
   }
};

TYPED_TEST_CASE(CopyBackup, ValueTypes);

TYPED_TEST(CopyBackup, CreateAndRestoreDefaultAlloc)
{
   using ValueType  = typename TestFixture::value_type;
   
   using Allocator = std::allocator<ValueType>;

   // Matrix size and block size
   int m = 1000;
   int n = 500;
   int lda = spral::ssids::cpu::align_lda<ValueType>(m);
   // Create matrix and fill it with zeros
   std::vector<ValueType> aval(lda*n, 0.0);
   // std::vector<ValueType> acpy(lda*n, 0.0);

   // Put random values into A
   this->populate_matrix(m, n, &aval[0], lda);
   // Copy A into buffer
   // acpy = aval;
   
   // Block size
   int blksz = 100;

   // Backup
   sylver::CopyBackup<ValueType, Allocator> backup(m, n, blksz);

   // TODO: loop over blocks
   
   // Block
   int iblk = 0; // Row index
   int jblk = 0; // Column index

   int blkm = std::min(m-iblk*blksz, blksz);
   int blkn = std::min(n-jblk*blksz, blksz);
   int ldblk = blkm;
   // Create block and fill it with zeros
   std::vector<ValueType> block(ldblk*blkn, 0.0);
   
   // Backup copy of block A(iblk, jblk)
   backup.create_restore_point(iblk, jblk, &aval[0], lda);

   // Restore the whole block in to copy of A
   int rfrom = 0;
   int cfrom = 0;
   backup.restore_part(iblk, jblk, rfrom, cfrom, &block[0], ldblk);

   this->check_block(m, n, &aval[0], lda, blksz, iblk, jblk, &block[0], ldblk);

}
   
}

#include "BlockBase.hxx"

#include <gtest/gtest.h>

#include <random>

namespace {

using ValueTypes =
   ::testing::Types<double>;
   // ::testing::Types<float, double>;

template <typename T>
class BlockBase : public ::testing::Test {
protected:
   BlockBase()
      : empty_block(0, 0, 0, 0, nullptr, 0),
        empty_block2(0, 0, 0, 0),
        m1(4), n1(4), lda1(4), 
        small_square_data(new T[n1*lda1]),
        small_block(0, 0, m1, n1, small_square_data.get(), lda1)

   {
      populate_block(m1, n1, small_square_data.get(), lda1);
   }

   void populate_block(int m, int n, T *data, int lda) {

      std::default_random_engine generator;
      std::uniform_real_distribution<T> dis(-1.0, 1.0);

      for (int j = 0; j < n; ++j) {
         for (int i = 0; i < m; ++i) {
            data[i+j*lda] = dis(generator);
         }
      }

   }
   
   sylver::BlockBase<T> empty_block;
   sylver::BlockBase<T> empty_block2;
   int m1, n1, lda1;
   std::unique_ptr<T> small_square_data;
   sylver::BlockBase<T> small_block;

};
   
TYPED_TEST_CASE(BlockBase, ValueTypes);
   
TYPED_TEST(BlockBase, EmptyBlock)
{

   ASSERT_EQ(this->empty_block.i(), 0);
   ASSERT_EQ(this->empty_block.j(), 0);
   ASSERT_EQ(this->empty_block.m(), 0);
   ASSERT_EQ(this->empty_block.n(), 0);
   ASSERT_EQ(this->empty_block.a(), nullptr);
   ASSERT_EQ(this->empty_block.lda(), 0);
#if defined(SPLDLT_USE_STARPU)
   ASSERT_EQ(this->empty_block.hdl(), nullptr);
#endif
}

TYPED_TEST(BlockBase, EmptyBlock2)
{
   ASSERT_EQ(this->empty_block.i(), 0);
   ASSERT_EQ(this->empty_block.j(), 0);
   ASSERT_EQ(this->empty_block.m(), 0);
   ASSERT_EQ(this->empty_block.n(), 0);
   ASSERT_EQ(this->empty_block.a(), nullptr);
   ASSERT_EQ(this->empty_block.lda(), 0);
#if defined(SPLDLT_USE_STARPU)
   ASSERT_EQ(this->empty_block.hdl(), nullptr);
#endif
}

TYPED_TEST(BlockBase, SmallBlock)
{
   ASSERT_EQ(this->small_block.i(), 0);
   ASSERT_EQ(this->small_block.j(), 0);
   ASSERT_EQ(this->small_block.m(), 4);
   ASSERT_EQ(this->small_block.n(), 4);
   ASSERT_NE(this->small_block.a(), nullptr);
   ASSERT_EQ(this->small_block.lda(), 4);
#if defined(SPLDLT_USE_STARPU)
   ASSERT_EQ(this->small_block.hdl(), nullptr);
#endif
  
}

TYPED_TEST(BlockBase, SmallBlockZero)
{
   this->small_block.zero();

   // make sure every entries is equla to zero
   for (int j = 0; j < this->small_block.n(); ++j) {
      for (int i = 0; i < this->small_block.m(); ++i) {
         ASSERT_EQ(this->small_block.a()[j*this->small_block.lda()+i], 0.0); 
      }
   }

}
   
}

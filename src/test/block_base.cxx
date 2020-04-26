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
      :
      // Empty block
      empty_block(0, 0, 0, 0, nullptr, 0),
      empty_block_no_a(0, 0, 0, 0),
      // Small square block
      m1(4), n1(4), lda1(4), 
      small_square_data(new T[n1*lda1]),
      small_block(0, 0, m1, n1, small_square_data.get(), lda1),
      // Small square block with padding i.e. lda /= m
      small_sqr_pad_m(12), small_sqr_pad_n(12), small_sqr_pad_lda(16),
      small_sqr_pad_data(new T[small_sqr_pad_n*small_sqr_pad_lda]),
      small_sqr_pad_block(11, 4, small_sqr_pad_m, small_sqr_pad_n, small_sqr_pad_data.get(), small_sqr_pad_lda),
      // Small rectangular block
      small_rec_m(8), small_rec_n(4), small_rec_lda(8),
      small_rec_data(new T[small_rec_n*small_rec_lda]),
      small_rec_block(2, 5, small_rec_m, small_rec_n, small_rec_data.get(), small_rec_lda)
      
   {
      // Populate small square block
      populate_block(m1, n1, small_square_data.get(), lda1);

      // Populate small rectangle block with padding
      populate_block(small_sqr_pad_m, small_sqr_pad_n, small_sqr_pad_data.get(), small_sqr_pad_lda);

      // Populate small rectangle block
      populate_block(small_rec_m, small_rec_n, small_rec_data.get(), small_rec_lda);

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

   // Empty block
   sylver::BlockBase<T> empty_block;
   // Empty block with no `a` and `lda` provided
   sylver::BlockBase<T> empty_block_no_a;
   // Small square block
   int m1, n1, lda1;
   std::unique_ptr<T> small_square_data;
   sylver::BlockBase<T> small_block;
   // Small square block with padding
   int small_sqr_pad_m, small_sqr_pad_n, small_sqr_pad_lda;
   std::unique_ptr<T> small_sqr_pad_data;
   sylver::BlockBase<T> small_sqr_pad_block;
   // Small rectangular block
   int small_rec_m, small_rec_n, small_rec_lda;
   std::unique_ptr<T> small_rec_data;
   sylver::BlockBase<T> small_rec_block;
   
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

TYPED_TEST(BlockBase, EmptyBlockNoA)
{
   ASSERT_EQ(this->empty_block_no_a.i(), 0);
   ASSERT_EQ(this->empty_block_no_a.j(), 0);
   ASSERT_EQ(this->empty_block_no_a.m(), 0);
   ASSERT_EQ(this->empty_block_no_a.n(), 0);
   ASSERT_EQ(this->empty_block_no_a.a(), nullptr);
   ASSERT_EQ(this->empty_block_no_a.lda(), 0);
#if defined(SPLDLT_USE_STARPU)
   ASSERT_EQ(this->empty_block_no_a.hdl(), nullptr);
#endif
}

// Small square block with lda == m
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

// Small square block with padding (lda /= m)
TYPED_TEST(BlockBase, SmallSqrPadBlock)
{
   ASSERT_EQ(this->small_sqr_pad_block.i(), 11);
   ASSERT_EQ(this->small_sqr_pad_block.j(), 4);
   ASSERT_EQ(this->small_sqr_pad_block.m(), this->small_sqr_pad_m);
   ASSERT_EQ(this->small_sqr_pad_block.n(), this->small_sqr_pad_n);
   ASSERT_NE(this->small_sqr_pad_block.a(), nullptr);
   ASSERT_EQ(this->small_sqr_pad_block.lda(), this->small_sqr_pad_lda);
#if defined(SPLDLT_USE_STARPU)
   ASSERT_EQ(this->small_sqr_pad_block.hdl(), nullptr);
#endif
   
}
   
// Small rectangular block with lda == m
TYPED_TEST(BlockBase, SmallRecBlock)
{
   ASSERT_EQ(this->small_rec_block.i(), 2);
   ASSERT_EQ(this->small_rec_block.j(), 5);
   ASSERT_EQ(this->small_rec_block.m(), this->small_rec_m);
   ASSERT_EQ(this->small_rec_block.n(), this->small_rec_n);
   ASSERT_NE(this->small_rec_block.a(), nullptr);
   ASSERT_EQ(this->small_rec_block.lda(), this->small_rec_lda);
#if defined(SPLDLT_USE_STARPU)
   ASSERT_EQ(this->small_rec_block.hdl(), nullptr);
#endif
   
}
   
// Zeroing small square with lda == m
TYPED_TEST(BlockBase, SmallBlockZero)
{
   // Zero block entries
   this->small_block.zero();

   // make sure every entries is equal to zero
   for (int j = 0; j < this->small_block.n(); ++j) {
      for (int i = 0; i < this->small_block.m(); ++i) {
         ASSERT_EQ(this->small_block.a()[j*this->small_block.lda()+i], 0.0); 
      }
   }

}

// Zeroing small square with lda != m
TYPED_TEST(BlockBase, SmallSqrPadBlockZero)
{

   // Zero block entries
   this->small_sqr_pad_block.zero();

   // make sure every entries is equal to zero
   for (int j = 0; j < this->small_sqr_pad_block.n(); ++j) {
      for (int i = 0; i < this->small_sqr_pad_block.m(); ++i) {
         ASSERT_EQ(this->small_sqr_pad_block.a()[j*this->small_sqr_pad_block.lda()+i], 0.0); 
      }
   }
   
}

// Zeroing small rectangular with lda == m
TYPED_TEST(BlockBase, SmallRecBlockZero)
{
   this->small_rec_block.zero();

   // make sure every entries is equal to zero
   for (int j = 0; j < this->small_rec_block.n(); ++j) {
      for (int i = 0; i < this->small_rec_block.m(); ++i) {
         ASSERT_EQ(this->small_rec_block.a()[j*this->small_rec_block.lda()+i], 0.0); 
      }
   }

}
   
#if defined(SPLDLT_USE_STARPU)
TYPED_TEST(BlockBase, SmallBlockRegHdl)
{

   // Register StarPU handle
   this->small_block.register_handle();
   
   ASSERT_NE(this->small_block.hdl(), nullptr);
   
}

TYPED_TEST(BlockBase, SmallBlockUnregHdl)
{

   // Register StarPU handle
   this->small_block.register_handle();

   ASSERT_NE(this->small_block.hdl(), nullptr);

   // Unregister StarPU handle (synchronously)
   bool const async = false;
   this->small_block.template unregister_handle<async>();

   ASSERT_EQ(this->small_block.hdl(), nullptr);

}
#endif
   
}

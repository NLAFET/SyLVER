#include "BlockBase.hxx"

#include <gtest/gtest.h>

namespace {

using ValueTypes =
   ::testing::Types<float, double>;

template <typename T>
class BlockBase : public ::testing::Test {
protected:
   BlockBase()
      : empty_block(0, 0, 0, 0, nullptr, 0),
        empty_block2(0, 0, 0, 0)

   {}
   
   sylver::BlockBase<T> empty_block;
   sylver::BlockBase<T> empty_block2;
};
   
TYPED_TEST_CASE(BlockBase, ValueTypes);
   
TYPED_TEST(BlockBase, EmptyBlockValues)
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

TYPED_TEST(BlockBase, EmptyBlock2Values)
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
   
}

#include "NumericFront.hxx"
#include "sylver/SymbolicFront.hxx"

#include <memory>

#include <gtest/gtest.h>

namespace {

using ValueTypes =
   ::testing::Types<double>;
   // ::testing::Types<float, double>;

template <typename T>
class NumericFrontIndef : public ::testing::Test {
protected:

   using ValueType = T;

   NumericFrontIndef()
   {}
   
};

TYPED_TEST_CASE(NumericFrontIndef, ValueTypes);

TYPED_TEST(NumericFrontIndef, ActivateFront)
{

   using ValueType = typename TestFixture::ValueType;
   using FactorAllocType = std::allocator<ValueType>;
   using PoolAllocType = std::allocator<ValueType>;

   using NumericFactorType = sylver::spldlt::NumericFront<ValueType, FactorAllocType, PoolAllocType>;
   
   FactorAllocType factor_alloc;
   PoolAllocType pool_alloc;
   int const blksz = 100;

   sylver::SymbolicFront symb;
   // Initialize symbolic front
   symb.idx = 0;

   symb.nrow = 350;
   symb.ncol = 250;
   symb.first_child = nullptr;
   symb.next_child = nullptr;
   symb.rlist = nullptr;
   symb.num_a = 0;
   symb.amap = nullptr;
   symb.parent = -1;

   // Simple Permute
   std::vector<int> rlist(symb.ncol);
   for (int i = 0; i < symb.ncol; ++i) {
      rlist[i] = i;
   }
   symb.rlist = &rlist[0];
   
   NumericFactorType front(symb, factor_alloc, pool_alloc, blksz);

   // Make sure variables are properly initialized
   ASSERT_EQ(front.lcol, nullptr);
   ASSERT_EQ(front.first_child, nullptr);
   ASSERT_EQ(front.next_child, nullptr);
   ASSERT_EQ(front.cperm, nullptr);
   ASSERT_EQ(front.perm, nullptr);

   ASSERT_EQ(front.symb().ncol, symb.ncol);
   ASSERT_EQ(front.symb().nrow, symb.nrow);
 
   void** child_contrib = nullptr;
   
   front.activate(child_contrib);

   ASSERT_EQ(front.ndelay_in(), 0);
   ASSERT_EQ(front.ndelay_out(), 0);
   ASSERT_NE(front.lcol, nullptr);
   ASSERT_NE(front.backup, nullptr);

}
   
}

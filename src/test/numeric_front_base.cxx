#include "NumericFrontBase.hxx"
#include "SymbolicFront.hxx"

#include <gtest/gtest.h>

#include <random>

#include "ssids/cpu/cpu_iface.hxx"

namespace {

using ValueTypes =
   ::testing::Types<double>;
   // ::testing::Types<float, double>;

template <typename T>
class NumericFrontBase : public ::testing::Test {
protected:

   using value_type = T;
   using default_allocator_type = std::allocator<T>;
   
   NumericFrontBase()
   {

      int blksz = 128;
         
      // Empty symbolic front
      empty_symb_front.idx = 0;
      empty_symb_front.nrow = 0;
      empty_symb_front.ncol = 0;
      empty_symb_front.first_child = nullptr;
      empty_symb_front.next_child = nullptr;
      empty_symb_front.rlist = nullptr;
      empty_symb_front.num_a = 0;
      empty_symb_front.amap = nullptr;
      empty_symb_front.parent = -1;
#if defined(SPLDLT_USE_STARPU)
      empty_symb_front.hdl = nullptr;
#endif 
      empty_symb_front.exec_loc = -1;
      
      empty_numeric_front.reset(
            new sylver::NumericFrontBase<T, default_allocator_type>(
                  empty_symb_front, default_allocator, blksz));

      // Small square symbolic front
      small_sqr_symb_front.idx = 0;
      small_sqr_symb_front.nrow = 21;
      small_sqr_symb_front.ncol = 21;
      small_sqr_symb_front.first_child = nullptr;
      small_sqr_symb_front.next_child = nullptr;
      small_sqr_symb_front.rlist = nullptr;
      small_sqr_symb_front.num_a = 0;
      small_sqr_symb_front.amap = nullptr;
      small_sqr_symb_front.parent = -1;
#if defined(SPLDLT_USE_STARPU)
      small_sqr_symb_front.hdl = nullptr;
#endif
      small_sqr_symb_front.exec_loc = -1;

      small_sqr_numeric_front.reset(
            new sylver::NumericFrontBase<T, default_allocator_type>(
                  small_sqr_symb_front, default_allocator, blksz));
   }

   default_allocator_type default_allocator;
   // Empty front
   sylver::SymbolicFront empty_symb_front;
   std::unique_ptr<sylver::NumericFrontBase<T, default_allocator_type>> empty_numeric_front;
   // Small square front
   sylver::SymbolicFront small_sqr_symb_front;
   std::unique_ptr<sylver::NumericFrontBase<T, default_allocator_type>> small_sqr_numeric_front;

};

TYPED_TEST_CASE(NumericFrontBase, ValueTypes);

TYPED_TEST(NumericFrontBase, EmptyNumericFront)
{
   ASSERT_EQ(this->empty_numeric_front->blksz(), 128);
   ASSERT_EQ(this->empty_numeric_front->hdl(), nullptr);
   ASSERT_EQ(this->empty_numeric_front->contrib_hdl(), nullptr);
   ASSERT_EQ(this->empty_numeric_front->ldl(), 0);
   ASSERT_EQ(this->empty_numeric_front->ndelay_in(), 0);
   ASSERT_EQ(this->empty_numeric_front->ndelay_out(), 0);
   ASSERT_EQ(this->empty_numeric_front->ncol(), 0);
   ASSERT_EQ(this->empty_numeric_front->nrow(), 0);
   ASSERT_EQ(this->empty_numeric_front->nc(), 0);
   ASSERT_EQ(this->empty_numeric_front->nr(), 0);
   ASSERT_EQ(this->empty_numeric_front->nelim(), 0);
   ASSERT_EQ(this->empty_numeric_front->nelim_first_pass(), 0);

   ASSERT_EQ(this->empty_numeric_front->symb().idx, this->empty_symb_front.idx);

#if defined(SPLDLT_USE_STARPU)
   this->empty_numeric_front->register_symb();
   
   ASSERT_NE(this->empty_numeric_front->hdl(), nullptr);

   this->empty_numeric_front->register_symb_contrib();

   ASSERT_NE(this->empty_numeric_front->hdl(), nullptr);

#endif
}

TYPED_TEST(NumericFrontBase, SmallSqrNumericFront)
{

   using value_type = typename TestFixture::value_type;

   ASSERT_EQ(this->small_sqr_numeric_front->blksz(), 128);
   ASSERT_EQ(this->small_sqr_numeric_front->hdl(), nullptr);
   ASSERT_EQ(this->small_sqr_numeric_front->contrib_hdl(), nullptr);
   ASSERT_EQ(this->small_sqr_numeric_front->ndelay_in(), 0);
   ASSERT_EQ(this->small_sqr_numeric_front->ndelay_out(), 0);
   ASSERT_EQ(this->small_sqr_numeric_front->ncol(), 21);
   ASSERT_EQ(this->small_sqr_numeric_front->nrow(), 21);
   ASSERT_EQ(this->small_sqr_numeric_front->ldl(),
             spral::ssids::cpu::align_lda<value_type>(21));
   ASSERT_EQ(this->small_sqr_numeric_front->nelim(), 0);
   ASSERT_EQ(this->small_sqr_numeric_front->nelim_first_pass(), 0);

   ASSERT_EQ(this->small_sqr_numeric_front->symb().idx, this->small_sqr_symb_front.idx);
   
#if defined(SPLDLT_USE_STARPU)

   this->small_sqr_numeric_front->register_symb();
   
   ASSERT_NE(this->small_sqr_numeric_front->hdl(), nullptr);

   this->small_sqr_numeric_front->register_symb_contrib();

   ASSERT_NE(this->small_sqr_numeric_front->hdl(), nullptr);

#endif

}
   
}

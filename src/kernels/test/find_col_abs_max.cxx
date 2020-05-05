#include "kernels/common.hxx"

#include <gtest/gtest.h>

#include <random>

namespace {

using ValueTypes =
   ::testing::Types<double>;
   // ::testing::Types<float, double>;

template <typename T>
class FindColAbsMax : public ::testing::Test {
protected:
   using value_type = T;
   FindColAbsMax()
      :
      col_dim(16), col(new T[col_dim])
   {
      this->populate_col(col_dim, col.get());
   }

   void populate_col(int dim, T*col) {
      std::default_random_engine generator;
      std::uniform_real_distribution<T> dis(-1.0, 1.0);

      for (int i = 0; i < dim; ++i) {
         col[i] = dis(generator);
      }
   }

   void print_col(int dim, T*col) {
      std::cout << "Col =";
      for (int i = 0; i < dim; ++i) {
         std::cout << " " << col[i] ;
      }
      std::cout << std::endl;
   }
   
   int col_dim;
   std::unique_ptr<T> col;

};

TYPED_TEST_CASE(FindColAbsMax, ValueTypes);

// Max absoute value in first position with positive element
TYPED_TEST(FindColAbsMax, FindColAbsMaxFirst)
{
   using value_type  = typename TestFixture::value_type;

   int maxidx;
   int maxidx_ret;

   // Reset
   maxidx = -1;
   maxidx_ret = -1;
   
   // First element
   maxidx = 0;
   this->col.get()[maxidx] = static_cast<value_type>(2.0);
   // this->print_col(this->col_dim, this->col.get());

   maxidx_ret = sylver::find_col_abs_max(0, this->col_dim-1, this->col.get());   

   ASSERT_EQ(maxidx_ret, maxidx);

}

// Max absoute value in first position with negative element
TYPED_TEST(FindColAbsMax, FindColAbsMaxFirstNeg)
{
   using value_type  = typename TestFixture::value_type;

   int maxidx;
   int maxidx_ret;

   // Reset
   maxidx = -1;
   maxidx_ret = -1;
   
   // First element
   maxidx = 0;
   this->col.get()[maxidx] = static_cast<value_type>(-2.0);
   // this->print_col(this->col_dim, this->col.get());

   maxidx_ret = sylver::find_col_abs_max(0, this->col_dim-1, this->col.get());   

   ASSERT_EQ(maxidx_ret, maxidx);

}

// Max absoute value in last position with positive element   
TYPED_TEST(FindColAbsMax, FindColAbsMaxLast)
{
   using value_type  = typename TestFixture::value_type;

   int maxidx;
   int res;

   // Reset
   maxidx = -1;
   res = -1;
   
   // First element
   maxidx = this->col_dim-1;
   this->col.get()[maxidx] = static_cast<value_type>(2.0);
   // this->print_col(this->col_dim, this->col.get());
   
   res = sylver::find_col_abs_max(0, this->col_dim-1, this->col.get());   

   ASSERT_EQ(res, maxidx);
}

// Max absoute value in last position with negative element   
TYPED_TEST(FindColAbsMax, FindColAbsMaxLastNeg)
{
   using value_type  = typename TestFixture::value_type;

   int maxidx;
   int res;

   // Reset
   maxidx = -1;
   res = -1;
   
   // First element
   maxidx = this->col_dim-1;
   this->col.get()[maxidx] = static_cast<value_type>(-2.0);
   // this->print_col(this->col_dim, this->col.get());
   
   res = sylver::find_col_abs_max(0, this->col_dim-1, this->col.get());   

   ASSERT_EQ(res, maxidx);

}

   
// Max absoute value in middle position with positive element
TYPED_TEST(FindColAbsMax, FindColAbsMaxMiddle)
{
   using value_type  = typename TestFixture::value_type;

   int maxidx;
   int res;

   // Reset
   maxidx = -1;
   res = -1;
   
   // First element
   maxidx = 5;
   this->col.get()[maxidx] = static_cast<value_type>(2.0);

   res = sylver::find_col_abs_max(0, this->col_dim-1, this->col.get());   

   ASSERT_EQ(res, maxidx);

}

TYPED_TEST(FindColAbsMax, FindColAbsMaxFirstFrom)
{
   using value_type  = typename TestFixture::value_type;

   int maxidx;
   int res;

   // Reset
   maxidx = -1;
   res = -1;

   // First element
   maxidx = 3;
   // col(maxid) = 2.0
   this->col.get()[maxidx] = static_cast<value_type>(2.0);
   // col(1:maxid-1) = 10.0
   for (int i = 0; i < maxidx; ++i) {
      this->col.get()[i] = static_cast<value_type>(10.0);
   }
   // col(maxid+1:n) in [-1,1]

   res = sylver::find_col_abs_max(0, this->col_dim-1, this->col.get());   
   ASSERT_NE(res, maxidx);
   res = sylver::find_col_abs_max(maxidx, this->col_dim-1, this->col.get());   
   ASSERT_EQ(res, maxidx);
}
   
}

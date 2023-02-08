#include <memory>
#include <vector>

#include <autograd/scalar.hpp>
#include <gtest/gtest.h>

using std::make_shared;
using std::vector;
using autograd::Scalar;

TEST(ScalarTest, Add) {
  auto a = make_shared<Scalar>(-1.0);
  auto b = make_shared<Scalar>(1.0);
  auto c = a + b;
  EXPECT_EQ(0, c->data());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

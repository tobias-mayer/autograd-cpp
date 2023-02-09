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

TEST(ScalarTest, Sub) {
    auto a = make_shared<Scalar>(5.0);
    auto b = make_shared<Scalar>(2.0);
    auto c = a - b;
    EXPECT_EQ(3.0, c->data());
}

TEST(ScalarTest, MUL) {
    auto a = make_shared<Scalar>(7.0);
    auto b = make_shared<Scalar>(3.0);
    auto c = a * b;
    EXPECT_EQ(21.0, c->data());
}

TEST(ScalarTest, DIV) {
    auto a = make_shared<Scalar>(9.0);
    auto b = make_shared<Scalar>(3.0);
    auto c = a / b;
    EXPECT_EQ(3.0, c->data());
}

TEST(ScalarTest, ChildrenTracking) {
    auto a = make_shared<Scalar>(-2.0);
    auto b = make_shared<Scalar>(2.0);
    auto c = a + b;
    EXPECT_EQ(0.0, c->data());
    EXPECT_EQ(a->data(), c->children()[0]->data());
    EXPECT_EQ(b->data(), c->children()[1]->data());
}

TEST(ScalarTest, ChildrenTrackingSelf) {
    auto a = make_shared<Scalar>(-3.0);
    auto const1 = make_shared<Scalar>(-5.0);
    a = a + const1;
    EXPECT_EQ(a->children().size(), 2);
}

TEST(ScalarTest, NestingOperations) {
    auto a = make_shared<Scalar>(-2.0);
    auto b = make_shared<Scalar>(3.0);
    auto const1 = make_shared<Scalar>(2.0);

    auto c = a + b;
    c = c + const1;

    ASSERT_EQ(c->children().size(), 2); // two children (a + b, const1)
    ASSERT_FALSE(c->children()[0]->children().empty()); // c = a + b -> two children
    ASSERT_TRUE(c->children()[1]->children().empty()); // const value does not have any children
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

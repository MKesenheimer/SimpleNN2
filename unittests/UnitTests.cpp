#include <vector>
#include <iostream>
#include <gtest/gtest.h>

int add(int a, int b) {return a + b;}

TEST(Addition, CanAddTwoNumbers) {
  EXPECT_TRUE(add(2, 2) == 4);
}
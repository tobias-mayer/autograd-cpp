add_executable(scalar_test scalar_test.cpp)
target_link_libraries(scalar_test autograd gtest_main)
add_test(NAME scalar_test COMMAND scalar_test)


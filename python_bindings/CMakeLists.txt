pybind11_add_module(pyautograd bindings.cpp)
target_link_libraries(pyautograd PRIVATE autograd)

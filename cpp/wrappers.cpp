#include <pybind11/pybind11.h>

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(kf_cpp, m) {
    m.doc() = "KF C++ implementation"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
}
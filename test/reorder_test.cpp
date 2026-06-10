#include <chrono>

#include "club/reorder.hpp"
#include <icecream.hpp>
using namespace club;

int main() {
    // Read the matrix from file
    CSR<float, size_t> mat;
    std::ifstream infile("matrices/1138_bus/1138_bus.mtx");
    mat.read_from_mtx(infile);
    IC(mat.rows, mat.cols, mat.nztot());

    club::reorder(mat, 16, 5);
    return 0;
}
#include <chrono>
#include <icecream.hpp>

#include "club/reorder.hpp"
using namespace club;

int main() {
    // Read the matrix from file
    CSR<float, size_t> mat;
    std::vector<std::string> test_matrices = {
        "matrices/1138_bus/1138_bus.mtx",
        "matrices/ash292/ash292.mtx",
        "matrices/lp_stocfor3/lp_stocfor3.mtx",
        "matrices/thermal2/thermal2.mtx",
        "matrices/webbase-2001/webbase-2001.mtx",
    };
    for ( const auto& path : test_matrices ) {
        std::ifstream infile( path );
        if ( !infile.is_open() ) {
            std::cerr << "Error: Could not open file " << path << std::endl;
            continue;
        }
        IC( "Processing", path );
        mat.read_from_mtx( infile );
        IC( mat.rows, mat.cols, mat.nztot() );
        auto start = std::chrono::high_resolution_clock::now();
        club::reorder( mat, 1, 64 );
        auto end = std::chrono::high_resolution_clock::now();
        LOG_INFO( "msg", "Reordering completed", "Time (s)", std::chrono::duration<double>( end - start ).count() );
    }
    return 0;
}
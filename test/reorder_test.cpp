#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <icecream.hpp>

#include "club/reorder.hpp"
using namespace club;
namespace fs = std::filesystem;

// Repo root, baked in by CMake. Falls back to the launch directory.
#ifndef REORDER_SOURCE_DIR
#define REORDER_SOURCE_DIR "."
#endif

namespace {

    const fs::path kRoot = REORDER_SOURCE_DIR;
    const fs::path kDatasets = kRoot / "datasets";

    std::string trim( const std::string& s ) {
        const size_t b = s.find_first_not_of( " \t\r\n" );
        if ( b == std::string::npos )
            return "";
        const size_t e = s.find_last_not_of( " \t\r\n" );
        return s.substr( b, e - b + 1 );
    }

    // List entries may be absolute paths from another machine.
    std::string resolve_listed( const std::string& raw ) {
        const std::string line = trim( raw );
        if ( line.empty() || line[0] == '#' )
            return "";
        const fs::path p( line );
        const std::vector<fs::path> candidates = {
            p,
            kRoot / p,
            kDatasets / p.stem() / p.filename(),
            kDatasets / p.filename(),
        };
        for ( const auto& c : candidates ) {
            std::error_code ec;
            if ( fs::is_regular_file( c, ec ) )
                return c.string();
        }
        return "";
    }

    std::vector<std::string> default_matrices() {
        std::vector<std::string> out;
        const fs::path list_file = kDatasets / "matrices_list_filtered.txt";
        if ( std::ifstream list{ list_file }; list.is_open() ) {
            std::string line;
            while ( std::getline( list, line ) ) {
                const std::string hit = resolve_listed( line );
                if ( !hit.empty() ) {
                    out.push_back( hit );
                } else if ( const std::string t = trim( line ); !t.empty() && t[0] != '#' ) {
                    std::cerr << "Warning: listed matrix not found: " << t << std::endl;
                }
            }
            if ( !out.empty() )
                return out;
        }
        std::error_code ec;
        if ( fs::exists( kDatasets, ec ) ) {
            for ( const auto& e : fs::recursive_directory_iterator( kDatasets, ec ) ) {
                if ( e.is_regular_file() && e.path().extension() == ".mtx" )
                    out.push_back( e.path().string() );
            }
            std::sort( out.begin(), out.end() );
        }
        return out;
    }

} // namespace

int main( int argc, char** argv ) {
    // Usage: ./build/reorder [matrix.mtx ...]
    std::vector<std::string> test_matrices;
    if ( argc > 1 ) {
        for ( int i = 1; i < argc; ++i )
            test_matrices.emplace_back( argv[i] );
    } else {
        test_matrices = default_matrices();
    }
    if ( test_matrices.empty() ) {
        std::cerr << "No matrices found.\n"
                     "Either fetch them:\n"
                     "  ./fetch_datasets.sh\n"
                     "or pass .mtx paths:\n"
                     "  ./build/reorder path/to/matrix.mtx [...]\n"
                     "Files are expected under datasets/ as\n"
                     "  datasets/<name>/<name>.mtx\n"
                     "(e.g. from the SuiteSparse collection), optionally listing them in\n"
                     "datasets/matrices_list_filtered.txt"
                  << std::endl;
        return 1;
    }
    // Read the matrix from file
    CSR<float, size_t> mat;
    bool ok = true;
    for ( const auto& path : test_matrices ) {
        std::ifstream infile( path );
        if ( !infile.is_open() ) {
            std::cerr << "Error: Could not open file " << path << std::endl;
            ok = false;
            continue;
        }
        IC( "Processing", path );
        mat.read_from_mtx( infile );
        IC( mat.rows, mat.cols, mat.nztot() );
        auto start = std::chrono::high_resolution_clock::now();
        club::reorder2( mat, 1, 64 );
        auto end = std::chrono::high_resolution_clock::now();
        LOG_INFO( "msg",
                  "Reordering completed",
                  "Time (s)",
                  std::chrono::duration<double>( end - start ).count() );
    }
    return ok ? 0 : 1;
}

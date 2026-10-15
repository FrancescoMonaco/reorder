/**
 * mt-kahypar Partitioning-based Permutation Generator
 *
 * Interface: ./kahypar_perm matrix.mtx output.perm [--nparts N] [--epsilon E]
 *                                [--preset P] [--objective O] [--threads T] [--seed S]
 *
 * Generates a reordering based on mt-kahypar (Mt-KaHyPar) graph partitioning.
 * Like the METIS k-way partition tool, this orders vertices by their partition
 * assignment: vertices sharing a partition are placed contiguously, producing a
 * block-diagonal structure that improves cache locality and enables block-based
 * parallel processing.
 *
 * The sparsity pattern of the (square) matrix is treated as a symmetric graph:
 *   - each row/column index is a graph node,
 *   - every stored nonzero A[i][j] (i != j) is an undirected edge {i, j}.
 * Self-loops (the diagonal) are dropped. The graph is handed to mt-kahypar via
 * its stable C interface; the resulting block assignment per node is then used
 * to build the permutation with a stable sort by block id (vertices keep their
 * relative order inside a block).
 *
 * Output is a 1-based permutation file (new-to-old mapping) compatible with the
 * reordering pipeline: perm[new_position] = original_row_id.
 *
 * Reference:
 *   Gottesbueren et al., "Scalable High-Quality Hypergraph Partitioning",
 *   ACM Trans. Algorithms 20(1), 2024.  https://github.com/kahypar/mt-kahypar
 */

#include <mtkahypar.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
double now_sec() {
    return std::chrono::duration<double>( Clock::now().time_since_epoch() ).count();
}

// Simple flag parser: supports "--flag value" and "--flag=value".
std::string get_flag_value( int argc, char* argv[], const std::string& flag ) {
    for ( int i = 1; i < argc; ++i ) {
        std::string arg = argv[i];
        if ( arg.rfind( flag + "=", 0 ) == 0 )
            return arg.substr( flag.size() + 1 );
        if ( arg == flag && i + 1 < argc )
            return argv[i + 1];
    }
    return "";
}

struct COO {
    long rows = 0, cols = 0, nnz = 0;
    std::vector<long> row, col;
};

// Minimal Matrix Market reader (coordinate format). Handles 'pattern',
// 'integer', 'real' and 'complex' value types (values are ignored). Comment
// lines starting with '%' are skipped. No dependency on SparseBase.
COO read_mtx( const std::string& path ) {
    std::ifstream in( path );
    if ( !in.is_open() ) {
        std::cerr << "[KaHyPar-Part] Error: cannot open " << path << "\n";
        std::exit( 1 );
    }

    std::string line;
    // Banner line
    if ( !std::getline( in, line ) ) {
        std::cerr << "[KaHyPar-Part] Error: empty file " << path << "\n";
        std::exit( 1 );
    }
    const bool is_pattern = line.find( "pattern" ) != std::string::npos ||
                            line.find( "Pattern" ) != std::string::npos;

    // Skip comment lines, then read the size header.
    while ( std::getline( in, line ) ) {
        if ( !line.empty() && line[0] == '%' )
            continue;
        break;
    }
    long m = 0, n = 0, nnz_file = 0;
    if ( std::sscanf( line.c_str(), "%ld %ld %ld", &m, &n, &nnz_file ) != 3 ) {
        std::cerr << "[KaHyPar-Part] Error: failed to read matrix dimensions\n";
        std::exit( 1 );
    }

    COO coo;
    coo.rows = m;
    coo.cols = n;
    coo.row.reserve( nnz_file );
    coo.col.reserve( nnz_file );

    for ( long e = 0; e < nnz_file; ++e ) {
        long r = 0, c = 0;
        if ( is_pattern ) {
            if ( !( in >> r >> c ) )
                break;
        } else {
            // Read the two indices; ignore the value(s) so both real (one
            // value) and complex (two values) rows are consumed.
            double v1 = 0;
            if ( !( in >> r >> c >> v1 ) )
                break;
            (void)v1;
        }
        coo.row.push_back( r - 1 );  // 1-based -> 0-based
        coo.col.push_back( c - 1 );
    }
    coo.nnz = static_cast<long>( coo.row.size() );
    return coo;
}

// Save a 0-based permutation array to file in the 1-based, single-line,
// space-separated format expected by the pipeline (see SPARSEBASE/reorder_utils.h).
bool save_permutation( const std::string& perm_file, const std::vector<long>& perm ) {
    std::ofstream out( perm_file );
    if ( !out.is_open() ) {
        std::cerr << "[KaHyPar-Part] Error: could not open output file "
                  << perm_file << "\n";
        return false;
    }
    for ( size_t i = 0; i < perm.size(); ++i ) {
        if ( i > 0 )
            out << ' ';
        out << ( perm[i] + 1 );
    }
    out << "\n";
    out.close();
    return true;
}

} // namespace


int main( int argc, char* argv[] ) {
    if ( argc < 3 ) {
        std::cerr << "Usage: " << argv[0]
                  << " <matrix.mtx> <output.perm> [--nparts N] [--epsilon E]\n"
                  << "       [--preset default|quality|deterministic|large_k]\n"
                  << "       [--objective cut|km1|soed] [--threads T] [--seed S]\n"
                  << "Generates an mt-kahypar partitioning-based permutation.\n"
                  << "Optional flags:\n"
                  << "  --nparts N      number of blocks/partitions (default: 128)\n"
                  << "  --epsilon E     allowed imbalance, e.g. 0.03 (default: 0.03)\n"
                  << "  --preset P      partitioning preset (default: default)\n"
                  << "  --objective O   cut, km1 or soed (default: cut)\n"
                  << "  --threads T     number of threads (default: all cores)\n"
                  << "  --seed S        random seed (default: 42)\n";
        return 1;
    }

    const std::string mtx_file = argv[1];
    const std::string perm_file = argv[2];

    // ---- Parse options (defaults mirror the METIS partition tool) ----
    long nparts = 128;
    if ( const std::string s = get_flag_value( argc, argv, "--nparts" ); !s.empty() )
        nparts = std::stol( s );
    double epsilon = 0.03;
    if ( const std::string s = get_flag_value( argc, argv, "--epsilon" ); !s.empty() )
        epsilon = std::stod( s );
    long seed = 42;
    if ( const std::string s = get_flag_value( argc, argv, "--seed" ); !s.empty() )
        seed = std::stol( s );

    const std::string preset_str = get_flag_value( argc, argv, "--preset" );
    mt_kahypar_preset_type_t preset = DEFAULT;
    if ( preset_str == "quality" || preset_str == "QUALITY" )
        preset = QUALITY;
    else if ( preset_str == "deterministic" || preset_str == "DETERMINISTIC" )
        preset = DETERMINISTIC;
    else if ( preset_str == "large_k" || preset_str == "LARGE_K" )
        preset = LARGE_K;
    else if ( !preset_str.empty() && preset_str != "default" && preset_str != "DEFAULT" )
        std::cerr << "[KaHyPar-Part] Warning: unknown preset '" << preset_str
                  << "', using 'default'\n";

    const std::string obj_str = get_flag_value( argc, argv, "--objective" );
    mt_kahypar_objective_t objective = CUT;
    if ( obj_str == "km1" || obj_str == "KM1" )
        objective = KM1;
    else if ( obj_str == "soed" || obj_str == "SOED" )
        objective = SOED;
    else if ( !obj_str.empty() && obj_str != "cut" && obj_str != "CUT" )
        std::cerr << "[KaHyPar-Part] Warning: unknown objective '" << obj_str
                  << "', using 'cut'\n";

    size_t threads = std::thread::hardware_concurrency();
    if ( const std::string s = get_flag_value( argc, argv, "--threads" ); !s.empty() )
        threads = static_cast<size_t>( std::stol( s ) );

    // ---- Load matrix ----
    double t0 = now_sec();
    COO coo = read_mtx( mtx_file );
    const long n = coo.rows;
    if ( n != coo.cols ) {
        std::cerr << "[KaHyPar-Part] Error: mt-kahypar partitioning requires a "
                  << "square matrix. Got " << coo.rows << " x " << coo.cols << "\n";
        return 1;
    }
    if ( nparts < 1 )
        nparts = 1;
    if ( static_cast<long>( nparts ) > n ) {
        std::cerr << "[KaHyPar-Part] Warning: nparts (" << nparts
                  << ") > #nodes (" << n << "); clamping to " << n << "\n";
        nparts = n;
    }
    std::cerr << "[KaHyPar-Part] Matrix: " << coo.rows << " x " << coo.cols
              << ", nnz=" << coo.nnz << ", nparts=" << nparts << "\n";
    double load_ms = ( now_sec() - t0 ) * 1000.0;

    // ---- Build the symmetric edge set (dedup, drop self-loops) ----
    // Store each undirected edge once as (min,max).
    std::vector<std::pair<long, long>> edges;
    edges.reserve( coo.nnz );
    {
        // The dense n*n marker is only safe for modest n; fall back for large
        // matrices to a hash set on 64-bit keys.
        const bool dense_ok = n > 0 && static_cast<double>( n ) * n <= 2e8;
        if ( dense_ok ) {
            std::vector<char> seen( static_cast<size_t>( n ) * n, 0 );
            for ( long e = 0; e < coo.nnz; ++e ) {
                long u = coo.row[e], v = coo.col[e];
                if ( u == v )
                    continue;  // drop diagonal self-loops
                if ( u > v )
                    std::swap( u, v );
                const size_t key = static_cast<size_t>( u ) * n + v;
                if ( seen[key] )
                    continue;
                seen[key] = 1;
                edges.emplace_back( u, v );
            }
        } else {
            std::vector<unsigned long long> keys;
            keys.reserve( coo.nnz );
            for ( long e = 0; e < coo.nnz; ++e ) {
                long u = coo.row[e], v = coo.col[e];
                if ( u == v )
                    continue;
                if ( u > v )
                    std::swap( u, v );
                const unsigned long long key =
                    ( static_cast<unsigned long long>( u ) << 32 ) |
                    static_cast<unsigned int>( v );
                keys.push_back( key );
            }
            std::sort( keys.begin(), keys.end() );
            keys.erase( std::unique( keys.begin(), keys.end() ), keys.end() );
            edges.reserve( keys.size() );
            for ( auto key : keys ) {
                const long u = static_cast<long>( key >> 32 );
                const long v = static_cast<long>( key & 0xffffffffULL );
                edges.emplace_back( u, v );
            }
        }
    }
    const long num_edges = static_cast<long>( edges.size() );
    std::cerr << "[KaHyPar-Part] Graph: " << n << " nodes, " << num_edges
              << " edges\n";

    // ---- Set up mt-kahypar ----
    mt_kahypar_initialize( threads, /* interleaved NUMA = */ true );
    mt_kahypar_set_seed( seed );

    mt_kahypar_error_t error{};
    mt_kahypar_context_t* context = mt_kahypar_context_from_preset( preset );
    if ( context == nullptr ) {
        std::cerr << "[KaHyPar-Part] Error: failed to create context\n";
        return 1;
    }
    mt_kahypar_set_partitioning_parameters(
        context, static_cast<mt_kahypar_partition_id_t>( nparts ), epsilon, objective );

    // ---- Construct the hypergraph in memory ----
    // A graph edge {u,v} is a 2-pin hyperedge. hyperedge_indices is the CSR-like
    // offset array (num_edges+1), hyperedges is the flattened pin list.
    std::vector<size_t> hyperedge_indices( static_cast<size_t>( num_edges ) + 1 );
    std::vector<mt_kahypar_hypernode_id_t> hyperedges( static_cast<size_t>( num_edges ) * 2 );
    std::vector<mt_kahypar_hypernode_weight_t> node_weights( n, 1 );
    std::vector<mt_kahypar_hyperedge_weight_t> edge_weights( num_edges, 1 );

    for ( long e = 0; e < num_edges; ++e ) {
        hyperedge_indices[e] = static_cast<size_t>( e ) * 2;
        hyperedges[static_cast<size_t>( e ) * 2 + 0] =
            static_cast<mt_kahypar_hypernode_id_t>( edges[e].first );
        hyperedges[static_cast<size_t>( e ) * 2 + 1] =
            static_cast<mt_kahypar_hypernode_id_t>( edges[e].second );
    }
    hyperedge_indices[num_edges] = static_cast<size_t>( num_edges ) * 2;

    mt_kahypar_hypergraph_t hg = mt_kahypar_create_hypergraph(
        context,
        static_cast<mt_kahypar_hypernode_id_t>( n ),
        static_cast<mt_kahypar_hyperedge_id_t>( num_edges ),
        hyperedge_indices.data(),
        hyperedges.data(),
        num_edges > 0 ? edge_weights.data() : nullptr,
        node_weights.data(),
        &error );
    if ( hg.hypergraph == nullptr ) {
        std::cerr << "[KaHyPar-Part] Error: create_hypergraph failed: "
                  << ( error.msg ? error.msg : "(unknown)" ) << "\n";
        mt_kahypar_free_context( context );
        return 1;
    }

    // ---- Partition ----
    double r0 = now_sec();
    mt_kahypar_partitioned_hypergraph_t phg = mt_kahypar_partition( hg, context, &error );
    double reorder_ms = ( now_sec() - r0 ) * 1000.0;
    if ( phg.partitioned_hg == nullptr ) {
        std::cerr << "[KaHyPar-Part] Error: partition failed: "
                  << ( error.msg ? error.msg : "(unknown)" ) << "\n";
        mt_kahypar_free_hypergraph( hg );
        mt_kahypar_free_context( context );
        return 1;
    }

    std::vector<mt_kahypar_partition_id_t> block( n );
    mt_kahypar_get_partition( phg, block.data() );

    // ---- Build permutation: stable-sort vertices by block id ----
    std::vector<long> order( n );
    std::iota( order.begin(), order.end(), 0 );
    std::stable_sort( order.begin(), order.end(),
                      [&]( long a, long b ) { return block[a] < block[b]; } );
    // order[new_position] = original vertex id  (new-to-old mapping)

    const bool ok = save_permutation( perm_file, order );

    std::cerr << "[KaHyPar-Part] Saving permutation to: " << perm_file << "\n";
    std::cerr << "[KaHyPar-Part] " << ( ok ? "Success" : "Failed" ) << "\n";
    // Timer lines (parsed by parse_results.py / test_pipeline.py).
    std::printf( "<Timer>[loading] %.6f ms\n", load_ms );
    std::printf( "<Timer>[reordering] %.6f ms\n", reorder_ms );

    mt_kahypar_free_partitioned_hypergraph( phg );
    mt_kahypar_free_hypergraph( hg );
    mt_kahypar_free_context( context );

    return ok ? 0 : 1;
}


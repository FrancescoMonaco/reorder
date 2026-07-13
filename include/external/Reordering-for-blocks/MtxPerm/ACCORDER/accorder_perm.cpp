// accorder_perm.cpp — Thin wrapper around Acc-SpMM's accOrder reordering
//
// Reads a Matrix Market file, runs the accOrder algorithm (community detection
// via Rabbit Order + greedy common-neighbor traversal), and writes a .perm file
// in our pipeline's 1-based format.
//
// The actual reordering logic lives in the unmodified Zenodo sources under src/.

// Workaround: Boost 1.82+ enforces trivially-copyable for boost::atomic<T>,
// but rabbit_order.hpp uses boost::atomic with non-trivially-copyable types.
// Disable the static_assert and the __builtin_bit_cast path that also checks.
// Safe: rabbit_order only uses lock-free 64-bit atomics via union access (raw member).
#define BOOST_ATOMIC_DETAIL_NO_CXX11_IS_TRIVIALLY_COPYABLE
#define BOOST_ATOMIC_DETAIL_NO_HAS_UNIQUE_OBJECT_REPRESENTATIONS

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

// ---- accOrder headers (unmodified Zenodo sources) ----
#include "my_order.hpp"   // pulls in edge_list.hpp -> rabbit_order.hpp

// ======================================================================
// Minimal MTX reader (no CUDA deps, mirrors the Zenodo utils/load_data.h)
// ======================================================================
namespace mtx_io {

using rabbit_order::vint;

struct COO {
    vint rows = 0, cols = 0, nnz = 0;
    std::vector<vint> row, col;

    // Write edge list in the format expected by edge_list::read()
    void write_edge_list(const std::string& path) const {
        FILE* f = fopen(path.c_str(), "w");
        if (!f) { perror("write_edge_list"); exit(1); }
        for (vint i = 0; i < nnz; ++i)
            fprintf(f, "%u %u\n", row[i], col[i]);
        fclose(f);
    }
};

COO read_mtx(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); exit(1); }

    char line[2048];
    // Read banner
    if (!fgets(line, sizeof(line), f)) { fclose(f); exit(1); }

    bool is_pattern   = (strstr(line, "pattern") != nullptr ||
                         strstr(line, "Pattern") != nullptr);

    // Skip comments
    while (fgets(line, sizeof(line), f) && line[0] == '%') {}

    vint m, n, nnz_file;
    if (sscanf(line, "%u %u %u", &m, &n, &nnz_file) != 3) {
        fprintf(stderr, "Failed to read matrix dimensions\n"); exit(1);
    }

    // Read entries (1-based in file -> 0-based in memory)
    // NOTE: Do NOT symmetrize here — make_adj_list() in the original accOrder
    // code already symmetrizes all edges.  Doing it here would double the data
    // before that second symmetrization, causing 4× edge count and much slower
    // sorting.
    std::vector<vint> rows_tmp, cols_tmp;
    rows_tmp.reserve(nnz_file);
    cols_tmp.reserve(nnz_file);

    for (vint i = 0; i < nnz_file; ++i) {
        vint r, c;
        double v1, v2;
        if (is_pattern) {
            if (fscanf(f, "%u %u", &r, &c) != 2) break;
        } else {
            // Try real or integer; also handles complex (just ignore imaginary)
            int nread = fscanf(f, "%u %u %lg %lg", &r, &c, &v1, &v2);
            if (nread < 2) break;
        }
        r--; c--;  // to 0-based
        rows_tmp.push_back(r);
        cols_tmp.push_back(c);
    }
    fclose(f);

    COO coo;
    coo.rows = m;
    coo.cols = n;
    coo.nnz  = static_cast<vint>(rows_tmp.size());
    coo.row  = std::move(rows_tmp);
    coo.col  = std::move(cols_tmp);
    return coo;
}

} // namespace mtx_io

// ======================================================================

static double now_sec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input.mtx> <output.perm>\n", argv[0]);
        return 1;
    }
    const char* mtx_path  = argv[1];
    const char* perm_path = argv[2];

    // --- 1. Read MTX ---
    double t0 = now_sec();
    auto coo = mtx_io::read_mtx(mtx_path);
    std::cerr << "Matrix: " << coo.rows << " x " << coo.cols
              << ", nnz=" << coo.nnz << std::endl;

    // Create a temp directory for intermediate files
    auto tmp_dir = std::filesystem::temp_directory_path() /
                   ("accorder_" + std::to_string(getpid()));
    std::filesystem::create_directories(tmp_dir);
    std::string edge_file = (tmp_dir / "edges.txt").string();

    // Write edge list
    coo.write_edge_list(edge_file);
    double t1 = now_sec();
    fprintf(stderr, "<Timer>[loading] %f ms\n", (t1 - t0) * 1000.0);

    // --- 2. Run accOrder reordering ---
    double t2 = now_sec();
    auto adj = read_graph(edge_file);

    // re2order_vertex writes "my_order.txt" in CWD — change to tmp dir
    auto orig_cwd = std::filesystem::current_path();
    std::filesystem::current_path(tmp_dir);

    std::deque<rabbit_order::vint> new2old = re2order_vertex(std::move(adj));
    // Clear global state from my_order.hpp
    all_vertex_nbrs.clear();
    std::map<rabbit_order::vint, std::vector<rabbit_order::vint>>().swap(all_vertex_nbrs);

    std::filesystem::current_path(orig_cwd);

    double t3 = now_sec();
    fprintf(stderr, "<Timer>[reordering] %f ms\n", (t3 - t2) * 1000.0);

    // --- 3. Write .perm file (1-based, space-separated, single line) ---
    // new2old[i] = original vertex at new position i (new-to-old mapping),
    // which is the format the pipeline expects (used as A[perm, :] indexing).
    rabbit_order::vint n = static_cast<rabbit_order::vint>(new2old.size());
    std::ofstream ofs(perm_path);
    if (!ofs) {
        fprintf(stderr, "Cannot open output %s\n", perm_path);
        return 1;
    }
    for (rabbit_order::vint i = 0; i < n; ++i) {
        if (i > 0) ofs << ' ';
        ofs << (new2old[i] + 1);  // convert to 1-based
    }
    ofs << '\n';
    ofs.close();

    // Clean up temp files
    std::filesystem::remove_all(tmp_dir);

    fprintf(stderr, "Wrote permutation (%u vertices) to %s\n", n, perm_path);
    return 0;
}

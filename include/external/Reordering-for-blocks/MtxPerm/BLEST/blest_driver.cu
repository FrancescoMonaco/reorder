/*
 * Custom BFS driver for the BLEST framework.
 * Replaces Benchmark::main() to support external reorderings via --no-reorder.
 *
 * Usage:
 *   blest_driver <input.mtx> [--no-reorder] [--jaccard 0|1] [--window W]
 *                             [--n-sources N] [--seed S]
 */

#include "CSC.cuh"
#include "BVSS.cuh"
#include "BFSKernel.cuh"
#include <cstdlib>
#include <cstring>
#include <iomanip>

static void print_usage(const char* prog)
{
    std::cerr
        << "Usage: " << prog
        << " <input.mtx> [options]\n"
        << "  <input.mtx>       Path to a Matrix Market file\n"
        << "  --no-reorder      Use identity permutation (skip BLEST internal reordering)\n"
        << "  --jaccard <0|1>   Enable Jaccard reordering (default: 1)\n"
        << "  --window <W>      Window size for Jaccard (default: 65536)\n"
        << "  --n-sources <N>   Number of random BFS source vertices (default: 64)\n"
        << "  --seed <S>        Random seed for source selection (default: 42)\n";
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        print_usage(argv[0]);
        return 1;
    }

    std::string inputFile = argv[1];
    if (inputFile == "-h" || inputFile == "--help")
    {
        print_usage(argv[0]);
        return 0;
    }

    bool noReorder = false;
    bool jackardEnabled = true;
    unsigned windowSize = 65536;
    unsigned nSources = 64;
    unsigned seed = 42;

    for (int i = 2; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--no-reorder")
        {
            noReorder = true;
        }
        else if (arg == "--jaccard" && i + 1 < argc)
        {
            jackardEnabled = (std::string(argv[++i]) == "1");
        }
        else if (arg == "--window" && i + 1 < argc)
        {
            windowSize = static_cast<unsigned>(std::stoul(argv[++i]));
        }
        else if (arg == "--n-sources" && i + 1 < argc)
        {
            nSources = static_cast<unsigned>(std::stoul(argv[++i]));
        }
        else if (arg == "--seed" && i + 1 < argc)
        {
            seed = static_cast<unsigned>(std::stoul(argv[++i]));
        }
        else
        {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Set global config used by BLEST internals
    JACKARD_ON = jackardEnabled;
    WINDOW_SIZE = windowSize;

    constexpr unsigned sliceSize = 8;
    constexpr unsigned noMasks = 32 / sliceSize;

    try
    {
        // --- Phase 1: CSC construction (loading) ---
        double startCSC = omp_get_wtime();
        // CSC constructor reads .mtx, treating it as undirected (symmetric) with non-binary values
        CSC* csc = new CSC(inputFile, true, false);
        double endCSC = omp_get_wtime();
        double loadingMs = (endCSC - startCSC) * 1000.0;
        std::cout << "<Timer>[loading] " << std::fixed << std::setprecision(6) << loadingMs << " ms" << std::endl;

        // --- Phase 2: Reordering + BVSS construction (preprocessing) ---
        double startPreprocess = omp_get_wtime();

        // Determine FULL_PADDING based on network type
        if (csc->isSocialNetwork())
        {
            FULL_PADDING = false;
        }
        else
        {
            FULL_PADDING = true;
        }

        unsigned* inversePermutation = nullptr;
        if (noReorder)
        {
            // Create identity permutation (natural() is private in CSC)
            unsigned N = csc->getN();
            inversePermutation = new unsigned[N];
            for (unsigned i = 0; i < N; ++i)
                inversePermutation[i] = i;
        }
        else
        {
            inversePermutation = csc->reorder(sliceSize);
        }

        // Construct BVSS
        std::ofstream devnull("/dev/null");
        BVSS* bvss = new BVSS(sliceSize, noMasks, devnull);
        bvss->constructFromCSCMatrix(csc);

        double endPreprocess = omp_get_wtime();
        double preprocessMs = (endPreprocess - startPreprocess) * 1000.0;
        std::cout << "<Timer>[preprocessing] " << std::fixed << std::setprecision(6) << preprocessMs << " ms" << std::endl;

        // --- Phase 3: Generate random source vertices ---
        std::mt19937 rng(seed);
        unsigned N = csc->getN();
        std::uniform_int_distribution<unsigned> dist(0, N - 1);
        std::vector<unsigned> sources(nSources);
        for (unsigned s = 0; s < nSources; ++s)
        {
            // Map source through inverse permutation (reordered space)
            unsigned origVertex = dist(rng);
            sources[s] = inversePermutation[origVertex];
        }

        // --- Phase 4: Run BFS ---
        BFSKernel* kernel = new BFSKernel(dynamic_cast<BitMatrix*>(bvss));
        std::vector<BFSResult> results = kernel->multiSourceRun(sources);

        // Compute average BFS time
        double totalTime = 0.0;
        for (auto& result : results)
        {
            totalTime += result.time;
        }
        double avgTimeMs = (totalTime / results.size()) * 1000.0;
        std::cout << "<Timer>[operation] " << std::fixed << std::setprecision(6) << avgTimeMs << " ms" << std::endl;

        // Cleanup
        for (auto& result : results)
        {
            delete[] result.levels;
        }
        delete kernel;
        delete bvss;
        devnull.close();
        delete[] inversePermutation;
        delete csc;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

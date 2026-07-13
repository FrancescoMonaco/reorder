# Distributed MMIO

Lightweight Templated `C++` library for local or distributed reading of Matrix Market files. Can be also used in a `C` project via a provided wrapper.

This repository integrates with MtxMan ([https://github.com/ThomasPasquali/MtxMan](https://github.com/ThomasPasquali/MtxMan)) to simplify the management of Matrix Market files, check it out!

## Usage in CMake Projects

This library can be included either as a submodule or by copying the source files directly into your project.

### CMake Usage (for C++ projects)

If you are using CMake, you can include this library as follows:

```cmake
add_subdirectory(distributed_mmio)
```

And ensure to link the target on the executables that require it:

```cmake
add_executable(main main.cpp)
target_link_libraries(main distributed_mmio <...> )
```

If you are not using CMake, make sure to include the `distributed_mmio/include` directory and `distributed_mmio/src/mmio.cpp`, `distributed_mmio/src/mmio_utils.cpp` source files.

### Makefile Usage (for C projects)

If you are using a Makefile, first build the library:

```bash
cd distributed_mmio
make
```
Then, you can include the library in your project by adding the following lines to your Makefile:

```makefile
CFLAGS += -I/path/to/distributed_mmio/include
LDFLAGS += /path/to/distributed_mmio/libdistributed_mmio.a
```

<!-- ## Including "Distributed MMIO" with CMake

Simply add to your `CMakeLists.txt` the following:
<!-- TODO ### Fetch and Build Automatically (Recommended)

```cmake
include(FetchContent)

FetchContent_Declare(
  distributed_mmio
  GIT_REPOSITORY https://github.com/HicrestLaboratory/distributed_mmio.git
  GIT_TAG        main # or a specific tag/commit
)

FetchContent_MakeAvailable(distributed_mmio)

target_link_libraries(my_target PRIVATE distributed_mmio)
``` -->

### Clone and Include



```cmake
add_subdirectory(distributed_mmio)
```

If you are not using CMake, make sure to include the `distributed_mmio/include` directory and `distributed_mmio/src/mmio.cpp`, `distributed_mmio/src/mmio_utils.cpp` source files.

<!-- ### Makefile Usage (for C projects)

If you are using a Makefile, first build the library:

```bash
cd distributed_mmio
make
```
Then, you can include the library in your project by adding the following lines to your Makefile:

```makefile
CFLAGS += -I/path/to/distributed_mmio/include
LDFLAGS += /path/to/distributed_mmio/libdistributed_mmio.a
``` -->

## Usage Examples

### Non-distributed Matrix Market File CSR Read 

```c++
#include "mmio.h"
// ...
CSR_local<uint32_t, float> *csr_matrix = Distr_MMIO_CSR_local_read<uint32_t, float>("path/to/mtx_file", false, &meta);
COO_local<uint64_t, double> *coo_matrix = Distr_MMIO_COO_local_read<uint64_t, double>("path/to/mtx_file", false, &meta);
```

The `Distr_MMIO_CSR_local_read` and `Distr_MMIO_COO_local_read` functions take the following parameters:

-   **`const char* filename`**: Path to the input matrix file.
-   **`bool expl_val_for_bin_mtx = false`** (optional): When set to `true`, this allows the function to allocate the `val` array.
-   **`Matrix_Metadata* meta = NULL`** (optional): A pointer to a `Matrix_Metadata` struct. If provided, the function will populate it with information from the matrix file's header.

Explicit template instantiation is currently available for types:

| Index Type | Value Type |
|------------|------------|
| uint32_t   | float      |
| uint32_t   | double     |
| uint64_t   | float      |
| uint64_t   | double     |
| uint64_t   | uint64_t   |
| int        | float      |
| int        | double     |

> [!INFO]  
> If you need other, add the declaration at the end of `mmio.cpp`, `mmio_structs.cpp`, `distributed_mmio.cpp` and `distributed_mmio_structs.cpp`. 

### Non-distributed Matrix Market File CSR Read (C wrapper)

```c
#include "mmio_c_wrapper.h"

// ...
mmio_csr_u32_f32_t *csr_matrix = mmio_read_csr_u32_f32("path/to/mtx_file", false);
mmio_coo_u64_f64_t *coo_matrix = mmio_read_coo_u64_f64("path/to/mtx_file", false);
```

The C wrapper functions (`mmio_read_*`) take the following arguments:

-   **`const char* filename`**: Path to the input matrix file.
-   **`bool alloc_val`**: When set to `true`, this allows the function to allocate the `val` array.

The C wrapper uses a consistent naming scheme for its types and functions:
-   **Structs**: `mmio_<format>_<index_type>_<value_type>_t`
-   **Functions**: `mmio_read_<format>_<index_type>_<value_type>` and `mmio_destroy_<format>_<index_type>_<value_type>`

Where:
-   `<format>` is `csr` or `coo`.
-   `<index_type>` is `u32` or `u64`.
-   `<value_type>` is `f32` (float) or `f64` (double).

# Binary Matrix Market (.bmtx)

This repository also allows to convert, read and write matrices into a binary format.

> [!IMPORTANT]  
> `distributed_mmio` recognizes a file as Binary Matrix Market ONLY if its file extension is `.bmtx`. Currently there are no explicit flags to override this behavior.

## MTX to BMTX Converter Tool

Building `distributed_mmio` setting the `DMMIO_TOOLS` option (`-DDMMIO_TOOLS=ON`) will allow to build the target `mtx_to_bmtx`.  
This executable will read a `.mtx` file and generate the corresponding `.bmtx`.

> [!NOTE]  
> This tool currently operates only locally (non-distributed). Converting large files may take a while. 

## How it works

The idea is quite simple. A `BMTX` file is formatted as follows:

```
%%MatrixMarket <original header entries> <indices bytes> <values bytes>   // Added two custom values to the header
% <original multiline custom header>
...
% <original multiline custom header>
<n rows> <n cols> <n entries>                                             // As in the original MM format
<
  triples or couples in binary
  (types sizes accordigly with indices bytes and values bytes)
>
```

## mtx_to_bmtx Converter

CMake has a target named `mtx_to_bmtx` which compiles the converter. Once compiled:

```bash
build/mtx_to_bmtx path/to/.mtx  # Converts an MTX file to BMTX. Values (if present) will be written using 4 bytes (float)
build/mtx_to_bmtx path/to/.bmtx # Converts an BMTX file to MTX

build/mtx_to_bmtx path/to/.mtx [-d|--double-val] # Converts an MTX file to BMTX using 8 bytes for values (double)
```

> **NOTE** The size of indices selected automatically in order to maximize compression while mantaining integrity.

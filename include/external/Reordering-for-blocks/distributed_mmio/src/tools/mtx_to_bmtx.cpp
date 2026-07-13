#include <stdio.h>
#include <string>
#include <algorithm>
#include <chrono>
#include <ccutils/colors.h>
#include <ccutils/timers.h>

#include <mmio/mmio.h>
#include <mmio/io.h>

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    // printf("Usage: %s <filename> [-r|--reverse] [-d|--double-val]\n", argv[0]);
    printf("Usage: %s <filename> [-d|--double-val]\n", argv[0]);
    return EXIT_FAILURE;
  }

  std::string filename = argv[1];
  // bool reverse = false;
  bool double_val = false;

  uint32_t arg_i = 2;
  while (arg_i < argc) {    
    std::string flag = argv[arg_i];
    // if (flag == "-r" || flag == "--reverse") {
    //   reverse = true;
    // } else
    if (flag == "-d" || flag == "--double-val") {
      double_val = true;
    } else {
      printf("Unknown option: %s\n", argv[arg_i]);
    }
  }

  using namespace mmio;

  Matrix_Metadata mtx_meta;
  mtx_meta.value_bytes = double_val ? 8 : 4;
  CPU_TIMER_INIT(COO_read)
  COO<uint64_t, double> *coo = COO_read<uint64_t, double>(filename.c_str(), false, &mtx_meta);
  CPU_TIMER_CLOSE(COO_read)
  if (coo == NULL) {
    fprintf(stderr, "Something went wrong\n");
    exit(EXIT_FAILURE);
  }

  std::string out_filename = filename;
  size_t last_dot = out_filename.find_last_of('.');
  if (last_dot != std::string::npos) {
    out_filename = out_filename.substr(0, last_dot);
  }

  bool converting_to_bmtx = !mmio::io::mm_is_file_extension_bmtx(filename);

  // DMMIO_print_COO(coo);
  // if (!converting_to_bmtx) exit(0);

  CPU_TIMER_INIT(Conversion)
  if (converting_to_bmtx) {
    printf("Converting MTX file to BMTX...\n");
    out_filename += ".bmtx";
    COO_write(coo, out_filename.c_str(), true, &mtx_meta);
    printf("BMTX file written to %s\n", out_filename.c_str());
  } else {
    printf("Converting BMTX file to MTX...\n");
    out_filename += ".mtx";
    COO_write(coo, out_filename.c_str(), false, &mtx_meta);
    printf("MTX file written to %s\n", out_filename.c_str());
  }
  CPU_TIMER_CLOSE(Conversion)

  COO_destroy(&coo);

  // Compare file sizes of input and output files
  FILE *f_in = fopen(filename.c_str(), "rb");
  FILE *f_out = fopen(out_filename.c_str(), "rb");
  if (!f_in || !f_out) {
    fprintf(stderr, "Error opening files for size comparison\n");
    if (f_in) fclose(f_in);
    if (f_out) fclose(f_out);
    exit(EXIT_FAILURE);
  }
  fseek(f_in, 0, SEEK_END);
  fseek(f_out, 0, SEEK_END);
  long size_in = ftell(f_in);
  long size_out = ftell(f_out);
  fclose(f_in);
  fclose(f_out);

  printf("Size of input  file (%s): %.3f MB\n", filename.c_str(), size_in/1000000.0f);
  printf("Size of output file (%s): %.3f MB\n", out_filename.c_str(), size_out/1000000.0f);
  printf(BRIGHT_GREEN "Ratio: %.2f%%" RESET "\n", (float)size_out/size_in*100.0f);

  return 0;
}

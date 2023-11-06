/*
 * Copyright 1993-2021 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#pragma once
#include <cuda_runtime_api.h>  // cudaMalloc, cudaMemcpy, etc.
#include <cusp/csr_matrix.h>   // cusp::csr_matrix
#include <cusp/io/matrix_market.h>
#include <cusparse.h>  // cusparseSpMM
#include <stdio.h>     // printf
#include <stdlib.h>    // EXIT_FAILURE
#include <utils/generate_random_data.h>
// renamed this source file to .cpp to allow cstddef. Source:
// https://talk.pokitto.com/t/sudden-error-cstddef-no-such-file-or-directory/711/4
// renamed to .cu to allow cusp::csr_matrix<.,.,cusp::device_memory> instants as
// elaborated here:
// https://talk.pokitto.com/t/sudden-error-cstddef-no-such-file-or-directory/711/4
#include <utils/helper_string.h>

#include <chrono>
#include <map>
#include <string>
#include <tuple>

#include "helper_CUDAGraphConstructor.cu.h"
#include "helper_cusp.cu.h"
#include "helper_kernels.cu.h"
#include "helper_loops.cu.h"
#include "npy.hpp"

#define CHECK_CUDA(func)                                                   \
  {                                                                        \
    cudaError_t status = (func);                                           \
    if (status != cudaSuccess) {                                           \
      printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, \
             cudaGetErrorString(status), status);                          \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  }

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, \
             cusparseGetErrorString(status), status);                          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

namespace BenchSpMMCSRPartitioned {
struct TimingResults {
  std::vector<cudaEvent_t> start_events;
  std::vector<cudaEvent_t> stop_events;
  std::map<std::string, std::tuple<cudaEvent_t, cudaEvent_t>>
      utility_timestamps;
};

struct ProblemSpec {
  int A_num_rows;
  int A_num_cols;
  int B_num_cols;
  int AA_num_rows;
  int AA_num_cols;
  int BB_num_cols;
  float A_sparsity;
  bool enable_dump;
  bool enable_timing;
  bool enable_per_stream_timing;
  bool enable_debug_timing;
  bool enable_graph;
  bool enable_preprocess;
  char *cli_result_path_and_prefix;
  bool flag_specify_result_path_and_prefix;
  bool test_API_on_stream;
  int nstreams;
};

struct RuntimeData {
  int A_nnz;
  std::vector<int> AA_nnz;
  int B_num_rows;
  int ldb;
  int ldc;
  int B_size;
  int C_size;
  float alpha;
  float beta;
  float *hB;
  float *dB, *dC;
  std::vector<cusparseHandle_t> handles;
  std::vector<cusparseSpMatDescr_t> matAA;
  std::vector<cusparseDnMatDescr_t> matBB, matCC;
  std::vector<void *> dBuffers;
  std::vector<size_t> bufferSizes;
  // Keep hA in case data needs to be dumped
  cusp::csr_matrix<int, float, cusp::host_memory> hA;
  std::vector<cusp::csr_matrix<int, float, cusp::host_memory>> hAA;
  std::vector<cusp::csr_matrix<int, float, cusp::device_memory>> dAA;
  // The recommended way is to switch stream before CUSPARSE compute APIs by
  // cusparseSetStream() And in practice concurrent kernels won't be larger than
  // 16 https://docs.nvidia.com/cuda/cusparse/index.html#thread-safety
  std::vector<cudaStream_t> streams;
  std::vector<cudaGraph_t> graphs;
  std::vector<cudaGraphExec_t> graphExecs;
};

void print_usage() {
  printf(
      "Usage: bench_spmm_csr_partitioned --A_num_rows=## --A_num_cols=## "
      "--B_num_cols=## "
      "--AA_num_rows=## --AA_num_cols=## --BB_num_cols=## "
      "--A_sparsity=0.## [--enable_dump] [--result_path_and_prefix=...] "
      "[--enable_timing] [--enable_per_stream_timing] [--enable_debug_timing] "
      "[--test_API_on_stream]\n");
  // Print the meaning of each argument
  printf(
      "--enable_timing records the elapsed time of the computation function\n"
      "--enable_debug_timing also records the elapsed time of the computation\n"
      "function; but it adds device synchronization and uses chrono to record\n"
      "the timing\n"
      "--enable_preprocess enables the cusparse preprocessing, i.e., "
      "autotuning to find the optimized schedule for the given problem size "
      "and sparse pattern\n"
      "--enable_per_stream_timing prints the elapsed time on every stream of\n"
      "the computation function (not implemented yet)\n"
      "When there is single stream, --enable_timing has the same meaning as\n"
      "--enable_per_stream_timing. To reduce confusion,\n"
      "--enable_per_stream_timing is not allowed when it is single stream\n"
      "When there are multiple streams, --enable_timing prints the elapsed\n"
      "time of the complete computation function:\n"
      "It waits all events on the first stream and print the elapsed time of\n"
      "the.\n");
}

bool report_elapsed_time_per_stream(bool enable_per_stream_timing,
                                    bool enable_timing, int nstreams) {
  return enable_per_stream_timing || (enable_timing && nstreams == 1);
}

bool wait_streams_on_first_and_report_that_as_elapsed_time(
    bool enable_per_stream_timing, bool enable_timing, int nstreams) {
  return (enable_timing && nstreams > 1);
}

std::tuple<ProblemSpec, std::shared_ptr<RuntimeData>> generate_data_and_prepare(
    const int argc, const char **argv, TimingResults &timing_results) {
  // Host problem definition
  int A_num_rows = getCmdLineArgumentInt(argc, argv, "A_num_rows");
  int A_num_cols = getCmdLineArgumentInt(argc, argv, "A_num_cols");
  int B_num_cols = getCmdLineArgumentInt(argc, argv, "B_num_cols");
  int AA_num_rows = getCmdLineArgumentInt(argc, argv, "AA_num_rows");
  int AA_num_cols = getCmdLineArgumentInt(argc, argv, "AA_num_cols");
  int BB_num_cols = getCmdLineArgumentInt(argc, argv, "BB_num_cols");
  float A_sparsity = getCmdLineArgumentFloat(argc, argv, "A_sparsity");
  int nstreams = getCmdLineArgumentInt(argc, argv, "nstreams");
  nstreams = (nstreams > 0) ? nstreams : 1;  // default value
  bool enable_dump = checkCmdLineFlag(argc, argv, "enable_dump");
  bool enable_graph = checkCmdLineFlag(argc, argv, "enable_graph");
  bool enable_timing = checkCmdLineFlag(argc, argv, "enable_timing");
  bool enable_per_stream_timing =
      checkCmdLineFlag(argc, argv, "enable_per_stream_timing");
  bool test_API_on_stream = checkCmdLineFlag(argc, argv, "test_API_on_stream");
  bool enable_debug_timing =
      checkCmdLineFlag(argc, argv, "enable_debug_timing");
  bool enable_preprocess = checkCmdLineFlag(argc, argv, "enable_preprocess");
  char *cli_result_path_and_prefix;
  bool flag_specify_result_path_and_prefix = getCmdLineArgumentString(
      argc, argv, "result_path_and_prefix", &cli_result_path_and_prefix);
  if (A_num_rows == 0 || A_num_cols == 0 || B_num_cols == 0 ||
      AA_num_rows == 0 || AA_num_cols == 0 || BB_num_cols == 0 ||
      A_sparsity == 0.0f) {
    print_usage();
    // Example: ./bench_spmm_csr_partitioned --A_num_rows=1024 --A_num_cols=512
    // --B_num_cols=512 --AA_num_rows=128 --AA_num_cols=64 --BB_num_cols=64
    // --A_sparsity=0.1 --test_API_on_stream
    // Example: ./bench_spmm_csr_partitioned --A_num_rows=128 --A_num_cols=256
    // --B_num_cols=256 --AA_num_rows=16 --AA_num_cols=64 --BB_num_cols=64
    // --A_sparsity=0.1
    exit(EXIT_FAILURE);
  }
  if (nstreams == 1 && enable_per_stream_timing) {
    printf(
        "please use --enable_timing instead of --enable_per_stream_timing "
        "when there is only one stream\n");
    print_usage();
    exit(EXIT_FAILURE);
  }
  if (test_API_on_stream) {
    printf(
        "This is to test if cusparse environment and matrix handle creation "
        "APIs are on any stream or not. There are three possibilities 1) the "
        "API is blocking, 2) the API is on default stream, and 3) the API is "
        "on the specified stream set by cusparseSetStream. We can use two "
        "events to check if either 2) or 3) is true. We use C++ chrono to "
        "check if 1) is true.\n");
  }
  if (A_num_rows % AA_num_rows != 0 || A_num_cols % AA_num_cols != 0 ||
      A_num_cols % BB_num_cols != 0) {
    print_usage();
    printf(
        "A_num_rows must be a multiple of AA_num_rows, A_num_cols must be a "
        "multiple of AA_num_cols, A_num_cols must be a multiple of "
        "BB_num_cols\n");
    exit(EXIT_FAILURE);
  }
  printf("A_num_rows: %d\n", A_num_rows);
  printf("A_num_cols: %d\n", A_num_cols);
  printf("B_num_cols: %d\n", B_num_cols);
  printf("AA_num_rows: %d\n", AA_num_rows);
  printf("AA_num_cols: %d\n", AA_num_cols);
  printf("BB_num_cols: %d\n", BB_num_cols);
  printf("A_sparsity: %f\n", A_sparsity);

  // ***** END OF HOST PROBLEM DEFINITION *****
  int A_nnz = A_num_rows * A_num_cols * A_sparsity;
  int B_num_rows = A_num_cols;
  int ldb = B_num_rows;
  int ldc = A_num_rows;
  int B_size = ldb * B_num_cols;
  int C_size = ldc * B_num_cols;
  float alpha = 1.0f;
  float beta = 0.0f;
  float *hB;
  float *dB, *dC;
  std::vector<cusparseHandle_t> handles;
  // std::vector<void *> dBuffers{};
  // std::vector<size_t> bufferSizes{};
  // std::vector<cusparseSpMatDescr_t> matAA{};
  // std::vector<cusparseDnMatDescr_t> matBB{};
  // std::vector<cusparseDnMatDescr_t> matCC{};
  std::vector<cudaStream_t> streams;
  std::shared_ptr<RuntimeData> runtime_data = std::make_shared<RuntimeData>();

  // instantiating data
  hB = (float *)malloc(sizeof(float) * B_size);
  generate_random_matrix(hB, B_size);
  cusp::csr_matrix<int, float, cusp::host_memory> hA =
      generate_random_sparse_matrix_nodup<
          cusp::csr_matrix<int, float, cusp::host_memory>>(A_num_rows,
                                                           A_num_cols, A_nnz);
  auto hAA = partitionMatrix(hA, AA_num_rows, AA_num_cols);
  std::vector<cusp::csr_matrix<int, float, cusp::device_memory>> dAA;
  std::vector<int> AA_nnz;
  for (auto &AA : hAA) {
    dAA.push_back(cusp::csr_matrix<int, float, cusp::device_memory>(AA));
    AA_nnz.push_back(AA.num_entries);
  }
  A_nnz = hA.values.size();
  printf("actual A_nnz using non-dup random data generation: %d\n", A_nnz);

  cudaEvent_t handle_creation_start, handle_creation_stop;
  cudaEvent_t test_API_0_handle_creation_start, test_API_0_handle_creation_stop;
  cudaEvent_t test_API_stream_handle_creation_start,
      test_API_stream_handle_creation_stop;
  cudaEvent_t data_copy_start, data_copy_stop;
  cudaEvent_t cusparse_data_handle_and_buffer_creation_start,
      cusparse_data_handle_and_buffer_creation_stop;
  cudaEvent_t test_API_0_cusparse_data_handle_and_buffer_creation_start,
      test_API_0_cusparse_data_handle_and_buffer_creation_stop;
  cudaEvent_t test_API_stream_cusparse_data_handle_and_buffer_creation_start,
      test_API_stream_cusparse_data_handle_and_buffer_creation_stop;

  CHECK_CUDA(cudaEventCreate(&handle_creation_start));
  CHECK_CUDA(cudaEventCreate(&handle_creation_stop));
  CHECK_CUDA(cudaEventCreate(&data_copy_start));
  CHECK_CUDA(cudaEventCreate(&data_copy_stop));
  CHECK_CUDA(cudaEventCreate(&cusparse_data_handle_and_buffer_creation_start));
  CHECK_CUDA(cudaEventCreate(&cusparse_data_handle_and_buffer_creation_stop));
  if (test_API_on_stream) {
    CHECK_CUDA(cudaEventCreate(&test_API_0_handle_creation_start));
    CHECK_CUDA(cudaEventCreate(&test_API_0_handle_creation_stop));
    CHECK_CUDA(cudaEventCreate(&test_API_stream_handle_creation_start));
    CHECK_CUDA(cudaEventCreate(&test_API_stream_handle_creation_stop));
    CHECK_CUDA(cudaEventCreate(
        &test_API_0_cusparse_data_handle_and_buffer_creation_start));
    CHECK_CUDA(cudaEventCreate(
        &test_API_0_cusparse_data_handle_and_buffer_creation_stop));
    CHECK_CUDA(cudaEventCreate(
        &test_API_stream_cusparse_data_handle_and_buffer_creation_start));
    CHECK_CUDA(cudaEventCreate(
        &test_API_stream_cusparse_data_handle_and_buffer_creation_stop));
  }

  for (int idx = 0; idx < nstreams; idx++) {
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    streams.push_back(stream);
  }
  //--------------------------------------------------------------------------
  // Create Handle
  std::chrono::time_point<std::chrono::system_clock> handle_creation_beg,
      handle_creation_end;
  if (enable_timing) {
    CHECK_CUDA(cudaEventRecord(handle_creation_start, streams.front()));
  }
  if (test_API_on_stream) {
    handle_creation_beg = std::chrono::system_clock::now();

    CHECK_CUDA(cudaEventRecord(test_API_0_handle_creation_start));
    CHECK_CUDA(cudaEventRecord(test_API_stream_handle_creation_start,
                               streams.front()));

    // CHECK_CUDA(cudaDeviceSynchronize());
  }
  for (int idx = 0; idx < nstreams; idx++) {
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle))
    handles.push_back(handle);
  }
  if (test_API_on_stream) {
    // CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(test_API_0_handle_creation_stop));
    CHECK_CUDA(
        cudaEventRecord(test_API_stream_handle_creation_stop, streams.front()));
    handle_creation_end = std::chrono::system_clock::now();
    timing_results.utility_timestamps["[DEBUG]API_0_handle_creation"] =
        std::make_tuple(test_API_0_handle_creation_start,
                        test_API_0_handle_creation_stop);
    timing_results.utility_timestamps["[DEBUG]API_stream_handle_creation"] =
        std::make_tuple(test_API_stream_handle_creation_start,
                        test_API_stream_handle_creation_stop);
    printf(
        "[DEBUG] cusparseSpMM+CSR handle creation chrono time "
        "(microseconds): %ld\n",
        std::chrono::duration_cast<std::chrono::microseconds>(
            handle_creation_end - handle_creation_beg)
            .count());
  }
  if (enable_timing) {
    CHECK_CUDA(cudaEventRecord(handle_creation_stop, streams.front()));
    timing_results.utility_timestamps["handle_creation"] =
        std::make_tuple(handle_creation_start, handle_creation_stop);
  }
  for (int idx = 0; idx < nstreams; idx++) {
    CHECK_CUSPARSE(cusparseSetStream(handles[idx], streams[idx]));
  }

  // Device memory management
  if (enable_timing) {
    CHECK_CUDA(cudaEventRecord(data_copy_start, streams.front()));
  }
  CHECK_CUDA(cudaMalloc((void **)&dB, B_size * sizeof(float)))
  CHECK_CUDA(cudaMalloc((void **)&dC, C_size * (1 + 1) * sizeof(float)))

  CHECK_CUDA(cudaMemcpy(dB, hB, B_size * sizeof(float), cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemset(dC, 0, C_size * (1 + 1) * sizeof(float)))
  if (enable_timing) {
    CHECK_CUDA(cudaEventRecord(data_copy_stop, streams.front()));
    timing_results.utility_timestamps["data_copy"] =
        std::make_tuple(data_copy_start, data_copy_stop);
  }
  //--------------------------------------------------------------------------
  // CUSPARSE APIs

  // Create the RuntimeData struct so that the cusparse handle creation could
  // get pointers to the matrix data
  runtime_data->A_nnz = A_nnz;
  runtime_data->AA_nnz = AA_nnz;
  runtime_data->B_num_rows = B_num_rows;
  runtime_data->ldb = ldb;
  runtime_data->ldc = ldc;
  runtime_data->B_size = B_size;
  runtime_data->C_size = C_size;
  runtime_data->alpha = alpha;
  runtime_data->beta = beta;
  runtime_data->hB = hB;
  runtime_data->dB = dB;
  runtime_data->dC = dC;
  runtime_data->handles = handles;
  //  .matAA empty vector,
  //  .matBB empty vector,
  //  .matCC empty vector,
  //  .dBuffers empty vector,
  //  .bufferSizes empty vector,
  runtime_data->hA = hA;
  runtime_data->hAA = hAA;
  runtime_data->dAA = dAA;
  runtime_data->streams = streams;

  std::chrono::time_point<std::chrono::system_clock>
      data_handle_and_buffer_creation_beg, data_handle_and_buffer_creation_end;

  if (enable_timing) {
    CHECK_CUDA(cudaEventRecord(cusparse_data_handle_and_buffer_creation_start,
                               streams.front()));
  }
  if (test_API_on_stream) {
    data_handle_and_buffer_creation_beg = std::chrono::system_clock::now();

    CHECK_CUDA(cudaEventRecord(
        test_API_0_cusparse_data_handle_and_buffer_creation_start));
    CHECK_CUDA(cudaEventRecord(
        test_API_stream_cusparse_data_handle_and_buffer_creation_start,
        streams.front()));
    // CHECK_CUDA(cudaDeviceSynchronize());
  }

  for (int BB_col_idx = 0; BB_col_idx < B_num_cols / BB_num_cols;
       BB_col_idx++) {
    for (int AA_col_idx = 0; AA_col_idx < A_num_cols / AA_num_cols;
         AA_col_idx++) {
      for (int AA_row_idx = 0; AA_row_idx < A_num_rows / AA_num_rows;
           AA_row_idx++) {
        if (BB_col_idx == 0) {
          // Create sparse matrix A in CSR format
          cusparseSpMatDescr_t curr_matAA;
          int curr_AA_nnz =
              runtime_data
                  ->dAA[AA_row_idx + AA_col_idx * A_num_rows / AA_num_rows]
                  .num_entries;
          CHECK_CUSPARSE(cusparseCreateCsr(
              &curr_matAA, AA_num_rows, AA_num_cols, curr_AA_nnz,
              // dA_csrOffsets, dA_columns, dA_values,
              (void *)thrust::raw_pointer_cast(
                  runtime_data
                      ->dAA[AA_row_idx + AA_col_idx * A_num_rows / AA_num_rows]
                      .row_offsets.data()),
              (void *)thrust::raw_pointer_cast(
                  runtime_data
                      ->dAA[AA_row_idx + AA_col_idx * A_num_rows / AA_num_rows]
                      .column_indices.data()),
              (void *)thrust::raw_pointer_cast(
                  runtime_data
                      ->dAA[AA_row_idx + AA_col_idx * A_num_rows / AA_num_rows]
                      .values.data()),
              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
              CUDA_R_32F))
          runtime_data->matAA.push_back(curr_matAA);
        }
        if (AA_row_idx == 0) {
          // Create dense matrix B
          cusparseDnMatDescr_t curr_matBB;
          CHECK_CUSPARSE(cusparseCreateDnMat(
              &curr_matBB, AA_num_cols, BB_num_cols, ldb,
              dB + /*row*/ (AA_col_idx * AA_num_cols) +
                  /*col*/ (BB_col_idx * BB_num_cols) * A_num_cols,
              CUDA_R_32F, CUSPARSE_ORDER_COL))
          runtime_data->matBB.push_back(curr_matBB);
        }
        if (AA_col_idx == 0) {
          // Create dense matrix C
          cusparseDnMatDescr_t curr_matCC;
          CHECK_CUSPARSE(
              cusparseCreateDnMat(&curr_matCC, AA_num_rows, BB_num_cols, ldc,
                                  dC + AA_row_idx * AA_num_rows +
                                      BB_col_idx * BB_num_cols * AA_num_rows,
                                  CUDA_R_32F, CUSPARSE_ORDER_COL))
          runtime_data->matCC.push_back(curr_matCC);
        }
      }
    }
  }
  for (int BB_col_idx = 0; BB_col_idx < B_num_cols / BB_num_cols;
       BB_col_idx++) {
    for (int AA_col_idx = 0; AA_col_idx < A_num_cols / AA_num_cols;
         AA_col_idx++) {
      for (int AA_row_idx = 0; AA_row_idx < A_num_rows / AA_num_rows;
           AA_row_idx++) {
        int idx_AA = AA_row_idx + AA_col_idx * A_num_rows / AA_num_rows;
        int idx_BB = AA_col_idx + BB_col_idx * A_num_cols / AA_num_cols;
        int idx_CC = AA_row_idx + BB_col_idx * A_num_rows / AA_num_rows;
        size_t curr_bufferSize;
        void *curr_dBuffer;
        int idx_stream =
            getCurrStream({BB_col_idx, AA_col_idx, AA_row_idx},
                          {B_num_cols / BB_num_cols, A_num_cols / AA_num_cols,
                           A_num_rows / AA_num_rows},
                          nstreams);
        // Allocate an external buffer if needed
        CHECK_CUSPARSE(cusparseSpMM_bufferSize(
            handles[idx_stream], CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &(alpha),
            runtime_data->matAA[idx_AA], runtime_data->matBB[idx_BB], &(beta),
            runtime_data->matCC[idx_CC], CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
            &curr_bufferSize))
        // TODO: switch to memcpy async
        CHECK_CUDA(cudaMalloc(&curr_dBuffer, curr_bufferSize))
        runtime_data->dBuffers.push_back(curr_dBuffer);
        runtime_data->bufferSizes.push_back(curr_bufferSize);
      }
    }
  }
  if (test_API_on_stream) {
    // CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(
        test_API_0_cusparse_data_handle_and_buffer_creation_stop));
    CHECK_CUDA(cudaEventRecord(
        test_API_stream_cusparse_data_handle_and_buffer_creation_stop,
        streams.front()));

    timing_results
        .utility_timestamps["[DEBUG]API_0_data_handle_and_buffer_creation"] =
        std::make_tuple(
            test_API_0_cusparse_data_handle_and_buffer_creation_start,
            test_API_0_cusparse_data_handle_and_buffer_creation_stop);
    timing_results.utility_timestamps
        ["[DEBUG]API_stream_data_handle_and_buffer_creation"] = std::make_tuple(
        test_API_stream_cusparse_data_handle_and_buffer_creation_start,
        test_API_stream_cusparse_data_handle_and_buffer_creation_stop);
    data_handle_and_buffer_creation_end = std::chrono::system_clock::now();
    printf(
        "[DEBUG] cusparseSpMM+CSR data handle and buffer creation chrono "
        "time "
        "(microseconds): %ld\n",
        std::chrono::duration_cast<std::chrono::microseconds>(
            data_handle_and_buffer_creation_end -
            data_handle_and_buffer_creation_beg)
            .count());
  }
  if (enable_timing) {
    CHECK_CUDA(cudaEventRecord(cusparse_data_handle_and_buffer_creation_stop,
                               streams.front()));
    timing_results
        .utility_timestamps["cusparse_data_handle_and_buffer_creation"] =
        std::make_tuple(cusparse_data_handle_and_buffer_creation_start,
                        cusparse_data_handle_and_buffer_creation_stop);
  }

  ProblemSpec problem_spec{
      .A_num_rows = A_num_rows,
      .A_num_cols = A_num_cols,
      .B_num_cols = B_num_cols,
      .AA_num_rows = AA_num_rows,
      .AA_num_cols = AA_num_cols,
      .BB_num_cols = BB_num_cols,
      .A_sparsity = A_sparsity,
      .enable_dump = enable_dump,
      .enable_timing = enable_timing,
      .enable_per_stream_timing = enable_per_stream_timing,
      .enable_debug_timing = enable_debug_timing,
      .enable_graph = enable_graph,
      .enable_preprocess = enable_preprocess,
      .cli_result_path_and_prefix = cli_result_path_and_prefix,
      .flag_specify_result_path_and_prefix =
          flag_specify_result_path_and_prefix,
      .test_API_on_stream = test_API_on_stream,
      .nstreams = nstreams};
  printf("dAA[0].values %p\n", runtime_data->dAA[0].values.data());
  auto bench_tuple = std::make_tuple(problem_spec, runtime_data);
  return bench_tuple;
}

std::tuple<ProblemSpec, std::shared_ptr<RuntimeData>> generate_data_and_prepare(
    std::vector<std::string> &args, TimingResults &timing_results) {
  std::vector<const char *> args_cstr;
  for (int idx = 0; idx < args.size(); idx++) {
    args_cstr.push_back(args[idx].c_str());
  }
  return generate_data_and_prepare(args_cstr.size(), args_cstr.data(),
                                   timing_results);
}

void preprocess(ProblemSpec &problem_spec, RuntimeData &runtime_data) {
  for (int BB_col_idx = 0;
       BB_col_idx < problem_spec.B_num_cols / problem_spec.BB_num_cols;
       BB_col_idx++) {
    for (int AA_col_idx = 0;
         AA_col_idx < problem_spec.A_num_cols / problem_spec.AA_num_cols;
         AA_col_idx++) {
      for (int AA_row_idx = 0;
           AA_row_idx < problem_spec.A_num_rows / problem_spec.AA_num_rows;
           AA_row_idx++) {
        auto [idx_spmm, num_spmms] = canonicalize_loop_index_and_bound(
            {BB_col_idx, AA_col_idx, AA_row_idx},
            {problem_spec.B_num_cols / problem_spec.BB_num_cols,
             problem_spec.A_num_cols / problem_spec.AA_num_cols,
             problem_spec.A_num_rows / problem_spec.AA_num_rows});
        int idx_stream =
            getCurrStream({BB_col_idx, AA_col_idx, AA_row_idx},
                          {problem_spec.B_num_cols / problem_spec.BB_num_cols,
                           problem_spec.A_num_cols / problem_spec.AA_num_cols,
                           problem_spec.A_num_rows / problem_spec.AA_num_rows},
                          problem_spec.nstreams);
        int idx_AA = AA_row_idx + AA_col_idx * problem_spec.A_num_rows /
                                      problem_spec.AA_num_rows;
        int idx_BB = AA_col_idx + BB_col_idx * problem_spec.A_num_cols /
                                      problem_spec.AA_num_cols;
        int idx_CC = AA_row_idx + BB_col_idx * problem_spec.A_num_rows /
                                      problem_spec.AA_num_rows;

        CHECK_CUSPARSE(cusparseSpMM_preprocess(
            runtime_data.handles[idx_stream], CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &(runtime_data.alpha),
            runtime_data.matAA[idx_AA], runtime_data.matBB[idx_BB],
            &(runtime_data.beta), runtime_data.matCC[idx_CC], CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, runtime_data.dBuffers[idx_spmm]))

        // idx_spmm++;
      }
    }
  }
}

// This function is written before GraphConstructor is implemented. It is no
// longer updated but stay here to 1) provide a reference implementation to
// facilitate debugging.
void _compute_reference(ProblemSpec &problem_spec, RuntimeData &runtime_data,
                        TimingResults &timing_results) {
  // Execute SpMM
  // We nest the cuda event timing with std::chrono to make sure the cuda
  // event is getting correct results, we will use the cuda event timing
  // results and ignore the std::chrono results
  std::chrono::time_point<std::chrono::system_clock> beg, end;
  // Start and stop when
  // wait_streams_on_first_and_report_that_as_elapsed_time

  printf("dAA[0].values %p\n", runtime_data.dAA[0].values.data());

  cudaEvent_t start, stop;
  std::vector<cudaEvent_t> starts_per_stream, stops_per_stream;
  if (wait_streams_on_first_and_report_that_as_elapsed_time(
          problem_spec.enable_per_stream_timing, problem_spec.enable_timing,
          problem_spec.nstreams)) {
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
  }
  // We need stop event per stream to synchronize before reduction no matter
  // timing is enabled or not
  for (int idx = 0; idx < problem_spec.nstreams; idx++) {
    cudaEvent_t stop_per_stream;
    CHECK_CUDA(cudaEventCreate(&stop_per_stream));
    stops_per_stream.push_back(stop_per_stream);
  }

  if (problem_spec.enable_timing) {
    // We need stop event per stream to synchronize before reduction no matter
    // timing is enabled or not
    for (int idx = 0; idx < problem_spec.nstreams; idx++) {
      cudaEvent_t start_per_stream;
      CHECK_CUDA(cudaEventCreate(&start_per_stream));
      starts_per_stream.push_back(start_per_stream);
    }
  }

  if (problem_spec.enable_debug_timing) {
    for (int idx = 0; idx < problem_spec.nstreams; idx++)
      CHECK_CUDA(cudaStreamSynchronize(runtime_data.streams[idx]));
    CHECK_CUDA(cudaDeviceSynchronize());
    beg = std::chrono::system_clock::now();
  }

  if (wait_streams_on_first_and_report_that_as_elapsed_time(
          problem_spec.enable_per_stream_timing, problem_spec.enable_timing,
          problem_spec.nstreams)) {
    CHECK_CUDA(cudaEventRecord(start, runtime_data.streams.front()));
  }

  if (problem_spec.enable_timing) {
    for (int idx = 0; idx < problem_spec.nstreams; idx++) {
      if (wait_streams_on_first_and_report_that_as_elapsed_time(
              problem_spec.enable_per_stream_timing, problem_spec.enable_timing,
              problem_spec.nstreams)) {
        CHECK_CUDA(cudaStreamWaitEvent(runtime_data.streams[idx], start));
      }

      CHECK_CUDA(
          cudaEventRecord(starts_per_stream[idx], runtime_data.streams[idx]));
    }
  }

  // TODO: skip if the sparse matrix size is 0
  for (int BB_col_idx = 0;
       BB_col_idx < problem_spec.B_num_cols / problem_spec.BB_num_cols;
       BB_col_idx++) {
    for (int AA_col_idx = 0;
         AA_col_idx < problem_spec.A_num_cols / problem_spec.AA_num_cols;
         AA_col_idx++) {
      for (int AA_row_idx = 0;
           AA_row_idx < problem_spec.A_num_rows / problem_spec.AA_num_rows;
           AA_row_idx++) {
        auto [idx_spmm, num_spmms] = canonicalize_loop_index_and_bound(
            {BB_col_idx, AA_col_idx, AA_row_idx},
            {problem_spec.B_num_cols / problem_spec.BB_num_cols,
             problem_spec.A_num_cols / problem_spec.AA_num_cols,
             problem_spec.A_num_rows / problem_spec.AA_num_rows});
        int idx_stream =
            getCurrStream({BB_col_idx, AA_col_idx, AA_row_idx},
                          {problem_spec.B_num_cols / problem_spec.BB_num_cols,
                           problem_spec.A_num_cols / problem_spec.AA_num_cols,
                           problem_spec.A_num_rows / problem_spec.AA_num_rows},
                          problem_spec.nstreams);
        int idx_AA = AA_row_idx + AA_col_idx * problem_spec.A_num_rows /
                                      problem_spec.AA_num_rows;
        int idx_BB = AA_col_idx + BB_col_idx * problem_spec.A_num_cols /
                                      problem_spec.AA_num_cols;
        int idx_CC = AA_row_idx + BB_col_idx * problem_spec.A_num_rows /
                                      problem_spec.AA_num_rows;

        CHECK_CUSPARSE(cusparseSpMM(
            runtime_data.handles[idx_stream], CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &(runtime_data.alpha),
            runtime_data.matAA[idx_AA], runtime_data.matBB[idx_BB],
            &(runtime_data.beta), runtime_data.matCC[idx_CC], CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, runtime_data.dBuffers[idx_spmm]))

        // idx_spmm++;
      }
    }
  }

  // Stream idx 0 waits for all other streams to finish before executing the
  // reduction kernel
  for (int idx = 1; idx < problem_spec.nstreams; idx++) {
    CHECK_CUDA(
        cudaEventRecord(stops_per_stream[idx], runtime_data.streams[idx]));
    CHECK_CUDA(cudaStreamWaitEvent(runtime_data.streams.front(),
                                   stops_per_stream[idx]));
  }

  // Accumulate the result
  // TODO: define BLOCK_SIZE and SHMEM_SIZE
  constexpr int BLOCK_SIZE = 256;
  constexpr int SHMEM_SIZE = 256;
  assert(problem_spec.A_num_rows * problem_spec.B_num_cols % SHMEM_SIZE == 0);
  dim3 nblocks(problem_spec.A_num_rows * problem_spec.B_num_cols / SHMEM_SIZE,
               problem_spec.A_num_cols / problem_spec.AA_num_cols, 1);
  dim3 nthreads(BLOCK_SIZE, 1, 1);
  reduce_segments<BLOCK_SIZE, SHMEM_SIZE, float>
      <<<nblocks, nthreads, 0, runtime_data.streams.front()>>>(
          runtime_data.dC,
          runtime_data.dC + problem_spec.A_num_cols / problem_spec.AA_num_cols *
                                problem_spec.AA_num_rows *
                                problem_spec.BB_num_cols,
          problem_spec.AA_num_rows, problem_spec.AA_num_rows,
          problem_spec.BB_num_cols,
          problem_spec.A_num_cols / problem_spec.AA_num_cols);

  if (problem_spec.enable_timing) {
    CHECK_CUDA(cudaEventRecord(stops_per_stream[0], runtime_data.streams[0]));
  }

  if (problem_spec.enable_timing)
    CHECK_CUDA(cudaEventRecord(stops_per_stream.front(),
                               runtime_data.streams.front()));
  if (wait_streams_on_first_and_report_that_as_elapsed_time(
          problem_spec.enable_per_stream_timing, problem_spec.enable_timing,
          problem_spec.nstreams)) {
    for (int idx = 0; idx < problem_spec.nstreams; idx++) {
      if (wait_streams_on_first_and_report_that_as_elapsed_time(
              problem_spec.enable_per_stream_timing, problem_spec.enable_timing,
              problem_spec.nstreams)) {
        CHECK_CUDA(cudaStreamWaitEvent(runtime_data.streams.front(),
                                       stops_per_stream[idx]));
      }
      CHECK_CUDA(cudaEventRecord(stop, runtime_data.streams.front()));
    }
  }

  if (problem_spec.enable_debug_timing) {
    for (int idx = 0; idx < problem_spec.nstreams; idx++)
      CHECK_CUDA(cudaStreamSynchronize(runtime_data.streams[idx]));
    CHECK_CUDA(cudaDeviceSynchronize());
    end = std::chrono::system_clock::now();
    printf(
        "[DEBUG] cusparseSpMM+CSR+Partitioned chrono time (microseconds): "
        "%ld\n",
        std::chrono::duration_cast<std::chrono::microseconds>(end - beg)
            .count());
  }

  // Add start, stop pair in each stream to the return value
  if (report_elapsed_time_per_stream(problem_spec.enable_per_stream_timing,
                                     problem_spec.enable_timing,
                                     problem_spec.nstreams)) {
    timing_results.start_events = starts_per_stream;
    timing_results.stop_events = stops_per_stream;
    return;
  }

  // Synchronize on the first stream in streams vector, and add the start, stop
  // pair to the return value
  if (wait_streams_on_first_and_report_that_as_elapsed_time(
          problem_spec.enable_per_stream_timing, problem_spec.enable_timing,
          problem_spec.nstreams)) {
    timing_results.start_events = std::vector<cudaEvent_t>({start});
    timing_results.stop_events = std::vector<cudaEvent_t>({stop});
    return;
  }

  // No timing should be reported
  for (int idx = 0; idx < problem_spec.nstreams; idx++) {
    CHECK_CUDA(cudaEventDestroy(stops_per_stream[idx]));
  }
  timing_results.start_events = std::vector<cudaEvent_t>();
  timing_results.stop_events = std::vector<cudaEvent_t>();
  return;
}

void compute(ProblemSpec &problem_spec, RuntimeData &runtime_data,
             TimingResults &timing_results,
             AbstractCUDAGraphConstructor<cudaStream_t> &graph_constructor) {
  // Execute SpMM
  // We nest the cuda event timing with std::chrono to make sure the cuda
  // event is getting correct results, we will use the cuda event timing
  // results and ignore the std::chrono results
  std::chrono::time_point<std::chrono::system_clock> beg, end;
  // Start and stop when
  // wait_streams_on_first_and_report_that_as_elapsed_time

  printf("dAA[0].values %p\n", runtime_data.dAA[0].values.data());

  cudaEvent_t start, stop;
  std::vector<cudaEvent_t> starts_per_stream, stops_per_stream;
  if (wait_streams_on_first_and_report_that_as_elapsed_time(
          problem_spec.enable_per_stream_timing, problem_spec.enable_timing,
          problem_spec.nstreams)) {
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
  }
  // We need stop event per stream to synchronize before reduction no matter
  // timing is enabled or not
  for (int idx = 0; idx < problem_spec.nstreams; idx++) {
    cudaEvent_t stop_per_stream;
    CHECK_CUDA(cudaEventCreate(&stop_per_stream));
    stops_per_stream.push_back(stop_per_stream);
  }

  if (problem_spec.enable_timing) {
    // We need stop event per stream to synchronize before reduction no matter
    // timing is enabled or not
    for (int idx = 0; idx < problem_spec.nstreams; idx++) {
      cudaEvent_t start_per_stream;
      CHECK_CUDA(cudaEventCreate(&start_per_stream));
      starts_per_stream.push_back(start_per_stream);
    }
  }

  if (problem_spec.enable_debug_timing) {
    for (int idx = 0; idx < problem_spec.nstreams; idx++)
      CHECK_CUDA(cudaStreamSynchronize(runtime_data.streams[idx]));
    CHECK_CUDA(cudaDeviceSynchronize());
    beg = std::chrono::system_clock::now();
  }

  if (wait_streams_on_first_and_report_that_as_elapsed_time(
          problem_spec.enable_per_stream_timing, problem_spec.enable_timing,
          problem_spec.nstreams)) {
    // Equivalent to error-checked cudaEventRecord
    graph_constructor.addEventRecordNode(start, runtime_data.streams.front());
  }

  if (problem_spec.enable_timing) {
    for (int idx = 0; idx < problem_spec.nstreams; idx++) {
      if (wait_streams_on_first_and_report_that_as_elapsed_time(
              problem_spec.enable_per_stream_timing, problem_spec.enable_timing,
              problem_spec.nstreams)) {
        // Equivalent to error-checked cudaStreamWaitEvent
        graph_constructor.addStreamWaitEventNode(runtime_data.streams[idx],
                                                 start);
      }

      // Equivalent to error-checked cudaEventRecord
      graph_constructor.addEventRecordNode(starts_per_stream[idx],
                                           runtime_data.streams[idx]);
    }
  }

  graph_constructor.notifyBeforeInvokingLibraryCall(runtime_data.streams[0]);
  // TODO: skip if the sparse matrix size is 0
  for (int BB_col_idx = 0;
       BB_col_idx < problem_spec.B_num_cols / problem_spec.BB_num_cols;
       BB_col_idx++) {
    for (int AA_col_idx = 0;
         AA_col_idx < problem_spec.A_num_cols / problem_spec.AA_num_cols;
         AA_col_idx++) {
      for (int AA_row_idx = 0;
           AA_row_idx < problem_spec.A_num_rows / problem_spec.AA_num_rows;
           AA_row_idx++) {
        auto [idx_spmm, num_spmms] = canonicalize_loop_index_and_bound(
            {BB_col_idx, AA_col_idx, AA_row_idx},
            {problem_spec.B_num_cols / problem_spec.BB_num_cols,
             problem_spec.A_num_cols / problem_spec.AA_num_cols,
             problem_spec.A_num_rows / problem_spec.AA_num_rows});
        int curr_stream_idx =
            getCurrStream({BB_col_idx, AA_col_idx, AA_row_idx},
                          {problem_spec.B_num_cols / problem_spec.BB_num_cols,
                           problem_spec.A_num_cols / problem_spec.AA_num_cols,
                           problem_spec.A_num_rows / problem_spec.AA_num_rows},
                          problem_spec.nstreams);
        std::vector<int> last_loop_idxes = get_last_loop_index(
            {BB_col_idx, AA_col_idx, AA_row_idx},
            {problem_spec.B_num_cols / problem_spec.BB_num_cols,
             problem_spec.A_num_cols / problem_spec.AA_num_cols,
             problem_spec.A_num_rows / problem_spec.AA_num_rows});
        int last_stream_idx =
            getCurrStream(last_loop_idxes,
                          {problem_spec.B_num_cols / problem_spec.BB_num_cols,
                           problem_spec.A_num_cols / problem_spec.AA_num_cols,
                           problem_spec.A_num_rows / problem_spec.AA_num_rows},
                          runtime_data.streams.size());
        if (BB_col_idx + AA_col_idx + AA_row_idx != 0 &&
            curr_stream_idx != last_stream_idx) {
          graph_constructor.notifyAfterInvokingLibraryCall(
              runtime_data.streams[last_stream_idx]);
          graph_constructor.notifyBeforeInvokingLibraryCall(
              runtime_data.streams[curr_stream_idx]);
        }
        int idx_AA = AA_row_idx + AA_col_idx * problem_spec.A_num_rows /
                                      problem_spec.AA_num_rows;
        int idx_BB = AA_col_idx + BB_col_idx * problem_spec.A_num_cols /
                                      problem_spec.AA_num_cols;
        int idx_CC = AA_row_idx + BB_col_idx * problem_spec.A_num_rows /
                                      problem_spec.AA_num_rows;

        CHECK_CUSPARSE(cusparseSpMM(
            runtime_data.handles[curr_stream_idx],
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &(runtime_data.alpha), runtime_data.matAA[idx_AA],
            runtime_data.matBB[idx_BB], &(runtime_data.beta),
            runtime_data.matCC[idx_CC], CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
            runtime_data.dBuffers[idx_spmm]))

        // idx_spmm++;
      }
    }
  }
  graph_constructor.notifyAfterInvokingLibraryCall(
      runtime_data.streams[problem_spec.nstreams - 1]);

  // Stream idx 0 waits for all other streams to finish before executing the
  // reduction kernel
  for (int idx = 1; idx < problem_spec.nstreams; idx++) {
    // Equivalent to error-checked cudaEventRecord
    graph_constructor.addEventRecordNode(stops_per_stream[idx],
                                         runtime_data.streams[idx]);
    // Equivalent to error-checked cudaStreamWaitEvent
    graph_constructor.addStreamWaitEventNode(runtime_data.streams[0],
                                             stops_per_stream[idx]);
  }

  graph_constructor.notifyBeforeInvokingLibraryCall(runtime_data.streams[0]);
  // Accumulate the result
  // TODO: define BLOCK_SIZE and SHMEM_SIZE
  constexpr int BLOCK_SIZE = 256;
  constexpr int SHMEM_SIZE = 256;
  assert(problem_spec.A_num_rows * problem_spec.B_num_cols % SHMEM_SIZE == 0);
  dim3 nblocks(problem_spec.A_num_rows * problem_spec.B_num_cols / SHMEM_SIZE,
               problem_spec.A_num_cols / problem_spec.AA_num_cols, 1);
  dim3 nthreads(BLOCK_SIZE, 1, 1);
  reduce_segments<BLOCK_SIZE, SHMEM_SIZE, float>
      <<<nblocks, nthreads, 0, runtime_data.streams.front()>>>(
          runtime_data.dC,
          runtime_data.dC + problem_spec.A_num_cols / problem_spec.AA_num_cols *
                                problem_spec.AA_num_rows *
                                problem_spec.BB_num_cols,
          problem_spec.AA_num_rows, problem_spec.AA_num_rows,
          problem_spec.BB_num_cols,
          problem_spec.A_num_cols / problem_spec.AA_num_cols);
  graph_constructor.notifyAfterInvokingLibraryCall(runtime_data.streams[0]);

  if (problem_spec.enable_timing) {
    // Equivalent to error-checked cudaEventRecord
    graph_constructor.addEventRecordNode(stops_per_stream[0],
                                         runtime_data.streams[0]);
  }

  if (problem_spec.enable_timing) {
    // Equivalent to error-checked cudaEventRecord
    graph_constructor.addEventRecordNode(stops_per_stream.front(),
                                         runtime_data.streams.front());
  }
  if (wait_streams_on_first_and_report_that_as_elapsed_time(
          problem_spec.enable_per_stream_timing, problem_spec.enable_timing,
          problem_spec.nstreams)) {
    for (int idx = 0; idx < problem_spec.nstreams; idx++) {
      if (wait_streams_on_first_and_report_that_as_elapsed_time(
              problem_spec.enable_per_stream_timing, problem_spec.enable_timing,
              problem_spec.nstreams)) {
        // Equivalent to error-checked cudaStreamWaitEvent
        graph_constructor.addStreamWaitEventNode(runtime_data.streams[0],
                                                 stops_per_stream[idx]);
      }
      // Equivalent to error-checked cudaEventRecord
      graph_constructor.addEventRecordNode(stop, runtime_data.streams[0]);
    }
  }

  if (problem_spec.enable_debug_timing) {
    for (int idx = 0; idx < problem_spec.nstreams; idx++)
      CHECK_CUDA(cudaStreamSynchronize(runtime_data.streams[idx]));
    CHECK_CUDA(cudaDeviceSynchronize());
    end = std::chrono::system_clock::now();
    printf(
        "[DEBUG] cusparseSpMM+CSR+Partitioned chrono time (microseconds): "
        "%ld\n",
        std::chrono::duration_cast<std::chrono::microseconds>(end - beg)
            .count());
  }

  // Add start, stop pair in each stream to the return value
  if (report_elapsed_time_per_stream(problem_spec.enable_per_stream_timing,
                                     problem_spec.enable_timing,
                                     problem_spec.nstreams)) {
    timing_results.start_events = starts_per_stream;
    timing_results.stop_events = stops_per_stream;
    return;
  }

  // Synchronize on the first stream in streams vector, and add the start, stop
  // pair to the return value
  if (wait_streams_on_first_and_report_that_as_elapsed_time(
          problem_spec.enable_per_stream_timing, problem_spec.enable_timing,
          problem_spec.nstreams)) {
    timing_results.start_events = std::vector<cudaEvent_t>({start});
    timing_results.stop_events = std::vector<cudaEvent_t>({stop});
    return;
  }

  // No timing should be reported
  for (int idx = 0; idx < problem_spec.nstreams; idx++) {
    CHECK_CUDA(cudaEventDestroy(stops_per_stream[idx]));
  }
  timing_results.start_events = std::vector<cudaEvent_t>();
  timing_results.stop_events = std::vector<cudaEvent_t>();
  return;
}

void consume_and_print_timing(ProblemSpec &problem_spec,
                              RuntimeData &runtime_data,
                              TimingResults &timing_results) {
  if (wait_streams_on_first_and_report_that_as_elapsed_time(
          problem_spec.enable_per_stream_timing, problem_spec.enable_timing,
          problem_spec.nstreams)) {
    cudaEvent_t start = timing_results.start_events.front();
    cudaEvent_t stop = timing_results.stop_events.front();
    CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsed_time = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("cusparseSpMM+CSR+Partitioned elapsed time (ms): %f\n",
           elapsed_time);
    printf("cusparseSpMM+CSR+Partitioned throughput (GFLOPS): %f\n",
           (2.0 * runtime_data.A_nnz * problem_spec.B_num_cols) /
               (elapsed_time / 1000.0) / 1e9);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
  }
  if (report_elapsed_time_per_stream(problem_spec.enable_per_stream_timing,
                                     problem_spec.enable_timing,
                                     problem_spec.nstreams)) {
    for (int idx = 0; idx < problem_spec.nstreams; idx++) {
      // Skip the first stream if it is already synchronized and destroyed
      if (idx == 0 && wait_streams_on_first_and_report_that_as_elapsed_time(
                          problem_spec.enable_per_stream_timing,
                          problem_spec.enable_timing, problem_spec.nstreams))
        continue;
      CHECK_CUDA(cudaEventSynchronize(timing_results.stop_events[idx]));
    }

    for (int idx = 0; idx < problem_spec.nstreams; idx++) {
      float elapsed_time = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&elapsed_time,
                                      timing_results.start_events[idx],
                                      timing_results.stop_events[idx]));
      printf(
          "cusparseSpMM+CSR+Partitioned elapsed time(streamIdx%d) (ms): %f\n",
          idx, elapsed_time);
      // TODO: enable throughput print
      CHECK_CUDA(cudaEventDestroy(timing_results.start_events[idx]));
      CHECK_CUDA(cudaEventDestroy(timing_results.stop_events[idx]));
    }
  }

  // Print elapsed time of utilities. Keyword "elapsed time(util) (ms):"
  for (const auto &keyval : timing_results.utility_timestamps) {
    const auto &key = keyval.first;
    const auto &value = keyval.second;
    float elapsed_time_util = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time_util, std::get<0>(value),
                                    std::get<1>(value)));
    printf("cusparseSpMM+CSR+Partitioned %s elapsed time(util) (ms): %f\n",
           key.c_str(), elapsed_time_util);
    CHECK_CUDA(cudaEventDestroy(std::get<0>(value)));
    CHECK_CUDA(cudaEventDestroy(std::get<1>(value)));
  }
}

void cleanUp(ProblemSpec &problem_spec, RuntimeData &runtime_data) {
  // Destroy matrix/vector descriptors
  int idx_spmm = 0;
  for (int BB_col_idx = 0;
       BB_col_idx < problem_spec.B_num_cols / problem_spec.BB_num_cols;
       BB_col_idx++) {
    for (int AA_col_idx = 0;
         AA_col_idx < problem_spec.A_num_cols / problem_spec.AA_num_cols;
         AA_col_idx++) {
      for (int AA_row_idx = 0;
           AA_row_idx < problem_spec.A_num_rows / problem_spec.AA_num_rows;
           AA_row_idx++) {
        int idx_AA = AA_row_idx + AA_col_idx * problem_spec.A_num_rows /
                                      problem_spec.AA_num_rows;
        int idx_BB = AA_col_idx + BB_col_idx * problem_spec.A_num_cols /
                                      problem_spec.AA_num_cols;
        int idx_CC = AA_row_idx + BB_col_idx * problem_spec.A_num_rows /
                                      problem_spec.AA_num_rows;
        if (BB_col_idx == 0) {
          // Destroy sparse matrix A in CSR format
          CHECK_CUSPARSE(cusparseDestroySpMat((runtime_data.matAA[idx_AA])))
        }

        if (AA_row_idx == 0) {
          // Destroy dense matrix B
          CHECK_CUSPARSE(cusparseDestroyDnMat((runtime_data.matBB[idx_BB])))
        }
        if (AA_col_idx == 0) {
          // Destroy dense matrix C
          CHECK_CUSPARSE(cusparseDestroyDnMat((runtime_data.matCC[idx_CC])))
        }
        // Destroy the external buffer
        CHECK_CUDA(cudaFree((runtime_data.dBuffers[idx_spmm])))
        idx_spmm++;
      }
    }
  }

  if (problem_spec.enable_dump) {
    float *hC;
    hC = (float *)malloc(sizeof(float) * runtime_data.C_size);
    CHECK_CUDA(cudaMemcpy(hC, runtime_data.dC,
                          runtime_data.C_size * sizeof(float),
                          cudaMemcpyDeviceToHost))
    // Get current timestamp
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    char time_str[64];
    std::strftime(time_str, sizeof(time_str), "%Y-%m-%d-%H-%M", &tm);
    // We should store the string in a std::string because when the .c_str()
    // pointer is referenced, the std::string object should not be destroyed
    std::string result_path_and_prefix;
    if (problem_spec.flag_specify_result_path_and_prefix) {
      result_path_and_prefix = problem_spec.cli_result_path_and_prefix;
    } else {
      result_path_and_prefix =
          std::string("cusparse_bench_spmm_csr_partitioned.") + time_str;
    }
    // Store m, n, k to a txt and store A, B, C to a numpy file
    FILE *fp = fopen((result_path_and_prefix + ".txt").c_str(), "w");
    assert(fp != nullptr);
    fprintf(fp, "%d %d %d %d %f\n", problem_spec.A_num_rows,
            problem_spec.A_num_cols, problem_spec.B_num_cols,
            runtime_data.A_nnz, problem_spec.A_sparsity);
    fclose(fp);
    cusp::io::write_matrix_market_file(runtime_data.hA,
                                       result_path_and_prefix + ".A.mtx");

    unsigned long b_shape[2] = {runtime_data.ldb, problem_spec.B_num_cols};
    unsigned long c_shape[2] = {runtime_data.ldc, problem_spec.B_num_cols};
    npy::SaveArrayAsNumpy(result_path_and_prefix + ".B.npy", false, 2, b_shape,
                          runtime_data.hB);
    npy::SaveArrayAsNumpy(result_path_and_prefix + ".C.npy", false, 2, c_shape,
                          hC);
    free(hC);
  }

  for (int idx = 0; idx < problem_spec.nstreams; idx++) {
    CHECK_CUSPARSE(cusparseDestroy(runtime_data.handles[idx]))
    CHECK_CUDA(cudaStreamDestroy(runtime_data.streams[idx]))
  }
  // device memory deallocation
  CHECK_CUDA(cudaFree(runtime_data.dB))
  CHECK_CUDA(cudaFree(runtime_data.dC))
  free(runtime_data.hB);
  return;
}

// This function is written before GraphConstructor is implemented. It is no
// longer updated but stay here to 1) provide a reference implementation to
// facilitate debugging.
// When cuda graph is enabled, the original compute stage is now creating
// the graph, and we need a new stage that launches the graph. The rest should
// be kept the same.
void _create_graph_reference(ProblemSpec &problem_spec,
                             RuntimeData &runtime_data,
                             TimingResults &timing_results) {
  std::vector<cudaGraph_t> graphs;
  CHECK_CUDA(cudaStreamBeginCapture(runtime_data.streams[0],
                                    cudaStreamCaptureModeGlobal));

  _compute_reference(problem_spec, runtime_data, timing_results);

  // Only stream idx 0 needs to be captured because other stream waits on the
  // start event of stream idx 0, and stream idx 0 waits on the stop event of
  // other streams Reference:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cross-stream-dependencies-and-events
  cudaGraph_t graph;
  CHECK_CUDA(cudaStreamEndCapture(runtime_data.streams[0], &graph));
  graphs.push_back(graph);
  runtime_data.graphs = graphs;

  return;
}

CUDAGraphConstructor<cudaStream_t> create_graph(ProblemSpec &problem_spec,
                                                RuntimeData &runtime_data,
                                                TimingResults &timing_results) {
  CUDAGraphConstructor<cudaStream_t> graph_constructor;
  for (cudaStream_t stream : runtime_data.streams) {
    graph_constructor.registerStream(stream);
  }

  compute(problem_spec, runtime_data, timing_results, graph_constructor);

  runtime_data.graphs.push_back(graph_constructor.getGraph());

  return graph_constructor;
}

void initiate_graph(RuntimeData &runtime_data) {
  std::vector<cudaGraphExec_t> graphExecs;
  cudaGraphExec_t graphExec = NULL;
  CHECK_CUDA(
      cudaGraphInstantiate(&graphExec, runtime_data.graphs[0], NULL, NULL, 0));
  graphExecs.push_back(graphExec);
  runtime_data.graphExecs = graphExecs;

  return;
}

void launch_graph_and_wait(RuntimeData &runtime_data) {
  CHECK_CUDA(
      cudaGraphLaunch(runtime_data.graphExecs[0], runtime_data.streams[0]));
  CHECK_CUDA(cudaStreamSynchronize(runtime_data.streams[0]));
}

// This function is written before GraphConstructor is implemented. It is no
// longer updated but stay here to 1) provide a reference implementation to
// facilitate debugging.
int _main_reference(const int argc, const char **argv) {
  TimingResults timing_results;
  auto bench_tuple = generate_data_and_prepare(argc, argv, timing_results);
  auto bench_spec = std::get<0>(bench_tuple);
  auto bench_data = std::get<1>(bench_tuple);

  if (bench_spec.enable_preprocess) {
    preprocess(bench_spec, *(bench_data.get()));
  }

  if (bench_spec.enable_graph) {
    std::chrono::time_point<std::chrono::system_clock> graph_creation_beg,
        graph_creation_end, graph_initialization_end, graph_execution_end;
    graph_creation_beg = std::chrono::system_clock::now();
    _create_graph_reference(bench_spec, *(bench_data.get()), timing_results);

    graph_creation_end = std::chrono::system_clock::now();
    initiate_graph(*(bench_data.get()));

    graph_initialization_end = std::chrono::system_clock::now();
    launch_graph_and_wait(*(bench_data.get()));
    graph_execution_end = std::chrono::system_clock::now();
    printf(
        "[DEBUG] cusparseSpMM+CSR+Partitioned graph creation chrono time "
        "(microseconds): "
        "%ld\n",
        std::chrono::duration_cast<std::chrono::microseconds>(
            graph_creation_end - graph_creation_beg)
            .count());
    printf(
        "[DEBUG] cusparseSpMM+CSR+Partitioned graph initialization chrono time "
        "(microseconds): "
        "%ld\n",
        std::chrono::duration_cast<std::chrono::microseconds>(
            graph_initialization_end - graph_creation_end)
            .count());
    printf(
        "[DEBUG] cusparseSpMM+CSR+Partitioned graph execution chrono time "
        "(microseconds): "
        "%ld\n",
        std::chrono::duration_cast<std::chrono::microseconds>(
            graph_execution_end - graph_initialization_end)
            .count());
  } else {
    _compute_reference(bench_spec, *(bench_data.get()), timing_results);
  }
  // When CUDA graph is enabled, we already wait until it finishes. Both
  // synchronize event inside the graph or measuring the elapsed time between
  // two events will trigger an error
  if ((bench_spec.enable_timing || bench_spec.test_API_on_stream) &&
      !bench_spec.enable_graph) {
    consume_and_print_timing(bench_spec, *(bench_data.get()), timing_results);
  }
  if (bench_spec.enable_graph) {
    CHECK_CUDA(cudaGraphExecDestroy(bench_data.get()->graphExecs[0]));
    CHECK_CUDA(cudaGraphDestroy(bench_data.get()->graphs[0]));
  }
  cleanUp(bench_spec, *(bench_data.get()));
  return 0;
}

int main(const int argc, const char **argv) {
  TimingResults timing_results;
  auto bench_tuple = generate_data_and_prepare(argc, argv, timing_results);
  auto bench_spec = std::get<0>(bench_tuple);
  auto bench_data = std::get<1>(bench_tuple);

  if (bench_spec.enable_preprocess) {
    preprocess(bench_spec, *(bench_data.get()));
  }

  if (bench_spec.enable_graph) {
    std::chrono::time_point<std::chrono::system_clock> graph_creation_beg,
        graph_creation_end, graph_initialization_end, graph_execution_end;
    graph_creation_beg = std::chrono::system_clock::now();

    auto graph_constructor =
        create_graph(bench_spec, *(bench_data.get()), timing_results);

    graph_creation_end = std::chrono::system_clock::now();
    initiate_graph(*(bench_data.get()));

    graph_initialization_end = std::chrono::system_clock::now();
    launch_graph_and_wait(*(bench_data.get()));
    graph_execution_end = std::chrono::system_clock::now();
    printf(
        "[DEBUG] cusparseSpMM+CSR+Partitioned graph creation chrono time "
        "(microseconds): "
        "%ld\n",
        std::chrono::duration_cast<std::chrono::microseconds>(
            graph_creation_end - graph_creation_beg)
            .count());
    printf(
        "[DEBUG] cusparseSpMM+CSR+Partitioned graph initialization chrono time "
        "(microseconds): "
        "%ld\n",
        std::chrono::duration_cast<std::chrono::microseconds>(
            graph_initialization_end - graph_creation_end)
            .count());
    printf(
        "[DEBUG] cusparseSpMM+CSR+Partitioned graph execution chrono time "
        "(microseconds): "
        "%ld\n",
        std::chrono::duration_cast<std::chrono::microseconds>(
            graph_execution_end - graph_initialization_end)
            .count());
  } else {
    _compute_reference(bench_spec, *(bench_data.get()), timing_results);
  }
  // When CUDA graph is enabled, we already wait until it finishes. Both
  // synchronize event inside the graph or measuring the elapsed time between
  // two events will trigger an error
  if ((bench_spec.enable_timing || bench_spec.test_API_on_stream) &&
      !bench_spec.enable_graph) {
    consume_and_print_timing(bench_spec, *(bench_data.get()), timing_results);
  }

  // Still needs to destroy graphExec though graph will be destroyed by
  // cudaGraphWrapper destructor
  if (bench_spec.enable_graph) {
    CHECK_CUDA(cudaGraphExecDestroy(bench_data.get()->graphExecs[0]));
  }
  cleanUp(bench_spec, *(bench_data.get()));
  return 0;
}
};  // namespace BenchSpMMCSRPartitioned
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

#include "helper_cusp.cu.h"
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

struct BenchSpmmCSRPartitionedProblemSpec {
  int A_num_rows;
  int A_num_cols;
  int B_num_cols;
  int AA_num_rows;
  int AA_num_cols;
  int BB_num_cols;
  float A_sparsity;
  bool enable_dump;
  bool enable_timing;
  bool enable_debug_timing;
  char *cli_result_path_and_prefix;
  bool flag_specify_result_path_and_prefix;
  bool test_API_on_stream;
};

struct BenchSpmmCSRPartitionedRuntimeData {
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
  cusparseHandle_t handle;
  std::vector<cusparseSpMatDescr_t> matAA;
  std::vector<cusparseDnMatDescr_t> matBB, matCC;
  std::vector<void *> dBuffers;
  std::vector<size_t> bufferSizes;
  // Keep hA in case data needs to be dumped
  cusp::csr_matrix<int, float, cusp::host_memory> hA;
  std::vector<cusp::csr_matrix<int, float, cusp::host_memory>> hAA;
  std::vector<cusp::csr_matrix<int, float, cusp::device_memory>> dAA;
  cudaStream_t stream;
};

std::tuple<BenchSpmmCSRPartitionedProblemSpec,
           BenchSpmmCSRPartitionedRuntimeData>
generate_data_and_prepare_bench_spmm_csr_partitioned(
    const int argc, const char **argv,
    std::map<std::string, std::tuple<cudaEvent_t, cudaEvent_t>>
        &utility_timestamps) {
  // Host problem definition
  int A_num_rows = getCmdLineArgumentInt(argc, argv, "A_num_rows");
  int A_num_cols = getCmdLineArgumentInt(argc, argv, "A_num_cols");
  int B_num_cols = getCmdLineArgumentInt(argc, argv, "B_num_cols");
  int AA_num_rows = getCmdLineArgumentInt(argc, argv, "AA_num_rows");
  int AA_num_cols = getCmdLineArgumentInt(argc, argv, "AA_num_cols");
  int BB_num_cols = getCmdLineArgumentInt(argc, argv, "BB_num_cols");
  float A_sparsity = getCmdLineArgumentFloat(argc, argv, "A_sparsity");
  bool enable_dump = checkCmdLineFlag(argc, argv, "enable_dump");
  bool enable_timing = checkCmdLineFlag(argc, argv, "enable_timing");
  bool test_API_on_stream = checkCmdLineFlag(argc, argv, "test_API_on_stream");
  bool enable_debug_timing =
      checkCmdLineFlag(argc, argv, "enable_debug_timing");
  char *cli_result_path_and_prefix;
  bool flag_specify_result_path_and_prefix = getCmdLineArgumentString(
      argc, argv, "result_path_and_prefix", &cli_result_path_and_prefix);
  if (A_num_rows == 0 || A_num_cols == 0 || B_num_cols == 0 ||
      AA_num_rows == 0 || AA_num_cols == 0 || BB_num_cols == 0 ||
      A_sparsity == 0.0f) {
    printf(
        "Usage: %s --A_num_rows=## --A_num_cols=## --B_num_cols=## "
        "--AA_num_rows=## --AA_num_cols=## --BB_num_cols=## "
        "--A_sparsity=0.## [--enable_dump] [--result_path_and_prefix=...] "
        "[--enable_timing] [--enable_debug_timing] [--test_API_on_stream]\n",
        argv[0]);
    // Example: ./bench_spmm_csr_partitioned --A_num_rows=1024 --A_num_cols=512
    // --B_num_cols=512 --AA_num_rows=128 --AA_num_cols=64 --BB_num_cols=64
    // --A_sparsity=0.1 --test_API_on_stream
    // Example: ./bench_spmm_csr_partitioned --A_num_rows=128 --A_num_cols=256
    // --B_num_cols=256 --AA_num_rows=16 --AA_num_cols=64 --BB_num_cols=64
    // --A_sparsity=0.1
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
    printf(
        "Usage: %s --A_num_rows=## --A_num_cols=## --B_num_cols=## "
        "--AA_num_rows=## --AA_num_cols=## --BB_num_cols=## "
        "--A_sparsity=0.## [--enable_dump] [--result_path_and_prefix=...] "
        "[--enable_timing] [--enable_debug_timing] [--test_API_on_stream]\n",
        argv[0]);
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
  cusparseHandle_t handle = NULL;
  std::vector<void *> dBuffers{};
  std::vector<size_t> bufferSizes{};
  std::vector<cusparseSpMatDescr_t> matAA{};
  std::vector<cusparseDnMatDescr_t> matBB{};
  std::vector<cusparseDnMatDescr_t> matCC{};
  cudaStream_t stream;

  // instantiating data
  hB = (float *)malloc(sizeof(float) * B_size);
  generate_random_matrix(hB, B_size);
  cusp::csr_matrix<int, float, cusp::host_memory> hA =
      generate_random_sparse_matrix<
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
  printf(
      "actual A_nnz due to deduplication during random data generation: %d\n",
      A_nnz);

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

  CHECK_CUDA(cudaStreamCreate(&stream));

  //--------------------------------------------------------------------------
  // Create Handle
  std::chrono::time_point<std::chrono::system_clock> handle_creation_beg,
      handle_creation_end;
  if (enable_timing) {
    CHECK_CUDA(cudaEventRecord(handle_creation_start, stream));
  }
  if (test_API_on_stream) {
    handle_creation_beg = std::chrono::system_clock::now();

    CHECK_CUDA(cudaEventRecord(test_API_0_handle_creation_start));
    CHECK_CUDA(cudaEventRecord(test_API_stream_handle_creation_start, stream));

    // CHECK_CUDA(cudaDeviceSynchronize());
  }
  CHECK_CUSPARSE(cusparseCreate(&handle))
  if (test_API_on_stream) {
    // CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(test_API_0_handle_creation_stop));
    CHECK_CUDA(cudaEventRecord(test_API_stream_handle_creation_stop, stream));
    handle_creation_end = std::chrono::system_clock::now();
    utility_timestamps["[DEBUG]API_0_handle_creation"] = std::make_tuple(
        test_API_0_handle_creation_start, test_API_0_handle_creation_stop);
    utility_timestamps["[DEBUG]API_stream_handle_creation"] =
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
    CHECK_CUDA(cudaEventRecord(handle_creation_stop, stream));
    utility_timestamps["handle_creation"] =
        std::make_tuple(handle_creation_start, handle_creation_stop);
  }
  CHECK_CUSPARSE(cusparseSetStream(handle, stream));
  // Device memory management
  if (enable_timing) {
    CHECK_CUDA(cudaEventRecord(data_copy_start, stream));
  }
  CHECK_CUDA(cudaMalloc((void **)&dB, B_size * sizeof(float)))
  CHECK_CUDA(cudaMalloc((void **)&dC, C_size * sizeof(float)))

  CHECK_CUDA(cudaMemcpy(dB, hB, B_size * sizeof(float), cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemset(dB, 0, B_size * sizeof(float)))
  if (enable_timing) {
    CHECK_CUDA(cudaEventRecord(data_copy_stop, stream));
    utility_timestamps["data_copy"] =
        std::make_tuple(data_copy_start, data_copy_stop);
  }
  //--------------------------------------------------------------------------
  // CUSPARSE APIs

  std::chrono::time_point<std::chrono::system_clock>
      data_handle_and_buffer_creation_beg, data_handle_and_buffer_creation_end;

  if (enable_timing) {
    CHECK_CUDA(cudaEventRecord(cusparse_data_handle_and_buffer_creation_start,
                               stream));
  }
  if (test_API_on_stream) {
    data_handle_and_buffer_creation_beg = std::chrono::system_clock::now();

    CHECK_CUDA(cudaEventRecord(
        test_API_0_cusparse_data_handle_and_buffer_creation_start));
    CHECK_CUDA(cudaEventRecord(
        test_API_stream_cusparse_data_handle_and_buffer_creation_start,
        stream));
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
              dAA[AA_row_idx + AA_col_idx * A_num_rows / AA_num_rows]
                  .num_entries;
          CHECK_CUSPARSE(cusparseCreateCsr(
              &curr_matAA, AA_num_rows, AA_num_cols, curr_AA_nnz,
              // dA_csrOffsets, dA_columns, dA_values,
              (void *)thrust::raw_pointer_cast(
                  dAA[AA_row_idx + AA_col_idx * A_num_rows / AA_num_rows]
                      .row_offsets.data()),
              (void *)thrust::raw_pointer_cast(
                  dAA[AA_row_idx + AA_col_idx * A_num_rows / AA_num_rows]
                      .column_indices.data()),
              (void *)thrust::raw_pointer_cast(
                  dAA[AA_row_idx + AA_col_idx * A_num_rows / AA_num_rows]
                      .values.data()),
              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
              CUDA_R_32F))
          matAA.push_back(curr_matAA);
          AA_nnz.push_back(curr_AA_nnz);
        }
        if (AA_row_idx == 0) {
          // Create dense matrix B
          cusparseDnMatDescr_t curr_matBB;
          CHECK_CUSPARSE(cusparseCreateDnMat(
              &curr_matBB, AA_num_cols, BB_num_cols, ldb,
              dB + /*row*/ (AA_col_idx * AA_num_cols) +
                  /*col*/ (BB_col_idx * BB_num_cols) * A_num_cols,
              CUDA_R_32F, CUSPARSE_ORDER_COL))
          matBB.push_back(curr_matBB);
        }
        if (AA_col_idx == 0) {
          // Create dense matrix C
          cusparseDnMatDescr_t curr_matCC;
          CHECK_CUSPARSE(
              cusparseCreateDnMat(&curr_matCC, AA_num_rows, BB_num_cols, ldc,
                                  dC + AA_row_idx * AA_num_rows +
                                      BB_col_idx * BB_num_cols * A_num_rows,
                                  CUDA_R_32F, CUSPARSE_ORDER_COL))
          matCC.push_back(curr_matCC);
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
        // Allocate an external buffer if needed
        CHECK_CUSPARSE(cusparseSpMM_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &(alpha), matAA[idx_AA],
            matBB[idx_BB], &(beta), matCC[idx_CC], CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, &curr_bufferSize))
        CHECK_CUDA(cudaMalloc(&curr_dBuffer, curr_bufferSize))
        dBuffers.push_back(curr_dBuffer);
        bufferSizes.push_back(curr_bufferSize);
      }
    }
  }
  if (test_API_on_stream) {
    // CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(
        test_API_0_cusparse_data_handle_and_buffer_creation_stop));
    CHECK_CUDA(cudaEventRecord(
        test_API_stream_cusparse_data_handle_and_buffer_creation_stop, stream));

    utility_timestamps["[DEBUG]API_0_data_handle_and_buffer_creation"] =
        std::make_tuple(
            test_API_0_cusparse_data_handle_and_buffer_creation_start,
            test_API_0_cusparse_data_handle_and_buffer_creation_stop);
    utility_timestamps["[DEBUG]API_stream_data_handle_and_buffer_creation"] =
        std::make_tuple(
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
    CHECK_CUDA(
        cudaEventRecord(cusparse_data_handle_and_buffer_creation_stop, stream));
    utility_timestamps["cusparse_data_handle_and_buffer_creation"] =
        std::make_tuple(cusparse_data_handle_and_buffer_creation_start,
                        cusparse_data_handle_and_buffer_creation_stop);
  }

  BenchSpmmCSRPartitionedProblemSpec problem_spec{
      .A_num_rows = A_num_rows,
      .A_num_cols = A_num_cols,
      .B_num_cols = B_num_cols,
      .AA_num_rows = AA_num_rows,
      .AA_num_cols = AA_num_cols,
      .BB_num_cols = BB_num_cols,
      .A_sparsity = A_sparsity,
      .enable_dump = enable_dump,
      .enable_timing = enable_timing,
      .enable_debug_timing = enable_debug_timing,
      .cli_result_path_and_prefix = cli_result_path_and_prefix,
      .flag_specify_result_path_and_prefix =
          flag_specify_result_path_and_prefix,
      .test_API_on_stream = test_API_on_stream};
  BenchSpmmCSRPartitionedRuntimeData runtime_data{.A_nnz = A_nnz,
                                                  .AA_nnz = AA_nnz,
                                                  .B_num_rows = B_num_rows,
                                                  .ldb = ldb,
                                                  .ldc = ldc,
                                                  .B_size = B_size,
                                                  .C_size = C_size,
                                                  .alpha = alpha,
                                                  .beta = beta,
                                                  .hB = hB,
                                                  .dB = dB,
                                                  .dC = dC,
                                                  .handle = handle,
                                                  .matAA = matAA,
                                                  .matBB = matBB,
                                                  .matCC = matCC,
                                                  .dBuffers = dBuffers,
                                                  .bufferSizes = bufferSizes,
                                                  .hA = hA,
                                                  .hAA = hAA,
                                                  .dAA = dAA,
                                                  .stream = stream};

  auto bench_tuple = std::make_tuple(problem_spec, runtime_data);
  return bench_tuple;
}

std::tuple<cudaEvent_t, cudaEvent_t> compute_bench_spmm_csr_partitioned(
    BenchSpmmCSRPartitionedProblemSpec &problem_spec,
    BenchSpmmCSRPartitionedRuntimeData &runtime_data,
    std::map<std::string, std::tuple<cudaEvent_t, cudaEvent_t>>
        &utility_timestamps) {
  // Execute SpMM
  // We nest the cuda event timing with std::chrono to make sure the cuda
  // event is getting correct results, we will use the cuda event timing
  // results and ignore the std::chrono results
  std::chrono::time_point<std::chrono::system_clock> beg, end;
  cudaEvent_t start, stop;
  if (problem_spec.enable_timing) {
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
  }

  if (problem_spec.enable_debug_timing) {
    CHECK_CUDA(cudaDeviceSynchronize());
    beg = std::chrono::system_clock::now();
  }
  if (problem_spec.enable_timing)
    CHECK_CUDA(cudaEventRecord(start, runtime_data.stream));
  // for (int idx_spmm = 0;
  //      idx_spmm < (problem_spec.A_num_rows / problem_spec.AA_num_rows) *
  //                     (problem_spec.A_num_cols / problem_spec.AA_num_cols)
  //                     * (problem_spec.B_num_cols /
  //                     problem_spec.BB_num_cols);
  //      idx_spmm++) {

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
        CHECK_CUSPARSE(cusparseSpMM(
            runtime_data.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &(runtime_data.alpha),
            runtime_data.matAA[idx_AA], runtime_data.matBB[idx_BB],
            &(runtime_data.beta), runtime_data.matCC[idx_CC], CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, runtime_data.dBuffers[idx_spmm]))

        idx_spmm++;
      }
    }
  }
  if (problem_spec.enable_timing)
    CHECK_CUDA(cudaEventRecord(stop, runtime_data.stream));
  if (problem_spec.enable_debug_timing) {
    CHECK_CUDA(cudaDeviceSynchronize());
    end = std::chrono::system_clock::now();
    printf(
        "[DEBUG] cusparseSpMM+CSR+Partitioned chrono time (microseconds): "
        "%ld\n",
        std::chrono::duration_cast<std::chrono::microseconds>(end - beg)
            .count());
  }

  return std::make_tuple(start, stop);
}

void print_timing_bench_spmm_csr_partitioned(
    cudaEvent_t start, cudaEvent_t stop,
    BenchSpmmCSRPartitionedProblemSpec &problem_spec,
    BenchSpmmCSRPartitionedRuntimeData &runtime_data,
    std::map<std::string, std::tuple<cudaEvent_t, cudaEvent_t>>
        &utility_timestamps) {
  CHECK_CUDA(cudaEventSynchronize(stop));
  // Print elapsed time of utilities. Keyword "elapsed time(util) (ms):"
  for (const auto &keyval : utility_timestamps) {
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

  if (problem_spec.enable_timing) {
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
}

void cleanup_bench_spmm_csr_partitioned(
    BenchSpmmCSRPartitionedProblemSpec &problem_spec,
    BenchSpmmCSRPartitionedRuntimeData &runtime_data) {
  CHECK_CUSPARSE(cusparseDestroy(runtime_data.handle))

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
    const char *result_path_and_prefix;
    if (problem_spec.flag_specify_result_path_and_prefix) {
      result_path_and_prefix = problem_spec.cli_result_path_and_prefix;
    } else {
      result_path_and_prefix =
          (std::string("cusparse_bench_spmm_csr_partitioned.") + time_str)
              .c_str();
    }
    // Store m, n, k to a txt and store A, B, C to a numpy file
    FILE *fp =
        fopen((std::string(result_path_and_prefix) + ".txt").c_str(), "w");
    assert(fp != nullptr);
    fprintf(fp, "%d %d %d %d %f\n", problem_spec.A_num_rows,
            problem_spec.A_num_cols, problem_spec.B_num_cols,
            runtime_data.A_nnz, problem_spec.A_sparsity);
    fclose(fp);
    cusp::io::write_matrix_market_file(
        runtime_data.hA, std::string(result_path_and_prefix) + ".A.mtx");

    unsigned long b_shape[2] = {runtime_data.ldb, problem_spec.B_num_cols};
    unsigned long c_shape[2] = {runtime_data.ldc, problem_spec.B_num_cols};
    npy::SaveArrayAsNumpy(std::string(result_path_and_prefix) + ".B.npy", false,
                          2, b_shape, runtime_data.hB);
    npy::SaveArrayAsNumpy(std::string(result_path_and_prefix) + ".C.npy", false,
                          2, c_shape, hC);
    free(hC);
  }
  // device memory deallocation
  CHECK_CUDA(cudaFree(runtime_data.dB))
  CHECK_CUDA(cudaFree(runtime_data.dC))
  free(runtime_data.hB);
  return;
}

int main_bench_spmm_csr_partitioned(const int argc, const char **argv) {
  std::map<std::string, std::tuple<cudaEvent_t, cudaEvent_t>>
      utility_timestamps;
  auto bench_tuple = generate_data_and_prepare_bench_spmm_csr_partitioned(
      argc, argv, utility_timestamps);
  auto bench_spec = std::get<0>(bench_tuple);
  auto bench_data = std::get<1>(bench_tuple);
  auto start_end_events = compute_bench_spmm_csr_partitioned(
      bench_spec, bench_data, utility_timestamps);
  auto start = std::get<0>(start_end_events);
  auto stop = std::get<1>(start_end_events);
  if (bench_spec.enable_timing || bench_spec.test_API_on_stream) {
    print_timing_bench_spmm_csr_partitioned(start, stop, bench_spec, bench_data,
                                            utility_timestamps);
  }
  cleanup_bench_spmm_csr_partitioned(bench_spec, bench_data);
  return 0;
}
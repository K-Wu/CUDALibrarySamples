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

struct BenchSpmmCSRProblemSpec {
  int A_num_rows;
  int A_num_cols;
  int B_num_cols;
  float A_sparsity;
  bool enable_dump;
  bool enable_timing;
  bool enable_debug_timing;
  char *cli_result_path_and_prefix;
  bool flag_specify_result_path_and_prefix;
};

struct BenchSpmmCSRRuntimeData {
  int A_nnz;
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
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  void *dBuffer;
  size_t bufferSize;
  cusp::csr_matrix<int, float, cusp::host_memory> hA;
  cusp::csr_matrix<int, float, cusp::device_memory> dA;
  cudaStream_t stream;
};

std::tuple<BenchSpmmCSRProblemSpec, BenchSpmmCSRRuntimeData>
generate_data_and_prepare_bench_spmm_csr(
    const int argc, const char **argv,
    std::map<std::string, std::tuple<cudaEvent_t, cudaEvent_t>>
        &utility_timestamps) {
  // Host problem definition
  int A_num_rows = getCmdLineArgumentInt(argc, argv, "A_num_rows");
  int A_num_cols = getCmdLineArgumentInt(argc, argv, "A_num_cols");
  int B_num_cols = getCmdLineArgumentInt(argc, argv, "B_num_cols");
  float A_sparsity = getCmdLineArgumentFloat(argc, argv, "A_sparsity");
  bool enable_dump = checkCmdLineFlag(argc, argv, "enable_dump");
  bool enable_timing = checkCmdLineFlag(argc, argv, "enable_timing");
  bool enable_debug_timing =
      checkCmdLineFlag(argc, argv, "enable_debug_timing");
  char *cli_result_path_and_prefix;
  bool flag_specify_result_path_and_prefix = getCmdLineArgumentString(
      argc, argv, "result_path_and_prefix", &cli_result_path_and_prefix);
  if (A_num_rows == 0 || A_num_cols == 0 || B_num_cols == 0 ||
      A_sparsity == 0.0f) {
    printf(
        "Usage: %s --A_num_rows=## --A_num_cols=## --B_num_cols=## "
        "--A_sparsity=0.## [--enable_dump] [--result_path_and_prefix=...] "
        "[--enable_timing] [--enable_debug_timing]\n",
        argv[0]);
    exit(EXIT_FAILURE);
  }
  printf("A_num_rows: %d\n", A_num_rows);
  printf("A_num_cols: %d\n", A_num_cols);
  printf("B_num_cols: %d\n", B_num_cols);
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
  void *dBuffer = NULL;
  size_t bufferSize = 0;
  cudaStream_t stream;

  // instantiating data
  hB = (float *)malloc(sizeof(float) * B_size);
  generate_random_matrix(hB, B_size);
  cusp::csr_matrix<int, float, cusp::host_memory> hA =
      generate_random_sparse_matrix<
          cusp::csr_matrix<int, float, cusp::host_memory>>(A_num_rows,
                                                           A_num_cols, A_nnz);
  cusp::csr_matrix<int, float, cusp::device_memory> dA(hA);
  A_nnz = hA.values.size();
  printf(
      "actual A_nnz due to deduplication during random data generation: %d\n",
      A_nnz);

  cudaEvent_t handle_creation_start, handle_creation_stop;
  cudaEvent_t data_copy_start, data_copy_stop;
  cudaEvent_t cusparse_data_handle_and_buffer_creation_start,
      cusparse_data_handle_and_buffer_creation_stop;

  CHECK_CUDA(cudaEventCreate(&handle_creation_start));
  CHECK_CUDA(cudaEventCreate(&handle_creation_stop));
  CHECK_CUDA(cudaEventCreate(&data_copy_start));
  CHECK_CUDA(cudaEventCreate(&data_copy_stop));
  CHECK_CUDA(cudaEventCreate(&cusparse_data_handle_and_buffer_creation_start));
  CHECK_CUDA(cudaEventCreate(&cusparse_data_handle_and_buffer_creation_stop));

  CHECK_CUDA(cudaStreamCreate(&stream));

  //--------------------------------------------------------------------------
  // Create Handle
  if (enable_timing) {
    CHECK_CUDA(cudaEventRecord(handle_creation_start, stream));
  }
  CHECK_CUSPARSE(cusparseCreate(&handle))
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

  std::chrono::time_point<std::chrono::system_clock> beg, end;
  beg = std::chrono::system_clock::now();
  CHECK_CUDA(cudaDeviceSynchronize());
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;

  if (enable_timing) {
    CHECK_CUDA(cudaEventRecord(cusparse_data_handle_and_buffer_creation_start,
                               stream));
  }

  // Create sparse matrix A in CSR format
  CHECK_CUSPARSE(cusparseCreateCsr(
      &matA, A_num_rows, A_num_cols, A_nnz,
      // dA_csrOffsets, dA_columns, dA_values,
      (void *)thrust::raw_pointer_cast(dA.row_offsets.data()),
      (void *)thrust::raw_pointer_cast(dA.column_indices.data()),
      (void *)thrust::raw_pointer_cast(dA.values.data()), CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
  // Create dense matrix B
  CHECK_CUSPARSE(cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                     CUDA_R_32F, CUSPARSE_ORDER_COL))
  // Create dense matrix C
  CHECK_CUSPARSE(cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                     CUDA_R_32F, CUSPARSE_ORDER_COL))
  // Allocate an external buffer if needed
  CHECK_CUSPARSE(cusparseSpMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize))
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

  if (enable_timing) {
    CHECK_CUDA(
        cudaEventRecord(cusparse_data_handle_and_buffer_creation_stop, stream));
    utility_timestamps["cusparse_data_handle_and_buffer_creation"] =
        std::make_tuple(cusparse_data_handle_and_buffer_creation_start,
                        cusparse_data_handle_and_buffer_creation_stop);
  }

  CHECK_CUDA(cudaDeviceSynchronize());

  end = std::chrono::system_clock::now();
  printf(
      "[DEBUG] cusparseSpMM+CSR data handle and buffer creation chrono time "
      "(microseconds): %ld\n",
      std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count());
  //--------------------------------------------------------------------------
  BenchSpmmCSRProblemSpec problem_spec{
      .A_num_rows = A_num_rows,
      .A_num_cols = A_num_cols,
      .B_num_cols = B_num_cols,
      .A_sparsity = A_sparsity,
      .enable_dump = enable_dump,
      .enable_timing = enable_timing,
      .enable_debug_timing = enable_debug_timing,
      .cli_result_path_and_prefix = cli_result_path_and_prefix,
      .flag_specify_result_path_and_prefix =
          flag_specify_result_path_and_prefix,
  };
  BenchSpmmCSRRuntimeData runtime_data{.A_nnz = A_nnz,
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
                                       .matA = matA,
                                       .matB = matB,
                                       .matC = matC,
                                       .dBuffer = dBuffer,
                                       .bufferSize = bufferSize,
                                       .hA = hA,
                                       .dA = dA,
                                       .stream = stream};

  auto bench_tuple = std::make_tuple(problem_spec, runtime_data);
  return bench_tuple;
}

std::tuple<cudaEvent_t, cudaEvent_t> compute_bench_spmm_csr(
    BenchSpmmCSRProblemSpec &problem_spec,
    BenchSpmmCSRRuntimeData &runtime_data,
    std::map<std::string, std::tuple<cudaEvent_t, cudaEvent_t>>
        &utility_timestamps) {
  // Execute SpMM
  // We nest the cuda event timing with std::chrono to make sure the cuda event
  // is getting correct results, we will use the cuda event timing results and
  // ignore the std::chrono results
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
  CHECK_CUSPARSE(
      cusparseSpMM(runtime_data.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                   CUSPARSE_OPERATION_NON_TRANSPOSE, &(runtime_data.alpha),
                   runtime_data.matA, runtime_data.matB, &(runtime_data.beta),
                   runtime_data.matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                   runtime_data.dBuffer))
  if (problem_spec.enable_timing)
    CHECK_CUDA(cudaEventRecord(stop, runtime_data.stream));
  if (problem_spec.enable_debug_timing) {
    CHECK_CUDA(cudaDeviceSynchronize());
    end = std::chrono::system_clock::now();
    printf("[DEBUG] cusparseSpMM+CSR chrono time (microseconds): %ld\n",
           std::chrono::duration_cast<std::chrono::microseconds>(end - beg)
               .count());
  }

  return std::make_tuple(start, stop);
}

void print_timing_bench_spmm_csr(
    cudaEvent_t start, cudaEvent_t stop, BenchSpmmCSRProblemSpec &problem_spec,
    BenchSpmmCSRRuntimeData &runtime_data,
    std::map<std::string, std::tuple<cudaEvent_t, cudaEvent_t>>
        &utility_timestamps) {
  CHECK_CUDA(cudaEventSynchronize(stop));
  float elapsed_time = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
  printf("cusparseSpMM+CSR elapsed time (ms): %f\n", elapsed_time);
  printf("cusparseSpMM+CSR throughput (GFLOPS): %f\n",
         (2.0 * runtime_data.A_nnz * problem_spec.B_num_cols) /
             (elapsed_time / 1000.0) / 1e9);
  // Print elapsed time of utilities. Keyword "elapsed time(util) (ms):"
  for (const auto &keyval : utility_timestamps) {
    const auto &key = keyval.first;
    const auto &value = keyval.second;
    float elapsed_time_util = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time_util, std::get<0>(value),
                                    std::get<1>(value)));
    printf("cusparseSpMM+CSR %s elapsed time(util) (ms): %f\n", key.c_str(),
           elapsed_time_util);
    CHECK_CUDA(cudaEventDestroy(std::get<0>(value)));
    CHECK_CUDA(cudaEventDestroy(std::get<1>(value)));
  }
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
}

void cleanup_bench_spmm_csr(BenchSpmmCSRProblemSpec &problem_spec,
                            BenchSpmmCSRRuntimeData &runtime_data) {
  // Destroy matrix/vector descriptors
  CHECK_CUSPARSE(cusparseDestroySpMat(runtime_data.matA))
  CHECK_CUSPARSE(cusparseDestroyDnMat(runtime_data.matB))
  CHECK_CUSPARSE(cusparseDestroyDnMat(runtime_data.matC))
  CHECK_CUSPARSE(cusparseDestroy(runtime_data.handle))

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
          (std::string("cusparse_bench_spmm_csr.") + time_str).c_str();
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
  CHECK_CUDA(cudaFree(runtime_data.dBuffer))
  CHECK_CUDA(cudaFree(runtime_data.dB))
  CHECK_CUDA(cudaFree(runtime_data.dC))
  free(runtime_data.hB);
  return;
}

int main_bench_spmm_csr(const int argc, const char **argv) {
  std::map<std::string, std::tuple<cudaEvent_t, cudaEvent_t>>
      utility_timestamps;
  auto bench_tuple =
      generate_data_and_prepare_bench_spmm_csr(argc, argv, utility_timestamps);
  auto bench_spec = std::get<0>(bench_tuple);
  auto bench_data = std::get<1>(bench_tuple);
  auto start_end_events =
      compute_bench_spmm_csr(bench_spec, bench_data, utility_timestamps);
  auto start = std::get<0>(start_end_events);
  auto stop = std::get<1>(start_end_events);
  if (bench_spec.enable_timing) {
    print_timing_bench_spmm_csr(start, stop, bench_spec, bench_data,
                                utility_timestamps);
  }
  cleanup_bench_spmm_csr(bench_spec, bench_data);
  return 0;
}
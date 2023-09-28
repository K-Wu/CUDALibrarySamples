/* test test by Lawrence
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
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
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <utils/generate_random_data.h>
#include <utils/helper_string.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "cublas_utils.h"
#include "helper_kernels.cu.h"
#include "npy.hpp"

using data_type = float;

struct BenchGEMMPartitionedProblemSpec {
  int m;
  int n;
  int k;
  int mm;
  int nn;
  int kk;
  bool enable_dump;
  bool enable_timing;
  bool enable_debug_timing;
  char *cli_result_path_and_prefix;
  bool flag_specify_result_path_and_prefix;
};

struct BenchGEMMPartitionedRuntimeData {
  int lda;
  int ldb;
  int ldc;
  const data_type alpha;
  const data_type beta;
  cublasOperation_t transa;
  cublasOperation_t transb;
  std::vector<data_type> A;
  std::vector<data_type> B;
  std::vector<data_type> C;
  data_type *d_A;
  data_type *d_B;
  data_type *d_C;
  cudaStream_t stream;
  cublasHandle_t cublasH;
};

std::tuple<BenchGEMMPartitionedProblemSpec, BenchGEMMPartitionedRuntimeData>
generate_data_and_prepare_bench_gemm_partitioned(
    const int argc, const char **argv,
    std::map<std::string, std::tuple<cudaEvent_t, cudaEvent_t>>
        &utility_timestamps) {
  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;
  // Host problem definition
  int m = getCmdLineArgumentInt(argc, argv, "m");
  int n = getCmdLineArgumentInt(argc, argv, "n");
  int k = getCmdLineArgumentInt(argc, argv, "k");
  int mm = getCmdLineArgumentInt(argc, argv, "mm");
  int nn = getCmdLineArgumentInt(argc, argv, "nn");
  int kk = getCmdLineArgumentInt(argc, argv, "kk");
  bool enable_timing = checkCmdLineFlag(argc, argv, "enable_timing");
  bool enable_debug_timing =
      checkCmdLineFlag(argc, argv, "enable_debug_timing");
  bool enable_dump = checkCmdLineFlag(argc, argv, "enable_dump");
  char *cli_result_path_and_prefix;
  bool flag_specify_result_path_and_prefix = getCmdLineArgumentString(
      argc, argv, "result_path_and_prefix", &cli_result_path_and_prefix);
  printf("m(%d) n(%d) k(%d) mm(%d) nn(%d) kk(%d)\n", m, n, k, mm, nn, kk);
  if (m == 0 || n == 0 || k == 0 || mm == 0 || nn == 0 || kk == 0) {
    printf("m == 0 || n == 0 || k == 0 || mm == 0 || nn == 0 || kk == 0\n");
    printf(
        "Usage: %s --m=## --n=## --k=## --mm=## --nn=## --kk=## "
        "[--enable_dump] "
        "[--result_path_and_prefix=...] [--enable_timing] "
        "[--enable_debug_timing]\n",
        argv[0]);
    exit(EXIT_FAILURE);
  }
  if (m % mm != 0 || n % nn != 0 || k % kk != 0) {
    printf("m % mm != 0 || n % nn != 0 || k % kk != 0\n");
    printf(
        "Usage: %s --m=## --n=## --k=## --mm=## --nn=## --kk=## "
        "[--enable_dump] "
        "[--result_path_and_prefix=...] [--enable_timing] "
        "[--enable_debug_timing]\n",
        argv[0]);
    printf("m, n, k must be divisible by mm, nn, kk, respectively\n");
    exit(EXIT_FAILURE);
  }
  const data_type alpha = 1.0;
  const data_type beta = 0.0;
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;
  int lda = m;
  int ldb = k;
  int ldc = m;
  std::vector<data_type> A(lda * k);
  std::vector<data_type> B(ldb * n);
  std::vector<data_type> C(m * n);
  data_type *d_A = nullptr;
  data_type *d_B = nullptr;
  data_type *d_C = nullptr;

  std::srand(unsigned(std::time(nullptr)));
  std::generate(A.begin(), A.end(), std::rand);
  std::generate(B.begin(), B.end(), std::rand);

  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  cudaEvent_t handle_creation_start, handle_creation_stop;
  cudaEvent_t data_copy_start, data_copy_stop;

  CUDA_CHECK(cudaEventCreate(&handle_creation_start));
  CUDA_CHECK(cudaEventCreate(&handle_creation_stop));
  CUDA_CHECK(cudaEventCreate(&data_copy_start));
  CUDA_CHECK(cudaEventCreate(&data_copy_stop));

  // TODO: the handle creation API should be mentioned using chrono instead
  // because the document says only computation is governed by cublasSetStream,
  // i.e., handle creation is blocking.
  /* step 1: create cublas handle, bind a stream */

  std::chrono::time_point<std::chrono::system_clock> beg, end;
  beg = std::chrono::system_clock::now();
  CUDA_CHECK(cudaDeviceSynchronize());
  if (enable_timing) {
    CUDA_CHECK(cudaEventRecord(handle_creation_start, stream));
  }
  CUBLAS_CHECK(cublasCreate(&cublasH));

  CUBLAS_CHECK(cublasSetStream(cublasH, stream));
  if (enable_timing) {
    CUDA_CHECK(cudaEventRecord(handle_creation_stop, stream));
    utility_timestamps["handle_creation"] =
        std::make_tuple(handle_creation_start, handle_creation_stop);
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  end = std::chrono::system_clock::now();
  printf(
      "[DEBUG] cublasSgemmPartitioned handle creation chrono time "
      "(microseconds): %ld\n",
      std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count());

  /* step 2: copy data to device */
  if (enable_timing) {
    CUDA_CHECK(cudaEventRecord(data_copy_start, stream));
  }
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A),
                        sizeof(data_type) * A.size()));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B),
                        sizeof(data_type) * B.size()));
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void **>(&d_C),
      sizeof(data_type) * C.size() *
          (k / kk + 1)));  // Need k times the size to store output of the
                           // partitioned kernels and 1 original size to store
                           // the final accumulated results

  CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(),
                             cudaMemcpyHostToDevice, stream));

  if (enable_timing) {
    CUDA_CHECK(cudaEventRecord(data_copy_stop, stream));
    utility_timestamps["data_copy"] =
        std::make_tuple(data_copy_start, data_copy_stop);
  }

  BenchGEMMPartitionedProblemSpec problem_spec = {
      .m = m,
      .n = n,
      .k = k,
      .mm = mm,
      .nn = nn,
      .kk = kk,
      .enable_dump = enable_dump,
      .enable_timing = enable_timing,
      .enable_debug_timing = enable_debug_timing,
      .cli_result_path_and_prefix = cli_result_path_and_prefix,
      .flag_specify_result_path_and_prefix =
          flag_specify_result_path_and_prefix};
  BenchGEMMPartitionedRuntimeData runtime_data = {
      //.lda = lda,
      //.ldb = ldb,
      //.ldc = ldc,
      .alpha = alpha, .beta = beta, .transa = transa, .transb = transb,
      .A = A,         .B = B,       .C = C,           .d_A = d_A,
      .d_B = d_B,     .d_C = d_C,   .stream = stream, .cublasH = cublasH};

  std::tuple<BenchGEMMPartitionedProblemSpec, BenchGEMMPartitionedRuntimeData>
      bench_gemm_partitioned_tuple =
          std::make_tuple(problem_spec, runtime_data);
  return bench_gemm_partitioned_tuple;
}

std::tuple<cudaEvent_t, cudaEvent_t> compute_bench_gemm_partitioned(
    BenchGEMMPartitionedProblemSpec &bench_spec,
    BenchGEMMPartitionedRuntimeData &bench_data,
    std::map<std::string, std::tuple<cudaEvent_t, cudaEvent_t>>
        &utility_timestamps) {
  /* step 3: compute */
  // We nest the cuda event timing with std::chrono to make sure the cuda event
  // is getting correct results, we will use the cuda event timing results and
  // ignore the std::chrono results

  // TODO: use local-scope variable lda, ldb, ldc
  int lda = bench_spec.m;
  int ldb = bench_spec.k;
  int ldc = bench_spec.m;
  std::chrono::time_point<std::chrono::system_clock> beg, end;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  if (bench_spec.enable_debug_timing) {
    CUDA_CHECK(cudaStreamSynchronize(bench_data.stream));
    CUDA_CHECK(cudaDeviceSynchronize());
    beg = std::chrono::system_clock::now();
  }
  if (bench_spec.enable_timing)
    CUDA_CHECK(cudaEventRecord(start, bench_data.stream));
  for (int i = 0; i < bench_spec.m / bench_spec.mm; i++) {
    for (int j = 0; j < bench_spec.n / bench_spec.nn; j++) {
      for (int l = 0; l < bench_spec.k / bench_spec.kk; l++) {
        CUBLAS_CHECK(cublasSgemm(
            bench_data.cublasH, bench_data.transa, bench_data.transb,
            bench_spec.mm, bench_spec.nn, bench_spec.kk, &(bench_data.alpha),
            bench_data.d_A + i * bench_spec.mm + l * bench_spec.kk * lda, lda,
            bench_data.d_B + l * bench_spec.kk + j * bench_spec.nn * ldb, ldb,
            &(bench_data.beta),
            bench_data.d_C + l * bench_spec.m * bench_spec.n +
                i * bench_spec.mm + j * bench_spec.nn * ldc,
            ldc));
      }
    }
  }
  // Accumulate the result
  // TODO: define BLOCK_SIZE and SHMEM_SIZE
  constexpr int BLOCK_SIZE = 256;
  constexpr int SHMEM_SIZE = 256;
  assert(bench_spec.m * bench_spec.n % SHMEM_SIZE == 0);
  dim3 nblocks(bench_spec.m * bench_spec.n / SHMEM_SIZE,
               bench_spec.k / bench_spec.kk, 1);
  dim3 nthreads(BLOCK_SIZE, 1, 1);
  reduce_segments<BLOCK_SIZE, SHMEM_SIZE, float>
      <<<nblocks, nthreads, 0, bench_data.stream>>>(
          bench_data.d_C,
          bench_data.d_C +
              bench_spec.k / bench_spec.kk * bench_spec.m * bench_spec.n,
          bench_spec.m, bench_spec.n, bench_spec.m,
          bench_spec.k / bench_spec.kk);

  if (bench_spec.enable_timing)
    CUDA_CHECK(cudaEventRecord(stop, bench_data.stream));
  if (bench_spec.enable_debug_timing) {
    CUDA_CHECK(cudaStreamSynchronize(bench_data.stream));
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::system_clock::now();
    printf("[DEBUG] cublasSgemmPartitioned chrono time (microseconds): %ld\n",
           std::chrono::duration_cast<std::chrono::microseconds>(end - beg)
               .count());
  }
  return std::make_tuple(start, stop);
}

void print_timing_bench_gemm_partitioned(
    cudaEvent_t start, cudaEvent_t stop,
    BenchGEMMPartitionedProblemSpec &bench_spec,
    std::map<std::string, std::tuple<cudaEvent_t, cudaEvent_t>>
        &utility_timestamps) {
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_time = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
  printf("cublasSgemmPartitioned elapsed time (ms): %f\n", elapsed_time);
  printf("cublasSgemmPartitioned throughput (GFLOPS): %f\n",
         (2.0 * bench_spec.m * bench_spec.n * bench_spec.k) /
             (elapsed_time / 1000.0) / 1e9);
  // Print elapsed time of utilities. Keyword "elapsed time(util) (ms):"
  for (const auto &keyval : utility_timestamps) {
    const auto &key = keyval.first;
    const auto &value = keyval.second;
    float elapsed_time_util = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time_util, std::get<0>(value),
                                    std::get<1>(value)));
    printf("cublasSgemmPartitioned %s elapsed time(util) (ms): %f\n",
           key.c_str(), elapsed_time_util);
    CUDA_CHECK(cudaEventDestroy(std::get<0>(value)));
    CUDA_CHECK(cudaEventDestroy(std::get<1>(value)));
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

void cleanup_bench_gemm_partitioned(
    BenchGEMMPartitionedProblemSpec &bench_spec,
    BenchGEMMPartitionedRuntimeData &bench_data) {
  int lda = bench_spec.m;
  int ldb = bench_spec.k;
  int ldc = bench_spec.m;
  if (bench_spec.enable_dump) {
    /* step 4: copy data to host */
    CUDA_CHECK(cudaMemcpyAsync(bench_data.C.data(), bench_data.d_C,
                               sizeof(data_type) * bench_data.C.size(),
                               cudaMemcpyDeviceToHost, bench_data.stream));
    CUDA_CHECK(cudaStreamSynchronize(bench_data.stream));

    // Get current timestamp
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    char time_str[64];
    std::strftime(time_str, sizeof(time_str), "%Y-%m-%d-%H-%M", &tm);
    const char *result_path_and_prefix;
    if (!bench_spec.flag_specify_result_path_and_prefix) {
      result_path_and_prefix =
          (std::string("cublas_bench_gemm_partitioned.") + time_str).c_str();
    } else {
      result_path_and_prefix = bench_spec.cli_result_path_and_prefix;
    }
    result_path_and_prefix = nullptr;
    // Store m, n, k to a txt and store A, B, C to a numpy file
    FILE *fp =
        fopen((std::string(result_path_and_prefix) + ".txt").c_str(), "w");
    assert(fp != nullptr);
    fprintf(fp, "%d %d %d\n", bench_spec.m, bench_spec.n, bench_spec.k);
    fclose(fp);
    unsigned long a_shape[2] = {lda, bench_spec.k};
    unsigned long b_shape[2] = {ldb, bench_spec.n};
    unsigned long c_shape[2] = {bench_spec.m, bench_spec.n};
    npy::SaveArrayAsNumpy(std::string(result_path_and_prefix) + ".C.npy", false,
                          2, c_shape, bench_data.C);
    npy::SaveArrayAsNumpy(std::string(result_path_and_prefix) + ".A.npy", false,
                          2, a_shape, bench_data.A);
    npy::SaveArrayAsNumpy(std::string(result_path_and_prefix) + ".B.npy", false,
                          2, b_shape, bench_data.B);
  }

  /* free resources */
  CUDA_CHECK(cudaFree(bench_data.d_A));
  CUDA_CHECK(cudaFree(bench_data.d_B));
  CUDA_CHECK(cudaFree(bench_data.d_C));

  CUBLAS_CHECK(cublasDestroy(bench_data.cublasH));
  CUDA_CHECK(cudaStreamDestroy(bench_data.stream));

  CUDA_CHECK(cudaDeviceReset());
  return;
}

int main_bench_gemm_partitioned(const int argc, const char **argv) {
  std::map<std::string, std::tuple<cudaEvent_t, cudaEvent_t>>
      utility_timestamps;
  auto bench_tuple = generate_data_and_prepare_bench_gemm_partitioned(
      argc, argv, utility_timestamps);
  auto bench_spec = std::get<0>(bench_tuple);
  auto bench_data = std::get<1>(bench_tuple);
  auto start_end_events = compute_bench_gemm_partitioned(bench_spec, bench_data,
                                                         utility_timestamps);
  auto start = std::get<0>(start_end_events);
  auto stop = std::get<1>(start_end_events);
  if (bench_spec.enable_timing) {
    print_timing_bench_gemm_partitioned(start, stop, bench_spec,
                                        utility_timestamps);
  }
  cleanup_bench_gemm_partitioned(bench_spec, bench_data);
  return 0;
}
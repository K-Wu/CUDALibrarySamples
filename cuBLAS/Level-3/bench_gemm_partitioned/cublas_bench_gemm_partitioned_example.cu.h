/*
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
#include "helper_CUDAGraphConstructor.cu.h"
#include "helper_kernels.cu.h"
#include "helper_loops.cu.h"
#include "npy.hpp"

using data_type = float;

namespace BenchGEMMPartitioned {
struct TimingResults {
  std::vector<cudaEvent_t> start_events;
  std::vector<cudaEvent_t> stop_events;
  std::map<std::string, std::tuple<cudaEvent_t, cudaEvent_t>>
      utility_timestamps;
};

struct ProblemSpec {
  int m;
  int n;
  int k;
  int mm;
  int nn;
  int kk;
  bool enable_dump;
  bool enable_timing;
  bool enable_per_stream_timing;
  bool enable_debug_timing;
  bool enable_graph;
  char *cli_result_path_and_prefix;
  bool flag_specify_result_path_and_prefix;
  int nstreams;
};

struct RuntimeData {
  // int lda;
  // int ldb;
  // int ldc;
  data_type alpha;
  data_type beta;
  cublasOperation_t transa;
  cublasOperation_t transb;
  std::vector<data_type> A;
  std::vector<data_type> B;
  std::vector<data_type> C;
  data_type *d_A;
  data_type *d_B;
  data_type *d_C;
  std::vector<cudaStream_t> streams;
  // One handle per stream to conserve reproducibility
  // https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
  std::vector<cublasHandle_t> cublasHs;
  std::vector<cudaGraph_t> graphs;
  std::vector<cudaGraphExec_t> graphExecs;
};

void print_usage() {
  printf(
      "Usage: cublas_bench_gemm_partitioned_example --m=## --n=## --k=## "
      "--mm=## --nn=## --kk=## "
      "[--nstreams=##] [--enable_graph] [--enable_dump] "
      "[--result_path_and_prefix=...] [--enable_timing] "
      "[--enable_debug_timing]\n");
  // Print the meaning of each argument
  printf(
      "--enable_timing records the elapsed time of the computation function\n"
      "--enable_debug_timing also records the elapsed time of the computation\n"
      "function; but it adds device synchronization and uses chrono to record\n"
      "the timing\n"
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

std::tuple<ProblemSpec, RuntimeData> generate_data_and_prepare(
    const int argc, const char **argv, TimingResults &timing_results) {
  std::vector<cublasHandle_t> cublasHs;
  std::vector<cudaStream_t> streams;
  // Host problem definition
  int m = getCmdLineArgumentInt(argc, argv, "m");
  int n = getCmdLineArgumentInt(argc, argv, "n");
  int k = getCmdLineArgumentInt(argc, argv, "k");
  int mm = getCmdLineArgumentInt(argc, argv, "mm");
  int nn = getCmdLineArgumentInt(argc, argv, "nn");
  int kk = getCmdLineArgumentInt(argc, argv, "kk");
  int nstreams = getCmdLineArgumentInt(argc, argv, "nstreams");
  nstreams = (nstreams > 0) ? nstreams : 1;  // default value
  bool enable_timing = checkCmdLineFlag(argc, argv, "enable_timing");
  bool enable_per_stream_timing =
      checkCmdLineFlag(argc, argv, "enable_per_stream_timing");
  bool enable_debug_timing =
      checkCmdLineFlag(argc, argv, "enable_debug_timing");
  bool enable_dump = checkCmdLineFlag(argc, argv, "enable_dump");
  bool enable_graph = checkCmdLineFlag(argc, argv, "enable_graph");
  char *cli_result_path_and_prefix;
  bool flag_specify_result_path_and_prefix = getCmdLineArgumentString(
      argc, argv, "result_path_and_prefix", &cli_result_path_and_prefix);
  printf("m(%d) n(%d) k(%d) mm(%d) nn(%d) kk(%d) nstreams(%d)\n", m, n, k, mm,
         nn, kk, nstreams);
  if (m == 0 || n == 0 || k == 0 || mm == 0 || nn == 0 || kk == 0) {
    printf("m == 0 || n == 0 || k == 0 || mm == 0 || nn == 0 || kk == 0\n");
    print_usage();

    exit(EXIT_FAILURE);
  }
  if (m % mm != 0 || n % nn != 0 || k % kk != 0) {
    printf("m % mm != 0 || n % nn != 0 || k % kk != 0\n");
    print_usage();
    printf("m, n, k must be divisible by mm, nn, kk, respectively\n");
    exit(EXIT_FAILURE);
  }
  if (nstreams == 1 && enable_per_stream_timing) {
    printf(
        "please use --enable_timing instead of --enable_per_stream_timing "
        "when there is only one stream\n");
    print_usage();
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

  for (int idx = 0; idx < nstreams; idx++) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    streams.push_back(stream);
  }
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
    CUDA_CHECK(cudaEventRecord(handle_creation_start, streams.front()));
  }
  for (int idx = 0; idx < nstreams; idx++) {
    cublasHandle_t cublasH;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUBLAS_CHECK(cublasSetStream(cublasH, streams[idx]));
    cublasHs.push_back(cublasH);
  }
  if (enable_timing) {
    CUDA_CHECK(cudaEventRecord(handle_creation_stop, streams.front()));
    timing_results.utility_timestamps["handle_creation"] =
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
    CUDA_CHECK(cudaEventRecord(data_copy_start, streams.front()));
  }
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A),
                        sizeof(data_type) * A.size()));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B),
                        sizeof(data_type) * B.size()));
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&d_C),
                 sizeof(data_type) * C.size() *
                     (1 + 1)));  // Need k times the size to store output of the
                                 // partitioned kernels and 1 original size to
                                 // store the final accumulated results

  CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(),
                             cudaMemcpyHostToDevice, streams.front()));
  CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(),
                             cudaMemcpyHostToDevice, streams.front()));

  if (enable_timing) {
    CUDA_CHECK(cudaEventRecord(data_copy_stop, streams.front()));
    timing_results.utility_timestamps["data_copy"] =
        std::make_tuple(data_copy_start, data_copy_stop);
  }

  ProblemSpec problem_spec = {
      .m = m,
      .n = n,
      .k = k,
      .mm = mm,
      .nn = nn,
      .kk = kk,
      .enable_dump = enable_dump,
      .enable_timing = enable_timing,
      .enable_per_stream_timing = enable_per_stream_timing,
      .enable_debug_timing = enable_debug_timing,
      .enable_graph = enable_graph,
      .cli_result_path_and_prefix = cli_result_path_and_prefix,
      .flag_specify_result_path_and_prefix =
          flag_specify_result_path_and_prefix,
      .nstreams = nstreams};
  RuntimeData runtime_data = {//.lda not set
                              //.ldb not set
                              //.ldc not set
                              .alpha = alpha,
                              .beta = beta,
                              .transa = transa,
                              .transb = transb,
                              .A = A,
                              .B = B,
                              .C = C,
                              .d_A = d_A,
                              .d_B = d_B,
                              .d_C = d_C,
                              .streams = streams,
                              .cublasHs = cublasHs};

  std::tuple<ProblemSpec, RuntimeData> bench_gemm_partitioned_tuple =
      std::make_tuple(problem_spec, runtime_data);
  return bench_gemm_partitioned_tuple;
}

std::tuple<ProblemSpec, RuntimeData> generate_data_and_prepare(
    std::vector<std::string> &args, TimingResults &timing_results) {
  std::vector<const char *> args_cstr;
  for (int idx = 0; idx < args.size(); idx++) {
    args_cstr.push_back(args[idx].c_str());
  }
  return generate_data_and_prepare(args_cstr.size(), args_cstr.data(),
                                   timing_results);
}

// This function is written before GraphConstructor is implemented. It is no
// longer updated but stay here to 1) provide a reference implementation to
// facilitate debugging.
void _compute_reference(ProblemSpec &bench_spec, RuntimeData &bench_data,
                        TimingResults &timing_results) {
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
  std::vector<cudaEvent_t> starts_per_stream, stops_per_stream;
  if (wait_streams_on_first_and_report_that_as_elapsed_time(
          bench_spec.enable_per_stream_timing, bench_spec.enable_timing,
          bench_spec.nstreams)) {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }

  // We need stop event per stream to synchronize before reduction no matter
  // timing is enabled or not
  for (int idx = 0; idx < bench_spec.nstreams; idx++) {
    cudaEvent_t stop_per_stream;
    CUDA_CHECK(cudaEventCreate(&stop_per_stream));
    stops_per_stream.push_back(stop_per_stream);
  }
  if (bench_spec.enable_timing) {
    for (int idx = 0; idx < bench_spec.nstreams; idx++) {
      cudaEvent_t start_per_stream;
      CUDA_CHECK(cudaEventCreate(&start_per_stream));
      starts_per_stream.push_back(start_per_stream);
    }
  }

  if (bench_spec.enable_debug_timing) {
    for (int idx = 0; idx < bench_spec.nstreams; idx++)
      CUDA_CHECK(cudaStreamSynchronize(bench_data.streams[idx]));
    CUDA_CHECK(cudaDeviceSynchronize());
    beg = std::chrono::system_clock::now();
  }

  if (wait_streams_on_first_and_report_that_as_elapsed_time(
          bench_spec.enable_per_stream_timing, bench_spec.enable_timing,
          bench_spec.nstreams)) {
    CUDA_CHECK(cudaEventRecord(start, bench_data.streams.front()));
  }
  if (bench_spec.enable_timing) {
    for (int idx = 0; idx < bench_spec.nstreams; idx++) {
      if (wait_streams_on_first_and_report_that_as_elapsed_time(
              bench_spec.enable_per_stream_timing, bench_spec.enable_timing,
              bench_spec.nstreams)) {
        CUDA_CHECK(cudaStreamWaitEvent(bench_data.streams[idx], start));
      }

      CUDA_CHECK(
          cudaEventRecord(starts_per_stream[idx], bench_data.streams[idx]));
    }
  }
  for (int i = 0; i < bench_spec.m / bench_spec.mm; i++) {
    for (int j = 0; j < bench_spec.n / bench_spec.nn; j++) {
      for (int l = 0; l < bench_spec.k / bench_spec.kk; l++) {
        int curr_stream_idx = getCurrStream(
            {i, j, l},
            {bench_spec.m / bench_spec.mm, bench_spec.n / bench_spec.nn,
             bench_spec.k / bench_spec.kk},
            bench_data.streams.size());
        CUBLAS_CHECK(cublasSgemm(
            bench_data.cublasHs[curr_stream_idx], bench_data.transa,
            bench_data.transb, bench_spec.mm, bench_spec.nn, bench_spec.kk,
            &(bench_data.alpha),
            bench_data.d_A + i * bench_spec.mm + l * bench_spec.kk * lda, lda,
            bench_data.d_B + l * bench_spec.kk + j * bench_spec.nn * ldb, ldb,
            &(bench_data.beta),
            bench_data.d_C + l * bench_spec.mm * bench_spec.nn +
                i * bench_spec.mm + j * bench_spec.nn * ldc,
            ldc));
      }
    }
  }

  // Stream idx 0 waits for all other streams to finish before executing the
  // reduction kernel
  for (int idx = 1; idx < bench_spec.nstreams; idx++) {
    CUDA_CHECK(cudaEventRecord(stops_per_stream[idx], bench_data.streams[idx]));
    CUDA_CHECK(
        cudaStreamWaitEvent(bench_data.streams.front(), stops_per_stream[idx]));
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
      <<<nblocks, nthreads, 0, bench_data.streams.front()>>>(
          bench_data.d_C,
          bench_data.d_C +
              bench_spec.k / bench_spec.kk * bench_spec.mm * bench_spec.nn,
          bench_spec.mm, bench_spec.mm, bench_spec.nn,
          bench_spec.k / bench_spec.kk);

  if (bench_spec.enable_timing) {
    CUDA_CHECK(cudaEventRecord(stops_per_stream[0], bench_data.streams[0]));
  }

  if (wait_streams_on_first_and_report_that_as_elapsed_time(
          bench_spec.enable_per_stream_timing, bench_spec.enable_timing,
          bench_spec.nstreams)) {
    for (int idx = 0; idx < bench_spec.nstreams; idx++) {
      if (wait_streams_on_first_and_report_that_as_elapsed_time(
              bench_spec.enable_per_stream_timing, bench_spec.enable_timing,
              bench_spec.nstreams)) {
        CUDA_CHECK(cudaStreamWaitEvent(bench_data.streams.front(),
                                       stops_per_stream[idx]));
      }
      CUDA_CHECK(cudaEventRecord(stop, bench_data.streams.front()));
    }
  }

  if (bench_spec.enable_debug_timing) {
    for (int idx = 0; idx < bench_spec.nstreams; idx++)
      CUDA_CHECK(cudaStreamSynchronize(bench_data.streams[idx]));
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::system_clock::now();
    printf("[DEBUG] cublasSgemmPartitioned chrono time (microseconds): %ld\n",
           std::chrono::duration_cast<std::chrono::microseconds>(end - beg)
               .count());
  }

  // Add start, stop pair in each stream to the return value
  if (report_elapsed_time_per_stream(bench_spec.enable_per_stream_timing,
                                     bench_spec.enable_timing,
                                     bench_spec.nstreams)) {
    timing_results.start_events = starts_per_stream;
    timing_results.stop_events = stops_per_stream;
    return;
  }

  // Synchronize on the first stream in streams vector, and add the start, stop
  // pair to the return value
  if (wait_streams_on_first_and_report_that_as_elapsed_time(
          bench_spec.enable_per_stream_timing, bench_spec.enable_timing,
          bench_spec.nstreams)) {
    timing_results.start_events = std::vector<cudaEvent_t>({start});
    timing_results.stop_events = std::vector<cudaEvent_t>({stop});
    return;
  }

  // No timing should be reported
  for (int idx = 0; idx < bench_spec.nstreams; idx++) {
    CUDA_CHECK(cudaEventDestroy(stops_per_stream[idx]));
  }
  timing_results.start_events = std::vector<cudaEvent_t>();
  timing_results.stop_events = std::vector<cudaEvent_t>();
}

void compute(ProblemSpec &bench_spec, RuntimeData &bench_data,
             TimingResults &timing_results,
             AbstractCUDAGraphConstructor<cudaStream_t> &graph_constructor) {
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
  std::vector<cudaEvent_t> starts_per_stream, stops_per_stream;
  if (wait_streams_on_first_and_report_that_as_elapsed_time(
          bench_spec.enable_per_stream_timing, bench_spec.enable_timing,
          bench_spec.nstreams)) {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }

  // We need stop event per stream to synchronize before reduction no matter
  // timing is enabled or not
  for (int idx = 0; idx < bench_spec.nstreams; idx++) {
    cudaEvent_t stop_per_stream;
    CUDA_CHECK(cudaEventCreate(&stop_per_stream));
    stops_per_stream.push_back(stop_per_stream);
  }
  if (bench_spec.enable_timing) {
    for (int idx = 0; idx < bench_spec.nstreams; idx++) {
      cudaEvent_t start_per_stream;
      CUDA_CHECK(cudaEventCreate(&start_per_stream));
      starts_per_stream.push_back(start_per_stream);
    }
  }

  if (bench_spec.enable_debug_timing) {
    for (int idx = 0; idx < bench_spec.nstreams; idx++)
      CUDA_CHECK(cudaStreamSynchronize(bench_data.streams[idx]));
    CUDA_CHECK(cudaDeviceSynchronize());
    beg = std::chrono::system_clock::now();
  }

  if (wait_streams_on_first_and_report_that_as_elapsed_time(
          bench_spec.enable_per_stream_timing, bench_spec.enable_timing,
          bench_spec.nstreams)) {
    // Equivalent to error-checked cudaEventRecord
    graph_constructor.addEventRecordNode(start, bench_data.streams.front());
  }
  if (bench_spec.enable_timing) {
    for (int idx = 0; idx < bench_spec.nstreams; idx++) {
      if (wait_streams_on_first_and_report_that_as_elapsed_time(
              bench_spec.enable_per_stream_timing, bench_spec.enable_timing,
              bench_spec.nstreams)) {
        // Equivalent to error-checked cudaStreamWaitEvent
        graph_constructor.addStreamWaitEventNode(bench_data.streams[idx],
                                                 start);
      }

      // Equivalent to error-checked cudaEventRecord
      graph_constructor.addEventRecordNode(starts_per_stream[idx],
                                           bench_data.streams[idx]);
    }
  }
  graph_constructor.notifyBeforeInvokingLibraryCall(bench_data.streams[0]);
  for (int i = 0; i < bench_spec.m / bench_spec.mm; i++) {
    for (int j = 0; j < bench_spec.n / bench_spec.nn; j++) {
      for (int l = 0; l < bench_spec.k / bench_spec.kk; l++) {
        int curr_stream_idx = getCurrStream(
            {i, j, l},
            {bench_spec.m / bench_spec.mm, bench_spec.n / bench_spec.nn,
             bench_spec.k / bench_spec.kk},
            bench_data.streams.size());
        std::vector<int> last_loop_idxes =
            get_last_loop_index({i, j, l}, {bench_spec.m / bench_spec.mm,
                                            bench_spec.n / bench_spec.nn,
                                            bench_spec.k / bench_spec.kk});
        int last_stream_idx = getCurrStream(
            last_loop_idxes,
            {bench_spec.m / bench_spec.mm, bench_spec.n / bench_spec.nn,
             bench_spec.k / bench_spec.kk},
            bench_data.streams.size());
        if (i + j + l != 0 && curr_stream_idx != last_stream_idx) {
          graph_constructor.notifyAfterInvokingLibraryCall(
              bench_data.streams[last_stream_idx]);
          graph_constructor.notifyBeforeInvokingLibraryCall(
              bench_data.streams[curr_stream_idx]);
        }
        CUBLAS_CHECK(cublasSgemm(
            bench_data.cublasHs[curr_stream_idx], bench_data.transa,
            bench_data.transb, bench_spec.mm, bench_spec.nn, bench_spec.kk,
            &(bench_data.alpha),
            bench_data.d_A + i * bench_spec.mm + l * bench_spec.kk * lda, lda,
            bench_data.d_B + l * bench_spec.kk + j * bench_spec.nn * ldb, ldb,
            &(bench_data.beta),
            bench_data.d_C + l * bench_spec.mm * bench_spec.nn +
                i * bench_spec.mm + j * bench_spec.nn * ldc,
            ldc));
      }
    }
  }
  graph_constructor.notifyAfterInvokingLibraryCall(bench_data.streams.back());

  // Stream idx 0 waits for all other streams to finish before executing the
  // reduction kernel
  for (int idx = 1; idx < bench_spec.nstreams; idx++) {
    // Equivalent to error-checked cudaEventRecord
    graph_constructor.addEventRecordNode(stops_per_stream[idx],
                                         bench_data.streams[idx]);
    // Equivalent to error-checked cudaStreamWaitEvent
    graph_constructor.addStreamWaitEventNode(bench_data.streams.front(),
                                             stops_per_stream[idx]);
  }

  graph_constructor.notifyBeforeInvokingLibraryCall(bench_data.streams.front());
  // Accumulate the result
  // TODO: define BLOCK_SIZE and SHMEM_SIZE
  constexpr int BLOCK_SIZE = 256;
  constexpr int SHMEM_SIZE = 256;
  assert(bench_spec.m * bench_spec.n % SHMEM_SIZE == 0);
  dim3 nblocks(bench_spec.m * bench_spec.n / SHMEM_SIZE,
               bench_spec.k / bench_spec.kk, 1);
  dim3 nthreads(BLOCK_SIZE, 1, 1);
  reduce_segments<BLOCK_SIZE, SHMEM_SIZE, float>
      <<<nblocks, nthreads, 0, bench_data.streams.front()>>>(
          bench_data.d_C,
          bench_data.d_C +
              bench_spec.k / bench_spec.kk * bench_spec.mm * bench_spec.nn,
          bench_spec.mm, bench_spec.mm, bench_spec.nn,
          bench_spec.k / bench_spec.kk);
  graph_constructor.notifyAfterInvokingLibraryCall(bench_data.streams.back());

  if (bench_spec.enable_timing) {
    // Equivalent to error-checked cudaEventRecord
    graph_constructor.addEventRecordNode(stops_per_stream[0],
                                         bench_data.streams[0]);
  }

  if (wait_streams_on_first_and_report_that_as_elapsed_time(
          bench_spec.enable_per_stream_timing, bench_spec.enable_timing,
          bench_spec.nstreams)) {
    for (int idx = 0; idx < bench_spec.nstreams; idx++) {
      if (wait_streams_on_first_and_report_that_as_elapsed_time(
              bench_spec.enable_per_stream_timing, bench_spec.enable_timing,
              bench_spec.nstreams)) {
        // Equivalent to error-checked cudaStreamWaitEvent
        graph_constructor.addStreamWaitEventNode(bench_data.streams.front(),
                                                 stops_per_stream[idx]);
      }
      // Equivalent to error-checked cudaEventRecord
      graph_constructor.addEventRecordNode(stop, bench_data.streams.front());
    }
  }

  if (bench_spec.enable_debug_timing) {
    for (int idx = 0; idx < bench_spec.nstreams; idx++)
      CUDA_CHECK(cudaStreamSynchronize(bench_data.streams[idx]));
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::system_clock::now();
    printf("[DEBUG] cublasSgemmPartitioned chrono time (microseconds): %ld\n",
           std::chrono::duration_cast<std::chrono::microseconds>(end - beg)
               .count());
  }

  // Add start, stop pair in each stream to the return value
  if (report_elapsed_time_per_stream(bench_spec.enable_per_stream_timing,
                                     bench_spec.enable_timing,
                                     bench_spec.nstreams)) {
    timing_results.start_events = starts_per_stream;
    timing_results.stop_events = stops_per_stream;
    return;
  }

  // Synchronize on the first stream in streams vector, and add the start, stop
  // pair to the return value
  if (wait_streams_on_first_and_report_that_as_elapsed_time(
          bench_spec.enable_per_stream_timing, bench_spec.enable_timing,
          bench_spec.nstreams)) {
    timing_results.start_events = std::vector<cudaEvent_t>({start});
    timing_results.stop_events = std::vector<cudaEvent_t>({stop});
    return;
  }

  // No timing should be reported
  for (int idx = 0; idx < bench_spec.nstreams; idx++) {
    CUDA_CHECK(cudaEventDestroy(stops_per_stream[idx]));
  }
  timing_results.start_events = std::vector<cudaEvent_t>();
  timing_results.stop_events = std::vector<cudaEvent_t>();
}

void consume_and_print_timing(ProblemSpec &bench_spec,
                              TimingResults &timing_results) {
  if (wait_streams_on_first_and_report_that_as_elapsed_time(
          bench_spec.enable_per_stream_timing, bench_spec.enable_timing,
          bench_spec.nstreams)) {
    cudaEvent_t start = timing_results.start_events.front();
    cudaEvent_t stop = timing_results.stop_events.front();
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_time = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("cublasSgemmPartitioned elapsed time (ms): %f\n", elapsed_time);
    printf("cublasSgemmPartitioned throughput (GFLOPS): %f\n",
           (2.0 * bench_spec.m * bench_spec.n * bench_spec.k) /
               (elapsed_time / 1000.0) / 1e9);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }
  if (report_elapsed_time_per_stream(bench_spec.enable_per_stream_timing,
                                     bench_spec.enable_timing,
                                     bench_spec.nstreams)) {
    for (int idx = 0; idx < bench_spec.nstreams; idx++) {
      // Skip the first stream if it is already synchronized and destroyed
      if (idx == 0 && wait_streams_on_first_and_report_that_as_elapsed_time(
                          bench_spec.enable_per_stream_timing,
                          bench_spec.enable_timing, bench_spec.nstreams))
        continue;
      CUDA_CHECK(cudaEventSynchronize(timing_results.stop_events[idx]));
    }

    for (int idx = 0; idx < bench_spec.nstreams; idx++) {
      float elapsed_time = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_time,
                                      timing_results.start_events[idx],
                                      timing_results.stop_events[idx]));
      printf("cublasSgemmPartitioned elapsed time(streamIdx%d) (ms): %f\n", idx,
             elapsed_time);
      // TODO: enable throughput print
      // printf("cublasSgemmPartitioned stream %d throughput (GFLOPS): %f\n",
      // idx,
      //        (2.0 * bench_spec.m * bench_spec.n * bench_spec.k) /
      //            (elapsed_time / 1000.0) / 1e9);
      CUDA_CHECK(cudaEventDestroy(timing_results.start_events[idx]));
      CUDA_CHECK(cudaEventDestroy(timing_results.stop_events[idx]));
    }
  }

  // Print elapsed time of utilities. Keyword "elapsed time(util) (ms):"
  for (const auto &keyval : timing_results.utility_timestamps) {
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
}

void cleanUp(ProblemSpec &bench_spec, RuntimeData &bench_data) {
  int lda = bench_spec.m;
  int ldb = bench_spec.k;
  int ldc = bench_spec.m;
  if (bench_spec.enable_dump) {
    /* step 4: copy data to host */
    CUDA_CHECK(cudaMemcpyAsync(bench_data.C.data(), bench_data.d_C,
                               sizeof(data_type) * bench_data.C.size(),
                               cudaMemcpyDeviceToHost,
                               bench_data.streams.front()));
    CUDA_CHECK(cudaStreamSynchronize(bench_data.streams.front()));

    // Get current timestamp
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    char time_str[64];
    std::strftime(time_str, sizeof(time_str), "%Y-%m-%d-%H-%M", &tm);
    // We should store the string in a std::string because when the .c_str()
    // pointer is referenced, the std::string object should not be destroyed
    std::string result_path_and_prefix;
    if (!bench_spec.flag_specify_result_path_and_prefix) {
      result_path_and_prefix =
          (std::string("cublas_bench_gemm_partitioned.") + time_str).c_str();
    } else {
      result_path_and_prefix = bench_spec.cli_result_path_and_prefix;
    }
    result_path_and_prefix = nullptr;
    // Store m, n, k to a txt and store A, B, C to a numpy file
    FILE *fp = fopen((result_path_and_prefix + ".txt").c_str(), "w");
    assert(fp != nullptr);
    fprintf(fp, "%d %d %d\n", bench_spec.m, bench_spec.n, bench_spec.k);
    fclose(fp);
    unsigned long a_shape[2] = {lda, bench_spec.k};
    unsigned long b_shape[2] = {ldb, bench_spec.n};
    unsigned long c_shape[2] = {bench_spec.m, bench_spec.n};
    npy::SaveArrayAsNumpy(result_path_and_prefix + ".C.npy", false, 2, c_shape,
                          bench_data.C);
    npy::SaveArrayAsNumpy(result_path_and_prefix + ".A.npy", false, 2, a_shape,
                          bench_data.A);
    npy::SaveArrayAsNumpy(result_path_and_prefix + ".B.npy", false, 2, b_shape,
                          bench_data.B);
  }

  /* free resources */
  CUDA_CHECK(cudaFree(bench_data.d_A));
  CUDA_CHECK(cudaFree(bench_data.d_B));
  CUDA_CHECK(cudaFree(bench_data.d_C));

  for (int idx = 0; idx < bench_spec.nstreams; idx++) {
    CUBLAS_CHECK(cublasDestroy(bench_data.cublasHs[idx]));
    CUDA_CHECK(cudaStreamDestroy(bench_data.streams[idx]));
  }
  return;
}

// This function is written before GraphConstructor is implemented. It is no
// longer updated but stay here to 1) provide a reference implementation to
// facilitate debugging.
// When cuda graph is enabled, the original compute stage is now creating
// the graph, and we need a new stage that launches the graph. The rest should
// be kept the same.
void _create_graph_reference(ProblemSpec &bench_spec, RuntimeData &bench_data,
                             TimingResults &timing_results) {
  std::vector<cudaGraph_t> graphs;
  CUDA_CHECK(cudaStreamBeginCapture(bench_data.streams[0],
                                    cudaStreamCaptureModeGlobal));

  _compute_reference(bench_spec, bench_data, timing_results);

  // Only stream idx 0 needs to be captured because other stream waits on
  // the start event of stream idx 0, and stream idx 0 waits on the stop event
  // of other streams
  // Reference:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cross-stream-dependencies-and-events

  cudaGraph_t graph;
  CUDA_CHECK(cudaStreamEndCapture(bench_data.streams[0], &graph));
  graphs.push_back(graph);
  bench_data.graphs = graphs;
  return;
}

CUDAExperimentalGraphConstructor<cudaStream_t> create_graph(
    ProblemSpec &bench_spec, RuntimeData &bench_data,
    TimingResults &timing_results) {
  CUDAExperimentalGraphConstructor<cudaStream_t> graph_constructor;
  for (cudaStream_t stream : bench_data.streams) {
    graph_constructor.registerStream(stream);
  }
  compute(bench_spec, bench_data, timing_results, graph_constructor);
  bench_data.graphs.push_back(graph_constructor.getGraph());
  return graph_constructor;
}

void initiate_graph(RuntimeData &bench_data) {
  std::vector<cudaGraphExec_t> graphExecs;
  cudaGraphExec_t graphExec = NULL;
  CUDA_CHECK(
      cudaGraphInstantiate(&graphExec, bench_data.graphs[0], NULL, NULL, 0));
  graphExecs.push_back(graphExec);
  bench_data.graphExecs = graphExecs;
  return;
}

void launch_graph_and_wait(RuntimeData &bench_data) {
  CUDA_CHECK(cudaGraphLaunch(bench_data.graphExecs[0], bench_data.streams[0]));
  CUDA_CHECK(cudaStreamSynchronize(bench_data.streams[0]));
}

// This function is written before GraphConstructor is implemented. It is no
// longer updated but stay here to 1) provide a reference implementation to
// facilitate debugging.
int _main_reference(const int argc, const char **argv) {
  TimingResults timing_results;

  auto bench_tuple = generate_data_and_prepare(argc, argv, timing_results);
  auto bench_spec = std::get<0>(bench_tuple);
  auto bench_data = std::get<1>(bench_tuple);

  if (bench_spec.enable_graph) {
    std::chrono::time_point<std::chrono::system_clock> graph_creation_beg,
        graph_creation_end, graph_initialization_end, graph_execution_end;
    graph_creation_beg = std::chrono::system_clock::now();
    _create_graph_reference(bench_spec, bench_data, timing_results);

    graph_creation_end = std::chrono::system_clock::now();
    initiate_graph(bench_data);

    graph_initialization_end = std::chrono::system_clock::now();
    launch_graph_and_wait(bench_data);
    graph_execution_end = std::chrono::system_clock::now();
    printf(
        "[DEBUG] cublasSgemmPartitioned graph creation chrono time "
        "(microseconds): "
        "%ld\n",
        std::chrono::duration_cast<std::chrono::microseconds>(
            graph_creation_end - graph_creation_beg)
            .count());
    printf(
        "[DEBUG] cublasSgemmPartitioned graph initialization chrono time "
        "(microseconds): "
        "%ld\n",
        std::chrono::duration_cast<std::chrono::microseconds>(
            graph_initialization_end - graph_creation_end)
            .count());
    printf(
        "[DEBUG] cublasSgemmPartitioned graph execution chrono time "
        "(microseconds): "
        "%ld\n",
        std::chrono::duration_cast<std::chrono::microseconds>(
            graph_execution_end - graph_initialization_end)
            .count());
  } else {
    _compute_reference(bench_spec, bench_data, timing_results);
  }

  // When CUDA graph is enabled, we already wait until it finishes. Both
  // synchronize event inside the graph or measuring the elapsed time between
  // two events will trigger an error
  if (bench_spec.enable_timing && !bench_spec.enable_graph) {
    consume_and_print_timing(bench_spec, timing_results);
  }
  if (bench_spec.enable_graph) {
    CUDA_CHECK(cudaGraphExecDestroy(bench_data.graphExecs[0]));
    CUDA_CHECK(cudaGraphDestroy(bench_data.graphs[0]));
  }
  cleanUp(bench_spec, bench_data);
  return 0;
}

int main(const int argc, const char **argv) {
  TimingResults timing_results;

  auto bench_tuple = generate_data_and_prepare(argc, argv, timing_results);
  auto bench_spec = std::get<0>(bench_tuple);
  auto bench_data = std::get<1>(bench_tuple);

  if (bench_spec.enable_graph) {
    std::chrono::time_point<std::chrono::system_clock> graph_creation_beg,
        graph_creation_end, graph_initialization_end, graph_execution_end;
    graph_creation_beg = std::chrono::system_clock::now();

    auto graph_constructor =
        create_graph(bench_spec, bench_data, timing_results);

    graph_creation_end = std::chrono::system_clock::now();
    initiate_graph(bench_data);

    graph_initialization_end = std::chrono::system_clock::now();
    launch_graph_and_wait(bench_data);
    graph_execution_end = std::chrono::system_clock::now();
    printf(
        "[DEBUG] cublasSgemmPartitioned graph creation chrono time "
        "(microseconds): "
        "%ld\n",
        std::chrono::duration_cast<std::chrono::microseconds>(
            graph_creation_end - graph_creation_beg)
            .count());
    printf(
        "[DEBUG] cublasSgemmPartitioned graph initialization chrono time "
        "(microseconds): "
        "%ld\n",
        std::chrono::duration_cast<std::chrono::microseconds>(
            graph_initialization_end - graph_creation_end)
            .count());
    printf(
        "[DEBUG] cublasSgemmPartitioned graph execution chrono time "
        "(microseconds): "
        "%ld\n",
        std::chrono::duration_cast<std::chrono::microseconds>(
            graph_execution_end - graph_initialization_end)
            .count());
  } else {
    _compute_reference(bench_spec, bench_data, timing_results);
  }

  // When CUDA graph is enabled, we already wait until it finishes. Both
  // synchronize event inside the graph or measuring the elapsed time between
  // two events will trigger an error
  if (bench_spec.enable_timing && !bench_spec.enable_graph) {
    consume_and_print_timing(bench_spec, timing_results);
  }

  // Still needs to destroy graphExec though graph will be destroyed by
  // cudaGraphWrapper destructor
  if (bench_spec.enable_graph) {
    CUDA_CHECK(cudaGraphExecDestroy(bench_data.graphExecs[0]));
  }

  cleanUp(bench_spec, bench_data);
  return 0;
}
};  // namespace BenchGEMMPartitioned

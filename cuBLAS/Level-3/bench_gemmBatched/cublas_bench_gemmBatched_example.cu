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

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <utils/generate_random_data.h>
#include <utils/helper_string.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

#include "cublas_utils.h"

using data_type = float;

int main(const int argc, const char *argv[]) {
  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;

  // const int m = 2;
  // const int n = 2;
  // const int k = 2;
  // const int lda = 2;
  // const int ldb = 2;
  // const int ldc = 2;
  // const int batch_count = 2;

  // Host problem definition
  int m = getCmdLineArgumentInt(argc, argv, "m");
  int n = getCmdLineArgumentInt(argc, argv, "n");
  int k = getCmdLineArgumentInt(argc, argv, "k");
  int batch_count = getCmdLineArgumentInt(argc, argv, "batch_count");
  if (argc != 5) {
    printf("Usage: %s --m=## --n=## --k=## --batch_count=##\n", argv[0]);
    return EXIT_FAILURE;
  }
  int lda = m;
  int ldb = k;
  int ldc = m;
  /*
   *   A = | 1.0 | 2.0 | 5.0 | 6.0 |
   *       | 3.0 | 4.0 | 7.0 | 8.0 |
   *
   *   B = | 5.0 | 6.0 |  9.0 | 10.0 |
   *       | 7.0 | 8.0 | 11.0 | 12.0 |
   */

  std::srand(unsigned(std::time(nullptr)));
  std::vector<std::vector<data_type>> A_array(batch_count,
                                              std::vector<data_type>(lda * k));
  std::vector<std::vector<data_type>> B_array(batch_count,
                                              std::vector<data_type>(ldb * n));
  for (int i = 0; i < batch_count; i++) {
    std::generate(A_array[i].begin(), A_array[i].end(), std::rand);
    std::generate(B_array[i].begin(), B_array[i].end(), std::rand);
  }
  // const std::vector<std::vector<data_type>> A_array = {{1.0 ,3.0, 2.0, 4.0},
  //                                                      {5.0, 7.0, 6.0, 8.0}};
  // const std::vector<std::vector<data_type>> B_array = {{5.0, 7.0, 6.0, 8.0},
  //                                                      {9.0, 11.0, 10.0, 12.0}};
  std::vector<std::vector<data_type>> C_array(batch_count,
                                              std::vector<data_type>(m * n));

  const data_type alpha = 1.0;
  const data_type beta = 0.0;

  data_type **d_A_array = nullptr;
  data_type **d_B_array = nullptr;
  data_type **d_C_array = nullptr;

  std::vector<data_type *> d_A(batch_count, nullptr);
  std::vector<data_type *> d_B(batch_count, nullptr);
  std::vector<data_type *> d_C(batch_count, nullptr);

  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;

  if (0) {
    printf("A[0]\n");
    print_matrix(m, k, A_array[0].data(), lda);
    printf("=====\n");

    printf("A[1]\n");
    print_matrix(m, k, A_array[1].data(), lda);
    printf("=====\n");

    printf("B[0]\n");
    print_matrix(k, n, B_array[0].data(), ldb);
    printf("=====\n");

    printf("B[1]\n");
    print_matrix(k, n, B_array[1].data(), ldb);
    printf("=====\n");
  }

  /* step 1: create cublas handle, bind a stream */
  CUBLAS_CHECK(cublasCreate(&cublasH));

  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasSetStream(cublasH, stream));

  /* step 2: copy data to device */
  for (int i = 0; i < batch_count; i++) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A[i]),
                          sizeof(data_type) * A_array[i].size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B[i]),
                          sizeof(data_type) * B_array[i].size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C[i]),
                          sizeof(data_type) * C_array[i].size()));
  }

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A_array),
                        sizeof(data_type *) * batch_count));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B_array),
                        sizeof(data_type *) * batch_count));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C_array),
                        sizeof(data_type *) * batch_count));

  for (int i = 0; i < batch_count; i++) {
    CUDA_CHECK(cudaMemcpyAsync(d_A[i], A_array[i].data(),
                               sizeof(data_type) * A_array[i].size(),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B[i], B_array[i].data(),
                               sizeof(data_type) * B_array[i].size(),
                               cudaMemcpyHostToDevice, stream));
  }

  CUDA_CHECK(cudaMemcpyAsync(d_A_array, d_A.data(),
                             sizeof(data_type *) * batch_count,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_B_array, d_B.data(),
                             sizeof(data_type *) * batch_count,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_C_array, d_C.data(),
                             sizeof(data_type *) * batch_count,
                             cudaMemcpyHostToDevice, stream));

  /* step 3: compute */
  // We nest the cuda event timing with std::chrono to make sure the cuda event
  // is getting correct results, we will use the cuda event timing results and
  // ignore the std::chrono results
  std::chrono::time_point<std::chrono::system_clock> beg, end;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaDeviceSynchronize());

  beg = std::chrono::system_clock::now();
  CUDA_CHECK(cudaEventRecord(start, stream));
  CUBLAS_CHECK(cublasSgemmBatched(cublasH, transa, transb, m, n, k, &alpha,
                                  d_A_array, lda, d_B_array, ldb, &beta,
                                  d_C_array, ldc, batch_count));
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaDeviceSynchronize());
  end = std::chrono::system_clock::now();
  float elapsed_time = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

  printf("cublas<X>gemmBatched elapsed time (ms): %f\n", elapsed_time);
  printf("throughput (GFLOPS): %f\n",
         (2.0 * m * n * k * batch_count) / (elapsed_time / 1000.0) / 1e9);
  printf("[DEBUG] cublas<X>gemmBatched chrono time (microseconds): %ld\n",
         std::chrono::duration_cast<std::chrono::microseconds>(end - beg)
             .count());

  /* step 4: copy data to host */
  for (int i = 0; i < batch_count; i++) {
    CUDA_CHECK(cudaMemcpyAsync(C_array[i].data(), d_C[i],
                               sizeof(data_type) * C_array[i].size(),
                               cudaMemcpyDeviceToHost, stream));
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));

  /*
   *   C = | 19.0 | 22.0 | 111.0 | 122.0 |
   *       | 43.0 | 50.0 | 151.0 | 166.0 |
   */

  if (0) {
    printf("C[0]\n");
    print_matrix(m, n, C_array[0].data(), ldc);
    printf("=====\n");

    printf("C[1]\n");
    print_matrix(m, n, C_array[1].data(), ldc);
    printf("=====\n");
  }

  /* free resources */
  CUDA_CHECK(cudaFree(d_A_array));
  CUDA_CHECK(cudaFree(d_B_array));
  CUDA_CHECK(cudaFree(d_C_array));
  for (int i = 0; i < batch_count; i++) {
    CUDA_CHECK(cudaFree(d_A[i]));
    CUDA_CHECK(cudaFree(d_B[i]));
    CUDA_CHECK(cudaFree(d_C[i]));
  }

  CUBLAS_CHECK(cublasDestroy(cublasH));

  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaDeviceReset());

  return EXIT_SUCCESS;
}

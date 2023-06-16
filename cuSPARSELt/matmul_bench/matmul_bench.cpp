/*
 * Copyright 1993-2023 NVIDIA Corporation.  All rights reserved.
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
#include <cuda_runtime_api.h>  // cudaMalloc, cudaMemcpy, etc.
#include <cusparseLt.h>        // cusparseLt header
#include <utils/helper_string.h>

#include <cstdio>   // printf
#include <cstdlib>  // std::rand

#define CHECK_CUDA(func)                                                   \
  {                                                                        \
    cudaError_t status = (func);                                           \
    if (status != cudaSuccess) {                                           \
      printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, \
             cudaGetErrorString(status), status);                          \
      return EXIT_FAILURE;                                                 \
    }                                                                      \
  }

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, \
             cusparseGetErrorString(status), status);                          \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  }

constexpr int EXIT_UNSUPPORTED = 2;

int main(const int argc, const char** argv) {
  int major_cc, minor_cc;
  CHECK_CUDA(
      cudaDeviceGetAttribute(&major_cc, cudaDevAttrComputeCapabilityMajor, 0))
  CHECK_CUDA(
      cudaDeviceGetAttribute(&minor_cc, cudaDevAttrComputeCapabilityMinor, 0))
  if (!(major_cc == 8 && minor_cc == 0) && !(major_cc == 8 && minor_cc == 6) &&
      !(major_cc == 8 && minor_cc == 9)) {
    std::printf(
        "\ncusparseLt is supported only on GPU devices with"
        " compute capability == 8.0, 8.6, 8.9 current: %d.%d\n\n",
        major_cc, minor_cc);
    return EXIT_UNSUPPORTED;
  }
  // CLI Input
  int m = getCmdLineArgumentInt(argc, argv, "m");
  int n = getCmdLineArgumentInt(argc, argv, "n");
  int k = getCmdLineArgumentInt(argc, argv, "k");
  bool tune_flag = checkCmdLineFlag(argc, argv, "tune");
  if (argc < 4) {
    printf("Usage: %s --m=## --n=## --k=## [--tune]\n", argv[0]);
    return EXIT_FAILURE;
  }
  printf("m: %d\n", m);
  printf("n: %d\n", n);
  printf("k: %d\n", k);
  printf("tune_flag: %d\n", tune_flag);
  // ***** END OF CLI Input *****
  // Host problem definition, row-major order
  // bigger sizes may require dynamic allocations
  // constexpr int m            = 32;
  // constexpr int n            = 32;
  // constexpr int k            = 32;
  auto order = CUSPARSE_ORDER_ROW;
  auto opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto type = CUDA_R_16F;
  auto compute_type = CUSPARSE_COMPUTE_16F;

  bool is_rowmajor = (order == CUSPARSE_ORDER_ROW);
  bool isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
  bool isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
  auto num_A_rows = (isA_transposed) ? k : m;
  auto num_A_cols = (isA_transposed) ? m : k;
  auto num_B_rows = (isB_transposed) ? n : k;
  auto num_B_cols = (isB_transposed) ? k : n;
  auto num_C_rows = m;
  auto num_C_cols = n;
  unsigned alignment = 16;
  auto lda = (is_rowmajor) ? num_A_cols : num_A_rows;
  auto ldb = (is_rowmajor) ? num_B_cols : num_B_rows;
  auto ldc = (is_rowmajor) ? num_C_cols : num_C_rows;
  auto A_height = (is_rowmajor) ? num_A_rows : num_A_cols;
  auto B_height = (is_rowmajor) ? num_B_rows : num_B_cols;
  auto C_height = (is_rowmajor) ? num_C_rows : num_C_cols;
  auto A_size = A_height * lda * sizeof(__half);
  auto B_size = B_height * ldb * sizeof(__half);
  auto C_size = C_height * ldc * sizeof(__half);

  __half* hA = (__half*)malloc(m * k * sizeof(__half));
  __half* hB = (__half*)malloc(k * n * sizeof(__half));
  __half* hC = (__half*)malloc(m * n * sizeof(__half));
  for (int i = 0; i < m * k; i++)
    hA[i] = static_cast<__half>(static_cast<float>(std::rand() % 10));
  for (int i = 0; i < k * n; i++)
    hB[i] = static_cast<__half>(static_cast<float>(std::rand() % 10));
  float alpha = 1.0f;
  float beta = 0.0f;
  //--------------------------------------------------------------------------
  // Device memory management
  __half *dA, *dB, *dC, *dD, *dA_compressed;
  int* d_valid;
  CHECK_CUDA(cudaMalloc((void**)&dA, A_size))
  CHECK_CUDA(cudaMalloc((void**)&dB, B_size))
  CHECK_CUDA(cudaMalloc((void**)&dC, C_size))
  CHECK_CUDA(cudaMalloc((void**)&d_valid, sizeof(int)))
  dD = dC;

  CHECK_CUDA(cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice))

  cudaEvent_t before_handle_creation, after_handle_creation, after_pruning,
      after_verification, after_compression, after_tuning, after_execution,
      after_destruction, after_workspace_alloc;
  CHECK_CUDA(cudaEventCreate(&before_handle_creation))
  CHECK_CUDA(cudaEventCreate(&after_handle_creation))
  CHECK_CUDA(cudaEventCreate(&after_pruning))
  CHECK_CUDA(cudaEventCreate(&after_verification))
  CHECK_CUDA(cudaEventCreate(&after_compression))
  CHECK_CUDA(cudaEventCreate(&after_tuning))
  CHECK_CUDA(cudaEventCreate(&after_workspace_alloc))
  CHECK_CUDA(cudaEventCreate(&after_execution))
  CHECK_CUDA(cudaEventCreate(&after_destruction))

  //--------------------------------------------------------------------------
  cusparseLtHandle_t handle;
  cusparseLtMatDescriptor_t matA, matB, matC;
  cusparseLtMatmulDescriptor_t matmul;
  cusparseLtMatmulAlgSelection_t alg_sel;
  cusparseLtMatmulPlan_t plan;
  cudaStream_t stream = nullptr;

  CHECK_CUDA(cudaDeviceSynchronize())
  CHECK_CUDA(cudaEventRecord(before_handle_creation))
  CHECK_CUSPARSE(cusparseLtInit(&handle))
  // matrix descriptor initialization
  CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
      &handle, &matA, num_A_rows, num_A_cols, lda, alignment, type, order,
      CUSPARSELT_SPARSITY_50_PERCENT))
  CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
      &handle, &matB, num_B_rows, num_B_cols, ldb, alignment, type, order))
  CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
      &handle, &matC, num_C_rows, num_C_cols, ldc, alignment, type, order))
  // matmul, algorithm selection, and plan initialization
  CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(
      &handle, &matmul, opA, opB, &matA, &matB, &matC, &matC, compute_type))
  CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(
      &handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT))
  CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel))

  CHECK_CUDA(cudaEventRecord(after_handle_creation))

  //--------------------------------------------------------------------------
  // Prune the A matrix (in-place) and check the correctness
  CHECK_CUSPARSE(cusparseLtSpMMAPrune(&handle, &matmul, dA, dA,
                                      CUSPARSELT_PRUNE_SPMMA_TILE, stream))

  CHECK_CUDA(cudaEventRecord(after_pruning))
  CHECK_CUSPARSE(
      cusparseLtSpMMAPruneCheck(&handle, &matmul, dA, d_valid, stream))
  int is_valid;
  CHECK_CUDA(cudaMemcpyAsync(&is_valid, d_valid, sizeof(int),
                             cudaMemcpyDeviceToHost, stream))
  CHECK_CUDA(cudaStreamSynchronize(stream))
  if (is_valid != 0) {
    std::printf(
        "!!!! The matrix has been pruned in a wrong way. "
        "cusparseLtMatmul will not provide correct results\n");
    return EXIT_FAILURE;
  }

  CHECK_CUDA(cudaEventRecord(after_verification))

  //--------------------------------------------------------------------------
  // Compress the A matrix
  size_t compressed_size, compressed_buffer_size;
  void* dA_compressedBuffer;
  CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size,
                                               &compressed_buffer_size))
  CHECK_CUDA(cudaMalloc((void**)&dA_compressed, compressed_size))
  CHECK_CUDA(cudaMalloc((void**)&dA_compressedBuffer, compressed_buffer_size))

  CHECK_CUSPARSE(cusparseLtSpMMACompress(&handle, &plan, dA, dA_compressed,
                                         dA_compressedBuffer, stream))
  CHECK_CUDA(cudaEventRecord(after_compression))
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Search the best kernel
  int num_streams = 0;
  cudaStream_t* streams = nullptr;
  if (tune_flag) {
    CHECK_CUSPARSE(cusparseLtMatmulSearch(&handle, &plan, &alpha, dA_compressed,
                                          dB, &beta, dC, dD, nullptr, streams,
                                          num_streams))
  } else {
    // otherwise, it is possible to set it directly:
    int alg = 0;
    CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(
        &handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg, sizeof(alg)))
  }
  CHECK_CUDA(cudaEventRecord(after_tuning))
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  size_t workspace_size;
  CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel))

  CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size))
  void* d_workspace;
  CHECK_CUDA(cudaMalloc((void**)&d_workspace, workspace_size))

  CHECK_CUDA(cudaEventRecord(after_workspace_alloc))

  // Perform the matrix multiplication
  CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB,
                                  &beta, dC, dD, d_workspace, streams,
                                  num_streams))

  CHECK_CUDA(cudaEventRecord(after_execution))
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // destroy plan and handle
  CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matA))
  CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matB))
  CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matC))
  CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&plan))
  CHECK_CUSPARSE(cusparseLtDestroy(&handle))

  CHECK_CUDA(cudaEventRecord(after_destruction))
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  CHECK_CUDA(cudaDeviceSynchronize())

  float time_handle_creation, time_pruning, time_verification, time_compression,
      time_tuning, time_workspace_alloc, time_execution, time_destruction;

  CHECK_CUDA(cudaEventElapsedTime(&time_handle_creation, before_handle_creation,
                                  after_handle_creation))
  CHECK_CUDA(
      cudaEventElapsedTime(&time_pruning, after_handle_creation, after_pruning))

  CHECK_CUDA(cudaEventElapsedTime(&time_verification, after_pruning,
                                  after_verification))
  CHECK_CUDA(cudaEventElapsedTime(&time_compression, after_verification,
                                  after_compression))
  CHECK_CUDA(
      cudaEventElapsedTime(&time_tuning, after_compression, after_tuning))
  CHECK_CUDA(cudaEventElapsedTime(&time_workspace_alloc, after_tuning,
                                  after_workspace_alloc))
  CHECK_CUDA(cudaEventElapsedTime(&time_execution, after_workspace_alloc,
                                  after_execution))
  CHECK_CUDA(cudaEventElapsedTime(&time_destruction, after_execution,
                                  after_destruction))

  printf("cusparseLtMatmul time breakdown:\n");
  printf("  handle creation: %f ms\n", time_handle_creation);
  printf("  pruning: %f ms\n", time_pruning);
  printf("  verification: %f ms\n", time_verification);
  printf("  compression: %f ms\n", time_compression);
  printf("  tuning: %f ms\n", time_tuning);

  printf("  workspace allocation: %f ms\n", time_workspace_alloc);

  printf("  execution: %f ms\n", time_execution);
  printf("  destruction: %f ms\n", time_destruction);
  printf("  total: %f ms\n", time_handle_creation + time_pruning +
                                 time_verification + time_compression +
                                 time_tuning + time_execution +
                                 time_destruction);
  //--------------------------------------------------------------------------
  // device result check
  // matrix A has been pruned
  CHECK_CUDA(cudaMemcpy(hA, dA, A_size, cudaMemcpyDeviceToHost))
  CHECK_CUDA(cudaMemcpy(hC, dC, C_size, cudaMemcpyDeviceToHost))

  bool A_std_layout = (is_rowmajor != isA_transposed);
  bool B_std_layout = (is_rowmajor != isB_transposed);
  // host computation
  float hC_result[m * n];
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0f;
      for (int k1 = 0; k1 < k; k1++) {
        auto posA = (A_std_layout) ? i * lda + k1 : i + k1 * lda;
        auto posB = (B_std_layout) ? k1 * ldb + j : k1 + j * ldb;
        sum += static_cast<float>(hA[posA]) *  // [i][k]
               static_cast<float>(hB[posB]);   // [k][j]
      }
      auto posC = (is_rowmajor) ? i * ldc + j : i + j * ldc;
      hC_result[posC] = sum;  // [i][j]
    }
  }
  // host-device comparison
  int correct = 1;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      auto pos = (is_rowmajor) ? i * ldc + j : i + j * ldc;
      auto device_value = static_cast<float>(hC[pos]);
      auto host_value = hC_result[pos];
      if (device_value != host_value) {
        // direct floating point comparison is not reliable
        std::printf("(%d, %d):\t%f vs. %f\n", i, j, host_value, device_value);
        correct = 0;
        break;
      }
    }
  }
  if (correct)
    std::printf("matmul_example test PASSED\n");
  else
    std::printf("matmul_example test FAILED: wrong result\n");
  //--------------------------------------------------------------------------
  // device memory deallocation
  CHECK_CUDA(cudaFree(dA_compressed))
  CHECK_CUDA(cudaFree(dA))
  CHECK_CUDA(cudaFree(dB))
  CHECK_CUDA(cudaFree(dC))
  CHECK_CUDA(cudaFree(d_valid))
  CHECK_CUDA(cudaFree(d_workspace))
  CHECK_CUDA(cudaFree(dA_compressedBuffer))

  // host memory deallocation
  free(hA);
  free(hB);
  free(hC);
  return EXIT_SUCCESS;
}

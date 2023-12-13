/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
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
#include <cusp/csr_matrix.h>   // cusp::csr_matrix
#include <cusparse.h>          // cusparseSpMM
#include <stdio.h>             // printf
#include <stdlib.h>            // EXIT_FAILURE
#include <utils/generate_random_data.h>
#include <utils/helper_string.h>

#include <chrono>

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

int main(const int argc, const char **argv) {
  // Host problem definition
  // int A_num_rows = 4;
  // int A_num_cols = 4;
  // int B_num_cols = 3;
  // int C_nnz = 9;
  // int num_batches = 2;

  int A_num_rows = getCmdLineArgumentInt(argc, argv, "A_num_rows");
  int A_num_cols = getCmdLineArgumentInt(argc, argv, "A_num_cols");
  int B_num_cols = getCmdLineArgumentInt(argc, argv, "B_num_cols");
  float C_density = getCmdLineArgumentFloat(argc, argv, "C_density");
  int num_batches = getCmdLineArgumentInt(argc, argv, "num_batches");
  bool enable_preprocess =
      getCmdLineArgumentInt(argc, argv, "enable_preprocess");

  if (A_num_rows == 0 || A_num_cols == 0 || B_num_cols == 0 || C_density == 0 ||
      num_batches == 0) {
    printf(
        "Usage: %s --A_num_rows=## --A_num_cols=## --B_num_cols=## "
        "--C_density=0.## --num_batches=## [--enable_preprocess]\n",
        argv[0]);
    return EXIT_FAILURE;
  }
  printf("A_num_rows: %d\n", A_num_rows);
  printf("A_num_cols: %d\n", A_num_cols);
  printf("B_num_cols: %d\n", B_num_cols);
  printf("C_density: %f\n", C_density);
  printf("num_batches: %d\n", num_batches);
  // ***** END OF HOST PROBLEM DEFINITION *****

  int B_num_rows = A_num_cols;
  int lda = A_num_rows;
  int ldb = A_num_cols;
  int A_size = lda * A_num_cols;
  int B_size = ldb * B_num_cols;
  int C_nnz = (int)(C_density * A_num_rows * B_num_cols);

  //   float hA1[] = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
  //                  9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  //   float hA2[] = {10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f,
  //                  18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f};
  //   float hB1[] = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
  //                  7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  //   float hB2[] = {6.0f, 4.0f, 2.0f, 3.0f, 7.0f, 1.0f,
  //                  9.0f, 5.0f, 2.0f, 8.0f, 4.0f, 7.0f};
  //   int hC_offsets[] = {0, 3, 4, 7, 9};
  //   int hC_columns1[] = {0, 1, 2, 1, 0, 1, 2, 0, 2};
  //   int hC_columns2[] = {0, 1, 2, 0, 0, 1, 2, 1, 2};
  //   float hC_values1[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  //   0.0f}; float hC_values2[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  //   0.0f, 0.0f}; float hC_result1[] = {70.0f,  80.0f,  90.0f,  184.0f,
  //   246.0f,
  //                         288.0f, 330.0f, 334.0f, 450.0f};
  //   float hC_result2[] = {305.0f, 229.0f, 146.0f, 409.0f, 513.0f,
  //                         389.0f, 242.0f, 469.0f, 290.0f};

  float *hA = (float *)malloc(A_size * sizeof(float));
  generate_random_matrix(hA, A_size);
  float *hB = (float *)malloc(B_size * sizeof(float));
  generate_random_matrix(hB, B_size);
  cusp::csr_matrix<int, float, cusp::host_memory> hC =
      generate_random_sparse_matrix<
          cusp::csr_matrix<int, float, cusp::host_memory>>(A_num_rows,
                                                           B_num_cols, C_nnz);
  C_nnz = hC.values.size();
  printf(
      "actual C_nnz due to deduplication during random data generation: %d\n",
      C_nnz);

  float alpha = 1.0f;
  float beta = 0.0f;
  //--------------------------------------------------------------------------
  // TODO: remove dC since it is not used
  cusp::csr_matrix<int, float, cusp::device_memory> dC(hC);
  // Device memory management
  int *dC_offsets, *dC_columns;
  float *dC_values, *dB, *dA;
  CHECK_CUDA(cudaMalloc((void **)&dA, A_size * num_batches * sizeof(float)))
  CHECK_CUDA(cudaMalloc((void **)&dB, B_size * num_batches * sizeof(float)))
  CHECK_CUDA(cudaMalloc((void **)&dC_offsets,
                        (A_num_rows + 1) * sizeof(int) * num_batches))
  CHECK_CUDA(cudaMalloc((void **)&dC_columns,
                        C_nnz * num_batches * sizeof(int) * num_batches))
  CHECK_CUDA(cudaMalloc((void **)&dC_values,
                        C_nnz * num_batches * sizeof(float) * num_batches))

  for (int idx = 0; idx < num_batches; idx++) {
    CHECK_CUDA(cudaMemcpy(dC_offsets + idx * (A_num_rows + 1),
                          hC.row_offsets.data(), (A_num_rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dC_columns + idx * C_nnz, hC.column_indices.data(),
                          C_nnz * sizeof(int), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dC_values + idx * C_nnz, hC.values.data(),
                          C_nnz * sizeof(float), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dA + idx * A_size, hA, A_size * sizeof(float),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dB + idx * B_size, hB, B_size * sizeof(float),
                          cudaMemcpyHostToDevice))
  }

  //   CHECK_CUDA(
  //       cudaMemcpy(dA, hA1, A_size * sizeof(float), cudaMemcpyHostToDevice))
  //   CHECK_CUDA(cudaMemcpy(dA + A_size, hA2, A_size * sizeof(float),
  //                         cudaMemcpyHostToDevice))
  //   CHECK_CUDA(
  //       cudaMemcpy(dB, hB1, B_size * sizeof(float), cudaMemcpyHostToDevice))
  //   CHECK_CUDA(cudaMemcpy(dB + B_size, hB2, B_size * sizeof(float),
  //                         cudaMemcpyHostToDevice))
  //   CHECK_CUDA(cudaMemcpy(dC_offsets, hC_offsets, (A_num_rows + 1) *
  //   sizeof(int),
  //                         cudaMemcpyHostToDevice))
  //   CHECK_CUDA(cudaMemcpy(dC_columns, hC_columns1, C_nnz * sizeof(int),
  //                         cudaMemcpyHostToDevice))
  //   CHECK_CUDA(cudaMemcpy(dC_columns + C_nnz, hC_columns2, C_nnz *
  //   sizeof(int),
  //                         cudaMemcpyHostToDevice))
  //   CHECK_CUDA(cudaMemcpy(dC_values, hC_values1, C_nnz * sizeof(float),
  //                         cudaMemcpyHostToDevice))
  //   CHECK_CUDA(cudaMemcpy(dC_values + C_nnz, hC_values2, C_nnz *
  //   sizeof(float),
  //                         cudaMemcpyHostToDevice))
  //--------------------------------------------------------------------------
  // CUSPARSE APIs
  cusparseHandle_t handle = NULL;
  cusparseDnMatDescr_t matA, matB;
  cusparseSpMatDescr_t matC;
  void *dBuffer = NULL;
  size_t bufferSize = 0;
  CHECK_CUSPARSE(cusparseCreate(&handle))
  // Create dense matrix A
  CHECK_CUSPARSE(cusparseCreateDnMat(&matA, A_num_rows, A_num_cols, lda, dA,
                                     CUDA_R_32F, CUSPARSE_ORDER_ROW))
  CHECK_CUSPARSE(cusparseDnMatSetStridedBatch(matA, num_batches, A_size))
  // Create dense matrix B
  CHECK_CUSPARSE(cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                     CUDA_R_32F, CUSPARSE_ORDER_ROW))
  CHECK_CUSPARSE(cusparseDnMatSetStridedBatch(matB, num_batches, B_size))
  // Create sparse matrix C in CSR format
  CHECK_CUSPARSE(cusparseCreateCsr(&matC, A_num_rows, B_num_cols, C_nnz,
                                   dC_offsets, dC_columns, dC_values,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
  CHECK_CUSPARSE(cusparseCsrSetStridedBatch(matC, num_batches, 0, C_nnz))
  // allocate an external buffer if needed
  CHECK_CUSPARSE(cusparseSDDMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize))
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

  // TODO: add option to control if preprocess is enabled
  // execute preprocess (optional)
  if (enable_preprocess) {
    CHECK_CUSPARSE(cusparseSDDMM_preprocess(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer))
  }
  // execute SpMM
  // We nest the cuda event timing with std::chrono to make sure the cuda event
  // is getting correct results, we will use the cuda event timing results and
  // ignore the std::chrono results
  std::chrono::time_point<std::chrono::system_clock> beg, end;
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaDeviceSynchronize());

  beg = std::chrono::system_clock::now();
  CHECK_CUDA(cudaEventRecord(start));

  CHECK_CUSPARSE(cusparseSDDMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                               matB, &beta, matC, CUDA_R_32F,
                               CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer))
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaDeviceSynchronize());
  end = std::chrono::system_clock::now();
  float elapsed_time = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
  printf("cusparseSDDMBatched+CSR elapsed time (ms): %f\n", elapsed_time);
  printf("cusparseSDDMBatched+CSR throughput (GFLOPS): %f\n",
         (2.0 * A_num_rows * B_num_cols * A_num_cols * num_batches) /
             (elapsed_time / 1000.0) / 1e9);
  printf(
      "[DEBUG] cusparseSDDMM chrono time (microseconds): %ld\n",
      std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count());

  // destroy matrix/vector descriptors
  CHECK_CUSPARSE(cusparseDestroyDnMat(matA))
  CHECK_CUSPARSE(cusparseDestroyDnMat(matB))
  CHECK_CUSPARSE(cusparseDestroySpMat(matC))
  CHECK_CUSPARSE(cusparseDestroy(handle))
  //--------------------------------------------------------------------------
  // device result check
  //   CHECK_CUDA(cudaMemcpy(hC_values1, dC_values, C_nnz * sizeof(float),
  //                         cudaMemcpyDeviceToHost))
  //   CHECK_CUDA(cudaMemcpy(hC_values2, dC_values + C_nnz, C_nnz *
  //   sizeof(float),
  //                         cudaMemcpyDeviceToHost))
  //   int correct = 1;
  //   for (int i = 0; i < C_nnz; i++) {
  //     if (hC_values1[i] != hC_result1[i]) {
  //       correct = 0;  // direct floating point comparison is not reliable
  //       break;
  //     }
  //     if (hC_values2[i] != hC_result2[i]) {
  //       correct = 0;  // direct floating point comparison is not reliable
  //       break;
  //     }
  //   }
  //   if (correct)
  //     printf("sddmm_csr_batched_example test PASSED\n");
  //   else
  //     printf("sddmm_csr_batched_example test FAILED: wrong result\n");
  //--------------------------------------------------------------------------
  // device memory deallocation
  CHECK_CUDA(cudaFree(dBuffer))
  CHECK_CUDA(cudaFree(dA))
  CHECK_CUDA(cudaFree(dB))
  CHECK_CUDA(cudaFree(dC_offsets))
  CHECK_CUDA(cudaFree(dC_columns))
  CHECK_CUDA(cudaFree(dC_values))
  return EXIT_SUCCESS;
}

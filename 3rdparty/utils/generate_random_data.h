#pragma once
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/gallery/random.h>

template <typename MatrixType>
MatrixType generate_random_sparse_matrix(int num_rows, int num_cols,
                                         int num_nonzeros) {
  MatrixType A(num_rows, num_cols, num_nonzeros);
  cusp::gallery::random(A, num_rows, num_cols, num_nonzeros);
  // Generate random values
  for (int i = 0; i < num_nonzeros; i++) {
    A.values[i] = (rand() + 0.0f) / RAND_MAX;
  }
  return A;
}

// This function generates a random sparse matrix with no duplicate entries.
// It is based on cusp::gallery::random from 3rdparty/CUDALibrarySamples/3rdparty/cusplibrary/cusp/gallery/detail/random.inl
template <typename MatrixType>
void random_nodup(MatrixType& matrix,
            const size_t m,
            const size_t n,
            const size_t num_samples)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;

    cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> coo(m, n, num_samples);

    std::vector<size_t> coordinates;
    coordinates.reserve(num_samples);
    for(size_t k = 0; k < num_samples; k++)
    {
        coordinates.push_back(k);
    }
    std::random_shuffle(coordinates.begin(), coordinates.end());

    for(size_t k = 0; k < num_samples; k++)
    {
        coo.row_indices[k]    = coordinates[k] / n;
        coo.column_indices[k] = coordinates[k] % n;
        coo.values[k]         = ValueType(1);
    }

    // sort indices by (row,column)
    coo.sort_by_row_and_column();

    matrix = coo;
}

// This function generates a random sparse matrix with no duplicate entries.
// It is based on cusp::gallery::random from 3rdparty/CUDALibrarySamples/3rdparty/cusplibrary/cusp/gallery/detail/random.inl
template <typename MatrixType>
MatrixType generate_random_sparse_matrix_nodup(int num_rows, int num_cols,
                                         int num_nonzeros) {
  MatrixType A(num_rows, num_cols, num_nonzeros);
  random_nodup(A, num_rows, num_cols, num_nonzeros);
  // Generate random values
  for (int i = 0; i < num_nonzeros; i++) {
    A.values[i] = (rand() + 0.0f) / RAND_MAX;
  }
  return A;
}

void generate_random_matrix(float *data, int len) {
  for (int i = 0; i < len; i++) {
    data[i] = (rand() + 0.0f) / RAND_MAX;
  }
}
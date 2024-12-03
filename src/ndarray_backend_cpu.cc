#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};

struct COOMatrix {
  size_t nnz;             // number of non-zero elements
  size_t rows;            // number of rows
  size_t cols;            // number of columns
  scalar_t* data;         // non-zero values
  int32_t* row_indices;   // row indices
  int32_t* col_indices;   // column indices

  COOMatrix(size_t nnz, size_t rows, size_t cols)
      : nnz(nnz), rows(rows), cols(cols) {
    int ret1 = posix_memalign((void**)&data, ALIGNMENT, nnz * sizeof(scalar_t));
    int ret2 = posix_memalign((void**)&row_indices, ALIGNMENT, nnz * sizeof(int32_t));
    int ret3 = posix_memalign((void**)&col_indices, ALIGNMENT, nnz * sizeof(int32_t));

    if (ret1 != 0 || ret2 != 0 || ret3 != 0) {
      // std::cout << "Could not allocate memory for COO matrix" << std::endl;
      throw std::bad_alloc();
    }
    // std::cout << "Allocated memory for COO matrix" << std::endl;
  }

  ~COOMatrix() {
    // std::cout << "Freeing data for COO matrix" << std::endl;
    free(data);
    // std::cout << "Freeing row_i for COO matrix" << std::endl;
    free(row_indices);
    // std::cout << "Freeing col_i for COO matrix" << std::endl;
    free(col_indices);
  }

  COOMatrix(const COOMatrix&) = delete;
  COOMatrix& operator=(const COOMatrix&) = delete;
};

std::unique_ptr<COOMatrix> DenseToSparse(const AlignedArray& dense_matrix, size_t rows, size_t cols) {
  // Count non-zero elements
  size_t nnz = 0;
  for (size_t i = 0; i < dense_matrix.size; ++i) {
    if (dense_matrix.ptr[i] != 0) {
      ++nnz;
    }
  }

  // Initialize the COO matrix
  std::unique_ptr<COOMatrix> sparse_matrix(new COOMatrix(nnz, rows, cols));
  // Fill the COO matrix with non-zero elements and their indices
  size_t idx = 0;
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      scalar_t value = dense_matrix.ptr[row * cols + col];
      if (value != 0) {
        sparse_matrix->data[idx] = value;
        sparse_matrix->row_indices[idx] = row;
        sparse_matrix->col_indices[idx] = col;
        ++idx;
      }
    }
  }

  return sparse_matrix;
}

std::unique_ptr<AlignedArray> SparseToDense(const COOMatrix& sparse_matrix) {
  // Create a new dense matrix
  std::unique_ptr<AlignedArray> dense_matrix(new AlignedArray(sparse_matrix.rows * sparse_matrix.cols));

  // Initialize the dense matrix to zero
  for (size_t i = 0; i < dense_matrix->size; ++i) {
    dense_matrix->ptr[i] = 0;
  }

  // Populate the dense matrix with non-zero values
  for (size_t idx = 0; idx < sparse_matrix.nnz; ++idx) {
    int32_t row = sparse_matrix.row_indices[idx];
    int32_t col = sparse_matrix.col_indices[idx];
    scalar_t value = sparse_matrix.data[idx];
    dense_matrix->ptr[row * sparse_matrix.cols + col] = value;
  }

  return dense_matrix;
}

void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}



void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  /// BEGIN SOLUTION
  size_t ndim = shape.size();
  std::vector<int32_t> indices(ndim, 0);

  // Calculate the total number of elements in the compact array.
  size_t total_elements = 1;
  for (size_t dim = 0; dim < ndim; ++dim) {
    total_elements *= shape[dim];
  }

  for (size_t i = 0; i < total_elements; ++i) {
    // Calculate the offset in `a` based on `strides` and `indices`.
    size_t a_offset = offset;
    for (size_t dim = 0; dim < ndim; ++dim) {
      a_offset += indices[dim] * strides[dim];
    }

    // Write the element from `a` to `out`.
    out->ptr[i] = a.ptr[a_offset];

    // Increment indices (for a compact layout).
    for (size_t dim = ndim; dim-- > 0; ) {
      indices[dim]++;
      if (indices[dim] < shape[dim]) break;
      indices[dim] = 0;
    }
  }
  /// END SOLUTION
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  size_t ndim = shape.size();
  std::vector<int32_t> indices(ndim, 0);
  for (size_t i = 0; i < a.size; i++){
    size_t a_offset = offset;
    for (size_t dim = 0; dim < ndim; ++dim) {
      a_offset += indices[dim] * strides[dim];
    }
    out->ptr[a_offset] = a.ptr[i];
    // Increment indices (for a compact layout).
    for (size_t dim = ndim; dim-- > 0; ) {
      indices[dim]++;
      if (indices[dim] < shape[dim]) break;
      indices[dim] = 0;
    }
  }
  /// END SOLUTION
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  /// BEGIN SOLUTION
  size_t ndim = shape.size();
  std::vector<int32_t> indices(ndim, 0);
  for (size_t i = 0; i < size; i++){
    size_t a_offset = offset;
    for (size_t dim = 0; dim < ndim; ++dim) {
      a_offset += indices[dim] * strides[dim];
    }
    out->ptr[a_offset] = val;
    // Increment indices (for a compact layout).
    for (size_t dim = ndim; dim-- > 0; ) {
      indices[dim]++;
      if (indices[dim] < shape[dim]) break;
      indices[dim] = 0;
    }
  }
  /// END SOLUTION
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}

void SparseDenseAdd(const COOMatrix& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Add a sparse matrix to a dense matrix.
   *
   * Args:
   *   a: sparse matrix
   *   b: dense matrix
   *   out: dense matrix to write to
   */
  for (size_t i = 0; i < b.size; i++) {
    out->ptr[i] = b.ptr[i];
  }
  for (size_t i = 0; i < a.nnz; i++) {
    out->ptr[a.row_indices[i] * a.cols + a.col_indices[i]] += a.data[i];
  }
}

struct PairHash {
  template <class T1, class T2>
  std::size_t operator () (const std::pair<T1, T2>& p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);
    return h1 ^ (h2 << 1);
  }
};

void SparseSparseAdd(const COOMatrix& a, const COOMatrix& b, COOMatrix* out) {
  /**
   * Add two sparse matrices.
   *
   * Args:
   *   a: first sparse matrix
   *   b: second sparse matrix
   *   out: sparse matrix to write to
   */
  // Copy the first matrix to the output matrix and keep a set of row, col indices.
  std::unordered_map<std::pair<int32_t, int32_t>, scalar_t, PairHash> out_map;
  out_map.reserve(a.nnz+b.nnz);
  for (size_t i = 0; i < a.nnz; i++) {
    out_map[std::make_pair(a.row_indices[i], a.col_indices[i])] = a.data[i];
  }
  for (size_t i = 0; i < b.nnz; i++) {
    if (out_map.find(std::make_pair(b.row_indices[i], b.col_indices[i])) != out_map.end()) {
      out_map[std::make_pair(b.row_indices[i], b.col_indices[i])] += b.data[i];
    } else {
      out_map[std::make_pair(b.row_indices[i], b.col_indices[i])] = b.data[i];
    }
  }
  // Copy the output map to the output matrix.
  out->nnz = out_map.size();
  out->rows = a.rows;
  out->cols = a.cols;
  void *data_ptr, *row_ptr, *col_ptr;
  int ret1 = posix_memalign(&data_ptr, ALIGNMENT, out->nnz * sizeof(scalar_t));
  int ret2 = posix_memalign(&row_ptr, ALIGNMENT, out->nnz * sizeof(int32_t));
  int ret3 = posix_memalign(&col_ptr, ALIGNMENT, out->nnz * sizeof(int32_t));
  if (ret1 != 0 || ret2 != 0 || ret3 != 0) throw std::bad_alloc();
  out->data = static_cast<scalar_t*>(data_ptr);
  out->row_indices = static_cast<int32_t*>(row_ptr);
  out->col_indices = static_cast<int32_t*>(col_ptr);
  size_t idx = 0;
  for (auto it = out_map.begin(); it != out_map.end(); it++) {
    out->data[idx] = it->second;
    out->row_indices[idx] = it->first.first;
    out->col_indices[idx] = it->first.second;
    idx++;
  }
}

/**
 * In the code the follows, use the above template to create analogous element-wise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */
void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] * b.ptr[i];
  }
}

void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] * val;
  }
}

void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] / b.ptr[i];
  }
}

void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] / val;
  }
}

void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = pow(a.ptr[i], val);
  }
}

void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::max(a.ptr[i], b.ptr[i]);
  }
}

void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::max(a.ptr[i], val);
  }
}

void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = scalar_t(a.ptr[i] == b.ptr[i]);
  }
}

void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = scalar_t(a.ptr[i] == val);
  }
}

void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = scalar_t(a.ptr[i] >= b.ptr[i]);
  }
}

void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = scalar_t(a.ptr[i] >= val);
  }
}

void EwiseLog(const AlignedArray& a, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = log(a.ptr[i]);
  }
}

void EwiseExp(const AlignedArray& a, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = exp(a.ptr[i]);
  }
}

void EwiseTanh(const AlignedArray& a, AlignedArray* out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = tanh(a.ptr[i]);
  }
}



void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) { 
      out->ptr[i * p + j] = 0;
      for (size_t k = 0; k < n; k++) {
        out->ptr[i * p + j] += a.ptr[i * n + k] * b.ptr[k * p + j];
      }
    }
  }
  /// END SOLUTION
}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN SOLUTION
  for (size_t i = 0; i < TILE; i++) {
    for (size_t j = 0; j < TILE; j++) {
      for (size_t k = 0; k < TILE; k++) {
        out[i * TILE + j] += a[i * TILE + k] * b[k * TILE + j];
      }
    }
  }
  /// END SOLUTION
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
  /// BEGIN SOLUTION
  // Calculate the number of tiles in each dimension
  uint32_t tile_m = m / TILE;
  uint32_t tile_n = n / TILE;
  uint32_t tile_p = p / TILE;

  // Iterate over the tiles in the output matrix
  for (int i = 0; i < m * p; i++){
    out->ptr[i] = 0;
  }
  for (int i = 0; i < tile_m; i++) {
    for (int j = 0; j < tile_p; j++) {
      for (int k = 0; k < tile_n; k++) {
        AlignedDot(&a.ptr[i * n * TILE + k * TILE * TILE], &b.ptr[k * p * TILE + j * TILE * TILE], &out->ptr[i * p * TILE + j * TILE * TILE]);
      }
    }
  }
  /// END SOLUTION
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < out->size; i++)
  {
    auto max_value = a.ptr[i * reduce_size];
    for (size_t j = 1; j < reduce_size; j++)
      max_value = std::max(max_value, a.ptr[i * reduce_size + j]);
    out->ptr[i] = max_value;
  }
  /// END SOLUTION
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < out->size; i++)
  {
    auto sum_value = a.ptr[i * reduce_size];
    for (size_t j = 1; j < reduce_size; j++)
      sum_value = sum_value + a.ptr[i * reduce_size + j];
    out->ptr[i] = sum_value;
  }
  /// END SOLUTION
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);
    
  py::class_<COOMatrix>(m, "COOMatrix")
      .def(py::init<size_t, size_t, size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("nnz", &COOMatrix::nnz)
      .def_readonly("rows", &COOMatrix::rows)
      .def_readonly("cols", &COOMatrix::cols)
      .def("get_data", [](const COOMatrix& mat) {
        return py::array_t<scalar_t>(
            {mat.nnz},            // Shape
            {sizeof(scalar_t)},   // Strides
            mat.data              // Data pointer
        );
      })
      .def("get_row_indices", [](const COOMatrix& mat) {
        return py::array_t<int32_t>(
            {mat.nnz},
            {sizeof(int32_t)},
            mat.row_indices
        );
      })
      .def("get_col_indices", [](const COOMatrix& mat) {
        return py::array_t<int32_t>(
            {mat.nnz},
            {sizeof(int32_t)},
            mat.col_indices
        );
      });

  m.def("dense_to_sparse", &DenseToSparse);
  m.def("sparse_to_dense", &SparseToDense);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);
  m.def("sparse_dense_add", SparseDenseAdd);
  m.def("sparse_sparse_add", SparseSparseAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}

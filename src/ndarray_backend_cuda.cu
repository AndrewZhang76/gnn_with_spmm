#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

namespace needle {
namespace needle_cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct COOMatrix {
  size_t nnz;
  size_t rows;
  size_t cols;
  scalar_t* data;
  int32_t* row_indices;
  int32_t* col_indices;

  COOMatrix(size_t nnz, size_t rows, size_t cols) : nnz(nnz), rows(rows), cols(cols) {
    size_t max_nnz = rows * cols;
    cudaError_t err1 = cudaMalloc(&data, max_nnz * sizeof(scalar_t));
    if (err1 != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err1));
    cudaError_t err2 = cudaMalloc(&row_indices, max_nnz * sizeof(int32_t));
    if (err2 != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err2));
    cudaError_t err3 = cudaMalloc(&col_indices, max_nnz * sizeof(int32_t));
    if (err3 != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err3));
  }

  ~COOMatrix() {
    cudaFree(data);
    cudaFree(row_indices);
    cudaFree(col_indices);
  }
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

__global__ void DenseToSparseKernel(const scalar_t* dense, scalar_t* data, int32_t* row_indices, int32_t* col_indices, size_t rows, size_t cols, unsigned int* nnz_counter) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_size = rows * cols;
  // printf("Idx: %d, Total Size: %d\n", static_cast<int>(idx), static_cast<int>(total_size));
  if (idx < total_size) {
    scalar_t val = dense[idx];
    // printf("Val: %f\n", val);
    if (val != 0) {
      // printf("Val not 0: %f\n", val);
      unsigned int nnz_idx = atomicAdd(nnz_counter, 1);
      data[nnz_idx] = val;
      row_indices[nnz_idx] = idx / cols;
      col_indices[nnz_idx] = idx % cols;
      // printf("NNZ: %d, Val: %f, Row: %d, Col: %d\n", nnz_idx, val, row_indices[nnz_idx], col_indices[nnz_idx]);
    }
  }
}

std::unique_ptr<COOMatrix> DenseToSparse(const CudaArray& dense_matrix, size_t rows, size_t cols) {
  unsigned int* d_nnz_counter;
  cudaMalloc(&d_nnz_counter, sizeof(unsigned int));
  cudaMemset(d_nnz_counter, 0, sizeof(unsigned int));

  size_t total_size = rows * cols;
  std::unique_ptr<COOMatrix> sparse_matrix (new COOMatrix(total_size, rows, cols));

  CudaDims dim = CudaOneDim(total_size);
  DenseToSparseKernel<<<dim.grid, dim.block>>>(dense_matrix.ptr, sparse_matrix->data, sparse_matrix->row_indices, sparse_matrix->col_indices, rows, cols, d_nnz_counter);
  cudaDeviceSynchronize();

  unsigned int h_nnz_counter;
  cudaMemcpy(&h_nnz_counter, d_nnz_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);

  // Allocate new arrays with the actual number of non-zero elements
  scalar_t* new_data;
  int32_t* new_row_indices;
  int32_t* new_col_indices;
  cudaMalloc(&new_data, h_nnz_counter * sizeof(scalar_t));
  cudaMalloc(&new_row_indices, h_nnz_counter * sizeof(int32_t));
  cudaMalloc(&new_col_indices, h_nnz_counter * sizeof(int32_t));

  // Copy the non-zero elements to the new arrays
  cudaMemcpy(new_data, sparse_matrix->data, h_nnz_counter * sizeof(scalar_t), cudaMemcpyDeviceToDevice);
  cudaMemcpy(new_row_indices, sparse_matrix->row_indices, h_nnz_counter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
  cudaMemcpy(new_col_indices, sparse_matrix->col_indices, h_nnz_counter * sizeof(int32_t), cudaMemcpyDeviceToDevice);

  // Free the old over-allocated memory
  cudaFree(sparse_matrix->data);
  cudaFree(sparse_matrix->row_indices);
  cudaFree(sparse_matrix->col_indices);

  // Update the sparse matrix with new pointers and actual nnz count
  sparse_matrix->data = new_data;
  sparse_matrix->row_indices = new_row_indices;
  sparse_matrix->col_indices = new_col_indices;
  sparse_matrix->nnz = h_nnz_counter;

  // Free the device nnz counter
  cudaFree(d_nnz_counter);

  // Return the updated sparse matrix
  return sparse_matrix;
}

__global__ void SparseToDenseKernel(scalar_t* dense, const scalar_t* data, const int32_t* row_indices, const int32_t* col_indices, size_t nnz, size_t rows, size_t cols) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nnz) {
    int row = row_indices[idx];
    int col = col_indices[idx];
    dense[row * cols + col] = data[idx];
  }
}

std::unique_ptr<CudaArray> SparseToDense(const COOMatrix& sparse_matrix) {
  std::unique_ptr<CudaArray> dense_matrix (new CudaArray(sparse_matrix.rows * sparse_matrix.cols));
  cudaMemset(dense_matrix->ptr, 0, dense_matrix->size * sizeof(scalar_t));

  CudaDims dim = CudaOneDim(sparse_matrix.nnz);
  SparseToDenseKernel<<<dim.grid, dim.block>>>(dense_matrix->ptr, sparse_matrix.data, sparse_matrix.row_indices, sparse_matrix.col_indices, sparse_matrix.nnz, sparse_matrix.rows, sparse_matrix.cols);
  return dense_matrix;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides
__device__ size_t get_index(size_t gid, CudaVec shape, CudaVec strides, size_t offset) {
  size_t idx = offset;
  for (int i = shape.size - 1; i >= 0; i--) {
    idx +=  strides.data[i] * (gid % shape.data[i]);
    gid /= shape.data[i];
  }
  return idx;
}


__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if (gid < size) {
    out[gid] = a[get_index(gid, shape, strides, offset)];
  }
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    out[get_index(gid, shape, strides, offset)] = a[gid];
  }
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
  /// END SOLUTION
}

__global__ void ScalarSetitemKernel(const scalar_t val, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    out[get_index(gid, shape, strides, offset)] = val;
  }
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape),
                                              VecToCuda(strides), offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void SparseDenseAddKernel(const scalar_t* data, const int32_t* row_indices, const int32_t* col_indices, size_t num_cols, size_t nnz, const scalar_t* b, scalar_t* out, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    out[idx] = b[idx];
  }

  if (idx < nnz) {
    int row = row_indices[idx];
    int col = col_indices[idx];
    size_t index = row * num_cols + col;
    atomicAdd(&out[index], data[idx]);
  }
}

void SparseDenseAdd(const COOMatrix& a, const CudaArray& b, CudaArray* out) {
  /**
   * Args:
   *   a: sparse matrix to add
   *   b: dense matrix to add
   *   out: dense matrix to write the output to
   */

  CudaDims dim = CudaOneDim(out->size);
  SparseDenseAddKernel<<<dim.grid, dim.block>>>(a.data, a.row_indices, a.col_indices, a.cols, a.nnz, b.ptr, out->ptr, out->size);
}

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void SparseSparseAdd(const COOMatrix& a, const COOMatrix& b, COOMatrix* out) {
  size_t total_nnz = a.nnz + b.nnz;

  // Create device vectors to hold combined data
  thrust::device_vector<scalar_t> data_vec(total_nnz);
  thrust::device_vector<int32_t> row_vec(total_nnz);
  thrust::device_vector<int32_t> col_vec(total_nnz);

  // Copy data from matrix 'a'
  thrust::copy_n(a.data, a.nnz, data_vec.begin());
  thrust::copy_n(a.row_indices, a.nnz, row_vec.begin());
  thrust::copy_n(a.col_indices, a.nnz, col_vec.begin());

  // Copy data from matrix 'b'
  thrust::copy_n(b.data, b.nnz, data_vec.begin() + a.nnz);
  thrust::copy_n(b.row_indices, b.nnz, row_vec.begin() + a.nnz);
  thrust::copy_n(b.col_indices, b.nnz, col_vec.begin() + a.nnz);

  // Zip the row and column indices
  auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(row_vec.begin(), col_vec.begin()));
  auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(row_vec.end(), col_vec.end()));

  // Sort by (row, col)
  thrust::sort_by_key(zip_begin, zip_end, data_vec.begin());

  // Reduce duplicates by summing data values
  thrust::device_vector<scalar_t> data_reduced(total_nnz);
  thrust::device_vector<int32_t> row_reduced(total_nnz);
  thrust::device_vector<int32_t> col_reduced(total_nnz);

  auto zip_keys_out = thrust::make_zip_iterator(thrust::make_tuple(row_reduced.begin(), col_reduced.begin()));
  auto new_end = thrust::reduce_by_key(
      zip_begin,
      zip_end,
      data_vec.begin(),
      zip_keys_out,
      data_reduced.begin());

  size_t nnz_new = thrust::distance(data_reduced.begin(), new_end.second);

  // Update the output COOMatrix
  out->nnz = nnz_new;
  out->rows = a.rows;
  out->cols = a.cols;

  // Allocate memory for the output data
  cudaMalloc(&(out->data), nnz_new * sizeof(scalar_t));
  cudaMalloc(&(out->row_indices), nnz_new * sizeof(int32_t));
  cudaMalloc(&(out->col_indices), nnz_new * sizeof(int32_t));
  cudaMemcpy(out->data, data_reduced.data().get(), nnz_new * sizeof(scalar_t), cudaMemcpyDeviceToDevice);
  cudaMemcpy(out->row_indices, row_reduced.data().get(), nnz_new * sizeof(int32_t), cudaMemcpyDeviceToDevice);
  cudaMemcpy(out->col_indices, col_reduced.data().get(), nnz_new * sizeof(int32_t), cudaMemcpyDeviceToDevice);
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
  // and store the result in array 'out'.
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
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

__global__ void EwiseMulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * b[gid];
}

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseMulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMulKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * val;
}

void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarMulKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseDivKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / b[gid];
}

void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseDivKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarDivKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / val;
}

void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarDivKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void ScalarPowerKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = pow(a[gid], val);
}

void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarPowerKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}
__global__ void EwiseMaximumKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = max(a[gid], b[gid]);
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, a.size);
}

__global__ void ScalarMaximumKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = max(a[gid], val);
}

void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, a.size);
}

__global__ void EwiseEqKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = scalar_t(a[gid] == b[gid]);
}

void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseEqKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, a.size);
}

__global__ void ScalarEqKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = scalar_t(a[gid] == val);
}

void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarEqKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, a.size);
}

__global__ void EwiseGeKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = scalar_t(a[gid] >= b[gid]);
}

void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseGeKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, a.size);
}

__global__ void ScalarGeKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = scalar_t(a[gid] >= val);
}

void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarGeKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, a.size);
}

__global__ void EwiseLogKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = log(a[gid]);
}

void EwiseLog(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseLogKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size);
}

__global__ void EwiseExpKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = exp(a[gid]);
}

void EwiseExp(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseExpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size);
}

__global__ void EwiseTanhKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = tanh(a[gid]);
}

void EwiseTanh(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseTanhKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size);
}
////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out,
        uint32_t N, uint32_t P, size_t size) {
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < size) {
        scalar_t val = 0.0;
        for (uint32_t k = 0; k < N; k++) {
            val += a[gid / P * N + k] * b[k * P + gid % P];
        }
        out[gid] = val;
    }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  MatmulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, N, P, out->size);  
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t out_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < out_size) {
    scalar_t max_val = a[gid * reduce_size];
    for (int i = 1; i < reduce_size; ++i) {
      max_val = max(max_val, a[gid * reduce_size + i]);
    }
    out[gid] = max_val;
  }
}

__global__ void SparseMatmulKernelCOO(
    const scalar_t*   A_data,
    const int32_t*    A_row_indices,
    const int32_t*    A_col_indices,
    size_t            A_nnz,
    const scalar_t*   B_data,
    const int32_t*    B_row_indices,
    const int32_t*    B_col_indices,
    size_t            B_nnz,
    scalar_t*         temp_data,
    int32_t*          temp_row_indices,
    int32_t*          temp_col_indices,
    unsigned long long int*    global_idx) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_nnz = A_nnz * B_nnz;

    if (idx < total_nnz) {
      size_t i = idx / B_nnz;
      size_t j = idx % B_nnz;

      int32_t col_a = A_col_indices[i];
      int32_t row_a = A_row_indices[i];
      scalar_t val_a = A_data[i];

      if (B_row_indices[j] == col_a) {
        int32_t col_b = B_col_indices[j];
        scalar_t val_b = B_data[j];

        scalar_t val_c = val_a * val_b;
        if (val_c != 0) {
          size_t temp_idx = atomicAdd(global_idx, 1);
          temp_data[temp_idx] = val_c;
          temp_row_indices[temp_idx] = row_a;
          temp_col_indices[temp_idx] = col_b;
        }
      }
    }
}

void SparseMatmulCOO(const COOMatrix& A, const COOMatrix& B, COOMatrix* C) {
  size_t temp_size = A.nnz * B.nnz;
  scalar_t* temp_data;
  int32_t* temp_row_indices;
  int32_t* temp_col_indices;
  unsigned long long int* global_idx;
  cudaMalloc(&temp_data, temp_size * sizeof(scalar_t));
  cudaMalloc(&temp_row_indices, temp_size * sizeof(int32_t));
  cudaMalloc(&temp_col_indices, temp_size * sizeof(int32_t));
  cudaMalloc(&global_idx, sizeof(unsigned long long int));
  cudaMemset(global_idx, 0, sizeof(unsigned long long int));
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to allocate memory: " + std::string(cudaGetErrorString(err)));
  }

  // Phase 2: Kernel Execution
  CudaDims dim = CudaOneDim(A.nnz*B.nnz);
  SparseMatmulKernelCOO<<<dim.grid, dim.block>>>(
      A.data, A.row_indices, A.col_indices, A.nnz,
      B.data, B.row_indices, B.col_indices, B.nnz,
      temp_data, temp_row_indices, temp_col_indices,
      global_idx);

  // Synchronize and check for errors
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to launch kernel: " + std::string(cudaGetErrorString(err)));
  }

  // Copy global_idx from device to host and print
  unsigned long long int host_global_idx;
  cudaMemcpy(&host_global_idx, global_idx, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
  cudaFree(global_idx);

  // Phase 3: Thrust Operations
  size_t nnz_temp = host_global_idx;

  thrust::device_ptr<scalar_t> temp_data_ptr(temp_data);
  thrust::device_ptr<int32_t> temp_row_ptr(temp_row_indices);
  thrust::device_ptr<int32_t> temp_col_ptr(temp_col_indices);

  auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(temp_row_ptr, temp_col_ptr));
  auto zip_end = zip_begin + nnz_temp;

  thrust::sort_by_key(zip_begin, zip_end, temp_data_ptr);

  // Phase 4: Reduction
  thrust::device_vector<int32_t> temp_rows(temp_row_indices, temp_row_indices + nnz_temp);
  thrust::device_vector<int32_t> temp_cols(temp_col_indices, temp_col_indices + nnz_temp);
  thrust::device_vector<scalar_t> temp_values(temp_data, temp_data + nnz_temp);
  cudaFree(temp_data);
  cudaFree(temp_row_indices);
  cudaFree(temp_col_indices);

  auto keys_in_begin = thrust::make_zip_iterator(thrust::make_tuple(temp_rows.begin(), temp_cols.begin()));
  auto keys_in_end = thrust::make_zip_iterator(thrust::make_tuple(temp_rows.end(), temp_cols.end()));

  thrust::device_vector<int32_t> out_rows(nnz_temp);
  thrust::device_vector<int32_t> out_cols(nnz_temp);
  thrust::device_vector<scalar_t> out_values(nnz_temp);

  // Create zipped iterators for output keys
  auto keys_out_begin = thrust::make_zip_iterator(thrust::make_tuple(out_rows.begin(), out_cols.begin()));
  auto new_end = thrust::reduce_by_key(
      keys_in_begin,
      keys_in_end,
      temp_values.begin(),
      keys_out_begin,
      out_values.begin());

  size_t nnz_C = thrust::distance(out_values.begin(), new_end.second);

  // Phase 5: Final Memory Operations
  C->nnz = nnz_C;
  C->rows = A.rows;
  C->cols = B.cols;

  cudaMalloc(&(C->data), nnz_C * sizeof(scalar_t));
  cudaMalloc(&(C->row_indices), nnz_C * sizeof(int32_t));
  cudaMalloc(&(C->col_indices), nnz_C * sizeof(int32_t));

  cudaMemcpy(C->data, out_values.data().get(), nnz_C * sizeof(scalar_t), cudaMemcpyDeviceToDevice);
  cudaMemcpy(C->row_indices, out_rows.data().get(), nnz_C * sizeof(int32_t), cudaMemcpyDeviceToDevice);
  cudaMemcpy(C->col_indices, out_cols.data().get(), nnz_C * sizeof(int32_t), cudaMemcpyDeviceToDevice);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to allocate memory at last " + std::string(cudaGetErrorString(err)));
  }
}

__global__ void SparseDenseMatmulKernel(
    const scalar_t* A_data,
    const scalar_t* B_data,
    const int32_t* A_row_indices,
    const int32_t* A_col_indices,
    size_t M,
    size_t N,
    size_t P,
    size_t A_nnz,
    scalar_t* C_data)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < A_nnz * P) {
    size_t a_idx = idx / P;
    size_t i = idx % P;

    int32_t row_A = A_row_indices[a_idx];
    int32_t col_A = A_col_indices[a_idx];
    scalar_t A_value = A_data[a_idx];
    scalar_t B_value = B_data[col_A * P + i];

    atomicAdd(&C_data[row_A * P + i], A_value * B_value);
  }
}

void SparseDenseMatmul(const COOMatrix& A, const CudaArray& B, CudaArray* C, uint32_t M, uint32_t N, uint32_t P) {
  cudaMemset(C->ptr, 0, C->size * sizeof(scalar_t));
  CudaDims dim = CudaOneDim(A.nnz * P);
  SparseDenseMatmulKernel<<<dim.grid, dim.block>>>(
      A.data, B.ptr, A.row_indices, A.col_indices, M, N, P, A.nnz, C->ptr);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

__global__ void DenseSparseMatmulKernel(
    const scalar_t* A_data,
    const scalar_t* B_data,
    const int32_t* B_row_indices,
    const int32_t* B_col_indices,
    size_t M,
    size_t N,
    size_t P,
    size_t B_nnz,
    scalar_t* C_data)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < B_nnz * M) {
        size_t b_idx = idx / M;
        size_t i = idx % M;

        int32_t row_B = B_row_indices[b_idx];
        int32_t col_B = B_col_indices[b_idx];
        scalar_t B_value = B_data[b_idx];
        scalar_t A_value = A_data[i * N + row_B];

        atomicAdd(&C_data[i * P + col_B], A_value * B_value);
    }
}

void DenseSparseMatmul(const CudaArray& A, const COOMatrix& B, CudaArray* C, uint32_t M, uint32_t N, uint32_t P) {
    cudaMemset(C->ptr, 0, C->size * sizeof(scalar_t));
    CudaDims dim = CudaOneDim(B.nnz * M);
    DenseSparseMatmulKernel<<<dim.grid, dim.block>>>(
        A.ptr, B.data, B.row_indices, B.col_indices, M, N, P, B.nnz, C->ptr);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }
}


__global__ void SparseSparseMatmulKernel(
    const scalar_t* A_data,
    const int32_t* A_row_indices,
    const int32_t* A_col_indices,
    const scalar_t* B_data,
    const int32_t* B_row_indices,
    const int32_t* B_col_indices,
    size_t M,
    size_t N,
    size_t P,
    size_t A_nnz,
    size_t B_nnz,
    scalar_t* C_data)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_nnz = A_nnz * B_nnz;

  if (idx < total_nnz) {
    size_t i = idx / B_nnz;
    size_t j = idx % B_nnz;

    int32_t col_a = A_col_indices[i];
    int32_t row_a = A_row_indices[i];
    scalar_t val_a = A_data[i];

    if (B_row_indices[j] == col_a) {
      int32_t col_b = B_col_indices[j];
      scalar_t val_b = B_data[j];

      scalar_t val_c = val_a * val_b;
      atomicAdd(&C_data[row_a * P + col_b], val_c);
    }
  }
}

void SparseSparseMatmul(const COOMatrix& A, const COOMatrix& B, CudaArray* C, uint32_t M, uint32_t N, uint32_t P) {
  cudaMemset(C->ptr, 0, C->size * sizeof(scalar_t));
  CudaDims dim = CudaOneDim(A.nnz * B.nnz);
  SparseSparseMatmulKernel<<<dim.grid, dim.block>>>(
      A.data, A.row_indices, A.col_indices, B.data, B.row_indices, B.col_indices, M, N, P, A.nnz, B.nnz, C->ptr);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  /// END SOLUTION
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t out_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < out_size) {
    scalar_t sum_val = a[gid * reduce_size];
    for (int i = 1; i < reduce_size; ++i) {
      sum_val += a[gid * reduce_size + i];
    }
    out[gid] = sum_val;
  }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  /// END SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace needle_cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  py::class_<COOMatrix>(m, "COOMatrix")
      .def(py::init<size_t, size_t, size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("nnz", &COOMatrix::nnz)
      .def_readonly("rows", &COOMatrix::rows)
      .def_readonly("cols", &COOMatrix::cols)
      .def("get_data", [](const COOMatrix& mat) {
          // Allocate host memory
          std::vector<scalar_t> host_data(mat.nnz);
          // Copy data from device to host
          cudaError_t err = cudaMemcpy(
              host_data.data(), mat.data, mat.nnz * sizeof(scalar_t), cudaMemcpyDeviceToHost);
          if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
          // Return as NumPy array
          return py::array_t<scalar_t>(mat.nnz, host_data.data());
      })
      .def("get_row_indices", [](const COOMatrix& mat) {
          std::vector<int32_t> host_row_indices(mat.nnz);
          cudaError_t err = cudaMemcpy(
              host_row_indices.data(), mat.row_indices, mat.nnz * sizeof(int32_t), cudaMemcpyDeviceToHost);
          if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
          return py::array_t<int32_t>(mat.nnz, host_row_indices.data());
      })
      .def("get_col_indices", [](const COOMatrix& mat) {
          std::vector<int32_t> host_col_indices(mat.nnz);
          cudaError_t err = cudaMemcpy(
              host_col_indices.data(), mat.col_indices, mat.nnz * sizeof(int32_t), cudaMemcpyDeviceToHost);
          if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
          return py::array_t<int32_t>(mat.nnz, host_col_indices.data());
      });

  m.def("dense_to_sparse", &DenseToSparse);
  m.def("sparse_to_dense", &SparseToDense);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
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
  m.def("sparse_matmul_coo", SparseMatmulCOO);
  m.def("sparse_dense_matmul_coo", SparseDenseMatmul);
  m.def("dense_sparse_matmul_coo", DenseSparseMatmul);
  m.def("sparse_sparse_matmul_coo", SparseSparseMatmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}

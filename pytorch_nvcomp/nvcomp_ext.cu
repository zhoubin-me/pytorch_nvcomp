#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "nvcomp/lz4.h"
#include "nvcomp/zstd.h"
#include "nvcomp/cascaded.h"

#include <vector>
#include <numeric>
#include <sstream>
#include <tuple>

namespace {

static_assert(sizeof(void*) == 8, "this extension requires 64-bit pointers");
static_assert(sizeof(size_t) == 8, "this extension requires 64-bit size_t");

inline void check_cuda(cudaError_t status, const char* msg)
{
  if (status != cudaSuccess) {
    std::ostringstream oss;
    oss << msg << ": " << cudaGetErrorString(status);
    throw std::runtime_error(oss.str());
  }
}

inline void check_nvcomp(nvcompStatus_t status, const char* msg)
{
  if (status != nvcompSuccess) {
    std::ostringstream oss;
    oss << msg << ": nvcomp status " << static_cast<int>(status);
    throw std::runtime_error(oss.str());
  }
}

struct CascadedTypeInfo {
  nvcompType_t type;
  size_t size;
};

CascadedTypeInfo get_cascaded_type(torch::ScalarType dtype)
{
  switch (dtype) {
    case torch::kUInt8:
      return {NVCOMP_TYPE_UCHAR, 1};
    case torch::kInt8:
      return {NVCOMP_TYPE_CHAR, 1};
    case torch::kInt16:
      return {NVCOMP_TYPE_SHORT, 2};
    case torch::kInt32:
      return {NVCOMP_TYPE_INT, 4};
    case torch::kInt64:
      return {NVCOMP_TYPE_LONGLONG, 8};
    default:
      throw std::runtime_error("cascaded only supports int8/uint8/int16/int32/int64 tensors");
  }
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t>
lz4_compress(torch::Tensor input, int64_t chunk_size)
{
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kUInt8, "input must be uint8");
  TORCH_CHECK(chunk_size > 0, "chunk_size must be > 0");

  c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getDefaultCUDAStream();
  cudaStream_t cuda_stream = stream.stream();

  const size_t in_bytes = static_cast<size_t>(input.numel());
  const size_t chunk_sz = static_cast<size_t>(chunk_size);
  const size_t batch_size = (in_bytes + chunk_sz - 1) / chunk_sz;

  auto sizes_cpu = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(torch::kCPU));
  auto orig_sizes_cpu = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(torch::kCPU));

  auto* orig_sizes_cpu_ptr = reinterpret_cast<int64_t*>(orig_sizes_cpu.data_ptr<int64_t>());
  for (size_t i = 0; i < batch_size; ++i) {
    const size_t remaining = in_bytes - (i * chunk_sz);
    const size_t this_size = remaining < chunk_sz ? remaining : chunk_sz;
    orig_sizes_cpu_ptr[i] = static_cast<int64_t>(this_size);
  }

  size_t temp_bytes = 0;
  check_nvcomp(
      nvcompBatchedLZ4CompressGetTempSizeAsync(
          batch_size,
          chunk_sz,
          nvcompBatchedLZ4CompressDefaultOpts,
          &temp_bytes,
          in_bytes),
      "nvcompBatchedLZ4CompressGetTempSizeAsync failed");

  size_t max_out_bytes = 0;
  check_nvcomp(
      nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
          chunk_sz,
          nvcompBatchedLZ4CompressDefaultOpts,
          &max_out_bytes),
      "nvcompBatchedLZ4CompressGetMaxOutputChunkSize failed");

  auto temp = torch::empty({static_cast<int64_t>(temp_bytes)}, torch::dtype(torch::kUInt8).device(input.device()));
  auto compressed = torch::empty({static_cast<int64_t>(batch_size * max_out_bytes)}, torch::dtype(torch::kUInt8).device(input.device()));
  auto comp_sizes_dev = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(input.device()));
  auto orig_sizes_dev = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(input.device()));

  check_cuda(
      cudaMemcpyAsync(
          orig_sizes_dev.data_ptr<int64_t>(),
          orig_sizes_cpu.data_ptr<int64_t>(),
          sizeof(int64_t) * batch_size,
          cudaMemcpyHostToDevice,
          cuda_stream),
      "cudaMemcpyAsync orig_sizes_dev failed");

  std::vector<void*> host_uncompressed_ptrs(batch_size);
  auto* input_ptr = input.data_ptr<uint8_t>();
  for (size_t i = 0; i < batch_size; ++i) {
    host_uncompressed_ptrs[i] = input_ptr + (i * chunk_sz);
  }

  std::vector<void*> host_compressed_ptrs(batch_size);
  auto* comp_ptr = compressed.data_ptr<uint8_t>();
  for (size_t i = 0; i < batch_size; ++i) {
    host_compressed_ptrs[i] = comp_ptr + (i * max_out_bytes);
  }

  auto uncompressed_ptrs_dev = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(input.device()));
  auto compressed_ptrs_dev = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(input.device()));

  check_cuda(
      cudaMemcpyAsync(
          uncompressed_ptrs_dev.data_ptr<int64_t>(),
          host_uncompressed_ptrs.data(),
          sizeof(void*) * batch_size,
          cudaMemcpyHostToDevice,
          cuda_stream),
      "cudaMemcpyAsync uncompressed_ptrs_dev failed");

  check_cuda(
      cudaMemcpyAsync(
          compressed_ptrs_dev.data_ptr<int64_t>(),
          host_compressed_ptrs.data(),
          sizeof(void*) * batch_size,
          cudaMemcpyHostToDevice,
          cuda_stream),
      "cudaMemcpyAsync compressed_ptrs_dev failed");

  auto comp_statuses = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt32).device(input.device()));
  check_nvcomp(
      nvcompBatchedLZ4CompressAsync(
          reinterpret_cast<void**>(uncompressed_ptrs_dev.data_ptr<int64_t>()),
          reinterpret_cast<size_t*>(orig_sizes_dev.data_ptr<int64_t>()),
          chunk_sz,
          batch_size,
          reinterpret_cast<void*>(temp.data_ptr<uint8_t>()),
          temp_bytes,
          reinterpret_cast<void**>(compressed_ptrs_dev.data_ptr<int64_t>()),
          reinterpret_cast<size_t*>(comp_sizes_dev.data_ptr<int64_t>()),
          nvcompBatchedLZ4CompressDefaultOpts,
          reinterpret_cast<nvcompStatus_t*>(comp_statuses.data_ptr<int32_t>()),
          cuda_stream),
      "nvcompBatchedLZ4CompressAsync failed");

  check_cuda(
      cudaMemcpyAsync(
          sizes_cpu.data_ptr<int64_t>(),
          comp_sizes_dev.data_ptr<int64_t>(),
          sizeof(int64_t) * batch_size,
          cudaMemcpyDeviceToHost,
          cuda_stream),
      "cudaMemcpyAsync sizes_cpu failed");

  check_cuda(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize failed");

  return std::make_tuple(
      compressed,
      sizes_cpu,
      orig_sizes_cpu,
      static_cast<int64_t>(max_out_bytes),
      static_cast<int64_t>(chunk_sz));
}

torch::Tensor lz4_decompress(
    torch::Tensor compressed,
    torch::Tensor compressed_sizes_cpu,
    torch::Tensor orig_sizes_cpu,
    int64_t max_out_bytes,
    int64_t chunk_size)
{
  TORCH_CHECK(compressed.is_cuda(), "compressed must be a CUDA tensor");
  TORCH_CHECK(compressed.is_contiguous(), "compressed must be contiguous");
  TORCH_CHECK(compressed.scalar_type() == torch::kUInt8, "compressed must be uint8");
  TORCH_CHECK(compressed_sizes_cpu.device().is_cpu(), "compressed_sizes must be on CPU");
  TORCH_CHECK(orig_sizes_cpu.device().is_cpu(), "orig_sizes must be on CPU");
  TORCH_CHECK(compressed_sizes_cpu.scalar_type() == torch::kInt64, "compressed_sizes must be int64");
  TORCH_CHECK(orig_sizes_cpu.scalar_type() == torch::kInt64, "orig_sizes must be int64");
  TORCH_CHECK(chunk_size > 0, "chunk_size must be > 0");
  TORCH_CHECK(max_out_bytes > 0, "max_out_bytes must be > 0");
  TORCH_CHECK(compressed_sizes_cpu.numel() == orig_sizes_cpu.numel(), "sizes must have same length");

  c10::cuda::CUDAGuard guard(compressed.device());
  auto stream = at::cuda::getDefaultCUDAStream();
  cudaStream_t cuda_stream = stream.stream();

  const size_t batch_size = static_cast<size_t>(compressed_sizes_cpu.numel());
  const size_t chunk_sz = static_cast<size_t>(chunk_size);
  const size_t max_out = static_cast<size_t>(max_out_bytes);

  auto* orig_sizes_ptr = reinterpret_cast<int64_t*>(orig_sizes_cpu.data_ptr<int64_t>());
  std::vector<size_t> orig_sizes(batch_size);
  size_t total_out = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    const size_t sz = static_cast<size_t>(orig_sizes_ptr[i]);
    orig_sizes[i] = sz;
    total_out += sz;
  }

  auto output = torch::empty({static_cast<int64_t>(total_out)}, torch::dtype(torch::kUInt8).device(compressed.device()));

  auto comp_sizes_dev = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(compressed.device()));
  auto orig_sizes_dev = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(compressed.device()));

  check_cuda(
      cudaMemcpyAsync(
          comp_sizes_dev.data_ptr<int64_t>(),
          compressed_sizes_cpu.data_ptr<int64_t>(),
          sizeof(int64_t) * batch_size,
          cudaMemcpyHostToDevice,
          cuda_stream),
      "cudaMemcpyAsync comp_sizes_dev failed");

  check_cuda(
      cudaMemcpyAsync(
          orig_sizes_dev.data_ptr<int64_t>(),
          orig_sizes_cpu.data_ptr<int64_t>(),
          sizeof(int64_t) * batch_size,
          cudaMemcpyHostToDevice,
          cuda_stream),
      "cudaMemcpyAsync orig_sizes_dev failed");

  std::vector<void*> host_compressed_ptrs(batch_size);
  auto* comp_ptr = compressed.data_ptr<uint8_t>();
  for (size_t i = 0; i < batch_size; ++i) {
    host_compressed_ptrs[i] = comp_ptr + (i * max_out);
  }

  std::vector<void*> host_output_ptrs(batch_size);
  auto* out_ptr = output.data_ptr<uint8_t>();
  size_t offset = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    host_output_ptrs[i] = out_ptr + offset;
    offset += orig_sizes[i];
  }

  auto compressed_ptrs_dev = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(compressed.device()));
  auto output_ptrs_dev = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(compressed.device()));

  check_cuda(
      cudaMemcpyAsync(
          compressed_ptrs_dev.data_ptr<int64_t>(),
          host_compressed_ptrs.data(),
          sizeof(void*) * batch_size,
          cudaMemcpyHostToDevice,
          cuda_stream),
      "cudaMemcpyAsync compressed_ptrs_dev failed");

  check_cuda(
      cudaMemcpyAsync(
          output_ptrs_dev.data_ptr<int64_t>(),
          host_output_ptrs.data(),
          sizeof(void*) * batch_size,
          cudaMemcpyHostToDevice,
          cuda_stream),
      "cudaMemcpyAsync output_ptrs_dev failed");

  size_t temp_bytes = 0;
  check_nvcomp(
      nvcompBatchedLZ4DecompressGetTempSizeAsync(
          batch_size,
          chunk_sz,
          nvcompBatchedLZ4DecompressDefaultOpts,
          &temp_bytes,
          total_out),
      "nvcompBatchedLZ4DecompressGetTempSizeAsync failed");

  auto temp = torch::empty({static_cast<int64_t>(temp_bytes)}, torch::dtype(torch::kUInt8).device(compressed.device()));
  auto statuses = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt32).device(compressed.device()));
  auto actual_sizes = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(compressed.device()));

  check_nvcomp(
      nvcompBatchedLZ4DecompressAsync(
          reinterpret_cast<void**>(compressed_ptrs_dev.data_ptr<int64_t>()),
          reinterpret_cast<size_t*>(comp_sizes_dev.data_ptr<int64_t>()),
          reinterpret_cast<size_t*>(orig_sizes_dev.data_ptr<int64_t>()),
          reinterpret_cast<size_t*>(actual_sizes.data_ptr<int64_t>()),
          batch_size,
          reinterpret_cast<void*>(temp.data_ptr<uint8_t>()),
          temp_bytes,
          reinterpret_cast<void**>(output_ptrs_dev.data_ptr<int64_t>()),
          nvcompBatchedLZ4DecompressDefaultOpts,
          reinterpret_cast<nvcompStatus_t*>(statuses.data_ptr<int32_t>()),
          cuda_stream),
      "nvcompBatchedLZ4DecompressAsync failed");

  check_cuda(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize failed");

  return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t>
zstd_compress(torch::Tensor input, int64_t chunk_size)
{
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kUInt8, "input must be uint8");
  TORCH_CHECK(chunk_size > 0, "chunk_size must be > 0");

  c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getDefaultCUDAStream();
  cudaStream_t cuda_stream = stream.stream();

  const size_t in_bytes = static_cast<size_t>(input.numel());
  const size_t chunk_sz = static_cast<size_t>(chunk_size);
  const size_t batch_size = (in_bytes + chunk_sz - 1) / chunk_sz;

  auto sizes_cpu = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(torch::kCPU));
  auto orig_sizes_cpu = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(torch::kCPU));

  auto* orig_sizes_cpu_ptr = reinterpret_cast<int64_t*>(orig_sizes_cpu.data_ptr<int64_t>());
  for (size_t i = 0; i < batch_size; ++i) {
    const size_t remaining = in_bytes - (i * chunk_sz);
    const size_t this_size = remaining < chunk_sz ? remaining : chunk_sz;
    orig_sizes_cpu_ptr[i] = static_cast<int64_t>(this_size);
  }

  size_t temp_bytes = 0;
  check_nvcomp(
      nvcompBatchedZstdCompressGetTempSizeAsync(
          batch_size,
          chunk_sz,
          nvcompBatchedZstdCompressDefaultOpts,
          &temp_bytes,
          in_bytes),
      "nvcompBatchedZstdCompressGetTempSizeAsync failed");

  size_t max_out_bytes = 0;
  check_nvcomp(
      nvcompBatchedZstdCompressGetMaxOutputChunkSize(
          chunk_sz,
          nvcompBatchedZstdCompressDefaultOpts,
          &max_out_bytes),
      "nvcompBatchedZstdCompressGetMaxOutputChunkSize failed");

  auto temp = torch::empty({static_cast<int64_t>(temp_bytes)}, torch::dtype(torch::kUInt8).device(input.device()));
  auto compressed = torch::empty({static_cast<int64_t>(batch_size * max_out_bytes)}, torch::dtype(torch::kUInt8).device(input.device()));
  auto comp_sizes_dev = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(input.device()));
  auto orig_sizes_dev = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(input.device()));

  check_cuda(
      cudaMemcpyAsync(
          orig_sizes_dev.data_ptr<int64_t>(),
          orig_sizes_cpu.data_ptr<int64_t>(),
          sizeof(int64_t) * batch_size,
          cudaMemcpyHostToDevice,
          cuda_stream),
      "cudaMemcpyAsync orig_sizes_dev failed");

  std::vector<void*> host_uncompressed_ptrs(batch_size);
  auto* input_ptr = input.data_ptr<uint8_t>();
  for (size_t i = 0; i < batch_size; ++i) {
    host_uncompressed_ptrs[i] = input_ptr + (i * chunk_sz);
  }

  std::vector<void*> host_compressed_ptrs(batch_size);
  auto* comp_ptr = compressed.data_ptr<uint8_t>();
  for (size_t i = 0; i < batch_size; ++i) {
    host_compressed_ptrs[i] = comp_ptr + (i * max_out_bytes);
  }

  auto uncompressed_ptrs_dev = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(input.device()));
  auto compressed_ptrs_dev = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(input.device()));

  check_cuda(
      cudaMemcpyAsync(
          uncompressed_ptrs_dev.data_ptr<int64_t>(),
          host_uncompressed_ptrs.data(),
          sizeof(void*) * batch_size,
          cudaMemcpyHostToDevice,
          cuda_stream),
      "cudaMemcpyAsync uncompressed_ptrs_dev failed");

  check_cuda(
      cudaMemcpyAsync(
          compressed_ptrs_dev.data_ptr<int64_t>(),
          host_compressed_ptrs.data(),
          sizeof(void*) * batch_size,
          cudaMemcpyHostToDevice,
          cuda_stream),
      "cudaMemcpyAsync compressed_ptrs_dev failed");

  auto comp_statuses = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt32).device(input.device()));
  check_nvcomp(
      nvcompBatchedZstdCompressAsync(
          reinterpret_cast<void**>(uncompressed_ptrs_dev.data_ptr<int64_t>()),
          reinterpret_cast<size_t*>(orig_sizes_dev.data_ptr<int64_t>()),
          chunk_sz,
          batch_size,
          reinterpret_cast<void*>(temp.data_ptr<uint8_t>()),
          temp_bytes,
          reinterpret_cast<void**>(compressed_ptrs_dev.data_ptr<int64_t>()),
          reinterpret_cast<size_t*>(comp_sizes_dev.data_ptr<int64_t>()),
          nvcompBatchedZstdCompressDefaultOpts,
          reinterpret_cast<nvcompStatus_t*>(comp_statuses.data_ptr<int32_t>()),
          cuda_stream),
      "nvcompBatchedZstdCompressAsync failed");

  check_cuda(
      cudaMemcpyAsync(
          sizes_cpu.data_ptr<int64_t>(),
          comp_sizes_dev.data_ptr<int64_t>(),
          sizeof(int64_t) * batch_size,
          cudaMemcpyDeviceToHost,
          cuda_stream),
      "cudaMemcpyAsync sizes_cpu failed");

  check_cuda(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize failed");

  return std::make_tuple(
      compressed,
      sizes_cpu,
      orig_sizes_cpu,
      static_cast<int64_t>(max_out_bytes),
      static_cast<int64_t>(chunk_sz));
}

torch::Tensor zstd_decompress(
    torch::Tensor compressed,
    torch::Tensor compressed_sizes_cpu,
    torch::Tensor orig_sizes_cpu,
    int64_t max_out_bytes,
    int64_t chunk_size)
{
  TORCH_CHECK(compressed.is_cuda(), "compressed must be a CUDA tensor");
  TORCH_CHECK(compressed.is_contiguous(), "compressed must be contiguous");
  TORCH_CHECK(compressed.scalar_type() == torch::kUInt8, "compressed must be uint8");
  TORCH_CHECK(compressed_sizes_cpu.device().is_cpu(), "compressed_sizes must be on CPU");
  TORCH_CHECK(orig_sizes_cpu.device().is_cpu(), "orig_sizes must be on CPU");
  TORCH_CHECK(compressed_sizes_cpu.scalar_type() == torch::kInt64, "compressed_sizes must be int64");
  TORCH_CHECK(orig_sizes_cpu.scalar_type() == torch::kInt64, "orig_sizes must be int64");
  TORCH_CHECK(chunk_size > 0, "chunk_size must be > 0");
  TORCH_CHECK(max_out_bytes > 0, "max_out_bytes must be > 0");
  TORCH_CHECK(compressed_sizes_cpu.numel() == orig_sizes_cpu.numel(), "sizes must have same length");

  c10::cuda::CUDAGuard guard(compressed.device());
  auto stream = at::cuda::getDefaultCUDAStream();
  cudaStream_t cuda_stream = stream.stream();

  const size_t batch_size = static_cast<size_t>(compressed_sizes_cpu.numel());
  const size_t chunk_sz = static_cast<size_t>(chunk_size);
  const size_t max_out = static_cast<size_t>(max_out_bytes);

  auto* orig_sizes_ptr = reinterpret_cast<int64_t*>(orig_sizes_cpu.data_ptr<int64_t>());
  std::vector<size_t> orig_sizes(batch_size);
  size_t total_out = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    const size_t sz = static_cast<size_t>(orig_sizes_ptr[i]);
    orig_sizes[i] = sz;
    total_out += sz;
  }

  auto output = torch::empty({static_cast<int64_t>(total_out)}, torch::dtype(torch::kUInt8).device(compressed.device()));

  auto comp_sizes_dev = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(compressed.device()));
  auto orig_sizes_dev = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(compressed.device()));

  check_cuda(
      cudaMemcpyAsync(
          comp_sizes_dev.data_ptr<int64_t>(),
          compressed_sizes_cpu.data_ptr<int64_t>(),
          sizeof(int64_t) * batch_size,
          cudaMemcpyHostToDevice,
          cuda_stream),
      "cudaMemcpyAsync comp_sizes_dev failed");

  check_cuda(
      cudaMemcpyAsync(
          orig_sizes_dev.data_ptr<int64_t>(),
          orig_sizes_cpu.data_ptr<int64_t>(),
          sizeof(int64_t) * batch_size,
          cudaMemcpyHostToDevice,
          cuda_stream),
      "cudaMemcpyAsync orig_sizes_dev failed");

  std::vector<void*> host_compressed_ptrs(batch_size);
  auto* comp_ptr = compressed.data_ptr<uint8_t>();
  for (size_t i = 0; i < batch_size; ++i) {
    host_compressed_ptrs[i] = comp_ptr + (i * max_out);
  }

  std::vector<void*> host_output_ptrs(batch_size);
  auto* out_ptr = output.data_ptr<uint8_t>();
  size_t offset = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    host_output_ptrs[i] = out_ptr + offset;
    offset += orig_sizes[i];
  }

  auto compressed_ptrs_dev = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(compressed.device()));
  auto output_ptrs_dev = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(compressed.device()));

  check_cuda(
      cudaMemcpyAsync(
          compressed_ptrs_dev.data_ptr<int64_t>(),
          host_compressed_ptrs.data(),
          sizeof(void*) * batch_size,
          cudaMemcpyHostToDevice,
          cuda_stream),
      "cudaMemcpyAsync compressed_ptrs_dev failed");

  check_cuda(
      cudaMemcpyAsync(
          output_ptrs_dev.data_ptr<int64_t>(),
          host_output_ptrs.data(),
          sizeof(void*) * batch_size,
          cudaMemcpyHostToDevice,
          cuda_stream),
      "cudaMemcpyAsync output_ptrs_dev failed");

  size_t temp_bytes = 0;
  check_nvcomp(
      nvcompBatchedZstdDecompressGetTempSizeAsync(
          batch_size,
          chunk_sz,
          nvcompBatchedZstdDecompressDefaultOpts,
          &temp_bytes,
          total_out),
      "nvcompBatchedZstdDecompressGetTempSizeAsync failed");

  auto temp = torch::empty({static_cast<int64_t>(temp_bytes)}, torch::dtype(torch::kUInt8).device(compressed.device()));
  auto statuses = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt32).device(compressed.device()));
  auto actual_sizes = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(compressed.device()));

  check_nvcomp(
      nvcompBatchedZstdDecompressAsync(
          reinterpret_cast<void**>(compressed_ptrs_dev.data_ptr<int64_t>()),
          reinterpret_cast<size_t*>(comp_sizes_dev.data_ptr<int64_t>()),
          reinterpret_cast<size_t*>(orig_sizes_dev.data_ptr<int64_t>()),
          reinterpret_cast<size_t*>(actual_sizes.data_ptr<int64_t>()),
          batch_size,
          reinterpret_cast<void*>(temp.data_ptr<uint8_t>()),
          temp_bytes,
          reinterpret_cast<void**>(output_ptrs_dev.data_ptr<int64_t>()),
          nvcompBatchedZstdDecompressDefaultOpts,
          reinterpret_cast<nvcompStatus_t*>(statuses.data_ptr<int32_t>()),
          cuda_stream),
      "nvcompBatchedZstdDecompressAsync failed");

  check_cuda(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize failed");

  return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t>
cascaded_compress(torch::Tensor input, int64_t chunk_size)
{
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(chunk_size > 0, "chunk_size must be > 0");

  const auto type_info = get_cascaded_type(input.scalar_type());

  c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getDefaultCUDAStream();
  cudaStream_t cuda_stream = stream.stream();

  const size_t in_bytes = static_cast<size_t>(input.numel()) * input.element_size();
  const size_t chunk_sz = static_cast<size_t>(chunk_size);
  TORCH_CHECK(in_bytes % type_info.size == 0, "input byte size must be a multiple of element size");
  TORCH_CHECK(chunk_sz % type_info.size == 0, "chunk_size must be a multiple of element size");

  const size_t batch_size = (in_bytes + chunk_sz - 1) / chunk_sz;

  auto sizes_cpu = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(torch::kCPU));
  auto orig_sizes_cpu = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(torch::kCPU));

  auto* orig_sizes_cpu_ptr = reinterpret_cast<int64_t*>(orig_sizes_cpu.data_ptr<int64_t>());
  for (size_t i = 0; i < batch_size; ++i) {
    const size_t remaining = in_bytes - (i * chunk_sz);
    const size_t this_size = remaining < chunk_sz ? remaining : chunk_sz;
    TORCH_CHECK(this_size % type_info.size == 0, "chunk size must be a multiple of element size");
    orig_sizes_cpu_ptr[i] = static_cast<int64_t>(this_size);
  }

  auto compress_opts = nvcompBatchedCascadedCompressDefaultOpts;
  compress_opts.type = type_info.type;

  size_t temp_bytes = 0;
  check_nvcomp(
      nvcompBatchedCascadedCompressGetTempSizeAsync(
          batch_size,
          chunk_sz,
          compress_opts,
          &temp_bytes,
          in_bytes),
      "nvcompBatchedCascadedCompressGetTempSizeAsync failed");

  size_t max_out_bytes = 0;
  check_nvcomp(
      nvcompBatchedCascadedCompressGetMaxOutputChunkSize(
          chunk_sz,
          compress_opts,
          &max_out_bytes),
      "nvcompBatchedCascadedCompressGetMaxOutputChunkSize failed");

  auto temp = torch::empty({static_cast<int64_t>(temp_bytes)}, torch::dtype(torch::kUInt8).device(input.device()));
  auto compressed = torch::empty({static_cast<int64_t>(batch_size * max_out_bytes)}, torch::dtype(torch::kUInt8).device(input.device()));
  auto comp_sizes_dev = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(input.device()));
  auto orig_sizes_dev = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(input.device()));

  check_cuda(
      cudaMemcpyAsync(
          orig_sizes_dev.data_ptr<int64_t>(),
          orig_sizes_cpu.data_ptr<int64_t>(),
          sizeof(int64_t) * batch_size,
          cudaMemcpyHostToDevice,
          cuda_stream),
      "cudaMemcpyAsync orig_sizes_dev failed");

  std::vector<void*> host_uncompressed_ptrs(batch_size);
  auto* input_ptr = reinterpret_cast<uint8_t*>(input.data_ptr());
  for (size_t i = 0; i < batch_size; ++i) {
    host_uncompressed_ptrs[i] = input_ptr + (i * chunk_sz);
  }

  std::vector<void*> host_compressed_ptrs(batch_size);
  auto* comp_ptr = compressed.data_ptr<uint8_t>();
  for (size_t i = 0; i < batch_size; ++i) {
    host_compressed_ptrs[i] = comp_ptr + (i * max_out_bytes);
  }

  auto uncompressed_ptrs_dev = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(input.device()));
  auto compressed_ptrs_dev = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(input.device()));

  check_cuda(
      cudaMemcpyAsync(
          uncompressed_ptrs_dev.data_ptr<int64_t>(),
          host_uncompressed_ptrs.data(),
          sizeof(void*) * batch_size,
          cudaMemcpyHostToDevice,
          cuda_stream),
      "cudaMemcpyAsync uncompressed_ptrs_dev failed");

  check_cuda(
      cudaMemcpyAsync(
          compressed_ptrs_dev.data_ptr<int64_t>(),
          host_compressed_ptrs.data(),
          sizeof(void*) * batch_size,
          cudaMemcpyHostToDevice,
          cuda_stream),
      "cudaMemcpyAsync compressed_ptrs_dev failed");

  auto comp_statuses = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt32).device(input.device()));
  check_nvcomp(
      nvcompBatchedCascadedCompressAsync(
          reinterpret_cast<void**>(uncompressed_ptrs_dev.data_ptr<int64_t>()),
          reinterpret_cast<size_t*>(orig_sizes_dev.data_ptr<int64_t>()),
          chunk_sz,
          batch_size,
          reinterpret_cast<void*>(temp.data_ptr<uint8_t>()),
          temp_bytes,
          reinterpret_cast<void**>(compressed_ptrs_dev.data_ptr<int64_t>()),
          reinterpret_cast<size_t*>(comp_sizes_dev.data_ptr<int64_t>()),
          compress_opts,
          reinterpret_cast<nvcompStatus_t*>(comp_statuses.data_ptr<int32_t>()),
          cuda_stream),
      "nvcompBatchedCascadedCompressAsync failed");

  check_cuda(
      cudaMemcpyAsync(
          sizes_cpu.data_ptr<int64_t>(),
          comp_sizes_dev.data_ptr<int64_t>(),
          sizeof(int64_t) * batch_size,
          cudaMemcpyDeviceToHost,
          cuda_stream),
      "cudaMemcpyAsync sizes_cpu failed");

  check_cuda(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize failed");

  return std::make_tuple(
      compressed,
      sizes_cpu,
      orig_sizes_cpu,
      static_cast<int64_t>(max_out_bytes),
      static_cast<int64_t>(chunk_sz));
}

torch::Tensor cascaded_decompress(
    torch::Tensor compressed,
    torch::Tensor compressed_sizes_cpu,
    torch::Tensor orig_sizes_cpu,
    int64_t max_out_bytes,
    int64_t chunk_size)
{
  TORCH_CHECK(compressed.is_cuda(), "compressed must be a CUDA tensor");
  TORCH_CHECK(compressed.is_contiguous(), "compressed must be contiguous");
  TORCH_CHECK(compressed.scalar_type() == torch::kUInt8, "compressed must be uint8");
  TORCH_CHECK(compressed_sizes_cpu.device().is_cpu(), "compressed_sizes must be on CPU");
  TORCH_CHECK(orig_sizes_cpu.device().is_cpu(), "orig_sizes must be on CPU");
  TORCH_CHECK(compressed_sizes_cpu.scalar_type() == torch::kInt64, "compressed_sizes must be int64");
  TORCH_CHECK(orig_sizes_cpu.scalar_type() == torch::kInt64, "orig_sizes must be int64");
  TORCH_CHECK(chunk_size > 0, "chunk_size must be > 0");
  TORCH_CHECK(max_out_bytes > 0, "max_out_bytes must be > 0");
  TORCH_CHECK(compressed_sizes_cpu.numel() == orig_sizes_cpu.numel(), "sizes must have same length");

  c10::cuda::CUDAGuard guard(compressed.device());
  auto stream = at::cuda::getDefaultCUDAStream();
  cudaStream_t cuda_stream = stream.stream();

  const size_t batch_size = static_cast<size_t>(compressed_sizes_cpu.numel());
  const size_t chunk_sz = static_cast<size_t>(chunk_size);
  const size_t max_out = static_cast<size_t>(max_out_bytes);

  auto* orig_sizes_ptr = reinterpret_cast<int64_t*>(orig_sizes_cpu.data_ptr<int64_t>());
  std::vector<size_t> orig_sizes(batch_size);
  size_t total_out = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    const size_t sz = static_cast<size_t>(orig_sizes_ptr[i]);
    orig_sizes[i] = sz;
    total_out += sz;
  }

  auto output = torch::empty({static_cast<int64_t>(total_out)}, torch::dtype(torch::kUInt8).device(compressed.device()));

  auto comp_sizes_dev = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(compressed.device()));
  auto orig_sizes_dev = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(compressed.device()));

  check_cuda(
      cudaMemcpyAsync(
          comp_sizes_dev.data_ptr<int64_t>(),
          compressed_sizes_cpu.data_ptr<int64_t>(),
          sizeof(int64_t) * batch_size,
          cudaMemcpyHostToDevice,
          cuda_stream),
      "cudaMemcpyAsync comp_sizes_dev failed");

  check_cuda(
      cudaMemcpyAsync(
          orig_sizes_dev.data_ptr<int64_t>(),
          orig_sizes_cpu.data_ptr<int64_t>(),
          sizeof(int64_t) * batch_size,
          cudaMemcpyHostToDevice,
          cuda_stream),
      "cudaMemcpyAsync orig_sizes_dev failed");

  std::vector<void*> host_compressed_ptrs(batch_size);
  auto* comp_ptr = compressed.data_ptr<uint8_t>();
  for (size_t i = 0; i < batch_size; ++i) {
    host_compressed_ptrs[i] = comp_ptr + (i * max_out);
  }

  std::vector<void*> host_output_ptrs(batch_size);
  auto* out_ptr = output.data_ptr<uint8_t>();
  size_t offset = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    host_output_ptrs[i] = out_ptr + offset;
    offset += orig_sizes[i];
  }

  auto compressed_ptrs_dev = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(compressed.device()));
  auto output_ptrs_dev = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(compressed.device()));

  check_cuda(
      cudaMemcpyAsync(
          compressed_ptrs_dev.data_ptr<int64_t>(),
          host_compressed_ptrs.data(),
          sizeof(void*) * batch_size,
          cudaMemcpyHostToDevice,
          cuda_stream),
      "cudaMemcpyAsync compressed_ptrs_dev failed");

  check_cuda(
      cudaMemcpyAsync(
          output_ptrs_dev.data_ptr<int64_t>(),
          host_output_ptrs.data(),
          sizeof(void*) * batch_size,
          cudaMemcpyHostToDevice,
          cuda_stream),
      "cudaMemcpyAsync output_ptrs_dev failed");

  size_t temp_bytes = 0;
  check_nvcomp(
      nvcompBatchedCascadedDecompressGetTempSizeAsync(
          batch_size,
          chunk_sz,
          nvcompBatchedCascadedDecompressDefaultOpts,
          &temp_bytes,
          total_out),
      "nvcompBatchedCascadedDecompressGetTempSizeAsync failed");

  auto temp = torch::empty({static_cast<int64_t>(temp_bytes)}, torch::dtype(torch::kUInt8).device(compressed.device()));
  auto statuses = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt32).device(compressed.device()));
  auto actual_sizes = torch::empty({static_cast<int64_t>(batch_size)}, torch::dtype(torch::kInt64).device(compressed.device()));

  check_nvcomp(
      nvcompBatchedCascadedDecompressAsync(
          reinterpret_cast<void**>(compressed_ptrs_dev.data_ptr<int64_t>()),
          reinterpret_cast<size_t*>(comp_sizes_dev.data_ptr<int64_t>()),
          reinterpret_cast<size_t*>(orig_sizes_dev.data_ptr<int64_t>()),
          reinterpret_cast<size_t*>(actual_sizes.data_ptr<int64_t>()),
          batch_size,
          reinterpret_cast<void*>(temp.data_ptr<uint8_t>()),
          temp_bytes,
          reinterpret_cast<void**>(output_ptrs_dev.data_ptr<int64_t>()),
          nvcompBatchedCascadedDecompressDefaultOpts,
          reinterpret_cast<nvcompStatus_t*>(statuses.data_ptr<int32_t>()),
          cuda_stream),
      "nvcompBatchedCascadedDecompressAsync failed");

  check_cuda(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize failed");

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("lz4_compress", &lz4_compress, "nvcomp LZ4 compress (CUDA)");
  m.def("lz4_decompress", &lz4_decompress, "nvcomp LZ4 decompress (CUDA)");
  m.def("zstd_compress", &zstd_compress, "nvcomp Zstd compress (CUDA)");
  m.def("zstd_decompress", &zstd_decompress, "nvcomp Zstd decompress (CUDA)");
  m.def("cascaded_compress", &cascaded_compress, "nvcomp Cascaded compress (CUDA)");
  m.def("cascaded_decompress", &cascaded_decompress, "nvcomp Cascaded decompress (CUDA)");
}

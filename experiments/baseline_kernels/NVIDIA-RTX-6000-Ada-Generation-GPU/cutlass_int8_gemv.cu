/*
 * Conservative INT8xINT8 GEMV kernel using CUTLASS tensor cores on Ada (SM89).
 *
 * This version intentionally keeps the tuning simple:
 *   - a single GEMV-oriented tile configuration
 *   - default swizzle (no extra L2 locality tuning)
 *   - a shallow pipeline
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

using ElementA = int8_t;
using LayoutA = cutlass::layout::RowMajor;
using ElementB = int8_t;
using LayoutB = cutlass::layout::ColumnMajor;
using ElementC = int32_t;
using LayoutC = cutlass::layout::RowMajor;
using ElementAccum = int32_t;

using GemvInt8 = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccum,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 128, 64>,
    cutlass::gemm::GemmShape<32, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<ElementC, 4, ElementAccum, ElementAccum>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2>;

torch::Tensor cutlass_int8_gemv(
    torch::Tensor A,
    torch::Tensor B,
    int M,
    int N,
    int K) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kInt8 && B.dtype() == torch::kInt8, "Inputs must be int8");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(M <= 32, "This kernel is intended for GEMV-sized M <= 32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be rank-2 tensors");
    TORCH_CHECK(A.size(0) == M && A.size(1) == K, "A shape does not match M/K");
    TORCH_CHECK(B.size(0) == N && B.size(1) == K, "B shape does not match N/K");

    auto C = torch::zeros(
        {M, N},
        torch::TensorOptions().dtype(torch::kInt32).device(A.device()));

    auto *ptr_A = A.data_ptr<int8_t>();
    auto *ptr_B = B.data_ptr<int8_t>();
    auto *ptr_C = C.data_ptr<int32_t>();

    int lda = K;
    int ldb = K;
    int ldc = N;

    GemvInt8 gemm_op;
    typename GemvInt8::Arguments args(
        {M, N, K},
        {ptr_A, lda},
        {ptr_B, ldb},
        {ptr_C, ldc},
        {ptr_C, ldc},
        {1, 0});

    cutlass::Status status = gemm_op.can_implement(args);
    TORCH_CHECK(
        status == cutlass::Status::kSuccess,
        "CUTLASS INT8 GEMV cannot implement: ",
        cutlassGetStatusString(status));

    status = gemm_op(args);
    TORCH_CHECK(
        status == cutlass::Status::kSuccess,
        "CUTLASS INT8 GEMV failed: ",
        cutlassGetStatusString(status));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8_gemv", &cutlass_int8_gemv, "Conservative INT8xINT8 GEMV via CUTLASS");
}

/*
 * Conservative INT4xINT4 GEMV kernel using CUTLASS tensor cores on Ada (SM89).
 *
 * Inputs are packed in CUTLASS int4b_t layout: two signed INT4 values per byte,
 * packed along the K dimension. The tuning is intentionally modest and keeps
 * a single GEMV-oriented tile.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/integer_subbyte.h"
#include "cutlass/layout/matrix.h"

using ElementA = cutlass::int4b_t;
using LayoutA = cutlass::layout::RowMajor;
using ElementB = cutlass::int4b_t;
using LayoutB = cutlass::layout::ColumnMajor;
using ElementC = int32_t;
using LayoutC = cutlass::layout::RowMajor;
using ElementAccum = int32_t;

using GemvInt4 = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccum,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 128, 128>,
    cutlass::gemm::GemmShape<32, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    cutlass::epilogue::thread::LinearCombination<ElementC, 4, ElementAccum, ElementAccum>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2>;

torch::Tensor cutlass_int4_gemv(
    torch::Tensor A_packed,
    torch::Tensor B_packed,
    int M,
    int N,
    int K) {
    TORCH_CHECK(A_packed.is_cuda() && B_packed.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(
        A_packed.dtype() == torch::kUInt8 && B_packed.dtype() == torch::kUInt8,
        "Packed INT4 inputs must use uint8 storage");
    TORCH_CHECK(A_packed.is_contiguous() && B_packed.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(M <= 32, "This kernel is intended for GEMV-sized M <= 32");
    TORCH_CHECK(K % 2 == 0, "K must be even for INT4 packing");
    TORCH_CHECK(A_packed.dim() == 2 && B_packed.dim() == 2, "Inputs must be rank-2 tensors");
    TORCH_CHECK(A_packed.size(0) == M && A_packed.size(1) == K / 2, "A_packed shape mismatch");
    TORCH_CHECK(B_packed.size(0) == N && B_packed.size(1) == K / 2, "B_packed shape mismatch");

    auto C = torch::zeros(
        {M, N},
        torch::TensorOptions().dtype(torch::kInt32).device(A_packed.device()));

    auto *ptr_A = reinterpret_cast<ElementA const *>(A_packed.data_ptr<uint8_t>());
    auto *ptr_B = reinterpret_cast<ElementB const *>(B_packed.data_ptr<uint8_t>());
    auto *ptr_C = C.data_ptr<int32_t>();

    int lda = K;
    int ldb = K;
    int ldc = N;

    GemvInt4 gemm_op;
    typename GemvInt4::Arguments args(
        {M, N, K},
        {ptr_A, lda},
        {ptr_B, ldb},
        {ptr_C, ldc},
        {ptr_C, ldc},
        {1, 0});

    cutlass::Status status = gemm_op.can_implement(args);
    TORCH_CHECK(
        status == cutlass::Status::kSuccess,
        "CUTLASS INT4 GEMV cannot implement: ",
        cutlassGetStatusString(status));

    status = gemm_op(args);
    TORCH_CHECK(
        status == cutlass::Status::kSuccess,
        "CUTLASS INT4 GEMV failed: ",
        cutlassGetStatusString(status));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int4_gemv", &cutlass_int4_gemv, "Conservative INT4xINT4 GEMV via CUTLASS");
}

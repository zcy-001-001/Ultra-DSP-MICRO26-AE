#pragma once

#include "forward.h"
#include <hls_stream.h>

// Task descriptor for a single matmul operation routed through the shared engine.
// All matmuls (Q, K, V, Wo, W1, W3, W2, Wcls) can be expressed as one of these tasks.
typedef struct {
    float       *xout;  // output vector [N]
    const int8_t *xq;   // quantized input vector [K]
    const float  *xs;   // input scale(s)
    const int8_t *wq;   // quantized weight matrix [N,K]
    const float  *ws;   // weight scales [N]
    int M;
    int N;
    int K;
    bool last;          // marks the final task in a sequence
} MatmulTask_t;

// Shared matmul engine worker: consumes MatmulTask_t commands from a stream
// and executes them one-by-one using the single matmul_engine implementation.
//
// NOTE: 调度/控制逻辑（例如在 transformer_layer_pipeline 中）会负责：
//   - 按顺序写入所有需要的 MatmulTask_t 到 task_stream
//   - 对最后一个任务将 last 置为 true
//   - 在需要使用结果前，从 done_stream 读取一次完成标记
// 这样可以在 FPGA 上通过一份统一的 PE 阵列串行完成所有 matmul 计算。
static void matmul_engine_worker(
    hls::stream<MatmulTask_t> &task_stream,
    hls::stream<bool>         &done_stream) {
#pragma HLS INLINE off

worker_loop:
    while (true) {
        // 阻塞式读取一条任务，保证调度和计算严格按顺序执行
        MatmulTask_t task = task_stream.read();

        // 执行一次通用 matmul（共享的 PE 阵列）
        matmul_engine(
            task.xout,
            task.xq,
            task.xs,
            task.wq,
            task.ws,
            task.M,
            task.N,
            task.K);

        // 向调度端报告本次任务已完成
        done_stream.write(true);

        // 如果这是最后一条任务，则退出循环
        if (task.last) {
            break;
        }
    }
}

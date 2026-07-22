#ifndef FORWARD_H
#define FORWARD_H

#include "hls_stream.h" // Needed for hls::stream
#include "ap_int.h"
#include "config.h"     // Defines constants like dim, GS, etc.
#include "typedefs.h"   // Defines Config, Transformer, QuantizedTensor structs etc. (NOW CORRECTED)
#include <cstring>      // For memcpy, memset
#include <cmath>        // For fabs, round, sqrtf, expf, cosf, sinf, powf


// ----------------------------------------------------------------------------
// Typedef for the specific TransformerWeights instantiation using config.h constants
// --- UPDATED typedef to pass GS ---
// ----------------------------------------------------------------------------
typedef TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> ModelWeights_t;

// Sharded weights typedef for 8-port matmul (SHARDS ports)
typedef MatmulShardWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                           vocab_size, seq_len, GS, SHARDS> ShardWeights_t;

// ----------------------------------------------------------------------------
// Declarations for the split kernels (No changes needed here)
// ----------------------------------------------------------------------------

extern "C" {

void initial_embedding_lookup(
    const float* token_embedding_table, // Note: This kernel directly accesses the float table part
    int token,
    hls::stream<float>& stream_out_x
);

// --- ADDED: 新的合并了层循环的 Kernel 声明 ---
void transformer_layer_pipeline(
    hls::stream<float>& stream_initial_in, // 来自 embedding 的输入流
    hls::stream<float>& stream_final_out,  // 输出到 final norm 的流
    const ShardWeights_t* w_p0, const ShardWeights_t* w_p1, const ShardWeights_t* w_p2, const ShardWeights_t* w_p3,
    const ShardWeights_t* w_p4, const ShardWeights_t* w_p5, const ShardWeights_t* w_p6, const ShardWeights_t* w_p7,
    const ShardWeights_t* w_p8, const ShardWeights_t* w_p9, const ShardWeights_t* w_p10, const ShardWeights_t* w_p11,
    const ShardWeights_t* w_p12, const ShardWeights_t* w_p13, const ShardWeights_t* w_p14, const ShardWeights_t* w_p15,
    const ShardWeights_t* w_p16, const ShardWeights_t* w_p17, const ShardWeights_t* w_p18, const ShardWeights_t* w_p19,
    const ShardWeights_t* w_p20, const ShardWeights_t* w_p21, const ShardWeights_t* w_p22, const ShardWeights_t* w_p23,
    const ShardWeights_t* w_p24, const ShardWeights_t* w_p25, const ShardWeights_t* w_p26, const ShardWeights_t* w_p27,
    const ShardWeights_t* w_p28, const ShardWeights_t* w_p29, const ShardWeights_t* w_p30, const ShardWeights_t* w_p31,
    int pos,                               // 当前位置
    half* key_cache,                      // KV Cache 指针
    half* value_cache                     // KV Cache 指针                       // 组大小 (可能未使用)
);

void final_norm_classifier(
    hls::stream<float>& stream_in_x,
    float* logits_out,
    const ModelWeights_t* w, // Uses the corrected typedef
    float GS_val // Keep this if quantize needs a runtime float value, otherwise remove/ignore
);

} // extern "C"

// ----------------------------------------------------------------------------
// Template function definitions (kept in header for template instantiation)
// --- UPDATED quantize/dequantize signatures ---
// ----------------------------------------------------------------------------

// --- UPDATED Signature ---
template <int col, int row> // Added GROUP_SIZE template parameter
void dequantize(const WeightQuantizedTensor<col, row> *qx, float x[col * row]) { // Made qx const, Renamed param
    // Interpret WeightQuantizedTensor as having shape [col (out_dim), row (in_dim)]
    // with one scale per output channel (per "col").
    for (int c = 0; c < col; c++) {
        float s = qx->s[c];
        int base = c * row;
        for (int r = 0; r < row; r++) {
            #pragma HLS PIPELINE II=1
            x[base + r] = qx->q[base + r] * s;
        }
    }
}

typedef union {
    float f32;
    uint32_t u32;
} float_uint32_union;

// 快速转换函数
inline uint32_t float_to_uint32(float a) {
    float_uint32_union u;
    u.f32 = a;
    return u.u32;
}

inline float uint32_to_float(uint32_t a) {
    float_uint32_union u;
    u.u32 = a;
    return u.f32;
}

// ============================================================================
// Helper: Pointer-based Quantization (Unified Memory Compatible)
// 使用位操作近似量化，避免除法
// ============================================================================
template <int col, int row>
void quantize_ptr(int8_t* q_out, float* s_out, const float* x_in) {
#pragma HLS INLINE off
    constexpr float Q_MAX = 127.0f;
    float x_local[col * row];
    #pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=32

    // 拷贝输入到本地缓冲
copy_input:
    for (int i = 0; i < col * row; i+=16) {
    #pragma HLS PIPELINE II=1
        for (int j = 0; j < 16; j++) {
        #pragma HLS UNROLL
            x_local[i + j] = x_in[i + j];
        }
    }

main_loop:
    for (int r = 0; r < row; r++) {
        const int base_idx = r * col;

        // 第一遍：找最大绝对值（32 路并行累加器，消除循环携带依赖）
        float wmax_part[32];
        #pragma HLS ARRAY_PARTITION variable=wmax_part complete

    init_wmax:
        for (int k = 0; k < 32; k++) {
        #pragma HLS UNROLL
            wmax_part[k] = 0.0f;
        }

    max_val_loop:
        for (int i = 0; i < col; i += 32) {
        #pragma HLS PIPELINE II=2
        max_val_inner:
            for (int k = 0; k < 32; k++) {
            #pragma HLS UNROLL
                int off = i + k;
                if (off < col) {
                    float v = x_local[base_idx + off];
                    v = (v >= 0.0f) ? v : -v;
                    // 每路只更新自己的 wmax_part[k]，无跨路依赖
                    wmax_part[k] = (v > wmax_part[k]) ? v : wmax_part[k];
                }
            }
        }

        // 树形归约 32 → 1（分层流水打拍，每层 1 cycle）
        float reduce_buf[32];
        #pragma HLS ARRAY_PARTITION variable=reduce_buf complete
        
    copy_to_reduce: // 拷贝到归约缓冲
        for (int k = 0; k < 32; k++) {
        #pragma HLS UNROLL
            reduce_buf[k] = wmax_part[k];
        }
        
    reduce_s1: // Stage 1: 32 → 16
        for (int k = 0; k < 16; k++) {
        #pragma HLS UNROLL
        #pragma HLS LATENCY min=1 max=1
            float a = reduce_buf[k * 2];
            float b = reduce_buf[k * 2 + 1];
            reduce_buf[k] = (a > b) ? a : b;
        }
        
    reduce_s2: // Stage 2: 16 → 8
        for (int k = 0; k < 8; k++) {
        #pragma HLS UNROLL
        #pragma HLS LATENCY min=1 max=1
            float a = reduce_buf[k * 2];
            float b = reduce_buf[k * 2 + 1];
            reduce_buf[k] = (a > b) ? a : b;
        }

    reduce_s3: // Stage 3: 8 → 4
        for (int k = 0; k < 4; k++) {
        #pragma HLS UNROLL
        #pragma HLS LATENCY min=1 max=1
            float a = reduce_buf[k * 2];
            float b = reduce_buf[k * 2 + 1];
            reduce_buf[k] = (a > b) ? a : b;
        }

    reduce_s4: // Stage 4: 4 → 2
        for (int k = 0; k < 2; k++) {
        #pragma HLS UNROLL
        #pragma HLS LATENCY min=1 max=1
            float a = reduce_buf[k * 2];
            float b = reduce_buf[k * 2 + 1];
            reduce_buf[k] = (a > b) ? a : b;
        }

        // Stage 5: 2 → 1
        float wmax = (reduce_buf[0] > reduce_buf[1]) ? reduce_buf[0] : reduce_buf[1];

        // 用位操作计算 scale = wmax / Q_MAX (避免除法)
        float_uint32_union wmu, qmu, su;
        wmu.f32 = wmax;
        qmu.f32 = Q_MAX;
        uint32_t tempu1 = wmu.u32 + 0x3F780000;
        uint32_t resultu1 = tempu1 - qmu.u32;
        su.u32 = resultu1;
        s_out[r] = su.f32;

        // 第二遍：量化（32 路并行，用位操作避免除法）
    quant_loop:
        for (int i = 0; i < col; i += 32) {
        #pragma HLS PIPELINE II=2
        quant_inner:
            for (int k = 0; k < 32; k++) {
            #pragma HLS UNROLL
                int off = i + k;
                if (off < col) {
                    int idx = base_idx + off;
                    float val = x_local[idx];
                    float_uint32_union xu, qu;
                    xu.f32 = val;
                    uint32_t tempu2 = xu.u32 + 0x3F780000;
                    uint32_t resultu2 = tempu2 - su.u32;
                    qu.u32 = resultu2;
                    q_out[idx] = (int8_t)qu.f32;
                }
            }
        }
    }
}

template <int S>
void rmsnorm(float o[S], const float x[S], const float weight[S]) {
#pragma HLS INLINE off
    
    float x_buff[S];
    float weight_buff[S];
    float out_buff[S];
    
    // 64 路并行分区（支持 64 路并行读取）
    #pragma HLS ARRAY_PARTITION variable=x_buff cyclic factor=64
    #pragma HLS ARRAY_PARTITION variable=weight_buff cyclic factor=64
    #pragma HLS ARRAY_PARTITION variable=out_buff cyclic factor=64

    // Stage 1: 加载数据 (16 路并行)
load_x:
    for (int j = 0; j < S; j += 16) {
    #pragma HLS PIPELINE II=1
        for (int k = 0; k < 16; k++) {
        #pragma HLS UNROLL
            x_buff[j + k] = x[j + k];
        }
    }
    
load_weight:
    for (int j = 0; j < S; j += 16) {
    #pragma HLS PIPELINE II=1
        for (int k = 0; k < 16; k++) {
        #pragma HLS UNROLL
            weight_buff[j + k] = weight[j + k];
        }
    }

    // Stage 2: 计算平方和 (64 路并行, II=4)
    // 消除 if (prev_valid) 条件分支，避免 select 在关键路径上
    const int BLOCK = 64;
    float ss[BLOCK];
    #pragma HLS ARRAY_PARTITION variable=ss complete
    float prev_x[BLOCK];
    #pragma HLS ARRAY_PARTITION variable=prev_x complete

init_ss:
    for (int k = 0; k < BLOCK; k++) {
    #pragma HLS UNROLL
        ss[k] = 0.0f;
        prev_x[k] = 0.0f;  // 初始化为 0，第一次迭代 0*0=0 不影响累加
    }

rms_sum_loop:
    for (int j = 0; j < S; j += BLOCK) {
    #pragma HLS PIPELINE II=8
    #pragma HLS LOOP_FLATTEN off
        for (int k = 0; k < BLOCK; k++) {
        #pragma HLS UNROLL
            float x_cur = x_buff[j + k];
            float x_prev = prev_x[k];
            // 无条件累加：第一次 prev_x=0，贡献为 0；后续正常累加
            ss[k] += x_prev * x_prev;
            prev_x[k] = x_cur;
        }
    }

    // 处理最后一批 prev_x（循环结束后还有一组数据未累加）
last_acc:
    for (int k = 0; k < BLOCK; k++) {
    #pragma HLS UNROLL
        ss[k] += prev_x[k] * prev_x[k];
    }

    // 树形归约 64 → 1（分层流水打拍，每层 1 cycle）
    float reduce_ss[64];
    #pragma HLS ARRAY_PARTITION variable=reduce_ss complete

copy_ss: // 拷贝到归约缓冲
    for (int k = 0; k < BLOCK; k++) {
    #pragma HLS UNROLL
        reduce_ss[k] = ss[k];
    }

reduce_ss_s0: // Stage 0: 64 → 32
    for (int k = 0; k < 32; k++) {
    #pragma HLS UNROLL
    #pragma HLS LATENCY min=1 max=1
        reduce_ss[k] = reduce_ss[k * 2] + reduce_ss[k * 2 + 1];
    }

reduce_ss_s1: // Stage 1: 32 → 16
    for (int k = 0; k < 16; k++) {
    #pragma HLS UNROLL
    #pragma HLS LATENCY min=1 max=1
        reduce_ss[k] = reduce_ss[k * 2] + reduce_ss[k * 2 + 1];
    }

reduce_ss_s2: // Stage 2: 16 → 8
    for (int k = 0; k < 8; k++) {
    #pragma HLS UNROLL
    #pragma HLS LATENCY min=1 max=1
        reduce_ss[k] = reduce_ss[k * 2] + reduce_ss[k * 2 + 1];
    }

reduce_ss_s3: // Stage 3: 8 → 4
    for (int k = 0; k < 4; k++) {
    #pragma HLS UNROLL
    #pragma HLS LATENCY min=1 max=1
        reduce_ss[k] = reduce_ss[k * 2] + reduce_ss[k * 2 + 1];
    }

reduce_ss_s4: // Stage 4: 4 → 2
    for (int k = 0; k < 2; k++) {
    #pragma HLS UNROLL
    #pragma HLS LATENCY min=1 max=1
        reduce_ss[k] = reduce_ss[k * 2] + reduce_ss[k * 2 + 1];
    }

    // Stage 5: 2 → 1
    float ss_total = reduce_ss[0] + reduce_ss[1];
    ss_total = ss_total / S + 1e-5f;
    float inv_ss = 1.0f / sqrtf(ss_total);

    // Stage 3: 归一化并缩放 (16 路并行)
norm_and_scale:
    for (int j = 0; j < S; j += 16) {
    #pragma HLS PIPELINE II=1
        for (int k = 0; k < 16; k++) {
        #pragma HLS UNROLL
            out_buff[j + k] = weight_buff[j + k] * (inv_ss * x_buff[j + k]);
        }
    }

    // Stage 4: 写回 (16 路并行)
write_out:
    for (int j = 0; j < S; j += 16) {
    #pragma HLS PIPELINE II=1
        for (int k = 0; k < 16; k++) {
        #pragma HLS UNROLL
            o[j + k] = out_buff[j + k];
        }
    }
}

// Softmax 定义保持不变 (它不直接依赖于 QuantizedTensor 的改动)
template <int MAXSIZE>
void softmax(float *x, int size) {
    float buffer[MAXSIZE];
    if (size <= 0) return;
    float max_val0 = x[0];
    float max_val1 = x[1];
max:
    for (int i = 1; i < size; i+=2) {
#pragma HLS loop_tripcount min = 1 max = seq_len avg = seq_len/2 // Use config constants
        if (x[i] > max_val0) { max_val0 = x[i]; }
        if (i + 1 < size) {
            if (x[i + 1] > max_val1) { max_val1 = x[i + 1]; }
        }
    }
    float max_val = max_val0 > max_val1 ? max_val0 : max_val1;
    float sum = 0.0f;
exp_sum: // Merged loop from previous example version
    {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
    exp_sum_loop:
        for (int i = 0; i < size; i += 2) {
        #pragma HLS loop_tripcount min = 1 max = seq_len avg = seq_len/2
            float b0 = expf(x[i] - max_val);
            buffer[i] = b0;
            sum0 = sum0 + b0;

            if (i + 1 < size) {
                float b1 = expf(x[i + 1] - max_val);
                buffer[i + 1] = b1;
                sum1 = sum1 + b1;
            }
        }
        sum = sum0 + sum1;
    }
    const float inv_sum = (sum == 0.0f) ? 0.0f : 1.0f / sum;
norm:
    for (int i = 0; i < size; i++) {
#pragma HLS loop_tripcount min = 1 max = seq_len avg = seq_len/2
        x[i] = buffer[i] * inv_sum;
    }
}



// Helper task: load weight rows into streams (supports M rows of inputs)
static void matmul_load_weights(
    const int8_t *wq,
    const float  *ws,
    int M,
    int N,
    int K,
    hls::stream<ap_uint<512> > &wq_stream,
    hls::stream<float>  &ws_stream) {
#pragma HLS INLINE off

    const ap_uint<512> *wq512 = reinterpret_cast<const ap_uint<512> *>(wq);
    const int words_per_row   = (K + 63) / 64; // number of 512-bit words per row

    float ws_local[vocab_size];

load_ws:
    for (int n = 0; n < N; ++n) {
#pragma HLS PIPELINE II=1
        ws_local[n] = ws[n];
    }

load_m:
    for (int m = 0; m < M; ++m) {
    load_rows:
        for (int n = 0; n < N; ++n) {
            const ap_uint<512> *row_ptr = wq512 + n * words_per_row;
        load_row_words:
            for (int w = 0; w < words_per_row; ++w) {
#pragma HLS PIPELINE II=1
                ap_uint<512> packed = row_ptr[w];
                wq_stream.write(packed);
            }
            ws_stream.write(ws_local[n]);
        }
    }
}

static void matmul_core(
    float *xout_row,
    const int8_t *x_buffer,
    float xs_val,
    int N,
    int K,
    hls::stream<ap_uint<512> > &wq_stream,
    hls::stream<float>  &ws_stream) {
#pragma HLS ALLOCATION function instances=matmul_core limit=1
    const int WORD_BYTES = 64;
    const int PE_COUNT   = 64;
    const int words_per_row = (K + WORD_BYTES - 1) / WORD_BYTES;

compute_n_core:
    for (int n = 0; n < N; ++n) {
        int32_t partial_sums[PE_COUNT];
#pragma HLS ARRAY_PARTITION variable=partial_sums complete

    init_partial_sums_core:
        for (int i = 0; i < PE_COUNT; ++i) {
#pragma HLS UNROLL
            partial_sums[i] = 0;
        }

    load_and_dot_core:
        for (int w = 0; w < words_per_row; ++w) {
#pragma HLS PIPELINE II=1
            ap_uint<512> packed = wq_stream.read();
            const int base = w * WORD_BYTES;
        dot_word_core:
            for (int b = 0; b < PE_COUNT; ++b) {
#pragma HLS UNROLL
                const int idx = base + b;
                if (idx < K) {
                    int8_t wb = (int8_t)packed.range(8 * b + 7, 8 * b);
                    int16_t xval = (int16_t)x_buffer[idx];
                    int32_t prod = (int32_t)xval * (int16_t)wb;
                    partial_sums[b] += prod;
                }
            }
        }

        float w_scale = ws_stream.read();
        int32_t ival = 0;
    reduce_partial_sums_core:
        for (int i = 0; i < PE_COUNT; ++i) {
#pragma HLS UNROLL
            ival += partial_sums[i];
        }

        float scale = xs_val * w_scale;
        xout_row[n] = (float)ival * scale;
    }
}


// Helper task: read weights from streams, perform matmul and write outputs
static void matmul_compute_outputs(
    float *xout,
    const int8_t *xq,
    const float  *xs,
    int M,
    int N,
    int K,
    hls::stream<ap_uint<512> > &wq_stream,
    hls::stream<float>  &ws_stream) {
#pragma HLS INLINE off

compute_m:
    for (int m = 0; m < M; ++m) {
        // Load one input row into local buffer
        int8_t x_buffer[hidden_dim];
        float  xs_val;

    load_x:
        for (int k = 0; k < K; ++k) {
#pragma HLS PIPELINE II=1
            x_buffer[k] = xq[m * K + k];
        }
        xs_val = xs[m];

        float *xout_row = xout + m * N;
        matmul_core(xout_row, x_buffer, xs_val, N, K, wq_stream, ws_stream);
    }
}


static void matmul_engine(
    float *xout,
    const int8_t *xq,
    const float *xs,
    const int8_t *wq,
    const float *ws,
    int M,
    int N,
    int K) {
    #pragma HLS INLINE off

    // Streams to connect weight-loading task and compute task
    hls::stream<ap_uint<512> > wq_stream("wq_stream");
    hls::stream<float>  ws_stream("ws_stream");
#pragma HLS STREAM variable=wq_stream depth=64
#pragma HLS STREAM variable=ws_stream depth=16

    // Enable task-level parallelism between load and compute
#pragma HLS DATAFLOW

    matmul_load_weights(wq, ws, M, N, K, wq_stream, ws_stream);
    matmul_compute_outputs(xout, xq, xs, M, N, K, wq_stream, ws_stream);
}




// ----------------------------------------------------------------------------
// W4A8 8-port matmul: load packed int4 weights over 8 AXI ports and compute
// Layout matches MatmulShardWeights shards: each port holds 1/8 output channels.
// ----------------------------------------------------------------------------

static void matmul_load_weights_w4a8(
    const int8_t *wq_base0, const int8_t *wq_base1, const int8_t *wq_base2, const int8_t *wq_base3, const int8_t *wq_base4, const int8_t *wq_base5, const int8_t *wq_base6, const int8_t *wq_base7,
    const int8_t *wq_base8, const int8_t *wq_base9, const int8_t *wq_base10, const int8_t *wq_base11, const int8_t *wq_base12, const int8_t *wq_base13, const int8_t *wq_base14, const int8_t *wq_base15,
    const int8_t *wq_base16, const int8_t *wq_base17, const int8_t *wq_base18, const int8_t *wq_base19, const int8_t *wq_base20, const int8_t *wq_base21, const int8_t *wq_base22, const int8_t *wq_base23,
    const int8_t *wq_base24, const int8_t *wq_base25, const int8_t *wq_base26, const int8_t *wq_base27, const int8_t *wq_base28, const int8_t *wq_base29, const int8_t *wq_base30, const int8_t *wq_base31,
    const int8_t *sg_base0, const int8_t *sg_base1, const int8_t *sg_base2, const int8_t *sg_base3, const int8_t *sg_base4, const int8_t *sg_base5, const int8_t *sg_base6, const int8_t *sg_base7,
    const int8_t *sg_base8, const int8_t *sg_base9, const int8_t *sg_base10, const int8_t *sg_base11, const int8_t *sg_base12, const int8_t *sg_base13, const int8_t *sg_base14, const int8_t *sg_base15,
    const int8_t *sg_base16, const int8_t *sg_base17, const int8_t *sg_base18, const int8_t *sg_base19, const int8_t *sg_base20, const int8_t *sg_base21, const int8_t *sg_base22, const int8_t *sg_base23,
    const int8_t *sg_base24, const int8_t *sg_base25, const int8_t *sg_base26, const int8_t *sg_base27, const int8_t *sg_base28, const int8_t *sg_base29, const int8_t *sg_base30, const int8_t *sg_base31,
    const float  *sc_base0, const float  *sc_base1, const float  *sc_base2, const float  *sc_base3, const float  *sc_base4, const float  *sc_base5, const float  *sc_base6, const float  *sc_base7,
    const float  *sc_base8, const float  *sc_base9, const float  *sc_base10, const float  *sc_base11, const float  *sc_base12, const float  *sc_base13, const float  *sc_base14, const float  *sc_base15,
    const float  *sc_base16, const float  *sc_base17, const float  *sc_base18, const float  *sc_base19, const float  *sc_base20, const float  *sc_base21, const float  *sc_base22, const float  *sc_base23,
    const float  *sc_base24, const float  *sc_base25, const float  *sc_base26, const float  *sc_base27, const float  *sc_base28, const float  *sc_base29, const float  *sc_base30, const float  *sc_base31,
    int M, int N, int K,
    hls::stream<ap_uint<512> > &wq_stream0, hls::stream<ap_uint<512> > &wq_stream1,
    hls::stream<ap_uint<512> > &wq_stream2, hls::stream<ap_uint<512> > &wq_stream3,
    hls::stream<ap_uint<512> > &wq_stream4, hls::stream<ap_uint<512> > &wq_stream5,
    hls::stream<ap_uint<512> > &wq_stream6, hls::stream<ap_uint<512> > &wq_stream7,
    hls::stream<ap_uint<512> > &wq_stream8, hls::stream<ap_uint<512> > &wq_stream9,
    hls::stream<ap_uint<512> > &wq_stream10, hls::stream<ap_uint<512> > &wq_stream11,
    hls::stream<ap_uint<512> > &wq_stream12, hls::stream<ap_uint<512> > &wq_stream13,
    hls::stream<ap_uint<512> > &wq_stream14, hls::stream<ap_uint<512> > &wq_stream15,
    hls::stream<ap_uint<512> > &wq_stream16, hls::stream<ap_uint<512> > &wq_stream17,
    hls::stream<ap_uint<512> > &wq_stream18, hls::stream<ap_uint<512> > &wq_stream19,
    hls::stream<ap_uint<512> > &wq_stream20, hls::stream<ap_uint<512> > &wq_stream21,
    hls::stream<ap_uint<512> > &wq_stream22, hls::stream<ap_uint<512> > &wq_stream23,
    hls::stream<ap_uint<512> > &wq_stream24, hls::stream<ap_uint<512> > &wq_stream25,
    hls::stream<ap_uint<512> > &wq_stream26, hls::stream<ap_uint<512> > &wq_stream27,
    hls::stream<ap_uint<512> > &wq_stream28, hls::stream<ap_uint<512> > &wq_stream29,
    hls::stream<ap_uint<512> > &wq_stream30, hls::stream<ap_uint<512> > &wq_stream31,
    hls::stream<int8_t> &sg_stream0, hls::stream<int8_t> &sg_stream1,
    hls::stream<int8_t> &sg_stream2, hls::stream<int8_t> &sg_stream3,
    hls::stream<int8_t> &sg_stream4, hls::stream<int8_t> &sg_stream5,
    hls::stream<int8_t> &sg_stream6, hls::stream<int8_t> &sg_stream7,
    hls::stream<int8_t> &sg_stream8, hls::stream<int8_t> &sg_stream9,
    hls::stream<int8_t> &sg_stream10, hls::stream<int8_t> &sg_stream11,
    hls::stream<int8_t> &sg_stream12, hls::stream<int8_t> &sg_stream13,
    hls::stream<int8_t> &sg_stream14, hls::stream<int8_t> &sg_stream15,
    hls::stream<int8_t> &sg_stream16, hls::stream<int8_t> &sg_stream17,
    hls::stream<int8_t> &sg_stream18, hls::stream<int8_t> &sg_stream19,
    hls::stream<int8_t> &sg_stream20, hls::stream<int8_t> &sg_stream21,
    hls::stream<int8_t> &sg_stream22, hls::stream<int8_t> &sg_stream23,
    hls::stream<int8_t> &sg_stream24, hls::stream<int8_t> &sg_stream25,
    hls::stream<int8_t> &sg_stream26, hls::stream<int8_t> &sg_stream27,
    hls::stream<int8_t> &sg_stream28, hls::stream<int8_t> &sg_stream29,
    hls::stream<int8_t> &sg_stream30, hls::stream<int8_t> &sg_stream31,
    hls::stream<float>  &sc_stream0, hls::stream<float>  &sc_stream1,
    hls::stream<float>  &sc_stream2, hls::stream<float>  &sc_stream3,
    hls::stream<float>  &sc_stream4, hls::stream<float>  &sc_stream5,
    hls::stream<float>  &sc_stream6, hls::stream<float>  &sc_stream7,
    hls::stream<float>  &sc_stream8, hls::stream<float>  &sc_stream9,
    hls::stream<float>  &sc_stream10, hls::stream<float>  &sc_stream11,
    hls::stream<float>  &sc_stream12, hls::stream<float>  &sc_stream13,
    hls::stream<float>  &sc_stream14, hls::stream<float>  &sc_stream15,
    hls::stream<float>  &sc_stream16, hls::stream<float>  &sc_stream17,
    hls::stream<float>  &sc_stream18, hls::stream<float>  &sc_stream19,
    hls::stream<float>  &sc_stream20, hls::stream<float>  &sc_stream21,
    hls::stream<float>  &sc_stream22, hls::stream<float>  &sc_stream23,
    hls::stream<float>  &sc_stream24, hls::stream<float>  &sc_stream25,
    hls::stream<float>  &sc_stream26, hls::stream<float>  &sc_stream27,
    hls::stream<float>  &sc_stream28, hls::stream<float>  &sc_stream29,
    hls::stream<float>  &sc_stream30, hls::stream<float>  &sc_stream31) {
#pragma HLS INLINE off

    // Each 512-bit word holds 128 int4 weights -> 64 bytes
    const int WORD_BYTES    = 64;
    const int groups_per_row = K / GS;           // number of W4 groups per output channel
    const int words_per_row  = groups_per_row;   // GS = 128 => one group per 512-bit word

    const ap_uint<512> *wq512_0 = reinterpret_cast<const ap_uint<512> *>(wq_base0); const ap_uint<512> *wq512_1 = reinterpret_cast<const ap_uint<512> *>(wq_base1);
    const ap_uint<512> *wq512_2 = reinterpret_cast<const ap_uint<512> *>(wq_base2); const ap_uint<512> *wq512_3 = reinterpret_cast<const ap_uint<512> *>(wq_base3);
    const ap_uint<512> *wq512_4 = reinterpret_cast<const ap_uint<512> *>(wq_base4); const ap_uint<512> *wq512_5 = reinterpret_cast<const ap_uint<512> *>(wq_base5);
    const ap_uint<512> *wq512_6 = reinterpret_cast<const ap_uint<512> *>(wq_base6); const ap_uint<512> *wq512_7 = reinterpret_cast<const ap_uint<512> *>(wq_base7);
    const ap_uint<512> *wq512_8 = reinterpret_cast<const ap_uint<512> *>(wq_base8); const ap_uint<512> *wq512_9 = reinterpret_cast<const ap_uint<512> *>(wq_base9);
    const ap_uint<512> *wq512_10 = reinterpret_cast<const ap_uint<512> *>(wq_base10); const ap_uint<512> *wq512_11 = reinterpret_cast<const ap_uint<512> *>(wq_base11);
    const ap_uint<512> *wq512_12 = reinterpret_cast<const ap_uint<512> *>(wq_base12); const ap_uint<512> *wq512_13 = reinterpret_cast<const ap_uint<512> *>(wq_base13);
    const ap_uint<512> *wq512_14 = reinterpret_cast<const ap_uint<512> *>(wq_base14); const ap_uint<512> *wq512_15 = reinterpret_cast<const ap_uint<512> *>(wq_base15);
    const ap_uint<512> *wq512_16 = reinterpret_cast<const ap_uint<512> *>(wq_base16); const ap_uint<512> *wq512_17 = reinterpret_cast<const ap_uint<512> *>(wq_base17);
    const ap_uint<512> *wq512_18 = reinterpret_cast<const ap_uint<512> *>(wq_base18); const ap_uint<512> *wq512_19 = reinterpret_cast<const ap_uint<512> *>(wq_base19);
    const ap_uint<512> *wq512_20 = reinterpret_cast<const ap_uint<512> *>(wq_base20); const ap_uint<512> *wq512_21 = reinterpret_cast<const ap_uint<512> *>(wq_base21);
    const ap_uint<512> *wq512_22 = reinterpret_cast<const ap_uint<512> *>(wq_base22); const ap_uint<512> *wq512_23 = reinterpret_cast<const ap_uint<512> *>(wq_base23);
    const ap_uint<512> *wq512_24 = reinterpret_cast<const ap_uint<512> *>(wq_base24); const ap_uint<512> *wq512_25 = reinterpret_cast<const ap_uint<512> *>(wq_base25);
    const ap_uint<512> *wq512_26 = reinterpret_cast<const ap_uint<512> *>(wq_base26); const ap_uint<512> *wq512_27 = reinterpret_cast<const ap_uint<512> *>(wq_base27);
    const ap_uint<512> *wq512_28 = reinterpret_cast<const ap_uint<512> *>(wq_base28); const ap_uint<512> *wq512_29 = reinterpret_cast<const ap_uint<512> *>(wq_base29);
    const ap_uint<512> *wq512_30 = reinterpret_cast<const ap_uint<512> *>(wq_base30); const ap_uint<512> *wq512_31 = reinterpret_cast<const ap_uint<512> *>(wq_base31);

    const int N_oct       = N / SHARDS;                 // outputs per shard (dim_shard)
    const int total_words = N_oct * words_per_row; // total 512-bit words per shard

    // Local buffers for group scales and channel scales
    const int MAX_SG_BUF = 16384 * 1;
    const int MAX_SC_BUF = 512 * 1;

    int8_t sg_buf0[MAX_SG_BUF], sg_buf1[MAX_SG_BUF], sg_buf2[MAX_SG_BUF], sg_buf3[MAX_SG_BUF];
    int8_t sg_buf4[MAX_SG_BUF], sg_buf5[MAX_SG_BUF], sg_buf6[MAX_SG_BUF], sg_buf7[MAX_SG_BUF];
    int8_t sg_buf8[MAX_SG_BUF], sg_buf9[MAX_SG_BUF], sg_buf10[MAX_SG_BUF], sg_buf11[MAX_SG_BUF];
    int8_t sg_buf12[MAX_SG_BUF], sg_buf13[MAX_SG_BUF], sg_buf14[MAX_SG_BUF], sg_buf15[MAX_SG_BUF];
    int8_t sg_buf16[MAX_SG_BUF], sg_buf17[MAX_SG_BUF], sg_buf18[MAX_SG_BUF], sg_buf19[MAX_SG_BUF];
    int8_t sg_buf20[MAX_SG_BUF], sg_buf21[MAX_SG_BUF], sg_buf22[MAX_SG_BUF], sg_buf23[MAX_SG_BUF];
    int8_t sg_buf24[MAX_SG_BUF], sg_buf25[MAX_SG_BUF], sg_buf26[MAX_SG_BUF], sg_buf27[MAX_SG_BUF];
    int8_t sg_buf28[MAX_SG_BUF], sg_buf29[MAX_SG_BUF], sg_buf30[MAX_SG_BUF], sg_buf31[MAX_SG_BUF];
    float  sc_buf0[MAX_SC_BUF], sc_buf1[MAX_SC_BUF], sc_buf2[MAX_SC_BUF], sc_buf3[MAX_SC_BUF];
    float  sc_buf4[MAX_SC_BUF], sc_buf5[MAX_SC_BUF], sc_buf6[MAX_SC_BUF], sc_buf7[MAX_SC_BUF];
    float  sc_buf8[MAX_SC_BUF], sc_buf9[MAX_SC_BUF], sc_buf10[MAX_SC_BUF], sc_buf11[MAX_SC_BUF];
    float  sc_buf12[MAX_SC_BUF], sc_buf13[MAX_SC_BUF], sc_buf14[MAX_SC_BUF], sc_buf15[MAX_SC_BUF];
    float  sc_buf16[MAX_SC_BUF], sc_buf17[MAX_SC_BUF], sc_buf18[MAX_SC_BUF], sc_buf19[MAX_SC_BUF];
    float  sc_buf20[MAX_SC_BUF], sc_buf21[MAX_SC_BUF], sc_buf22[MAX_SC_BUF], sc_buf23[MAX_SC_BUF];
    float  sc_buf24[MAX_SC_BUF], sc_buf25[MAX_SC_BUF], sc_buf26[MAX_SC_BUF], sc_buf27[MAX_SC_BUF];
    float  sc_buf28[MAX_SC_BUF], sc_buf29[MAX_SC_BUF], sc_buf30[MAX_SC_BUF], sc_buf31[MAX_SC_BUF];

    // 512-bit views for vectorized prefetch of sg/sc
    const ap_uint<512> *sg512_0 = reinterpret_cast<const ap_uint<512> *>(sg_base0); const ap_uint<512> *sg512_1 = reinterpret_cast<const ap_uint<512> *>(sg_base1);
    const ap_uint<512> *sg512_2 = reinterpret_cast<const ap_uint<512> *>(sg_base2); const ap_uint<512> *sg512_3 = reinterpret_cast<const ap_uint<512> *>(sg_base3);
    const ap_uint<512> *sg512_4 = reinterpret_cast<const ap_uint<512> *>(sg_base4); const ap_uint<512> *sg512_5 = reinterpret_cast<const ap_uint<512> *>(sg_base5);
    const ap_uint<512> *sg512_6 = reinterpret_cast<const ap_uint<512> *>(sg_base6); const ap_uint<512> *sg512_7 = reinterpret_cast<const ap_uint<512> *>(sg_base7);
    const ap_uint<512> *sg512_8 = reinterpret_cast<const ap_uint<512> *>(sg_base8); const ap_uint<512> *sg512_9 = reinterpret_cast<const ap_uint<512> *>(sg_base9);
    const ap_uint<512> *sg512_10 = reinterpret_cast<const ap_uint<512> *>(sg_base10); const ap_uint<512> *sg512_11 = reinterpret_cast<const ap_uint<512> *>(sg_base11);
    const ap_uint<512> *sg512_12 = reinterpret_cast<const ap_uint<512> *>(sg_base12); const ap_uint<512> *sg512_13 = reinterpret_cast<const ap_uint<512> *>(sg_base13);
    const ap_uint<512> *sg512_14 = reinterpret_cast<const ap_uint<512> *>(sg_base14); const ap_uint<512> *sg512_15 = reinterpret_cast<const ap_uint<512> *>(sg_base15);
    const ap_uint<512> *sg512_16 = reinterpret_cast<const ap_uint<512> *>(sg_base16); const ap_uint<512> *sg512_17 = reinterpret_cast<const ap_uint<512> *>(sg_base17);
    const ap_uint<512> *sg512_18 = reinterpret_cast<const ap_uint<512> *>(sg_base18); const ap_uint<512> *sg512_19 = reinterpret_cast<const ap_uint<512> *>(sg_base19);
    const ap_uint<512> *sg512_20 = reinterpret_cast<const ap_uint<512> *>(sg_base20); const ap_uint<512> *sg512_21 = reinterpret_cast<const ap_uint<512> *>(sg_base21);
    const ap_uint<512> *sg512_22 = reinterpret_cast<const ap_uint<512> *>(sg_base22); const ap_uint<512> *sg512_23 = reinterpret_cast<const ap_uint<512> *>(sg_base23);
    const ap_uint<512> *sg512_24 = reinterpret_cast<const ap_uint<512> *>(sg_base24); const ap_uint<512> *sg512_25 = reinterpret_cast<const ap_uint<512> *>(sg_base25);
    const ap_uint<512> *sg512_26 = reinterpret_cast<const ap_uint<512> *>(sg_base26); const ap_uint<512> *sg512_27 = reinterpret_cast<const ap_uint<512> *>(sg_base27);
    const ap_uint<512> *sg512_28 = reinterpret_cast<const ap_uint<512> *>(sg_base28); const ap_uint<512> *sg512_29 = reinterpret_cast<const ap_uint<512> *>(sg_base29);
    const ap_uint<512> *sg512_30 = reinterpret_cast<const ap_uint<512> *>(sg_base30); const ap_uint<512> *sg512_31 = reinterpret_cast<const ap_uint<512> *>(sg_base31);
    
    const ap_uint<512> *sc512_0 = reinterpret_cast<const ap_uint<512> *>(sc_base0); const ap_uint<512> *sc512_1 = reinterpret_cast<const ap_uint<512> *>(sc_base1);
    const ap_uint<512> *sc512_2 = reinterpret_cast<const ap_uint<512> *>(sc_base2); const ap_uint<512> *sc512_3 = reinterpret_cast<const ap_uint<512> *>(sc_base3);
    const ap_uint<512> *sc512_4 = reinterpret_cast<const ap_uint<512> *>(sc_base4); const ap_uint<512> *sc512_5 = reinterpret_cast<const ap_uint<512> *>(sc_base5);
    const ap_uint<512> *sc512_6 = reinterpret_cast<const ap_uint<512> *>(sc_base6); const ap_uint<512> *sc512_7 = reinterpret_cast<const ap_uint<512> *>(sc_base7);
    const ap_uint<512> *sc512_8 = reinterpret_cast<const ap_uint<512> *>(sc_base8); const ap_uint<512> *sc512_9 = reinterpret_cast<const ap_uint<512> *>(sc_base9);
    const ap_uint<512> *sc512_10 = reinterpret_cast<const ap_uint<512> *>(sc_base10); const ap_uint<512> *sc512_11 = reinterpret_cast<const ap_uint<512> *>(sc_base11);
    const ap_uint<512> *sc512_12 = reinterpret_cast<const ap_uint<512> *>(sc_base12); const ap_uint<512> *sc512_13 = reinterpret_cast<const ap_uint<512> *>(sc_base13);
    const ap_uint<512> *sc512_14 = reinterpret_cast<const ap_uint<512> *>(sc_base14); const ap_uint<512> *sc512_15 = reinterpret_cast<const ap_uint<512> *>(sc_base15);
    const ap_uint<512> *sc512_16 = reinterpret_cast<const ap_uint<512> *>(sc_base16); const ap_uint<512> *sc512_17 = reinterpret_cast<const ap_uint<512> *>(sc_base17);
    const ap_uint<512> *sc512_18 = reinterpret_cast<const ap_uint<512> *>(sc_base18); const ap_uint<512> *sc512_19 = reinterpret_cast<const ap_uint<512> *>(sc_base19);
    const ap_uint<512> *sc512_20 = reinterpret_cast<const ap_uint<512> *>(sc_base20); const ap_uint<512> *sc512_21 = reinterpret_cast<const ap_uint<512> *>(sc_base21);
    const ap_uint<512> *sc512_22 = reinterpret_cast<const ap_uint<512> *>(sc_base22); const ap_uint<512> *sc512_23 = reinterpret_cast<const ap_uint<512> *>(sc_base23);
    const ap_uint<512> *sc512_24 = reinterpret_cast<const ap_uint<512> *>(sc_base24); const ap_uint<512> *sc512_25 = reinterpret_cast<const ap_uint<512> *>(sc_base25);
    const ap_uint<512> *sc512_26 = reinterpret_cast<const ap_uint<512> *>(sc_base26); const ap_uint<512> *sc512_27 = reinterpret_cast<const ap_uint<512> *>(sc_base27);
    const ap_uint<512> *sc512_28 = reinterpret_cast<const ap_uint<512> *>(sc_base28); const ap_uint<512> *sc512_29 = reinterpret_cast<const ap_uint<512> *>(sc_base29);
    const ap_uint<512> *sc512_30 = reinterpret_cast<const ap_uint<512> *>(sc_base30); const ap_uint<512> *sc512_31 = reinterpret_cast<const ap_uint<512> *>(sc_base31);

load_m_w4a8:
    for (int m = 0; m < M; ++m) {

        // --------------------------------------------------------------------
        // Stage 1: Vectorized prefetch of group scales sg into local buffers
        // sg layout in W4A8Tensor shard: [col_shard][groups_per_row]
        // --------------------------------------------------------------------
        const int num_sg_vec = (total_words + 63) / 64; // 64 int8 per 512-bit word
    prefetch_sg_loop:
        for (int v = 0; v < num_sg_vec; ++v) {
        #pragma HLS PIPELINE II=1
            ap_uint<512> v0 = sg512_0[v]; ap_uint<512> v1 = sg512_1[v]; ap_uint<512> v2 = sg512_2[v]; ap_uint<512> v3 = sg512_3[v];
            ap_uint<512> v4 = sg512_4[v]; ap_uint<512> v5 = sg512_5[v]; ap_uint<512> v6 = sg512_6[v]; ap_uint<512> v7 = sg512_7[v];
            ap_uint<512> v8 = sg512_8[v]; ap_uint<512> v9 = sg512_9[v]; ap_uint<512> v10 = sg512_10[v]; ap_uint<512> v11 = sg512_11[v];
            ap_uint<512> v12 = sg512_12[v]; ap_uint<512> v13 = sg512_13[v]; ap_uint<512> v14 = sg512_14[v]; ap_uint<512> v15 = sg512_15[v];
            ap_uint<512> v16 = sg512_16[v]; ap_uint<512> v17 = sg512_17[v]; ap_uint<512> v18 = sg512_18[v]; ap_uint<512> v19 = sg512_19[v];
            ap_uint<512> v20 = sg512_20[v]; ap_uint<512> v21 = sg512_21[v]; ap_uint<512> v22 = sg512_22[v]; ap_uint<512> v23 = sg512_23[v];
            ap_uint<512> v24 = sg512_24[v]; ap_uint<512> v25 = sg512_25[v]; ap_uint<512> v26 = sg512_26[v]; ap_uint<512> v27 = sg512_27[v];
            ap_uint<512> v28 = sg512_28[v]; ap_uint<512> v29 = sg512_29[v]; ap_uint<512> v30 = sg512_30[v]; ap_uint<512> v31 = sg512_31[v];

            for (int b = 0; b < 64; ++b) {
            #pragma HLS UNROLL
                int t = (v << 6) + b; // t in [0, total_words)
                if (t < total_words) {
                    sg_buf0[t] = v0.range(b * 8 + 7, b * 8); sg_buf1[t] = v1.range(b * 8 + 7, b * 8);
                    sg_buf2[t] = v2.range(b * 8 + 7, b * 8); sg_buf3[t] = v3.range(b * 8 + 7, b * 8);
                    sg_buf4[t] = v4.range(b * 8 + 7, b * 8); sg_buf5[t] = v5.range(b * 8 + 7, b * 8);
                    sg_buf6[t] = v6.range(b * 8 + 7, b * 8); sg_buf7[t] = v7.range(b * 8 + 7, b * 8);
                    sg_buf8[t] = v8.range(b * 8 + 7, b * 8); sg_buf9[t] = v9.range(b * 8 + 7, b * 8);
                    sg_buf10[t] = v10.range(b * 8 + 7, b * 8); sg_buf11[t] = v11.range(b * 8 + 7, b * 8);
                    sg_buf12[t] = v12.range(b * 8 + 7, b * 8); sg_buf13[t] = v13.range(b * 8 + 7, b * 8);
                    sg_buf14[t] = v14.range(b * 8 + 7, b * 8); sg_buf15[t] = v15.range(b * 8 + 7, b * 8);
                    sg_buf16[t] = v16.range(b * 8 + 7, b * 8); sg_buf17[t] = v17.range(b * 8 + 7, b * 8);
                    sg_buf18[t] = v18.range(b * 8 + 7, b * 8); sg_buf19[t] = v19.range(b * 8 + 7, b * 8);
                    sg_buf20[t] = v20.range(b * 8 + 7, b * 8); sg_buf21[t] = v21.range(b * 8 + 7, b * 8);
                    sg_buf22[t] = v22.range(b * 8 + 7, b * 8); sg_buf23[t] = v23.range(b * 8 + 7, b * 8);
                    sg_buf24[t] = v24.range(b * 8 + 7, b * 8); sg_buf25[t] = v25.range(b * 8 + 7, b * 8);
                    sg_buf26[t] = v26.range(b * 8 + 7, b * 8); sg_buf27[t] = v27.range(b * 8 + 7, b * 8);
                    sg_buf28[t] = v28.range(b * 8 + 7, b * 8); sg_buf29[t] = v29.range(b * 8 + 7, b * 8);
                    sg_buf30[t] = v30.range(b * 8 + 7, b * 8); sg_buf31[t] = v31.range(b * 8 + 7, b * 8);
                }
            }
        }

        // --------------------------------------------------------------------
        // Stage 2: Vectorized prefetch of channel scales sc into local buffers
        // sc layout in W4A8Tensor shard: [col_shard]
        // --------------------------------------------------------------------
        const int num_sc_vec = (N_oct + 15) / 16; // 16 float per 512-bit word
    prefetch_sc_loop:
        for (int v = 0; v < num_sc_vec; ++v) {
        #pragma HLS PIPELINE II=1
            ap_uint<512> v0 = sc512_0[v]; ap_uint<512> v1 = sc512_1[v]; ap_uint<512> v2 = sc512_2[v]; ap_uint<512> v3 = sc512_3[v];
            ap_uint<512> v4 = sc512_4[v]; ap_uint<512> v5 = sc512_5[v]; ap_uint<512> v6 = sc512_6[v]; ap_uint<512> v7 = sc512_7[v];
            ap_uint<512> v8 = sc512_8[v]; ap_uint<512> v9 = sc512_9[v]; ap_uint<512> v10 = sc512_10[v]; ap_uint<512> v11 = sc512_11[v];
            ap_uint<512> v12 = sc512_12[v]; ap_uint<512> v13 = sc512_13[v]; ap_uint<512> v14 = sc512_14[v]; ap_uint<512> v15 = sc512_15[v];
            ap_uint<512> v16 = sc512_16[v]; ap_uint<512> v17 = sc512_17[v]; ap_uint<512> v18 = sc512_18[v]; ap_uint<512> v19 = sc512_19[v];
            ap_uint<512> v20 = sc512_20[v]; ap_uint<512> v21 = sc512_21[v]; ap_uint<512> v22 = sc512_22[v]; ap_uint<512> v23 = sc512_23[v];
            ap_uint<512> v24 = sc512_24[v]; ap_uint<512> v25 = sc512_25[v]; ap_uint<512> v26 = sc512_26[v]; ap_uint<512> v27 = sc512_27[v];
            ap_uint<512> v28 = sc512_28[v]; ap_uint<512> v29 = sc512_29[v]; ap_uint<512> v30 = sc512_30[v]; ap_uint<512> v31 = sc512_31[v];

            for (int b = 0; b < 16; ++b) {
            #pragma HLS UNROLL
                int p = (v << 4) + b; // p in [0, N_oct)
                if (p < N_oct) {
                    union { unsigned int u; float f; } u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18, u19, u20, u21, u22, u23, u24, u25, u26, u27, u28, u29, u30, u31;
                    u0.u = v0.range(b * 32 + 31, b * 32); sc_buf0[p] = u0.f; u1.u = v1.range(b * 32 + 31, b * 32); sc_buf1[p] = u1.f;
                    u2.u = v2.range(b * 32 + 31, b * 32); sc_buf2[p] = u2.f; u3.u = v3.range(b * 32 + 31, b * 32); sc_buf3[p] = u3.f;
                    u4.u = v4.range(b * 32 + 31, b * 32); sc_buf4[p] = u4.f; u5.u = v5.range(b * 32 + 31, b * 32); sc_buf5[p] = u5.f;
                    u6.u = v6.range(b * 32 + 31, b * 32); sc_buf6[p] = u6.f; u7.u = v7.range(b * 32 + 31, b * 32); sc_buf7[p] = u7.f;
                    u8.u = v8.range(b * 32 + 31, b * 32); sc_buf8[p] = u8.f; u9.u = v9.range(b * 32 + 31, b * 32); sc_buf9[p] = u9.f;
                    u10.u = v10.range(b * 32 + 31, b * 32); sc_buf10[p] = u10.f; u11.u = v11.range(b * 32 + 31, b * 32); sc_buf11[p] = u11.f;
                    u12.u = v12.range(b * 32 + 31, b * 32); sc_buf12[p] = u12.f; u13.u = v13.range(b * 32 + 31, b * 32); sc_buf13[p] = u13.f;
                    u14.u = v14.range(b * 32 + 31, b * 32); sc_buf14[p] = u14.f; u15.u = v15.range(b * 32 + 31, b * 32); sc_buf15[p] = u15.f;
                    u16.u = v16.range(b * 32 + 31, b * 32); sc_buf16[p] = u16.f; u17.u = v17.range(b * 32 + 31, b * 32); sc_buf17[p] = u17.f;
                    u18.u = v18.range(b * 32 + 31, b * 32); sc_buf18[p] = u18.f; u19.u = v19.range(b * 32 + 31, b * 32); sc_buf19[p] = u19.f;
                    u20.u = v20.range(b * 32 + 31, b * 32); sc_buf20[p] = u20.f; u21.u = v21.range(b * 32 + 31, b * 32); sc_buf21[p] = u21.f;
                    u22.u = v22.range(b * 32 + 31, b * 32); sc_buf22[p] = u22.f; u23.u = v23.range(b * 32 + 31, b * 32); sc_buf23[p] = u23.f;
                    u24.u = v24.range(b * 32 + 31, b * 32); sc_buf24[p] = u24.f; u25.u = v25.range(b * 32 + 31, b * 32); sc_buf25[p] = u25.f;
                    u26.u = v26.range(b * 32 + 31, b * 32); sc_buf26[p] = u26.f; u27.u = v27.range(b * 32 + 31, b * 32); sc_buf27[p] = u27.f;
                    u28.u = v28.range(b * 32 + 31, b * 32); sc_buf28[p] = u28.f; u29.u = v29.range(b * 32 + 31, b * 32); sc_buf29[p] = u29.f;
                    u30.u = v30.range(b * 32 + 31, b * 32); sc_buf30[p] = u30.f; u31.u = v31.range(b * 32 + 31, b * 32); sc_buf31[p] = u31.f;
                }
            }
        }

        // --------------------------------------------------------------------
        // Stage 3: Main streaming loop
        // - Read packed W4 weights from wq_base* (AXI)
        // - Read group scales and channel scales from local buffers
        //   using the same flattened ordering as [p * words_per_row + g].
        // --------------------------------------------------------------------
        int w     = 0;
        int buf_t = 0; // 0..total_words-1
        int buf_p = 0; // 0..N_oct-1

    load_g_w4a8:
        for (int t = 0; t < total_words; ++t) {
        #pragma HLS PIPELINE II=1
            int word_index = t; // p * words_per_row + g, flattened

            // Load packed int4 weights: each port holds one shard (dim_shard outputs)
            wq_stream0.write(wq512_0[word_index]); wq_stream1.write(wq512_1[word_index]); wq_stream2.write(wq512_2[word_index]); wq_stream3.write(wq512_3[word_index]); 
            wq_stream4.write(wq512_4[word_index]); wq_stream5.write(wq512_5[word_index]); wq_stream6.write(wq512_6[word_index]); wq_stream7.write(wq512_7[word_index]);
            wq_stream8.write(wq512_8[word_index]); wq_stream9.write(wq512_9[word_index]); wq_stream10.write(wq512_10[word_index]); wq_stream11.write(wq512_11[word_index]);
            wq_stream12.write(wq512_12[word_index]); wq_stream13.write(wq512_13[word_index]); wq_stream14.write(wq512_14[word_index]); wq_stream15.write(wq512_15[word_index]);
            wq_stream16.write(wq512_16[word_index]); wq_stream17.write(wq512_17[word_index]); wq_stream18.write(wq512_18[word_index]); wq_stream19.write(wq512_19[word_index]);
            wq_stream20.write(wq512_20[word_index]); wq_stream21.write(wq512_21[word_index]); wq_stream22.write(wq512_22[word_index]); wq_stream23.write(wq512_23[word_index]);
            wq_stream24.write(wq512_24[word_index]); wq_stream25.write(wq512_25[word_index]); wq_stream26.write(wq512_26[word_index]); wq_stream27.write(wq512_27[word_index]);
            wq_stream28.write(wq512_28[word_index]); wq_stream29.write(wq512_29[word_index]); wq_stream30.write(wq512_30[word_index]); wq_stream31.write(wq512_31[word_index]);

            // Group scales: layout [col_shard][groups_per_row], mirrored in sg_buf*
            sg_stream0.write(sg_buf0[buf_t]); sg_stream1.write(sg_buf1[buf_t]); sg_stream2.write(sg_buf2[buf_t]); sg_stream3.write(sg_buf3[buf_t]);
            sg_stream4.write(sg_buf4[buf_t]); sg_stream5.write(sg_buf5[buf_t]); sg_stream6.write(sg_buf6[buf_t]); sg_stream7.write(sg_buf7[buf_t]);
            sg_stream8.write(sg_buf8[buf_t]); sg_stream9.write(sg_buf9[buf_t]); sg_stream10.write(sg_buf10[buf_t]); sg_stream11.write(sg_buf11[buf_t]);
            sg_stream12.write(sg_buf12[buf_t]); sg_stream13.write(sg_buf13[buf_t]); sg_stream14.write(sg_buf14[buf_t]); sg_stream15.write(sg_buf15[buf_t]);
            sg_stream16.write(sg_buf16[buf_t]); sg_stream17.write(sg_buf17[buf_t]); sg_stream18.write(sg_buf18[buf_t]); sg_stream19.write(sg_buf19[buf_t]);
            sg_stream20.write(sg_buf20[buf_t]); sg_stream21.write(sg_buf21[buf_t]); sg_stream22.write(sg_buf22[buf_t]); sg_stream23.write(sg_buf23[buf_t]);
            sg_stream24.write(sg_buf24[buf_t]); sg_stream25.write(sg_buf25[buf_t]); sg_stream26.write(sg_buf26[buf_t]); sg_stream27.write(sg_buf27[buf_t]);
            sg_stream28.write(sg_buf28[buf_t]); sg_stream29.write(sg_buf29[buf_t]); sg_stream30.write(sg_buf30[buf_t]); sg_stream31.write(sg_buf31[buf_t]);
            ++buf_t;

            // After finishing all groups for this output channel, emit channel scales
            if (w == words_per_row - 1) {
                sc_stream0.write(sc_buf0[buf_p]); sc_stream1.write(sc_buf1[buf_p]); sc_stream2.write(sc_buf2[buf_p]); sc_stream3.write(sc_buf3[buf_p]);
                sc_stream4.write(sc_buf4[buf_p]); sc_stream5.write(sc_buf5[buf_p]); sc_stream6.write(sc_buf6[buf_p]); sc_stream7.write(sc_buf7[buf_p]);
                sc_stream8.write(sc_buf8[buf_p]); sc_stream9.write(sc_buf9[buf_p]); sc_stream10.write(sc_buf10[buf_p]); sc_stream11.write(sc_buf11[buf_p]);
                sc_stream12.write(sc_buf12[buf_p]); sc_stream13.write(sc_buf13[buf_p]); sc_stream14.write(sc_buf14[buf_p]); sc_stream15.write(sc_buf15[buf_p]);
                sc_stream16.write(sc_buf16[buf_p]); sc_stream17.write(sc_buf17[buf_p]); sc_stream18.write(sc_buf18[buf_p]); sc_stream19.write(sc_buf19[buf_p]);
                sc_stream20.write(sc_buf20[buf_p]); sc_stream21.write(sc_buf21[buf_p]); sc_stream22.write(sc_buf22[buf_p]); sc_stream23.write(sc_buf23[buf_p]);
                sc_stream24.write(sc_buf24[buf_p]); sc_stream25.write(sc_buf25[buf_p]); sc_stream26.write(sc_buf26[buf_p]); sc_stream27.write(sc_buf27[buf_p]);
                sc_stream28.write(sc_buf28[buf_p]); sc_stream29.write(sc_buf29[buf_p]); sc_stream30.write(sc_buf30[buf_p]); sc_stream31.write(sc_buf31[buf_p]);
                ++buf_p;
                w = 0;
            } else {
                ++w;
            }
        }
    }
}

static void compute_core_w4a8(
    hls::stream<float> &res_stream0, hls::stream<float> &res_stream1,
    hls::stream<float> &res_stream2, hls::stream<float> &res_stream3,
    hls::stream<float> &res_stream4, hls::stream<float> &res_stream5,
    hls::stream<float> &res_stream6, hls::stream<float> &res_stream7,
    hls::stream<float> &res_stream8, hls::stream<float> &res_stream9,
    hls::stream<float> &res_stream10, hls::stream<float> &res_stream11,
    hls::stream<float> &res_stream12, hls::stream<float> &res_stream13,
    hls::stream<float> &res_stream14, hls::stream<float> &res_stream15,
    hls::stream<float> &res_stream16, hls::stream<float> &res_stream17,
    hls::stream<float> &res_stream18, hls::stream<float> &res_stream19,
    hls::stream<float> &res_stream20, hls::stream<float> &res_stream21,
    hls::stream<float> &res_stream22, hls::stream<float> &res_stream23,
    hls::stream<float> &res_stream24, hls::stream<float> &res_stream25,
    hls::stream<float> &res_stream26, hls::stream<float> &res_stream27,
    hls::stream<float> &res_stream28, hls::stream<float> &res_stream29,
    hls::stream<float> &res_stream30, hls::stream<float> &res_stream31,
    const int8_t *x_buffer, float xs_val, int N, int K,
    hls::stream<ap_uint<512> > &wq_stream0, hls::stream<ap_uint<512> > &wq_stream1,
    hls::stream<ap_uint<512> > &wq_stream2, hls::stream<ap_uint<512> > &wq_stream3,
    hls::stream<ap_uint<512> > &wq_stream4, hls::stream<ap_uint<512> > &wq_stream5,
    hls::stream<ap_uint<512> > &wq_stream6, hls::stream<ap_uint<512> > &wq_stream7,
    hls::stream<ap_uint<512> > &wq_stream8, hls::stream<ap_uint<512> > &wq_stream9,
    hls::stream<ap_uint<512> > &wq_stream10, hls::stream<ap_uint<512> > &wq_stream11,
    hls::stream<ap_uint<512> > &wq_stream12, hls::stream<ap_uint<512> > &wq_stream13,
    hls::stream<ap_uint<512> > &wq_stream14, hls::stream<ap_uint<512> > &wq_stream15,
    hls::stream<ap_uint<512> > &wq_stream16, hls::stream<ap_uint<512> > &wq_stream17,
    hls::stream<ap_uint<512> > &wq_stream18, hls::stream<ap_uint<512> > &wq_stream19,
    hls::stream<ap_uint<512> > &wq_stream20, hls::stream<ap_uint<512> > &wq_stream21,
    hls::stream<ap_uint<512> > &wq_stream22, hls::stream<ap_uint<512> > &wq_stream23,
    hls::stream<ap_uint<512> > &wq_stream24, hls::stream<ap_uint<512> > &wq_stream25,
    hls::stream<ap_uint<512> > &wq_stream26, hls::stream<ap_uint<512> > &wq_stream27,
    hls::stream<ap_uint<512> > &wq_stream28, hls::stream<ap_uint<512> > &wq_stream29,
    hls::stream<ap_uint<512> > &wq_stream30, hls::stream<ap_uint<512> > &wq_stream31,
    hls::stream<int8_t> &sg_stream0, hls::stream<int8_t> &sg_stream1,
    hls::stream<int8_t> &sg_stream2, hls::stream<int8_t> &sg_stream3,
    hls::stream<int8_t> &sg_stream4, hls::stream<int8_t> &sg_stream5,
    hls::stream<int8_t> &sg_stream6, hls::stream<int8_t> &sg_stream7,
    hls::stream<int8_t> &sg_stream8, hls::stream<int8_t> &sg_stream9,
    hls::stream<int8_t> &sg_stream10, hls::stream<int8_t> &sg_stream11,
    hls::stream<int8_t> &sg_stream12, hls::stream<int8_t> &sg_stream13,
    hls::stream<int8_t> &sg_stream14, hls::stream<int8_t> &sg_stream15,
    hls::stream<int8_t> &sg_stream16, hls::stream<int8_t> &sg_stream17,
    hls::stream<int8_t> &sg_stream18, hls::stream<int8_t> &sg_stream19,
    hls::stream<int8_t> &sg_stream20, hls::stream<int8_t> &sg_stream21,
    hls::stream<int8_t> &sg_stream22, hls::stream<int8_t> &sg_stream23,
    hls::stream<int8_t> &sg_stream24, hls::stream<int8_t> &sg_stream25,
    hls::stream<int8_t> &sg_stream26, hls::stream<int8_t> &sg_stream27,
    hls::stream<int8_t> &sg_stream28, hls::stream<int8_t> &sg_stream29,
    hls::stream<int8_t> &sg_stream30, hls::stream<int8_t> &sg_stream31,
    hls::stream<float>  &sc_stream0, hls::stream<float>  &sc_stream1,
    hls::stream<float>  &sc_stream2, hls::stream<float>  &sc_stream3,
    hls::stream<float>  &sc_stream4, hls::stream<float>  &sc_stream5,
    hls::stream<float>  &sc_stream6, hls::stream<float>  &sc_stream7,
    hls::stream<float>  &sc_stream8, hls::stream<float>  &sc_stream9,
    hls::stream<float>  &sc_stream10, hls::stream<float>  &sc_stream11,
    hls::stream<float>  &sc_stream12, hls::stream<float>  &sc_stream13,
    hls::stream<float>  &sc_stream14, hls::stream<float>  &sc_stream15,
    hls::stream<float>  &sc_stream16, hls::stream<float>  &sc_stream17,
    hls::stream<float>  &sc_stream18, hls::stream<float>  &sc_stream19,
    hls::stream<float>  &sc_stream20, hls::stream<float>  &sc_stream21,
    hls::stream<float>  &sc_stream22, hls::stream<float>  &sc_stream23,
    hls::stream<float>  &sc_stream24, hls::stream<float>  &sc_stream25,
    hls::stream<float>  &sc_stream26, hls::stream<float>  &sc_stream27,
    hls::stream<float>  &sc_stream28, hls::stream<float>  &sc_stream29,
    hls::stream<float>  &sc_stream30, hls::stream<float>  &sc_stream31) {
#pragma HLS INLINE off

    const int GROUP_SIZE = GS;
    const int groups_per_row = K / GROUP_SIZE;
    const int N_oct = N / SHARDS;

compute_n_core_w4a8:
    for (int p = 0; p < N_oct; ++p) {
// #pragma HLS LOOP_FLATTEN off
        // Accumulators for 32 shards
        int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0, acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;
        int32_t acc8 = 0, acc9 = 0, acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0, acc14 = 0, acc15 = 0;
        int32_t acc16 = 0, acc17 = 0, acc18 = 0, acc19 = 0, acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0;
        int32_t acc24 = 0, acc25 = 0, acc26 = 0, acc27 = 0, acc28 = 0, acc29 = 0, acc30 = 0, acc31 = 0;

        // Pipeline registers for delayed accumulation (break critical path)
        int32_t prev_partial0 = 0, prev_partial1 = 0, prev_partial2 = 0, prev_partial3 = 0;
        int32_t prev_partial4 = 0, prev_partial5 = 0, prev_partial6 = 0, prev_partial7 = 0;
        int32_t prev_partial8 = 0, prev_partial9 = 0, prev_partial10 = 0, prev_partial11 = 0;
        int32_t prev_partial12 = 0, prev_partial13 = 0, prev_partial14 = 0, prev_partial15 = 0;
        int32_t prev_partial16 = 0, prev_partial17 = 0, prev_partial18 = 0, prev_partial19 = 0;
        int32_t prev_partial20 = 0, prev_partial21 = 0, prev_partial22 = 0, prev_partial23 = 0;
        int32_t prev_partial24 = 0, prev_partial25 = 0, prev_partial26 = 0, prev_partial27 = 0;
        int32_t prev_partial28 = 0, prev_partial29 = 0, prev_partial30 = 0, prev_partial31 = 0;
        int8_t prev_sg0 = 0, prev_sg1 = 0, prev_sg2 = 0, prev_sg3 = 0;
        int8_t prev_sg4 = 0, prev_sg5 = 0, prev_sg6 = 0, prev_sg7 = 0;
        int8_t prev_sg8 = 0, prev_sg9 = 0, prev_sg10 = 0, prev_sg11 = 0;
        int8_t prev_sg12 = 0, prev_sg13 = 0, prev_sg14 = 0, prev_sg15 = 0;
        int8_t prev_sg16 = 0, prev_sg17 = 0, prev_sg18 = 0, prev_sg19 = 0;
        int8_t prev_sg20 = 0, prev_sg21 = 0, prev_sg22 = 0, prev_sg23 = 0;
        int8_t prev_sg24 = 0, prev_sg25 = 0, prev_sg26 = 0, prev_sg27 = 0;
        int8_t prev_sg28 = 0, prev_sg29 = 0, prev_sg30 = 0, prev_sg31 = 0;

    group_loop_w4a8:
        for (int g = 0; g < groups_per_row; ++g) {
#pragma HLS PIPELINE II=1
            // Read packed weights and group scales
            ap_uint<512> packed0 = wq_stream0.read(); ap_uint<512> packed1 = wq_stream1.read();
            ap_uint<512> packed2 = wq_stream2.read(); ap_uint<512> packed3 = wq_stream3.read();
            ap_uint<512> packed4 = wq_stream4.read(); ap_uint<512> packed5 = wq_stream5.read();
            ap_uint<512> packed6 = wq_stream6.read(); ap_uint<512> packed7 = wq_stream7.read();
            ap_uint<512> packed8 = wq_stream8.read(); ap_uint<512> packed9 = wq_stream9.read();
            ap_uint<512> packed10 = wq_stream10.read(); ap_uint<512> packed11 = wq_stream11.read();
            ap_uint<512> packed12 = wq_stream12.read(); ap_uint<512> packed13 = wq_stream13.read();
            ap_uint<512> packed14 = wq_stream14.read(); ap_uint<512> packed15 = wq_stream15.read();
            ap_uint<512> packed16 = wq_stream16.read(); ap_uint<512> packed17 = wq_stream17.read();
            ap_uint<512> packed18 = wq_stream18.read(); ap_uint<512> packed19 = wq_stream19.read();
            ap_uint<512> packed20 = wq_stream20.read(); ap_uint<512> packed21 = wq_stream21.read();
            ap_uint<512> packed22 = wq_stream22.read(); ap_uint<512> packed23 = wq_stream23.read();
            ap_uint<512> packed24 = wq_stream24.read(); ap_uint<512> packed25 = wq_stream25.read();
            ap_uint<512> packed26 = wq_stream26.read(); ap_uint<512> packed27 = wq_stream27.read();
            ap_uint<512> packed28 = wq_stream28.read(); ap_uint<512> packed29 = wq_stream29.read();
            ap_uint<512> packed30 = wq_stream30.read(); ap_uint<512> packed31 = wq_stream31.read();

            int8_t sg0 = sg_stream0.read(); int8_t sg1 = sg_stream1.read();
            int8_t sg2 = sg_stream2.read(); int8_t sg3 = sg_stream3.read();
            int8_t sg4 = sg_stream4.read(); int8_t sg5 = sg_stream5.read();
            int8_t sg6 = sg_stream6.read(); int8_t sg7 = sg_stream7.read();
            int8_t sg8 = sg_stream8.read(); int8_t sg9 = sg_stream9.read();
            int8_t sg10 = sg_stream10.read(); int8_t sg11 = sg_stream11.read();
            int8_t sg12 = sg_stream12.read(); int8_t sg13 = sg_stream13.read();
            int8_t sg14 = sg_stream14.read(); int8_t sg15 = sg_stream15.read();
            int8_t sg16 = sg_stream16.read(); int8_t sg17 = sg_stream17.read();
            int8_t sg18 = sg_stream18.read(); int8_t sg19 = sg_stream19.read();
            int8_t sg20 = sg_stream20.read(); int8_t sg21 = sg_stream21.read();
            int8_t sg22 = sg_stream22.read(); int8_t sg23 = sg_stream23.read();
            int8_t sg24 = sg_stream24.read(); int8_t sg25 = sg_stream25.read();
            int8_t sg26 = sg_stream26.read(); int8_t sg27 = sg_stream27.read();
            int8_t sg28 = sg_stream28.read(); int8_t sg29 = sg_stream29.read();
            int8_t sg30 = sg_stream30.read(); int8_t sg31 = sg_stream31.read();

            // Delayed accumulation: add PREVIOUS iteration's sg*partial (breaks critical path)
            acc0 += prev_sg0 * prev_partial0; acc1 += prev_sg1 * prev_partial1;
            acc2 += prev_sg2 * prev_partial2; acc3 += prev_sg3 * prev_partial3;
            acc4 += prev_sg4 * prev_partial4; acc5 += prev_sg5 * prev_partial5;
            acc6 += prev_sg6 * prev_partial6; acc7 += prev_sg7 * prev_partial7;
            acc8 += prev_sg8 * prev_partial8; acc9 += prev_sg9 * prev_partial9;
            acc10 += prev_sg10 * prev_partial10; acc11 += prev_sg11 * prev_partial11;
            acc12 += prev_sg12 * prev_partial12; acc13 += prev_sg13 * prev_partial13;
            acc14 += prev_sg14 * prev_partial14; acc15 += prev_sg15 * prev_partial15;
            acc16 += prev_sg16 * prev_partial16; acc17 += prev_sg17 * prev_partial17;
            acc18 += prev_sg18 * prev_partial18; acc19 += prev_sg19 * prev_partial19;
            acc20 += prev_sg20 * prev_partial20; acc21 += prev_sg21 * prev_partial21;
            acc22 += prev_sg22 * prev_partial22; acc23 += prev_sg23 * prev_partial23;
            acc24 += prev_sg24 * prev_partial24; acc25 += prev_sg25 * prev_partial25;
            acc26 += prev_sg26 * prev_partial26; acc27 += prev_sg27 * prev_partial27;
            acc28 += prev_sg28 * prev_partial28; acc29 += prev_sg29 * prev_partial29;
            acc30 += prev_sg30 * prev_partial30; acc31 += prev_sg31 * prev_partial31;

            const int base_idx = g * GROUP_SIZE;
            int32_t partial0 = 0, partial1 = 0, partial2 = 0, partial3 = 0, partial4 = 0, partial5 = 0, partial6 = 0, partial7 = 0;
            int32_t partial8 = 0, partial9 = 0, partial10 = 0, partial11 = 0, partial12 = 0, partial13 = 0, partial14 = 0, partial15 = 0;
            int32_t partial16 = 0, partial17 = 0, partial18 = 0, partial19 = 0, partial20 = 0, partial21 = 0, partial22 = 0, partial23 = 0;
            int32_t partial24 = 0, partial25 = 0, partial26 = 0, partial27 = 0, partial28 = 0, partial29 = 0, partial30 = 0, partial31 = 0;

            // Stage 1: Unpack weights (break combinational path from wq read to mul)
            ap_int<4> w_unpacked[SHARDS][GS];
            #pragma HLS ARRAY_PARTITION variable=w_unpacked complete dim=0

        unpack_loop_w4a8:
            for (int k = 0; k < GROUP_SIZE; ++k) {
#pragma HLS UNROLL
                // Extract 4-bit nibbles as signed values (two's complement in 4 bits)
                ap_int<4> w0 = packed0.range(4 * k + 3, 4 * k); ap_int<4> w1 = packed1.range(4 * k + 3, 4 * k);
                ap_int<4> w2 = packed2.range(4 * k + 3, 4 * k); ap_int<4> w3 = packed3.range(4 * k + 3, 4 * k);
                ap_int<4> w4 = packed4.range(4 * k + 3, 4 * k); ap_int<4> w5 = packed5.range(4 * k + 3, 4 * k);
                ap_int<4> w6 = packed6.range(4 * k + 3, 4 * k); ap_int<4> w7 = packed7.range(4 * k + 3, 4 * k);
                ap_int<4> w8 = packed8.range(4 * k + 3, 4 * k); ap_int<4> w9 = packed9.range(4 * k + 3, 4 * k);
                ap_int<4> w10 = packed10.range(4 * k + 3, 4 * k); ap_int<4> w11 = packed11.range(4 * k + 3, 4 * k);
                ap_int<4> w12 = packed12.range(4 * k + 3, 4 * k); ap_int<4> w13 = packed13.range(4 * k + 3, 4 * k);
                ap_int<4> w14 = packed14.range(4 * k + 3, 4 * k); ap_int<4> w15 = packed15.range(4 * k + 3, 4 * k);
                ap_int<4> w16 = packed16.range(4 * k + 3, 4 * k); ap_int<4> w17 = packed17.range(4 * k + 3, 4 * k);
                ap_int<4> w18 = packed18.range(4 * k + 3, 4 * k); ap_int<4> w19 = packed19.range(4 * k + 3, 4 * k);
                ap_int<4> w20 = packed20.range(4 * k + 3, 4 * k); ap_int<4> w21 = packed21.range(4 * k + 3, 4 * k);
                ap_int<4> w22 = packed22.range(4 * k + 3, 4 * k); ap_int<4> w23 = packed23.range(4 * k + 3, 4 * k);
                ap_int<4> w24 = packed24.range(4 * k + 3, 4 * k); ap_int<4> w25 = packed25.range(4 * k + 3, 4 * k);
                ap_int<4> w26 = packed26.range(4 * k + 3, 4 * k); ap_int<4> w27 = packed27.range(4 * k + 3, 4 * k);
                ap_int<4> w28 = packed28.range(4 * k + 3, 4 * k); ap_int<4> w29 = packed29.range(4 * k + 3, 4 * k);
                ap_int<4> w30 = packed30.range(4 * k + 3, 4 * k); ap_int<4> w31 = packed31.range(4 * k + 3, 4 * k);

                w_unpacked[0][k] = w0; w_unpacked[1][k] = w1; w_unpacked[2][k] = w2; w_unpacked[3][k] = w3;
                w_unpacked[4][k] = w4; w_unpacked[5][k] = w5; w_unpacked[6][k] = w6; w_unpacked[7][k] = w7;
                w_unpacked[8][k] = w8; w_unpacked[9][k] = w9; w_unpacked[10][k] = w10; w_unpacked[11][k] = w11;
                w_unpacked[12][k] = w12; w_unpacked[13][k] = w13; w_unpacked[14][k] = w14; w_unpacked[15][k] = w15;
                w_unpacked[16][k] = w16; w_unpacked[17][k] = w17; w_unpacked[18][k] = w18; w_unpacked[19][k] = w19;
                w_unpacked[20][k] = w20; w_unpacked[21][k] = w21; w_unpacked[22][k] = w22; w_unpacked[23][k] = w23;
                w_unpacked[24][k] = w24; w_unpacked[25][k] = w25; w_unpacked[26][k] = w26; w_unpacked[27][k] = w27;
                w_unpacked[28][k] = w28; w_unpacked[29][k] = w29; w_unpacked[30][k] = w30; w_unpacked[31][k] = w31;
            }

            // Stage 2: Compute dot product (using registered unpacked weights)
        compute_loop_w4a8:
            for (int k = 0; k < GROUP_SIZE; ++k) {
#pragma HLS UNROLL
                int16_t xval = (int16_t)x_buffer[base_idx + k];

                partial0 += xval * w_unpacked[0][k]; partial1 += xval * w_unpacked[1][k];
                partial2 += xval * w_unpacked[2][k]; partial3 += xval * w_unpacked[3][k];
                partial4 += xval * w_unpacked[4][k]; partial5 += xval * w_unpacked[5][k];
                partial6 += xval * w_unpacked[6][k]; partial7 += xval * w_unpacked[7][k];
                partial8 += xval * w_unpacked[8][k]; partial9 += xval * w_unpacked[9][k];
                partial10 += xval * w_unpacked[10][k]; partial11 += xval * w_unpacked[11][k];
                partial12 += xval * w_unpacked[12][k]; partial13 += xval * w_unpacked[13][k];
                partial14 += xval * w_unpacked[14][k]; partial15 += xval * w_unpacked[15][k];
                partial16 += xval * w_unpacked[16][k]; partial17 += xval * w_unpacked[17][k];
                partial18 += xval * w_unpacked[18][k]; partial19 += xval * w_unpacked[19][k];
                partial20 += xval * w_unpacked[20][k]; partial21 += xval * w_unpacked[21][k];
                partial22 += xval * w_unpacked[22][k]; partial23 += xval * w_unpacked[23][k];
                partial24 += xval * w_unpacked[24][k]; partial25 += xval * w_unpacked[25][k];
                partial26 += xval * w_unpacked[26][k]; partial27 += xval * w_unpacked[27][k];
                partial28 += xval * w_unpacked[28][k]; partial29 += xval * w_unpacked[29][k];
                partial30 += xval * w_unpacked[30][k]; partial31 += xval * w_unpacked[31][k];
            }

            // Save current partial and sg for next iteration (pipeline register)
            prev_partial0 = partial0; prev_partial1 = partial1; prev_partial2 = partial2; prev_partial3 = partial3;
            prev_partial4 = partial4; prev_partial5 = partial5; prev_partial6 = partial6; prev_partial7 = partial7;
            prev_partial8 = partial8; prev_partial9 = partial9; prev_partial10 = partial10; prev_partial11 = partial11;
            prev_partial12 = partial12; prev_partial13 = partial13; prev_partial14 = partial14; prev_partial15 = partial15;
            prev_partial16 = partial16; prev_partial17 = partial17; prev_partial18 = partial18; prev_partial19 = partial19;
            prev_partial20 = partial20; prev_partial21 = partial21; prev_partial22 = partial22; prev_partial23 = partial23;
            prev_partial24 = partial24; prev_partial25 = partial25; prev_partial26 = partial26; prev_partial27 = partial27;
            prev_partial28 = partial28; prev_partial29 = partial29; prev_partial30 = partial30; prev_partial31 = partial31;
            prev_sg0 = sg0; prev_sg1 = sg1; prev_sg2 = sg2; prev_sg3 = sg3;
            prev_sg4 = sg4; prev_sg5 = sg5; prev_sg6 = sg6; prev_sg7 = sg7;
            prev_sg8 = sg8; prev_sg9 = sg9; prev_sg10 = sg10; prev_sg11 = sg11;
            prev_sg12 = sg12; prev_sg13 = sg13; prev_sg14 = sg14; prev_sg15 = sg15;
            prev_sg16 = sg16; prev_sg17 = sg17; prev_sg18 = sg18; prev_sg19 = sg19;
            prev_sg20 = sg20; prev_sg21 = sg21; prev_sg22 = sg22; prev_sg23 = sg23;
            prev_sg24 = sg24; prev_sg25 = sg25; prev_sg26 = sg26; prev_sg27 = sg27;
            prev_sg28 = sg28; prev_sg29 = sg29; prev_sg30 = sg30; prev_sg31 = sg31;
        }

        // Accumulate the LAST iteration's sg*partial (not done in loop)
        acc0 += prev_sg0 * prev_partial0; acc1 += prev_sg1 * prev_partial1;
        acc2 += prev_sg2 * prev_partial2; acc3 += prev_sg3 * prev_partial3;
        acc4 += prev_sg4 * prev_partial4; acc5 += prev_sg5 * prev_partial5;
        acc6 += prev_sg6 * prev_partial6; acc7 += prev_sg7 * prev_partial7;
        acc8 += prev_sg8 * prev_partial8; acc9 += prev_sg9 * prev_partial9;
        acc10 += prev_sg10 * prev_partial10; acc11 += prev_sg11 * prev_partial11;
        acc12 += prev_sg12 * prev_partial12; acc13 += prev_sg13 * prev_partial13;
        acc14 += prev_sg14 * prev_partial14; acc15 += prev_sg15 * prev_partial15;
        acc16 += prev_sg16 * prev_partial16; acc17 += prev_sg17 * prev_partial17;
        acc18 += prev_sg18 * prev_partial18; acc19 += prev_sg19 * prev_partial19;
        acc20 += prev_sg20 * prev_partial20; acc21 += prev_sg21 * prev_partial21;
        acc22 += prev_sg22 * prev_partial22; acc23 += prev_sg23 * prev_partial23;
        acc24 += prev_sg24 * prev_partial24; acc25 += prev_sg25 * prev_partial25;
        acc26 += prev_sg26 * prev_partial26; acc27 += prev_sg27 * prev_partial27;
        acc28 += prev_sg28 * prev_partial28; acc29 += prev_sg29 * prev_partial29;
        acc30 += prev_sg30 * prev_partial30; acc31 += prev_sg31 * prev_partial31;

        // Read channel scales and compute final output
        float sc0 = sc_stream0.read(); float sc1 = sc_stream1.read(); float sc2 = sc_stream2.read(); float sc3 = sc_stream3.read();
        float sc4 = sc_stream4.read(); float sc5 = sc_stream5.read(); float sc6 = sc_stream6.read(); float sc7 = sc_stream7.read();
        float sc8 = sc_stream8.read(); float sc9 = sc_stream9.read(); float sc10 = sc_stream10.read(); float sc11 = sc_stream11.read();
        float sc12 = sc_stream12.read(); float sc13 = sc_stream13.read(); float sc14 = sc_stream14.read(); float sc15 = sc_stream15.read();
        float sc16 = sc_stream16.read(); float sc17 = sc_stream17.read(); float sc18 = sc_stream18.read(); float sc19 = sc_stream19.read();
        float sc20 = sc_stream20.read(); float sc21 = sc_stream21.read(); float sc22 = sc_stream22.read(); float sc23 = sc_stream23.read();
        float sc24 = sc_stream24.read(); float sc25 = sc_stream25.read(); float sc26 = sc_stream26.read(); float sc27 = sc_stream27.read();
        float sc28 = sc_stream28.read(); float sc29 = sc_stream29.read(); float sc30 = sc_stream30.read(); float sc31 = sc_stream31.read();

        res_stream0.write((float)acc0 * (xs_val * sc0)); res_stream1.write((float)acc1 * (xs_val * sc1));
        res_stream2.write((float)acc2 * (xs_val * sc2)); res_stream3.write((float)acc3 * (xs_val * sc3));
        res_stream4.write((float)acc4 * (xs_val * sc4)); res_stream5.write((float)acc5 * (xs_val * sc5));
        res_stream6.write((float)acc6 * (xs_val * sc6)); res_stream7.write((float)acc7 * (xs_val * sc7));
        res_stream8.write((float)acc8 * (xs_val * sc8)); res_stream9.write((float)acc9 * (xs_val * sc9));
        res_stream10.write((float)acc10 * (xs_val * sc10)); res_stream11.write((float)acc11 * (xs_val * sc11));
        res_stream12.write((float)acc12 * (xs_val * sc12)); res_stream13.write((float)acc13 * (xs_val * sc13));
        res_stream14.write((float)acc14 * (xs_val * sc14)); res_stream15.write((float)acc15 * (xs_val * sc15));
        res_stream16.write((float)acc16 * (xs_val * sc16)); res_stream17.write((float)acc17 * (xs_val * sc17));
        res_stream18.write((float)acc18 * (xs_val * sc18)); res_stream19.write((float)acc19 * (xs_val * sc19));
        res_stream20.write((float)acc20 * (xs_val * sc20)); res_stream21.write((float)acc21 * (xs_val * sc21));
        res_stream22.write((float)acc22 * (xs_val * sc22)); res_stream23.write((float)acc23 * (xs_val * sc23));
        res_stream24.write((float)acc24 * (xs_val * sc24)); res_stream25.write((float)acc25 * (xs_val * sc25));
        res_stream26.write((float)acc26 * (xs_val * sc26)); res_stream27.write((float)acc27 * (xs_val * sc27));
        res_stream28.write((float)acc28 * (xs_val * sc28)); res_stream29.write((float)acc29 * (xs_val * sc29));
        res_stream30.write((float)acc30 * (xs_val * sc30)); res_stream31.write((float)acc31 * (xs_val * sc31));
    }
}

static void store_core_32(
    float *xout_row,
    int N_oct,
    hls::stream<float> &res_stream0, hls::stream<float> &res_stream1,
    hls::stream<float> &res_stream2, hls::stream<float> &res_stream3,
    hls::stream<float> &res_stream4, hls::stream<float> &res_stream5,
    hls::stream<float> &res_stream6, hls::stream<float> &res_stream7,
    hls::stream<float> &res_stream8, hls::stream<float> &res_stream9,
    hls::stream<float> &res_stream10, hls::stream<float> &res_stream11,
    hls::stream<float> &res_stream12, hls::stream<float> &res_stream13,
    hls::stream<float> &res_stream14, hls::stream<float> &res_stream15,
    hls::stream<float> &res_stream16, hls::stream<float> &res_stream17,
    hls::stream<float> &res_stream18, hls::stream<float> &res_stream19,
    hls::stream<float> &res_stream20, hls::stream<float> &res_stream21,
    hls::stream<float> &res_stream22, hls::stream<float> &res_stream23,
    hls::stream<float> &res_stream24, hls::stream<float> &res_stream25,
    hls::stream<float> &res_stream26, hls::stream<float> &res_stream27,
    hls::stream<float> &res_stream28, hls::stream<float> &res_stream29,
    hls::stream<float> &res_stream30, hls::stream<float> &res_stream31
) {
    // 本地 buffer：两维都完全分区，支持任意 32 路并行读取
    float buf[32][64];
#pragma HLS ARRAY_PARTITION variable=buf complete dim=0

    // Phase 1: 32 路并行读取 stream（II=1，latency=N_oct）
read_streams:
    for (int k = 0; k < N_oct; ++k) {
#pragma HLS PIPELINE II=1
        buf[0][k]  = res_stream0.read();  buf[1][k]  = res_stream1.read();
        buf[2][k]  = res_stream2.read();  buf[3][k]  = res_stream3.read();
        buf[4][k]  = res_stream4.read();  buf[5][k]  = res_stream5.read();
        buf[6][k]  = res_stream6.read();  buf[7][k]  = res_stream7.read();
        buf[8][k]  = res_stream8.read();  buf[9][k]  = res_stream9.read();
        buf[10][k] = res_stream10.read(); buf[11][k] = res_stream11.read();
        buf[12][k] = res_stream12.read(); buf[13][k] = res_stream13.read();
        buf[14][k] = res_stream14.read(); buf[15][k] = res_stream15.read();
        buf[16][k] = res_stream16.read(); buf[17][k] = res_stream17.read();
        buf[18][k] = res_stream18.read(); buf[19][k] = res_stream19.read();
        buf[20][k] = res_stream20.read(); buf[21][k] = res_stream21.read();
        buf[22][k] = res_stream22.read(); buf[23][k] = res_stream23.read();
        buf[24][k] = res_stream24.read(); buf[25][k] = res_stream25.read();
        buf[26][k] = res_stream26.read(); buf[27][k] = res_stream27.read();
        buf[28][k] = res_stream28.read(); buf[29][k] = res_stream29.read();
        buf[30][k] = res_stream30.read(); buf[31][k] = res_stream31.read();
    }

    // Phase 2: 斜向遍历，32 路并行写入 32 个不同分区
    // shard s 在 k_base 时访问 k = (k_base + s) % N_oct
    // 分区 = k % 32，32 个 shard 访问 32 个不同分区，可并行！
write_skewed:
    for (int k_base = 0; k_base < N_oct; ++k_base) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=xout_row inter false
        // 计算每个 shard 的斜向 k 索引
        int k0  = (k_base + 0)  % N_oct, k1  = (k_base + 1)  % N_oct;
        int k2  = (k_base + 2)  % N_oct, k3  = (k_base + 3)  % N_oct;
        int k4  = (k_base + 4)  % N_oct, k5  = (k_base + 5)  % N_oct;
        int k6  = (k_base + 6)  % N_oct, k7  = (k_base + 7)  % N_oct;
        int k8  = (k_base + 8)  % N_oct, k9  = (k_base + 9)  % N_oct;
        int k10 = (k_base + 10) % N_oct, k11 = (k_base + 11) % N_oct;
        int k12 = (k_base + 12) % N_oct, k13 = (k_base + 13) % N_oct;
        int k14 = (k_base + 14) % N_oct, k15 = (k_base + 15) % N_oct;
        int k16 = (k_base + 16) % N_oct, k17 = (k_base + 17) % N_oct;
        int k18 = (k_base + 18) % N_oct, k19 = (k_base + 19) % N_oct;
        int k20 = (k_base + 20) % N_oct, k21 = (k_base + 21) % N_oct;
        int k22 = (k_base + 22) % N_oct, k23 = (k_base + 23) % N_oct;
        int k24 = (k_base + 24) % N_oct, k25 = (k_base + 25) % N_oct;
        int k26 = (k_base + 26) % N_oct, k27 = (k_base + 27) % N_oct;
        int k28 = (k_base + 28) % N_oct, k29 = (k_base + 29) % N_oct;
        int k30 = (k_base + 30) % N_oct, k31 = (k_base + 31) % N_oct;

        // 32 路并行写入（每个访问不同分区）
        xout_row[0  * N_oct + k0]  = buf[0][k0];   xout_row[1  * N_oct + k1]  = buf[1][k1];
        xout_row[2  * N_oct + k2]  = buf[2][k2];   xout_row[3  * N_oct + k3]  = buf[3][k3];
        xout_row[4  * N_oct + k4]  = buf[4][k4];   xout_row[5  * N_oct + k5]  = buf[5][k5];
        xout_row[6  * N_oct + k6]  = buf[6][k6];   xout_row[7  * N_oct + k7]  = buf[7][k7];
        xout_row[8  * N_oct + k8]  = buf[8][k8];   xout_row[9  * N_oct + k9]  = buf[9][k9];
        xout_row[10 * N_oct + k10] = buf[10][k10]; xout_row[11 * N_oct + k11] = buf[11][k11];
        xout_row[12 * N_oct + k12] = buf[12][k12]; xout_row[13 * N_oct + k13] = buf[13][k13];
        xout_row[14 * N_oct + k14] = buf[14][k14]; xout_row[15 * N_oct + k15] = buf[15][k15];
        xout_row[16 * N_oct + k16] = buf[16][k16]; xout_row[17 * N_oct + k17] = buf[17][k17];
        xout_row[18 * N_oct + k18] = buf[18][k18]; xout_row[19 * N_oct + k19] = buf[19][k19];
        xout_row[20 * N_oct + k20] = buf[20][k20]; xout_row[21 * N_oct + k21] = buf[21][k21];
        xout_row[22 * N_oct + k22] = buf[22][k22]; xout_row[23 * N_oct + k23] = buf[23][k23];
        xout_row[24 * N_oct + k24] = buf[24][k24]; xout_row[25 * N_oct + k25] = buf[25][k25];
        xout_row[26 * N_oct + k26] = buf[26][k26]; xout_row[27 * N_oct + k27] = buf[27][k27];
        xout_row[28 * N_oct + k28] = buf[28][k28]; xout_row[29 * N_oct + k29] = buf[29][k29];
        xout_row[30 * N_oct + k30] = buf[30][k30]; xout_row[31 * N_oct + k31] = buf[31][k31];
    }
}

static void matmul_core_w4a8(
    float *xout_row,
    const int8_t *x_buffer,
    float xs_val,
    int N,
    int K,
    hls::stream<ap_uint<512> > &wq_stream0, hls::stream<ap_uint<512> > &wq_stream1,
    hls::stream<ap_uint<512> > &wq_stream2, hls::stream<ap_uint<512> > &wq_stream3,
    hls::stream<ap_uint<512> > &wq_stream4, hls::stream<ap_uint<512> > &wq_stream5,
    hls::stream<ap_uint<512> > &wq_stream6, hls::stream<ap_uint<512> > &wq_stream7,
    hls::stream<ap_uint<512> > &wq_stream8, hls::stream<ap_uint<512> > &wq_stream9,
    hls::stream<ap_uint<512> > &wq_stream10, hls::stream<ap_uint<512> > &wq_stream11,
    hls::stream<ap_uint<512> > &wq_stream12, hls::stream<ap_uint<512> > &wq_stream13,
    hls::stream<ap_uint<512> > &wq_stream14, hls::stream<ap_uint<512> > &wq_stream15,
    hls::stream<ap_uint<512> > &wq_stream16, hls::stream<ap_uint<512> > &wq_stream17,
    hls::stream<ap_uint<512> > &wq_stream18, hls::stream<ap_uint<512> > &wq_stream19,
    hls::stream<ap_uint<512> > &wq_stream20, hls::stream<ap_uint<512> > &wq_stream21,
    hls::stream<ap_uint<512> > &wq_stream22, hls::stream<ap_uint<512> > &wq_stream23,
    hls::stream<ap_uint<512> > &wq_stream24, hls::stream<ap_uint<512> > &wq_stream25,
    hls::stream<ap_uint<512> > &wq_stream26, hls::stream<ap_uint<512> > &wq_stream27,
    hls::stream<ap_uint<512> > &wq_stream28, hls::stream<ap_uint<512> > &wq_stream29,
    hls::stream<ap_uint<512> > &wq_stream30, hls::stream<ap_uint<512> > &wq_stream31,
    hls::stream<int8_t> &sg_stream0, hls::stream<int8_t> &sg_stream1,
    hls::stream<int8_t> &sg_stream2, hls::stream<int8_t> &sg_stream3,
    hls::stream<int8_t> &sg_stream4, hls::stream<int8_t> &sg_stream5,
    hls::stream<int8_t> &sg_stream6, hls::stream<int8_t> &sg_stream7,
    hls::stream<int8_t> &sg_stream8, hls::stream<int8_t> &sg_stream9,
    hls::stream<int8_t> &sg_stream10, hls::stream<int8_t> &sg_stream11,
    hls::stream<int8_t> &sg_stream12, hls::stream<int8_t> &sg_stream13,
    hls::stream<int8_t> &sg_stream14, hls::stream<int8_t> &sg_stream15,
    hls::stream<int8_t> &sg_stream16, hls::stream<int8_t> &sg_stream17,
    hls::stream<int8_t> &sg_stream18, hls::stream<int8_t> &sg_stream19,
    hls::stream<int8_t> &sg_stream20, hls::stream<int8_t> &sg_stream21,
    hls::stream<int8_t> &sg_stream22, hls::stream<int8_t> &sg_stream23,
    hls::stream<int8_t> &sg_stream24, hls::stream<int8_t> &sg_stream25,
    hls::stream<int8_t> &sg_stream26, hls::stream<int8_t> &sg_stream27,
    hls::stream<int8_t> &sg_stream28, hls::stream<int8_t> &sg_stream29,
    hls::stream<int8_t> &sg_stream30, hls::stream<int8_t> &sg_stream31,
    hls::stream<float>  &sc_stream0, hls::stream<float>  &sc_stream1,
    hls::stream<float>  &sc_stream2, hls::stream<float>  &sc_stream3,
    hls::stream<float>  &sc_stream4, hls::stream<float>  &sc_stream5,
    hls::stream<float>  &sc_stream6, hls::stream<float>  &sc_stream7,
    hls::stream<float>  &sc_stream8, hls::stream<float>  &sc_stream9,
    hls::stream<float>  &sc_stream10, hls::stream<float>  &sc_stream11,
    hls::stream<float>  &sc_stream12, hls::stream<float>  &sc_stream13,
    hls::stream<float>  &sc_stream14, hls::stream<float>  &sc_stream15,
    hls::stream<float>  &sc_stream16, hls::stream<float>  &sc_stream17,
    hls::stream<float>  &sc_stream18, hls::stream<float>  &sc_stream19,
    hls::stream<float>  &sc_stream20, hls::stream<float>  &sc_stream21,
    hls::stream<float>  &sc_stream22, hls::stream<float>  &sc_stream23,
    hls::stream<float>  &sc_stream24, hls::stream<float>  &sc_stream25,
    hls::stream<float>  &sc_stream26, hls::stream<float>  &sc_stream27,
    hls::stream<float>  &sc_stream28, hls::stream<float>  &sc_stream29,
    hls::stream<float>  &sc_stream30, hls::stream<float>  &sc_stream31) {
#pragma HLS ALLOCATION function instances=matmul_core_w4a8 limit=1

    const int N_oct = N / SHARDS;
    hls::stream<float> res_stream0("res_stream_0"); hls::stream<float> res_stream1("res_stream_1");
    hls::stream<float> res_stream2("res_stream_2"); hls::stream<float> res_stream3("res_stream_3");
    hls::stream<float> res_stream4("res_stream_4"); hls::stream<float> res_stream5("res_stream_5");
    hls::stream<float> res_stream6("res_stream_6"); hls::stream<float> res_stream7("res_stream_7");
    hls::stream<float> res_stream8("res_stream_8"); hls::stream<float> res_stream9("res_stream_9");
    hls::stream<float> res_stream10("res_stream_10"); hls::stream<float> res_stream11("res_stream_11");
    hls::stream<float> res_stream12("res_stream_12"); hls::stream<float> res_stream13("res_stream_13");
    hls::stream<float> res_stream14("res_stream_14"); hls::stream<float> res_stream15("res_stream_15");
    hls::stream<float> res_stream16("res_stream_16"); hls::stream<float> res_stream17("res_stream_17");
    hls::stream<float> res_stream18("res_stream_18"); hls::stream<float> res_stream19("res_stream_19");
    hls::stream<float> res_stream20("res_stream_20"); hls::stream<float> res_stream21("res_stream_21");
    hls::stream<float> res_stream22("res_stream_22"); hls::stream<float> res_stream23("res_stream_23");
    hls::stream<float> res_stream24("res_stream_24"); hls::stream<float> res_stream25("res_stream_25");
    hls::stream<float> res_stream26("res_stream_26"); hls::stream<float> res_stream27("res_stream_27");
    hls::stream<float> res_stream28("res_stream_28"); hls::stream<float> res_stream29("res_stream_29");
    hls::stream<float> res_stream30("res_stream_30"); hls::stream<float> res_stream31("res_stream_31");
#pragma HLS STREAM variable=res_stream0 depth=16
#pragma HLS STREAM variable=res_stream1 depth=16
#pragma HLS STREAM variable=res_stream2 depth=16
#pragma HLS STREAM variable=res_stream3 depth=16
#pragma HLS STREAM variable=res_stream4 depth=16
#pragma HLS STREAM variable=res_stream5 depth=16
#pragma HLS STREAM variable=res_stream6 depth=16
#pragma HLS STREAM variable=res_stream7 depth=16
#pragma HLS STREAM variable=res_stream8 depth=16
#pragma HLS STREAM variable=res_stream9 depth=16
#pragma HLS STREAM variable=res_stream10 depth=16
#pragma HLS STREAM variable=res_stream11 depth=16
#pragma HLS STREAM variable=res_stream12 depth=16
#pragma HLS STREAM variable=res_stream13 depth=16
#pragma HLS STREAM variable=res_stream14 depth=16
#pragma HLS STREAM variable=res_stream15 depth=16
#pragma HLS STREAM variable=res_stream16 depth=16
#pragma HLS STREAM variable=res_stream17 depth=16
#pragma HLS STREAM variable=res_stream18 depth=16
#pragma HLS STREAM variable=res_stream19 depth=16
#pragma HLS STREAM variable=res_stream20 depth=16
#pragma HLS STREAM variable=res_stream21 depth=16
#pragma HLS STREAM variable=res_stream22 depth=16
#pragma HLS STREAM variable=res_stream23 depth=16
#pragma HLS STREAM variable=res_stream24 depth=16
#pragma HLS STREAM variable=res_stream25 depth=16
#pragma HLS STREAM variable=res_stream26 depth=16
#pragma HLS STREAM variable=res_stream27 depth=16
#pragma HLS STREAM variable=res_stream28 depth=16
#pragma HLS STREAM variable=res_stream29 depth=16
#pragma HLS STREAM variable=res_stream30 depth=16
#pragma HLS STREAM variable=res_stream31 depth=16
#pragma HLS DATAFLOW
    compute_core_w4a8(res_stream0, res_stream1, res_stream2, res_stream3, res_stream4, res_stream5, res_stream6, res_stream7,
                      res_stream8, res_stream9, res_stream10, res_stream11, res_stream12, res_stream13, res_stream14, res_stream15,
                      res_stream16, res_stream17, res_stream18, res_stream19, res_stream20, res_stream21, res_stream22, res_stream23,
                      res_stream24, res_stream25, res_stream26, res_stream27, res_stream28, res_stream29, res_stream30, res_stream31,
                      x_buffer, xs_val, N, K,
                      wq_stream0, wq_stream1, wq_stream2, wq_stream3, wq_stream4, wq_stream5, wq_stream6, wq_stream7,
                      wq_stream8, wq_stream9, wq_stream10, wq_stream11, wq_stream12, wq_stream13, wq_stream14, wq_stream15,
                      wq_stream16, wq_stream17, wq_stream18, wq_stream19, wq_stream20, wq_stream21, wq_stream22, wq_stream23,
                      wq_stream24, wq_stream25, wq_stream26, wq_stream27, wq_stream28, wq_stream29, wq_stream30, wq_stream31,
                      sg_stream0, sg_stream1, sg_stream2, sg_stream3, sg_stream4, sg_stream5, sg_stream6, sg_stream7,
                      sg_stream8, sg_stream9, sg_stream10, sg_stream11, sg_stream12, sg_stream13, sg_stream14, sg_stream15,
                      sg_stream16, sg_stream17, sg_stream18, sg_stream19, sg_stream20, sg_stream21, sg_stream22, sg_stream23,
                      sg_stream24, sg_stream25, sg_stream26, sg_stream27, sg_stream28, sg_stream29, sg_stream30, sg_stream31,
                      sc_stream0, sc_stream1, sc_stream2, sc_stream3, sc_stream4, sc_stream5, sc_stream6, sc_stream7,
                      sc_stream8, sc_stream9, sc_stream10, sc_stream11, sc_stream12, sc_stream13, sc_stream14, sc_stream15,
                      sc_stream16, sc_stream17, sc_stream18, sc_stream19, sc_stream20, sc_stream21, sc_stream22, sc_stream23,
                      sc_stream24, sc_stream25, sc_stream26, sc_stream27, sc_stream28, sc_stream29, sc_stream30, sc_stream31);

    // Reuse existing store_core_32 for layout (32-way N splitting)
    store_core_32(xout_row, N_oct,
                  res_stream0, res_stream1, res_stream2, res_stream3, res_stream4, res_stream5, res_stream6, res_stream7,
                  res_stream8, res_stream9, res_stream10, res_stream11, res_stream12, res_stream13, res_stream14, res_stream15,
                  res_stream16, res_stream17, res_stream18, res_stream19, res_stream20, res_stream21, res_stream22, res_stream23,
                  res_stream24, res_stream25, res_stream26, res_stream27, res_stream28, res_stream29, res_stream30, res_stream31);
}

static void matmul_compute_outputs_w4a8(
    float *xout,
    const int8_t *xq,
    const float  *xs,
    int M, int N, int K,
    hls::stream<ap_uint<512> > &wq_stream0, hls::stream<ap_uint<512> > &wq_stream1,
    hls::stream<ap_uint<512> > &wq_stream2, hls::stream<ap_uint<512> > &wq_stream3,
    hls::stream<ap_uint<512> > &wq_stream4, hls::stream<ap_uint<512> > &wq_stream5,
    hls::stream<ap_uint<512> > &wq_stream6, hls::stream<ap_uint<512> > &wq_stream7,
    hls::stream<ap_uint<512> > &wq_stream8, hls::stream<ap_uint<512> > &wq_stream9,
    hls::stream<ap_uint<512> > &wq_stream10, hls::stream<ap_uint<512> > &wq_stream11,
    hls::stream<ap_uint<512> > &wq_stream12, hls::stream<ap_uint<512> > &wq_stream13,
    hls::stream<ap_uint<512> > &wq_stream14, hls::stream<ap_uint<512> > &wq_stream15,
    hls::stream<ap_uint<512> > &wq_stream16, hls::stream<ap_uint<512> > &wq_stream17,
    hls::stream<ap_uint<512> > &wq_stream18, hls::stream<ap_uint<512> > &wq_stream19,
    hls::stream<ap_uint<512> > &wq_stream20, hls::stream<ap_uint<512> > &wq_stream21,
    hls::stream<ap_uint<512> > &wq_stream22, hls::stream<ap_uint<512> > &wq_stream23,
    hls::stream<ap_uint<512> > &wq_stream24, hls::stream<ap_uint<512> > &wq_stream25,
    hls::stream<ap_uint<512> > &wq_stream26, hls::stream<ap_uint<512> > &wq_stream27,
    hls::stream<ap_uint<512> > &wq_stream28, hls::stream<ap_uint<512> > &wq_stream29,
    hls::stream<ap_uint<512> > &wq_stream30, hls::stream<ap_uint<512> > &wq_stream31,
    hls::stream<int8_t> &sg_stream0, hls::stream<int8_t> &sg_stream1,
    hls::stream<int8_t> &sg_stream2, hls::stream<int8_t> &sg_stream3,
    hls::stream<int8_t> &sg_stream4, hls::stream<int8_t> &sg_stream5,
    hls::stream<int8_t> &sg_stream6, hls::stream<int8_t> &sg_stream7,
    hls::stream<int8_t> &sg_stream8, hls::stream<int8_t> &sg_stream9,
    hls::stream<int8_t> &sg_stream10, hls::stream<int8_t> &sg_stream11,
    hls::stream<int8_t> &sg_stream12, hls::stream<int8_t> &sg_stream13,
    hls::stream<int8_t> &sg_stream14, hls::stream<int8_t> &sg_stream15,
    hls::stream<int8_t> &sg_stream16, hls::stream<int8_t> &sg_stream17,
    hls::stream<int8_t> &sg_stream18, hls::stream<int8_t> &sg_stream19,
    hls::stream<int8_t> &sg_stream20, hls::stream<int8_t> &sg_stream21,
    hls::stream<int8_t> &sg_stream22, hls::stream<int8_t> &sg_stream23,
    hls::stream<int8_t> &sg_stream24, hls::stream<int8_t> &sg_stream25,
    hls::stream<int8_t> &sg_stream26, hls::stream<int8_t> &sg_stream27,
    hls::stream<int8_t> &sg_stream28, hls::stream<int8_t> &sg_stream29,
    hls::stream<int8_t> &sg_stream30, hls::stream<int8_t> &sg_stream31,
    hls::stream<float>  &sc_stream0, hls::stream<float>  &sc_stream1,
    hls::stream<float>  &sc_stream2, hls::stream<float>  &sc_stream3,
    hls::stream<float>  &sc_stream4, hls::stream<float>  &sc_stream5,
    hls::stream<float>  &sc_stream6, hls::stream<float>  &sc_stream7,
    hls::stream<float>  &sc_stream8, hls::stream<float>  &sc_stream9,
    hls::stream<float>  &sc_stream10, hls::stream<float>  &sc_stream11,
    hls::stream<float>  &sc_stream12, hls::stream<float>  &sc_stream13,
    hls::stream<float>  &sc_stream14, hls::stream<float>  &sc_stream15,
    hls::stream<float>  &sc_stream16, hls::stream<float>  &sc_stream17,
    hls::stream<float>  &sc_stream18, hls::stream<float>  &sc_stream19,
    hls::stream<float>  &sc_stream20, hls::stream<float>  &sc_stream21,
    hls::stream<float>  &sc_stream22, hls::stream<float>  &sc_stream23,
    hls::stream<float>  &sc_stream24, hls::stream<float>  &sc_stream25,
    hls::stream<float>  &sc_stream26, hls::stream<float>  &sc_stream27,
    hls::stream<float>  &sc_stream28, hls::stream<float>  &sc_stream29,
    hls::stream<float>  &sc_stream30, hls::stream<float>  &sc_stream31) {
#pragma HLS INLINE off
        
    const int INT_PER_VEC = 64;
    const int NUM_VECS = (K + INT_PER_VEC - 1) / INT_PER_VEC;
compute_m_q8:
    for (int m = 0; m < M; ++m) {
        int8_t x_buffer[hidden_dim];
        float  xs_val;
        #pragma HLS ARRAY_PARTITION variable=x_buffer cyclic factor=64

    load_x_q8:
        for (int v = 0; v < NUM_VECS; ++v) {
#pragma HLS PIPELINE II=1
            int base = v << 6; // v * 64
            load_x_inner:
            for (int i = 0; i < INT_PER_VEC; ++i) {
#pragma HLS UNROLL
                if (base + i < K) x_buffer[base + i] = xq[m * K + base + i];
            }
        }
        xs_val = xs[m];

        float *xout_row = xout + m * N;
        matmul_core_w4a8(xout_row, x_buffer, xs_val, N, K,
                         wq_stream0, wq_stream1, wq_stream2, wq_stream3, wq_stream4, wq_stream5, wq_stream6, wq_stream7,
                         wq_stream8, wq_stream9, wq_stream10, wq_stream11, wq_stream12, wq_stream13, wq_stream14, wq_stream15,
                         wq_stream16, wq_stream17, wq_stream18, wq_stream19, wq_stream20, wq_stream21, wq_stream22, wq_stream23,
                         wq_stream24, wq_stream25, wq_stream26, wq_stream27, wq_stream28, wq_stream29, wq_stream30, wq_stream31,
                         sg_stream0, sg_stream1, sg_stream2, sg_stream3, sg_stream4, sg_stream5, sg_stream6, sg_stream7,
                         sg_stream8, sg_stream9, sg_stream10, sg_stream11, sg_stream12, sg_stream13, sg_stream14, sg_stream15,
                         sg_stream16, sg_stream17, sg_stream18, sg_stream19, sg_stream20, sg_stream21, sg_stream22, sg_stream23,
                         sg_stream24, sg_stream25, sg_stream26, sg_stream27, sg_stream28, sg_stream29, sg_stream30, sg_stream31,
                         sc_stream0, sc_stream1, sc_stream2, sc_stream3, sc_stream4, sc_stream5, sc_stream6, sc_stream7,
                         sc_stream8, sc_stream9, sc_stream10, sc_stream11, sc_stream12, sc_stream13, sc_stream14, sc_stream15,
                         sc_stream16, sc_stream17, sc_stream18, sc_stream19, sc_stream20, sc_stream21, sc_stream22, sc_stream23,
                         sc_stream24, sc_stream25, sc_stream26, sc_stream27, sc_stream28, sc_stream29, sc_stream30, sc_stream31);
    }
}

static void matmul_engine_w4a8_32port(
    float *xout,
    const int8_t *xq,
    const float *xs,
    const int8_t *wq_base0, const int8_t *wq_base1, const int8_t *wq_base2, const int8_t *wq_base3, const int8_t *wq_base4, const int8_t *wq_base5, const int8_t *wq_base6, const int8_t *wq_base7,
    const int8_t *wq_base8, const int8_t *wq_base9, const int8_t *wq_base10, const int8_t *wq_base11, const int8_t *wq_base12, const int8_t *wq_base13, const int8_t *wq_base14, const int8_t *wq_base15,
    const int8_t *wq_base16, const int8_t *wq_base17, const int8_t *wq_base18, const int8_t *wq_base19, const int8_t *wq_base20, const int8_t *wq_base21, const int8_t *wq_base22, const int8_t *wq_base23,
    const int8_t *wq_base24, const int8_t *wq_base25, const int8_t *wq_base26, const int8_t *wq_base27, const int8_t *wq_base28, const int8_t *wq_base29, const int8_t *wq_base30, const int8_t *wq_base31,
    const int8_t *sg_base0, const int8_t *sg_base1, const int8_t *sg_base2, const int8_t *sg_base3, const int8_t *sg_base4, const int8_t *sg_base5, const int8_t *sg_base6, const int8_t *sg_base7,
    const int8_t *sg_base8, const int8_t *sg_base9, const int8_t *sg_base10, const int8_t *sg_base11, const int8_t *sg_base12, const int8_t *sg_base13, const int8_t *sg_base14, const int8_t *sg_base15,
    const int8_t *sg_base16, const int8_t *sg_base17, const int8_t *sg_base18, const int8_t *sg_base19, const int8_t *sg_base20, const int8_t *sg_base21, const int8_t *sg_base22, const int8_t *sg_base23,
    const int8_t *sg_base24, const int8_t *sg_base25, const int8_t *sg_base26, const int8_t *sg_base27, const int8_t *sg_base28, const int8_t *sg_base29, const int8_t *sg_base30, const int8_t *sg_base31,
    const float  *sc_base0, const float  *sc_base1, const float  *sc_base2, const float  *sc_base3, const float  *sc_base4, const float  *sc_base5, const float  *sc_base6, const float  *sc_base7,
    const float  *sc_base8, const float  *sc_base9, const float  *sc_base10, const float  *sc_base11, const float  *sc_base12, const float  *sc_base13, const float  *sc_base14, const float  *sc_base15,
    const float  *sc_base16, const float  *sc_base17, const float  *sc_base18, const float  *sc_base19, const float  *sc_base20, const float  *sc_base21, const float  *sc_base22, const float  *sc_base23,
    const float  *sc_base24, const float  *sc_base25, const float  *sc_base26, const float  *sc_base27, const float  *sc_base28, const float  *sc_base29, const float  *sc_base30, const float  *sc_base31,
    int M, int N, int K) {
#pragma HLS INLINE off

    hls::stream<ap_uint<512> > wq_stream0("wq_stream0_32p"); hls::stream<ap_uint<512> > wq_stream1("wq_stream1_32p");
    hls::stream<ap_uint<512> > wq_stream2("wq_stream2_32p"); hls::stream<ap_uint<512> > wq_stream3("wq_stream3_32p");
    hls::stream<ap_uint<512> > wq_stream4("wq_stream4_32p"); hls::stream<ap_uint<512> > wq_stream5("wq_stream5_32p");
    hls::stream<ap_uint<512> > wq_stream6("wq_stream6_32p"); hls::stream<ap_uint<512> > wq_stream7("wq_stream7_32p");
    hls::stream<ap_uint<512> > wq_stream8("wq_stream8_32p"); hls::stream<ap_uint<512> > wq_stream9("wq_stream9_32p");
    hls::stream<ap_uint<512> > wq_stream10("wq_stream10_32p"); hls::stream<ap_uint<512> > wq_stream11("wq_stream11_32p");
    hls::stream<ap_uint<512> > wq_stream12("wq_stream12_32p"); hls::stream<ap_uint<512> > wq_stream13("wq_stream13_32p");
    hls::stream<ap_uint<512> > wq_stream14("wq_stream14_32p"); hls::stream<ap_uint<512> > wq_stream15("wq_stream15_32p");
    hls::stream<ap_uint<512> > wq_stream16("wq_stream16_32p"); hls::stream<ap_uint<512> > wq_stream17("wq_stream17_32p");
    hls::stream<ap_uint<512> > wq_stream18("wq_stream18_32p"); hls::stream<ap_uint<512> > wq_stream19("wq_stream19_32p");
    hls::stream<ap_uint<512> > wq_stream20("wq_stream20_32p"); hls::stream<ap_uint<512> > wq_stream21("wq_stream21_32p");
    hls::stream<ap_uint<512> > wq_stream22("wq_stream22_32p"); hls::stream<ap_uint<512> > wq_stream23("wq_stream23_32p");
    hls::stream<ap_uint<512> > wq_stream24("wq_stream24_32p"); hls::stream<ap_uint<512> > wq_stream25("wq_stream25_32p");
    hls::stream<ap_uint<512> > wq_stream26("wq_stream26_32p"); hls::stream<ap_uint<512> > wq_stream27("wq_stream27_32p");
    hls::stream<ap_uint<512> > wq_stream28("wq_stream28_32p"); hls::stream<ap_uint<512> > wq_stream29("wq_stream29_32p");
    hls::stream<ap_uint<512> > wq_stream30("wq_stream30_32p"); hls::stream<ap_uint<512> > wq_stream31("wq_stream31_32p");

    hls::stream<int8_t> sg_stream0("sg_stream0_32p"); hls::stream<int8_t> sg_stream1("sg_stream1_32p");
    hls::stream<int8_t> sg_stream2("sg_stream2_32p"); hls::stream<int8_t> sg_stream3("sg_stream3_32p");
    hls::stream<int8_t> sg_stream4("sg_stream4_32p"); hls::stream<int8_t> sg_stream5("sg_stream5_32p");
    hls::stream<int8_t> sg_stream6("sg_stream6_32p"); hls::stream<int8_t> sg_stream7("sg_stream7_32p");
    hls::stream<int8_t> sg_stream8("sg_stream8_32p"); hls::stream<int8_t> sg_stream9("sg_stream9_32p");
    hls::stream<int8_t> sg_stream10("sg_stream10_32p"); hls::stream<int8_t> sg_stream11("sg_stream11_32p");
    hls::stream<int8_t> sg_stream12("sg_stream12_32p"); hls::stream<int8_t> sg_stream13("sg_stream13_32p");
    hls::stream<int8_t> sg_stream14("sg_stream14_32p"); hls::stream<int8_t> sg_stream15("sg_stream15_32p");
    hls::stream<int8_t> sg_stream16("sg_stream16_32p"); hls::stream<int8_t> sg_stream17("sg_stream17_32p");
    hls::stream<int8_t> sg_stream18("sg_stream18_32p"); hls::stream<int8_t> sg_stream19("sg_stream19_32p");
    hls::stream<int8_t> sg_stream20("sg_stream20_32p"); hls::stream<int8_t> sg_stream21("sg_stream21_32p");
    hls::stream<int8_t> sg_stream22("sg_stream22_32p"); hls::stream<int8_t> sg_stream23("sg_stream23_32p");
    hls::stream<int8_t> sg_stream24("sg_stream24_32p"); hls::stream<int8_t> sg_stream25("sg_stream25_32p");
    hls::stream<int8_t> sg_stream26("sg_stream26_32p"); hls::stream<int8_t> sg_stream27("sg_stream27_32p");
    hls::stream<int8_t> sg_stream28("sg_stream28_32p"); hls::stream<int8_t> sg_stream29("sg_stream29_32p");
    hls::stream<int8_t> sg_stream30("sg_stream30_32p"); hls::stream<int8_t> sg_stream31("sg_stream31_32p");

    hls::stream<float> sc_stream0("sc_stream0_32p"); hls::stream<float> sc_stream1("sc_stream1_32p");
    hls::stream<float> sc_stream2("sc_stream2_32p"); hls::stream<float> sc_stream3("sc_stream3_32p");
    hls::stream<float> sc_stream4("sc_stream4_32p"); hls::stream<float> sc_stream5("sc_stream5_32p");
    hls::stream<float> sc_stream6("sc_stream6_32p"); hls::stream<float> sc_stream7("sc_stream7_32p");
    hls::stream<float> sc_stream8("sc_stream8_32p"); hls::stream<float> sc_stream9("sc_stream9_32p");
    hls::stream<float> sc_stream10("sc_stream10_32p"); hls::stream<float> sc_stream11("sc_stream11_32p");
    hls::stream<float> sc_stream12("sc_stream12_32p"); hls::stream<float> sc_stream13("sc_stream13_32p");
    hls::stream<float> sc_stream14("sc_stream14_32p"); hls::stream<float> sc_stream15("sc_stream15_32p");
    hls::stream<float> sc_stream16("sc_stream16_32p"); hls::stream<float> sc_stream17("sc_stream17_32p");
    hls::stream<float> sc_stream18("sc_stream18_32p"); hls::stream<float> sc_stream19("sc_stream19_32p");
    hls::stream<float> sc_stream20("sc_stream20_32p"); hls::stream<float> sc_stream21("sc_stream21_32p");
    hls::stream<float> sc_stream22("sc_stream22_32p"); hls::stream<float> sc_stream23("sc_stream23_32p");
    hls::stream<float> sc_stream24("sc_stream24_32p"); hls::stream<float> sc_stream25("sc_stream25_32p");
    hls::stream<float> sc_stream26("sc_stream26_32p"); hls::stream<float> sc_stream27("sc_stream27_32p");
    hls::stream<float> sc_stream28("sc_stream28_32p"); hls::stream<float> sc_stream29("sc_stream29_32p");
    hls::stream<float> sc_stream30("sc_stream30_32p"); hls::stream<float> sc_stream31("sc_stream31_32p");
#pragma HLS STREAM variable=wq_stream0 depth=32
#pragma HLS STREAM variable=wq_stream1 depth=32
#pragma HLS STREAM variable=wq_stream2 depth=32
#pragma HLS STREAM variable=wq_stream3 depth=32
#pragma HLS STREAM variable=wq_stream4 depth=32
#pragma HLS STREAM variable=wq_stream5 depth=32
#pragma HLS STREAM variable=wq_stream6 depth=32
#pragma HLS STREAM variable=wq_stream7 depth=32
#pragma HLS STREAM variable=wq_stream8 depth=32
#pragma HLS STREAM variable=wq_stream9 depth=32
#pragma HLS STREAM variable=wq_stream10 depth=32
#pragma HLS STREAM variable=wq_stream11 depth=32
#pragma HLS STREAM variable=wq_stream12 depth=32
#pragma HLS STREAM variable=wq_stream13 depth=32
#pragma HLS STREAM variable=wq_stream14 depth=32
#pragma HLS STREAM variable=wq_stream15 depth=32
#pragma HLS STREAM variable=wq_stream16 depth=32
#pragma HLS STREAM variable=wq_stream17 depth=32
#pragma HLS STREAM variable=wq_stream18 depth=32
#pragma HLS STREAM variable=wq_stream19 depth=32
#pragma HLS STREAM variable=wq_stream20 depth=32
#pragma HLS STREAM variable=wq_stream21 depth=32
#pragma HLS STREAM variable=wq_stream22 depth=32
#pragma HLS STREAM variable=wq_stream23 depth=32
#pragma HLS STREAM variable=wq_stream24 depth=32
#pragma HLS STREAM variable=wq_stream25 depth=32
#pragma HLS STREAM variable=wq_stream26 depth=32
#pragma HLS STREAM variable=wq_stream27 depth=32
#pragma HLS STREAM variable=wq_stream28 depth=32
#pragma HLS STREAM variable=wq_stream29 depth=32
#pragma HLS STREAM variable=wq_stream30 depth=32
#pragma HLS STREAM variable=wq_stream31 depth=32
#pragma HLS STREAM variable=sg_stream0 depth=8
#pragma HLS STREAM variable=sg_stream1 depth=8
#pragma HLS STREAM variable=sg_stream2 depth=8
#pragma HLS STREAM variable=sg_stream3 depth=8
#pragma HLS STREAM variable=sg_stream4 depth=8
#pragma HLS STREAM variable=sg_stream5 depth=8
#pragma HLS STREAM variable=sg_stream6 depth=8
#pragma HLS STREAM variable=sg_stream7 depth=8
#pragma HLS STREAM variable=sg_stream8 depth=8
#pragma HLS STREAM variable=sg_stream9 depth=8
#pragma HLS STREAM variable=sg_stream10 depth=8
#pragma HLS STREAM variable=sg_stream11 depth=8
#pragma HLS STREAM variable=sg_stream12 depth=8
#pragma HLS STREAM variable=sg_stream13 depth=8
#pragma HLS STREAM variable=sg_stream14 depth=8
#pragma HLS STREAM variable=sg_stream15 depth=8
#pragma HLS STREAM variable=sg_stream16 depth=8
#pragma HLS STREAM variable=sg_stream17 depth=8
#pragma HLS STREAM variable=sg_stream18 depth=8
#pragma HLS STREAM variable=sg_stream19 depth=8
#pragma HLS STREAM variable=sg_stream20 depth=8
#pragma HLS STREAM variable=sg_stream21 depth=8
#pragma HLS STREAM variable=sg_stream22 depth=8
#pragma HLS STREAM variable=sg_stream23 depth=8
#pragma HLS STREAM variable=sg_stream24 depth=8
#pragma HLS STREAM variable=sg_stream25 depth=8
#pragma HLS STREAM variable=sg_stream26 depth=8
#pragma HLS STREAM variable=sg_stream27 depth=8
#pragma HLS STREAM variable=sg_stream28 depth=8
#pragma HLS STREAM variable=sg_stream29 depth=8
#pragma HLS STREAM variable=sg_stream30 depth=8
#pragma HLS STREAM variable=sg_stream31 depth=8
#pragma HLS STREAM variable=sc_stream0 depth=4
#pragma HLS STREAM variable=sc_stream1 depth=4
#pragma HLS STREAM variable=sc_stream2 depth=4
#pragma HLS STREAM variable=sc_stream3 depth=4
#pragma HLS STREAM variable=sc_stream4 depth=4
#pragma HLS STREAM variable=sc_stream5 depth=4
#pragma HLS STREAM variable=sc_stream6 depth=4
#pragma HLS STREAM variable=sc_stream7 depth=4
#pragma HLS STREAM variable=sc_stream8 depth=4
#pragma HLS STREAM variable=sc_stream9 depth=4
#pragma HLS STREAM variable=sc_stream10 depth=4
#pragma HLS STREAM variable=sc_stream11 depth=4
#pragma HLS STREAM variable=sc_stream12 depth=4
#pragma HLS STREAM variable=sc_stream13 depth=4
#pragma HLS STREAM variable=sc_stream14 depth=4
#pragma HLS STREAM variable=sc_stream15 depth=4
#pragma HLS STREAM variable=sc_stream16 depth=4
#pragma HLS STREAM variable=sc_stream17 depth=4
#pragma HLS STREAM variable=sc_stream18 depth=4
#pragma HLS STREAM variable=sc_stream19 depth=4
#pragma HLS STREAM variable=sc_stream20 depth=4
#pragma HLS STREAM variable=sc_stream21 depth=4
#pragma HLS STREAM variable=sc_stream22 depth=4
#pragma HLS STREAM variable=sc_stream23 depth=4
#pragma HLS STREAM variable=sc_stream24 depth=4
#pragma HLS STREAM variable=sc_stream25 depth=4
#pragma HLS STREAM variable=sc_stream26 depth=4
#pragma HLS STREAM variable=sc_stream27 depth=4
#pragma HLS STREAM variable=sc_stream28 depth=4
#pragma HLS STREAM variable=sc_stream29 depth=4
#pragma HLS STREAM variable=sc_stream30 depth=4
#pragma HLS STREAM variable=sc_stream31 depth=4
#pragma HLS DATAFLOW

    matmul_load_weights_w4a8(wq_base0, wq_base1, wq_base2, wq_base3, wq_base4, wq_base5, wq_base6, wq_base7,
                             wq_base8, wq_base9, wq_base10, wq_base11, wq_base12, wq_base13, wq_base14, wq_base15,
                             wq_base16, wq_base17, wq_base18, wq_base19, wq_base20, wq_base21, wq_base22, wq_base23,
                             wq_base24, wq_base25, wq_base26, wq_base27, wq_base28, wq_base29, wq_base30, wq_base31,
                             sg_base0, sg_base1, sg_base2, sg_base3, sg_base4, sg_base5, sg_base6, sg_base7,
                             sg_base8, sg_base9, sg_base10, sg_base11, sg_base12, sg_base13, sg_base14, sg_base15,
                             sg_base16, sg_base17, sg_base18, sg_base19, sg_base20, sg_base21, sg_base22, sg_base23,
                             sg_base24, sg_base25, sg_base26, sg_base27, sg_base28, sg_base29, sg_base30, sg_base31,
                             sc_base0, sc_base1, sc_base2, sc_base3, sc_base4, sc_base5, sc_base6, sc_base7,
                             sc_base8, sc_base9, sc_base10, sc_base11, sc_base12, sc_base13, sc_base14, sc_base15,
                             sc_base16, sc_base17, sc_base18, sc_base19, sc_base20, sc_base21, sc_base22, sc_base23,
                             sc_base24, sc_base25, sc_base26, sc_base27, sc_base28, sc_base29, sc_base30, sc_base31,
                             M, N, K,
                             wq_stream0, wq_stream1, wq_stream2, wq_stream3, wq_stream4, wq_stream5, wq_stream6, wq_stream7,
                             wq_stream8, wq_stream9, wq_stream10, wq_stream11, wq_stream12, wq_stream13, wq_stream14, wq_stream15,
                             wq_stream16, wq_stream17, wq_stream18, wq_stream19, wq_stream20, wq_stream21, wq_stream22, wq_stream23,
                             wq_stream24, wq_stream25, wq_stream26, wq_stream27, wq_stream28, wq_stream29, wq_stream30, wq_stream31,
                             sg_stream0, sg_stream1, sg_stream2, sg_stream3, sg_stream4, sg_stream5, sg_stream6, sg_stream7,
                             sg_stream8, sg_stream9, sg_stream10, sg_stream11, sg_stream12, sg_stream13, sg_stream14, sg_stream15,
                             sg_stream16, sg_stream17, sg_stream18, sg_stream19, sg_stream20, sg_stream21, sg_stream22, sg_stream23,
                             sg_stream24, sg_stream25, sg_stream26, sg_stream27, sg_stream28, sg_stream29, sg_stream30, sg_stream31,
                             sc_stream0, sc_stream1, sc_stream2, sc_stream3, sc_stream4, sc_stream5, sc_stream6, sc_stream7,
                             sc_stream8, sc_stream9, sc_stream10, sc_stream11, sc_stream12, sc_stream13, sc_stream14, sc_stream15,
                             sc_stream16, sc_stream17, sc_stream18, sc_stream19, sc_stream20, sc_stream21, sc_stream22, sc_stream23,
                             sc_stream24, sc_stream25, sc_stream26, sc_stream27, sc_stream28, sc_stream29, sc_stream30, sc_stream31);

    matmul_compute_outputs_w4a8(xout, xq, xs, M, N, K,
                                wq_stream0, wq_stream1, wq_stream2, wq_stream3, wq_stream4, wq_stream5, wq_stream6, wq_stream7,
                                wq_stream8, wq_stream9, wq_stream10, wq_stream11, wq_stream12, wq_stream13, wq_stream14, wq_stream15,
                                wq_stream16, wq_stream17, wq_stream18, wq_stream19, wq_stream20, wq_stream21, wq_stream22, wq_stream23,
                                wq_stream24, wq_stream25, wq_stream26, wq_stream27, wq_stream28, wq_stream29, wq_stream30, wq_stream31,
                                sg_stream0, sg_stream1, sg_stream2, sg_stream3, sg_stream4, sg_stream5, sg_stream6, sg_stream7,
                                sg_stream8, sg_stream9, sg_stream10, sg_stream11, sg_stream12, sg_stream13, sg_stream14, sg_stream15,
                                sg_stream16, sg_stream17, sg_stream18, sg_stream19, sg_stream20, sg_stream21, sg_stream22, sg_stream23,
                                sg_stream24, sg_stream25, sg_stream26, sg_stream27, sg_stream28, sg_stream29, sg_stream30, sg_stream31,
                                sc_stream0, sc_stream1, sc_stream2, sc_stream3, sc_stream4, sc_stream5, sc_stream6, sc_stream7,
                                sc_stream8, sc_stream9, sc_stream10, sc_stream11, sc_stream12, sc_stream13, sc_stream14, sc_stream15,
                                sc_stream16, sc_stream17, sc_stream18, sc_stream19, sc_stream20, sc_stream21, sc_stream22, sc_stream23,
                                sc_stream24, sc_stream25, sc_stream26, sc_stream27, sc_stream28, sc_stream29, sc_stream30, sc_stream31);
}


static void matmul_engine_q(
    float *xout,
    const int8_t *xq,
    const float *xs,
    const ShardWeights_t* w_p0, const ShardWeights_t* w_p1, const ShardWeights_t* w_p2, const ShardWeights_t* w_p3,
    const ShardWeights_t* w_p4, const ShardWeights_t* w_p5, const ShardWeights_t* w_p6, const ShardWeights_t* w_p7,
    const ShardWeights_t* w_p8, const ShardWeights_t* w_p9, const ShardWeights_t* w_p10, const ShardWeights_t* w_p11,
    const ShardWeights_t* w_p12, const ShardWeights_t* w_p13, const ShardWeights_t* w_p14, const ShardWeights_t* w_p15,
    const ShardWeights_t* w_p16, const ShardWeights_t* w_p17, const ShardWeights_t* w_p18, const ShardWeights_t* w_p19,
    const ShardWeights_t* w_p20, const ShardWeights_t* w_p21, const ShardWeights_t* w_p22, const ShardWeights_t* w_p23,
    const ShardWeights_t* w_p24, const ShardWeights_t* w_p25, const ShardWeights_t* w_p26, const ShardWeights_t* w_p27,
    const ShardWeights_t* w_p28, const ShardWeights_t* w_p29, const ShardWeights_t* w_p30, const ShardWeights_t* w_p31,
    int layer, int M, int N, int K) {
    #pragma HLS INLINE

    const int8_t *wq_base0 = (w_p0->wq + layer)->q; const int8_t *wq_base1 = (w_p1->wq + layer)->q;
    const int8_t *wq_base2 = (w_p2->wq + layer)->q; const int8_t *wq_base3 = (w_p3->wq + layer)->q;
    const int8_t *wq_base4 = (w_p4->wq + layer)->q; const int8_t *wq_base5 = (w_p5->wq + layer)->q;
    const int8_t *wq_base6 = (w_p6->wq + layer)->q; const int8_t *wq_base7 = (w_p7->wq + layer)->q;
    const int8_t *wq_base8 = (w_p8->wq + layer)->q; const int8_t *wq_base9 = (w_p9->wq + layer)->q;
    const int8_t *wq_base10 = (w_p10->wq + layer)->q; const int8_t *wq_base11 = (w_p11->wq + layer)->q;
    const int8_t *wq_base12 = (w_p12->wq + layer)->q; const int8_t *wq_base13 = (w_p13->wq + layer)->q;
    const int8_t *wq_base14 = (w_p14->wq + layer)->q; const int8_t *wq_base15 = (w_p15->wq + layer)->q;
    const int8_t *wq_base16 = (w_p16->wq + layer)->q; const int8_t *wq_base17 = (w_p17->wq + layer)->q;
    const int8_t *wq_base18 = (w_p18->wq + layer)->q; const int8_t *wq_base19 = (w_p19->wq + layer)->q;
    const int8_t *wq_base20 = (w_p20->wq + layer)->q; const int8_t *wq_base21 = (w_p21->wq + layer)->q;
    const int8_t *wq_base22 = (w_p22->wq + layer)->q; const int8_t *wq_base23 = (w_p23->wq + layer)->q;
    const int8_t *wq_base24 = (w_p24->wq + layer)->q; const int8_t *wq_base25 = (w_p25->wq + layer)->q;
    const int8_t *wq_base26 = (w_p26->wq + layer)->q; const int8_t *wq_base27 = (w_p27->wq + layer)->q;
    const int8_t *wq_base28 = (w_p28->wq + layer)->q; const int8_t *wq_base29 = (w_p29->wq + layer)->q;
    const int8_t *wq_base30 = (w_p30->wq + layer)->q; const int8_t *wq_base31 = (w_p31->wq + layer)->q;
    
    const int8_t *sg_base0 = (w_p0->wq + layer)->sg; const int8_t *sg_base1 = (w_p1->wq + layer)->sg;
    const int8_t *sg_base2 = (w_p2->wq + layer)->sg; const int8_t *sg_base3 = (w_p3->wq + layer)->sg;
    const int8_t *sg_base4 = (w_p4->wq + layer)->sg; const int8_t *sg_base5 = (w_p5->wq + layer)->sg;
    const int8_t *sg_base6 = (w_p6->wq + layer)->sg; const int8_t *sg_base7 = (w_p7->wq + layer)->sg;
    const int8_t *sg_base8 = (w_p8->wq + layer)->sg; const int8_t *sg_base9 = (w_p9->wq + layer)->sg;
    const int8_t *sg_base10 = (w_p10->wq + layer)->sg; const int8_t *sg_base11 = (w_p11->wq + layer)->sg;
    const int8_t *sg_base12 = (w_p12->wq + layer)->sg; const int8_t *sg_base13 = (w_p13->wq + layer)->sg;
    const int8_t *sg_base14 = (w_p14->wq + layer)->sg; const int8_t *sg_base15 = (w_p15->wq + layer)->sg;
    const int8_t *sg_base16 = (w_p16->wq + layer)->sg; const int8_t *sg_base17 = (w_p17->wq + layer)->sg;
    const int8_t *sg_base18 = (w_p18->wq + layer)->sg; const int8_t *sg_base19 = (w_p19->wq + layer)->sg;
    const int8_t *sg_base20 = (w_p20->wq + layer)->sg; const int8_t *sg_base21 = (w_p21->wq + layer)->sg;
    const int8_t *sg_base22 = (w_p22->wq + layer)->sg; const int8_t *sg_base23 = (w_p23->wq + layer)->sg;
    const int8_t *sg_base24 = (w_p24->wq + layer)->sg; const int8_t *sg_base25 = (w_p25->wq + layer)->sg;
    const int8_t *sg_base26 = (w_p26->wq + layer)->sg; const int8_t *sg_base27 = (w_p27->wq + layer)->sg;
    const int8_t *sg_base28 = (w_p28->wq + layer)->sg; const int8_t *sg_base29 = (w_p29->wq + layer)->sg;
    const int8_t *sg_base30 = (w_p30->wq + layer)->sg; const int8_t *sg_base31 = (w_p31->wq + layer)->sg;
    
    const float  *sc_base0 = (w_p0->wq + layer)->sc; const float  *sc_base1 = (w_p1->wq + layer)->sc;
    const float  *sc_base2 = (w_p2->wq + layer)->sc; const float  *sc_base3 = (w_p3->wq + layer)->sc;
    const float  *sc_base4 = (w_p4->wq + layer)->sc; const float  *sc_base5 = (w_p5->wq + layer)->sc;
    const float  *sc_base6 = (w_p6->wq + layer)->sc; const float  *sc_base7 = (w_p7->wq + layer)->sc;
    const float  *sc_base8 = (w_p8->wq + layer)->sc; const float  *sc_base9 = (w_p9->wq + layer)->sc;
    const float  *sc_base10 = (w_p10->wq + layer)->sc; const float  *sc_base11 = (w_p11->wq + layer)->sc;
    const float  *sc_base12 = (w_p12->wq + layer)->sc; const float  *sc_base13 = (w_p13->wq + layer)->sc;
    const float  *sc_base14 = (w_p14->wq + layer)->sc; const float  *sc_base15 = (w_p15->wq + layer)->sc;
    const float  *sc_base16 = (w_p16->wq + layer)->sc; const float  *sc_base17 = (w_p17->wq + layer)->sc;
    const float  *sc_base18 = (w_p18->wq + layer)->sc; const float  *sc_base19 = (w_p19->wq + layer)->sc;
    const float  *sc_base20 = (w_p20->wq + layer)->sc; const float  *sc_base21 = (w_p21->wq + layer)->sc;
    const float  *sc_base22 = (w_p22->wq + layer)->sc; const float  *sc_base23 = (w_p23->wq + layer)->sc;
    const float  *sc_base24 = (w_p24->wq + layer)->sc; const float  *sc_base25 = (w_p25->wq + layer)->sc;
    const float  *sc_base26 = (w_p26->wq + layer)->sc; const float  *sc_base27 = (w_p27->wq + layer)->sc;
    const float  *sc_base28 = (w_p28->wq + layer)->sc; const float  *sc_base29 = (w_p29->wq + layer)->sc;
    const float  *sc_base30 = (w_p30->wq + layer)->sc; const float  *sc_base31 = (w_p31->wq + layer)->sc;

    matmul_engine_w4a8_32port(xout, xq, xs,
                             wq_base0, wq_base1, wq_base2, wq_base3, wq_base4, wq_base5, wq_base6, wq_base7,
                             wq_base8, wq_base9, wq_base10, wq_base11, wq_base12, wq_base13, wq_base14, wq_base15,
                             wq_base16, wq_base17, wq_base18, wq_base19, wq_base20, wq_base21, wq_base22, wq_base23,
                             wq_base24, wq_base25, wq_base26, wq_base27, wq_base28, wq_base29, wq_base30, wq_base31,
                             sg_base0, sg_base1, sg_base2, sg_base3, sg_base4, sg_base5, sg_base6, sg_base7,
                             sg_base8, sg_base9, sg_base10, sg_base11, sg_base12, sg_base13, sg_base14, sg_base15,
                             sg_base16, sg_base17, sg_base18, sg_base19, sg_base20, sg_base21, sg_base22, sg_base23,
                             sg_base24, sg_base25, sg_base26, sg_base27, sg_base28, sg_base29, sg_base30, sg_base31,
                             sc_base0, sc_base1, sc_base2, sc_base3, sc_base4, sc_base5, sc_base6, sc_base7,
                             sc_base8, sc_base9, sc_base10, sc_base11, sc_base12, sc_base13, sc_base14, sc_base15,
                             sc_base16, sc_base17, sc_base18, sc_base19, sc_base20, sc_base21, sc_base22, sc_base23,
                             sc_base24, sc_base25, sc_base26, sc_base27, sc_base28, sc_base29, sc_base30, sc_base31,
                             M, N, K);
}

static void matmul_engine_k(
    float *xout,
    const int8_t *xq,
    const float *xs,
    const ShardWeights_t* w_p0, const ShardWeights_t* w_p1, const ShardWeights_t* w_p2, const ShardWeights_t* w_p3,
    const ShardWeights_t* w_p4, const ShardWeights_t* w_p5, const ShardWeights_t* w_p6, const ShardWeights_t* w_p7,
    const ShardWeights_t* w_p8, const ShardWeights_t* w_p9, const ShardWeights_t* w_p10, const ShardWeights_t* w_p11,
    const ShardWeights_t* w_p12, const ShardWeights_t* w_p13, const ShardWeights_t* w_p14, const ShardWeights_t* w_p15,
    const ShardWeights_t* w_p16, const ShardWeights_t* w_p17, const ShardWeights_t* w_p18, const ShardWeights_t* w_p19,
    const ShardWeights_t* w_p20, const ShardWeights_t* w_p21, const ShardWeights_t* w_p22, const ShardWeights_t* w_p23,
    const ShardWeights_t* w_p24, const ShardWeights_t* w_p25, const ShardWeights_t* w_p26, const ShardWeights_t* w_p27,
    const ShardWeights_t* w_p28, const ShardWeights_t* w_p29, const ShardWeights_t* w_p30, const ShardWeights_t* w_p31,
    int layer, int M, int N, int K) {
    #pragma HLS INLINE

    const int8_t *wq_base0 = (w_p0->wk + layer)->q; const int8_t *wq_base1 = (w_p1->wk + layer)->q;
    const int8_t *wq_base2 = (w_p2->wk + layer)->q; const int8_t *wq_base3 = (w_p3->wk + layer)->q;
    const int8_t *wq_base4 = (w_p4->wk + layer)->q; const int8_t *wq_base5 = (w_p5->wk + layer)->q;
    const int8_t *wq_base6 = (w_p6->wk + layer)->q; const int8_t *wq_base7 = (w_p7->wk + layer)->q;
    const int8_t *wq_base8 = (w_p8->wk + layer)->q; const int8_t *wq_base9 = (w_p9->wk + layer)->q;
    const int8_t *wq_base10 = (w_p10->wk + layer)->q; const int8_t *wq_base11 = (w_p11->wk + layer)->q;
    const int8_t *wq_base12 = (w_p12->wk + layer)->q; const int8_t *wq_base13 = (w_p13->wk + layer)->q;
    const int8_t *wq_base14 = (w_p14->wk + layer)->q; const int8_t *wq_base15 = (w_p15->wk + layer)->q;
    const int8_t *wq_base16 = (w_p16->wk + layer)->q; const int8_t *wq_base17 = (w_p17->wk + layer)->q;
    const int8_t *wq_base18 = (w_p18->wk + layer)->q; const int8_t *wq_base19 = (w_p19->wk + layer)->q;
    const int8_t *wq_base20 = (w_p20->wk + layer)->q; const int8_t *wq_base21 = (w_p21->wk + layer)->q;
    const int8_t *wq_base22 = (w_p22->wk + layer)->q; const int8_t *wq_base23 = (w_p23->wk + layer)->q;
    const int8_t *wq_base24 = (w_p24->wk + layer)->q; const int8_t *wq_base25 = (w_p25->wk + layer)->q;
    const int8_t *wq_base26 = (w_p26->wk + layer)->q; const int8_t *wq_base27 = (w_p27->wk + layer)->q;
    const int8_t *wq_base28 = (w_p28->wk + layer)->q; const int8_t *wq_base29 = (w_p29->wk + layer)->q;
    const int8_t *wq_base30 = (w_p30->wk + layer)->q; const int8_t *wq_base31 = (w_p31->wk + layer)->q;
    
    const int8_t *sg_base0 = (w_p0->wk + layer)->sg; const int8_t *sg_base1 = (w_p1->wk + layer)->sg;
    const int8_t *sg_base2 = (w_p2->wk + layer)->sg; const int8_t *sg_base3 = (w_p3->wk + layer)->sg;
    const int8_t *sg_base4 = (w_p4->wk + layer)->sg; const int8_t *sg_base5 = (w_p5->wk + layer)->sg;
    const int8_t *sg_base6 = (w_p6->wk + layer)->sg; const int8_t *sg_base7 = (w_p7->wk + layer)->sg;
    const int8_t *sg_base8 = (w_p8->wk + layer)->sg; const int8_t *sg_base9 = (w_p9->wk + layer)->sg;
    const int8_t *sg_base10 = (w_p10->wk + layer)->sg; const int8_t *sg_base11 = (w_p11->wk + layer)->sg;
    const int8_t *sg_base12 = (w_p12->wk + layer)->sg; const int8_t *sg_base13 = (w_p13->wk + layer)->sg;
    const int8_t *sg_base14 = (w_p14->wk + layer)->sg; const int8_t *sg_base15 = (w_p15->wk + layer)->sg;
    const int8_t *sg_base16 = (w_p16->wk + layer)->sg; const int8_t *sg_base17 = (w_p17->wk + layer)->sg;
    const int8_t *sg_base18 = (w_p18->wk + layer)->sg; const int8_t *sg_base19 = (w_p19->wk + layer)->sg;
    const int8_t *sg_base20 = (w_p20->wk + layer)->sg; const int8_t *sg_base21 = (w_p21->wk + layer)->sg;
    const int8_t *sg_base22 = (w_p22->wk + layer)->sg; const int8_t *sg_base23 = (w_p23->wk + layer)->sg;
    const int8_t *sg_base24 = (w_p24->wk + layer)->sg; const int8_t *sg_base25 = (w_p25->wk + layer)->sg;
    const int8_t *sg_base26 = (w_p26->wk + layer)->sg; const int8_t *sg_base27 = (w_p27->wk + layer)->sg;
    const int8_t *sg_base28 = (w_p28->wk + layer)->sg; const int8_t *sg_base29 = (w_p29->wk + layer)->sg;
    const int8_t *sg_base30 = (w_p30->wk + layer)->sg; const int8_t *sg_base31 = (w_p31->wk + layer)->sg;
    
    const float  *sc_base0 = (w_p0->wk + layer)->sc; const float  *sc_base1 = (w_p1->wk + layer)->sc;
    const float  *sc_base2 = (w_p2->wk + layer)->sc; const float  *sc_base3 = (w_p3->wk + layer)->sc;
    const float  *sc_base4 = (w_p4->wk + layer)->sc; const float  *sc_base5 = (w_p5->wk + layer)->sc;
    const float  *sc_base6 = (w_p6->wk + layer)->sc; const float  *sc_base7 = (w_p7->wk + layer)->sc;
    const float  *sc_base8 = (w_p8->wk + layer)->sc; const float  *sc_base9 = (w_p9->wk + layer)->sc;
    const float  *sc_base10 = (w_p10->wk + layer)->sc; const float  *sc_base11 = (w_p11->wk + layer)->sc;
    const float  *sc_base12 = (w_p12->wk + layer)->sc; const float  *sc_base13 = (w_p13->wk + layer)->sc;
    const float  *sc_base14 = (w_p14->wk + layer)->sc; const float  *sc_base15 = (w_p15->wk + layer)->sc;
    const float  *sc_base16 = (w_p16->wk + layer)->sc; const float  *sc_base17 = (w_p17->wk + layer)->sc;
    const float  *sc_base18 = (w_p18->wk + layer)->sc; const float  *sc_base19 = (w_p19->wk + layer)->sc;
    const float  *sc_base20 = (w_p20->wk + layer)->sc; const float  *sc_base21 = (w_p21->wk + layer)->sc;
    const float  *sc_base22 = (w_p22->wk + layer)->sc; const float  *sc_base23 = (w_p23->wk + layer)->sc;
    const float  *sc_base24 = (w_p24->wk + layer)->sc; const float  *sc_base25 = (w_p25->wk + layer)->sc;
    const float  *sc_base26 = (w_p26->wk + layer)->sc; const float  *sc_base27 = (w_p27->wk + layer)->sc;
    const float  *sc_base28 = (w_p28->wk + layer)->sc; const float  *sc_base29 = (w_p29->wk + layer)->sc;
    const float  *sc_base30 = (w_p30->wk + layer)->sc; const float  *sc_base31 = (w_p31->wk + layer)->sc;

    matmul_engine_w4a8_32port(xout, xq, xs,
                             wq_base0, wq_base1, wq_base2, wq_base3, wq_base4, wq_base5, wq_base6, wq_base7,
                             wq_base8, wq_base9, wq_base10, wq_base11, wq_base12, wq_base13, wq_base14, wq_base15,
                             wq_base16, wq_base17, wq_base18, wq_base19, wq_base20, wq_base21, wq_base22, wq_base23,
                             wq_base24, wq_base25, wq_base26, wq_base27, wq_base28, wq_base29, wq_base30, wq_base31,
                             sg_base0, sg_base1, sg_base2, sg_base3, sg_base4, sg_base5, sg_base6, sg_base7,
                             sg_base8, sg_base9, sg_base10, sg_base11, sg_base12, sg_base13, sg_base14, sg_base15,
                             sg_base16, sg_base17, sg_base18, sg_base19, sg_base20, sg_base21, sg_base22, sg_base23,
                             sg_base24, sg_base25, sg_base26, sg_base27, sg_base28, sg_base29, sg_base30, sg_base31,
                             sc_base0, sc_base1, sc_base2, sc_base3, sc_base4, sc_base5, sc_base6, sc_base7,
                             sc_base8, sc_base9, sc_base10, sc_base11, sc_base12, sc_base13, sc_base14, sc_base15,
                             sc_base16, sc_base17, sc_base18, sc_base19, sc_base20, sc_base21, sc_base22, sc_base23,
                             sc_base24, sc_base25, sc_base26, sc_base27, sc_base28, sc_base29, sc_base30, sc_base31,
                             M, N, K);
}

static void matmul_engine_v(
    float *xout,
    const int8_t *xq,
    const float *xs,
    const ShardWeights_t* w_p0, const ShardWeights_t* w_p1, const ShardWeights_t* w_p2, const ShardWeights_t* w_p3,
    const ShardWeights_t* w_p4, const ShardWeights_t* w_p5, const ShardWeights_t* w_p6, const ShardWeights_t* w_p7,
    const ShardWeights_t* w_p8, const ShardWeights_t* w_p9, const ShardWeights_t* w_p10, const ShardWeights_t* w_p11,
    const ShardWeights_t* w_p12, const ShardWeights_t* w_p13, const ShardWeights_t* w_p14, const ShardWeights_t* w_p15,
    const ShardWeights_t* w_p16, const ShardWeights_t* w_p17, const ShardWeights_t* w_p18, const ShardWeights_t* w_p19,
    const ShardWeights_t* w_p20, const ShardWeights_t* w_p21, const ShardWeights_t* w_p22, const ShardWeights_t* w_p23,
    const ShardWeights_t* w_p24, const ShardWeights_t* w_p25, const ShardWeights_t* w_p26, const ShardWeights_t* w_p27,
    const ShardWeights_t* w_p28, const ShardWeights_t* w_p29, const ShardWeights_t* w_p30, const ShardWeights_t* w_p31,
    int layer, int M, int N, int K) {
    #pragma HLS INLINE

    const int8_t *wq_base0 = (w_p0->wv + layer)->q; const int8_t *wq_base1 = (w_p1->wv + layer)->q;
    const int8_t *wq_base2 = (w_p2->wv + layer)->q; const int8_t *wq_base3 = (w_p3->wv + layer)->q;
    const int8_t *wq_base4 = (w_p4->wv + layer)->q; const int8_t *wq_base5 = (w_p5->wv + layer)->q;
    const int8_t *wq_base6 = (w_p6->wv + layer)->q; const int8_t *wq_base7 = (w_p7->wv + layer)->q;
    const int8_t *wq_base8 = (w_p8->wv + layer)->q; const int8_t *wq_base9 = (w_p9->wv + layer)->q;
    const int8_t *wq_base10 = (w_p10->wv + layer)->q; const int8_t *wq_base11 = (w_p11->wv + layer)->q;
    const int8_t *wq_base12 = (w_p12->wv + layer)->q; const int8_t *wq_base13 = (w_p13->wv + layer)->q;
    const int8_t *wq_base14 = (w_p14->wv + layer)->q; const int8_t *wq_base15 = (w_p15->wv + layer)->q;
    const int8_t *wq_base16 = (w_p16->wv + layer)->q; const int8_t *wq_base17 = (w_p17->wv + layer)->q;
    const int8_t *wq_base18 = (w_p18->wv + layer)->q; const int8_t *wq_base19 = (w_p19->wv + layer)->q;
    const int8_t *wq_base20 = (w_p20->wv + layer)->q; const int8_t *wq_base21 = (w_p21->wv + layer)->q;
    const int8_t *wq_base22 = (w_p22->wv + layer)->q; const int8_t *wq_base23 = (w_p23->wv + layer)->q;
    const int8_t *wq_base24 = (w_p24->wv + layer)->q; const int8_t *wq_base25 = (w_p25->wv + layer)->q;
    const int8_t *wq_base26 = (w_p26->wv + layer)->q; const int8_t *wq_base27 = (w_p27->wv + layer)->q;
    const int8_t *wq_base28 = (w_p28->wv + layer)->q; const int8_t *wq_base29 = (w_p29->wv + layer)->q;
    const int8_t *wq_base30 = (w_p30->wv + layer)->q; const int8_t *wq_base31 = (w_p31->wv + layer)->q;
    
    const int8_t *sg_base0 = (w_p0->wv + layer)->sg; const int8_t *sg_base1 = (w_p1->wv + layer)->sg;
    const int8_t *sg_base2 = (w_p2->wv + layer)->sg; const int8_t *sg_base3 = (w_p3->wv + layer)->sg;
    const int8_t *sg_base4 = (w_p4->wv + layer)->sg; const int8_t *sg_base5 = (w_p5->wv + layer)->sg;
    const int8_t *sg_base6 = (w_p6->wv + layer)->sg; const int8_t *sg_base7 = (w_p7->wv + layer)->sg;
    const int8_t *sg_base8 = (w_p8->wv + layer)->sg; const int8_t *sg_base9 = (w_p9->wv + layer)->sg;
    const int8_t *sg_base10 = (w_p10->wv + layer)->sg; const int8_t *sg_base11 = (w_p11->wv + layer)->sg;
    const int8_t *sg_base12 = (w_p12->wv + layer)->sg; const int8_t *sg_base13 = (w_p13->wv + layer)->sg;
    const int8_t *sg_base14 = (w_p14->wv + layer)->sg; const int8_t *sg_base15 = (w_p15->wv + layer)->sg;
    const int8_t *sg_base16 = (w_p16->wv + layer)->sg; const int8_t *sg_base17 = (w_p17->wv + layer)->sg;
    const int8_t *sg_base18 = (w_p18->wv + layer)->sg; const int8_t *sg_base19 = (w_p19->wv + layer)->sg;
    const int8_t *sg_base20 = (w_p20->wv + layer)->sg; const int8_t *sg_base21 = (w_p21->wv + layer)->sg;
    const int8_t *sg_base22 = (w_p22->wv + layer)->sg; const int8_t *sg_base23 = (w_p23->wv + layer)->sg;
    const int8_t *sg_base24 = (w_p24->wv + layer)->sg; const int8_t *sg_base25 = (w_p25->wv + layer)->sg;
    const int8_t *sg_base26 = (w_p26->wv + layer)->sg; const int8_t *sg_base27 = (w_p27->wv + layer)->sg;
    const int8_t *sg_base28 = (w_p28->wv + layer)->sg; const int8_t *sg_base29 = (w_p29->wv + layer)->sg;
    const int8_t *sg_base30 = (w_p30->wv + layer)->sg; const int8_t *sg_base31 = (w_p31->wv + layer)->sg;
    
    const float  *sc_base0 = (w_p0->wv + layer)->sc; const float  *sc_base1 = (w_p1->wv + layer)->sc;
    const float  *sc_base2 = (w_p2->wv + layer)->sc; const float  *sc_base3 = (w_p3->wv + layer)->sc;
    const float  *sc_base4 = (w_p4->wv + layer)->sc; const float  *sc_base5 = (w_p5->wv + layer)->sc;
    const float  *sc_base6 = (w_p6->wv + layer)->sc; const float  *sc_base7 = (w_p7->wv + layer)->sc;
    const float  *sc_base8 = (w_p8->wv + layer)->sc; const float  *sc_base9 = (w_p9->wv + layer)->sc;
    const float  *sc_base10 = (w_p10->wv + layer)->sc; const float  *sc_base11 = (w_p11->wv + layer)->sc;
    const float  *sc_base12 = (w_p12->wv + layer)->sc; const float  *sc_base13 = (w_p13->wv + layer)->sc;
    const float  *sc_base14 = (w_p14->wv + layer)->sc; const float  *sc_base15 = (w_p15->wv + layer)->sc;
    const float  *sc_base16 = (w_p16->wv + layer)->sc; const float  *sc_base17 = (w_p17->wv + layer)->sc;
    const float  *sc_base18 = (w_p18->wv + layer)->sc; const float  *sc_base19 = (w_p19->wv + layer)->sc;
    const float  *sc_base20 = (w_p20->wv + layer)->sc; const float  *sc_base21 = (w_p21->wv + layer)->sc;
    const float  *sc_base22 = (w_p22->wv + layer)->sc; const float  *sc_base23 = (w_p23->wv + layer)->sc;
    const float  *sc_base24 = (w_p24->wv + layer)->sc; const float  *sc_base25 = (w_p25->wv + layer)->sc;
    const float  *sc_base26 = (w_p26->wv + layer)->sc; const float  *sc_base27 = (w_p27->wv + layer)->sc;
    const float  *sc_base28 = (w_p28->wv + layer)->sc; const float  *sc_base29 = (w_p29->wv + layer)->sc;
    const float  *sc_base30 = (w_p30->wv + layer)->sc; const float  *sc_base31 = (w_p31->wv + layer)->sc;

    matmul_engine_w4a8_32port(xout, xq, xs,
                             wq_base0, wq_base1, wq_base2, wq_base3, wq_base4, wq_base5, wq_base6, wq_base7,
                             wq_base8, wq_base9, wq_base10, wq_base11, wq_base12, wq_base13, wq_base14, wq_base15,
                             wq_base16, wq_base17, wq_base18, wq_base19, wq_base20, wq_base21, wq_base22, wq_base23,
                             wq_base24, wq_base25, wq_base26, wq_base27, wq_base28, wq_base29, wq_base30, wq_base31,
                             sg_base0, sg_base1, sg_base2, sg_base3, sg_base4, sg_base5, sg_base6, sg_base7,
                             sg_base8, sg_base9, sg_base10, sg_base11, sg_base12, sg_base13, sg_base14, sg_base15,
                             sg_base16, sg_base17, sg_base18, sg_base19, sg_base20, sg_base21, sg_base22, sg_base23,
                             sg_base24, sg_base25, sg_base26, sg_base27, sg_base28, sg_base29, sg_base30, sg_base31,
                             sc_base0, sc_base1, sc_base2, sc_base3, sc_base4, sc_base5, sc_base6, sc_base7,
                             sc_base8, sc_base9, sc_base10, sc_base11, sc_base12, sc_base13, sc_base14, sc_base15,
                             sc_base16, sc_base17, sc_base18, sc_base19, sc_base20, sc_base21, sc_base22, sc_base23,
                             sc_base24, sc_base25, sc_base26, sc_base27, sc_base28, sc_base29, sc_base30, sc_base31,
                             M, N, K);
}

static void matmul_engine_o(
    float *xout,
    const int8_t *xq,
    const float *xs,
    const ShardWeights_t* w_p0, const ShardWeights_t* w_p1, const ShardWeights_t* w_p2, const ShardWeights_t* w_p3,
    const ShardWeights_t* w_p4, const ShardWeights_t* w_p5, const ShardWeights_t* w_p6, const ShardWeights_t* w_p7,
    const ShardWeights_t* w_p8, const ShardWeights_t* w_p9, const ShardWeights_t* w_p10, const ShardWeights_t* w_p11,
    const ShardWeights_t* w_p12, const ShardWeights_t* w_p13, const ShardWeights_t* w_p14, const ShardWeights_t* w_p15,
    const ShardWeights_t* w_p16, const ShardWeights_t* w_p17, const ShardWeights_t* w_p18, const ShardWeights_t* w_p19,
    const ShardWeights_t* w_p20, const ShardWeights_t* w_p21, const ShardWeights_t* w_p22, const ShardWeights_t* w_p23,
    const ShardWeights_t* w_p24, const ShardWeights_t* w_p25, const ShardWeights_t* w_p26, const ShardWeights_t* w_p27,
    const ShardWeights_t* w_p28, const ShardWeights_t* w_p29, const ShardWeights_t* w_p30, const ShardWeights_t* w_p31,
    int layer, int M, int N, int K) {
    #pragma HLS INLINE

    const int8_t *wq_base0 = (w_p0->wo + layer)->q; const int8_t *wq_base1 = (w_p1->wo + layer)->q;
    const int8_t *wq_base2 = (w_p2->wo + layer)->q; const int8_t *wq_base3 = (w_p3->wo + layer)->q;
    const int8_t *wq_base4 = (w_p4->wo + layer)->q; const int8_t *wq_base5 = (w_p5->wo + layer)->q;
    const int8_t *wq_base6 = (w_p6->wo + layer)->q; const int8_t *wq_base7 = (w_p7->wo + layer)->q;
    const int8_t *wq_base8 = (w_p8->wo + layer)->q; const int8_t *wq_base9 = (w_p9->wo + layer)->q;
    const int8_t *wq_base10 = (w_p10->wo + layer)->q; const int8_t *wq_base11 = (w_p11->wo + layer)->q;
    const int8_t *wq_base12 = (w_p12->wo + layer)->q; const int8_t *wq_base13 = (w_p13->wo + layer)->q;
    const int8_t *wq_base14 = (w_p14->wo + layer)->q; const int8_t *wq_base15 = (w_p15->wo + layer)->q;
    const int8_t *wq_base16 = (w_p16->wo + layer)->q; const int8_t *wq_base17 = (w_p17->wo + layer)->q;
    const int8_t *wq_base18 = (w_p18->wo + layer)->q; const int8_t *wq_base19 = (w_p19->wo + layer)->q;
    const int8_t *wq_base20 = (w_p20->wo + layer)->q; const int8_t *wq_base21 = (w_p21->wo + layer)->q;
    const int8_t *wq_base22 = (w_p22->wo + layer)->q; const int8_t *wq_base23 = (w_p23->wo + layer)->q;
    const int8_t *wq_base24 = (w_p24->wo + layer)->q; const int8_t *wq_base25 = (w_p25->wo + layer)->q;
    const int8_t *wq_base26 = (w_p26->wo + layer)->q; const int8_t *wq_base27 = (w_p27->wo + layer)->q;
    const int8_t *wq_base28 = (w_p28->wo + layer)->q; const int8_t *wq_base29 = (w_p29->wo + layer)->q;
    const int8_t *wq_base30 = (w_p30->wo + layer)->q; const int8_t *wq_base31 = (w_p31->wo + layer)->q;

    const int8_t *sg_base0 = (w_p0->wo + layer)->sg; const int8_t *sg_base1 = (w_p1->wo + layer)->sg;
    const int8_t *sg_base2 = (w_p2->wo + layer)->sg; const int8_t *sg_base3 = (w_p3->wo + layer)->sg;
    const int8_t *sg_base4 = (w_p4->wo + layer)->sg; const int8_t *sg_base5 = (w_p5->wo + layer)->sg;
    const int8_t *sg_base6 = (w_p6->wo + layer)->sg; const int8_t *sg_base7 = (w_p7->wo + layer)->sg;
    const int8_t *sg_base8 = (w_p8->wo + layer)->sg; const int8_t *sg_base9 = (w_p9->wo + layer)->sg;
    const int8_t *sg_base10 = (w_p10->wo + layer)->sg; const int8_t *sg_base11 = (w_p11->wo + layer)->sg;
    const int8_t *sg_base12 = (w_p12->wo + layer)->sg; const int8_t *sg_base13 = (w_p13->wo + layer)->sg;
    const int8_t *sg_base14 = (w_p14->wo + layer)->sg; const int8_t *sg_base15 = (w_p15->wo + layer)->sg;
    const int8_t *sg_base16 = (w_p16->wo + layer)->sg; const int8_t *sg_base17 = (w_p17->wo + layer)->sg;
    const int8_t *sg_base18 = (w_p18->wo + layer)->sg; const int8_t *sg_base19 = (w_p19->wo + layer)->sg;
    const int8_t *sg_base20 = (w_p20->wo + layer)->sg; const int8_t *sg_base21 = (w_p21->wo + layer)->sg;
    const int8_t *sg_base22 = (w_p22->wo + layer)->sg; const int8_t *sg_base23 = (w_p23->wo + layer)->sg;
    const int8_t *sg_base24 = (w_p24->wo + layer)->sg; const int8_t *sg_base25 = (w_p25->wo + layer)->sg;
    const int8_t *sg_base26 = (w_p26->wo + layer)->sg; const int8_t *sg_base27 = (w_p27->wo + layer)->sg;
    const int8_t *sg_base28 = (w_p28->wo + layer)->sg; const int8_t *sg_base29 = (w_p29->wo + layer)->sg;
    const int8_t *sg_base30 = (w_p30->wo + layer)->sg; const int8_t *sg_base31 = (w_p31->wo + layer)->sg;
    
    const float  *sc_base0 = (w_p0->wo + layer)->sc; const float  *sc_base1 = (w_p1->wo + layer)->sc;
    const float  *sc_base2 = (w_p2->wo + layer)->sc; const float  *sc_base3 = (w_p3->wo + layer)->sc;
    const float  *sc_base4 = (w_p4->wo + layer)->sc; const float  *sc_base5 = (w_p5->wo + layer)->sc;
    const float  *sc_base6 = (w_p6->wo + layer)->sc; const float  *sc_base7 = (w_p7->wo + layer)->sc;
    const float  *sc_base8 = (w_p8->wo + layer)->sc; const float  *sc_base9 = (w_p9->wo + layer)->sc;
    const float  *sc_base10 = (w_p10->wo + layer)->sc; const float  *sc_base11 = (w_p11->wo + layer)->sc;
    const float  *sc_base12 = (w_p12->wo + layer)->sc; const float  *sc_base13 = (w_p13->wo + layer)->sc;
    const float  *sc_base14 = (w_p14->wo + layer)->sc; const float  *sc_base15 = (w_p15->wo + layer)->sc;
    const float  *sc_base16 = (w_p16->wo + layer)->sc; const float  *sc_base17 = (w_p17->wo + layer)->sc;
    const float  *sc_base18 = (w_p18->wo + layer)->sc; const float  *sc_base19 = (w_p19->wo + layer)->sc;
    const float  *sc_base20 = (w_p20->wo + layer)->sc; const float  *sc_base21 = (w_p21->wo + layer)->sc;
    const float  *sc_base22 = (w_p22->wo + layer)->sc; const float  *sc_base23 = (w_p23->wo + layer)->sc;
    const float  *sc_base24 = (w_p24->wo + layer)->sc; const float  *sc_base25 = (w_p25->wo + layer)->sc;
    const float  *sc_base26 = (w_p26->wo + layer)->sc; const float  *sc_base27 = (w_p27->wo + layer)->sc;
    const float  *sc_base28 = (w_p28->wo + layer)->sc; const float  *sc_base29 = (w_p29->wo + layer)->sc;
    const float  *sc_base30 = (w_p30->wo + layer)->sc; const float  *sc_base31 = (w_p31->wo + layer)->sc;

    matmul_engine_w4a8_32port(xout, xq, xs,
                             wq_base0, wq_base1, wq_base2, wq_base3, wq_base4, wq_base5, wq_base6, wq_base7,
                             wq_base8, wq_base9, wq_base10, wq_base11, wq_base12, wq_base13, wq_base14, wq_base15,
                             wq_base16, wq_base17, wq_base18, wq_base19, wq_base20, wq_base21, wq_base22, wq_base23,
                             wq_base24, wq_base25, wq_base26, wq_base27, wq_base28, wq_base29, wq_base30, wq_base31,
                             sg_base0, sg_base1, sg_base2, sg_base3, sg_base4, sg_base5, sg_base6, sg_base7,
                             sg_base8, sg_base9, sg_base10, sg_base11, sg_base12, sg_base13, sg_base14, sg_base15,
                             sg_base16, sg_base17, sg_base18, sg_base19, sg_base20, sg_base21, sg_base22, sg_base23,
                             sg_base24, sg_base25, sg_base26, sg_base27, sg_base28, sg_base29, sg_base30, sg_base31,
                             sc_base0, sc_base1, sc_base2, sc_base3, sc_base4, sc_base5, sc_base6, sc_base7,
                             sc_base8, sc_base9, sc_base10, sc_base11, sc_base12, sc_base13, sc_base14, sc_base15,
                             sc_base16, sc_base17, sc_base18, sc_base19, sc_base20, sc_base21, sc_base22, sc_base23,
                             sc_base24, sc_base25, sc_base26, sc_base27, sc_base28, sc_base29, sc_base30, sc_base31,
                             M, N, K);
}

static void matmul_engine_w1(
    float *xout,
    const int8_t *xq,
    const float *xs,
    const ShardWeights_t* w_p0, const ShardWeights_t* w_p1, const ShardWeights_t* w_p2, const ShardWeights_t* w_p3,
    const ShardWeights_t* w_p4, const ShardWeights_t* w_p5, const ShardWeights_t* w_p6, const ShardWeights_t* w_p7,
    const ShardWeights_t* w_p8, const ShardWeights_t* w_p9, const ShardWeights_t* w_p10, const ShardWeights_t* w_p11,
    const ShardWeights_t* w_p12, const ShardWeights_t* w_p13, const ShardWeights_t* w_p14, const ShardWeights_t* w_p15,
    const ShardWeights_t* w_p16, const ShardWeights_t* w_p17, const ShardWeights_t* w_p18, const ShardWeights_t* w_p19,
    const ShardWeights_t* w_p20, const ShardWeights_t* w_p21, const ShardWeights_t* w_p22, const ShardWeights_t* w_p23,
    const ShardWeights_t* w_p24, const ShardWeights_t* w_p25, const ShardWeights_t* w_p26, const ShardWeights_t* w_p27,
    const ShardWeights_t* w_p28, const ShardWeights_t* w_p29, const ShardWeights_t* w_p30, const ShardWeights_t* w_p31,
    int layer, int M, int N, int K) {
    #pragma HLS INLINE

    const int8_t *wq_base0 = (w_p0->w1 + layer)->q; const int8_t *wq_base1 = (w_p1->w1 + layer)->q;
    const int8_t *wq_base2 = (w_p2->w1 + layer)->q; const int8_t *wq_base3 = (w_p3->w1 + layer)->q;
    const int8_t *wq_base4 = (w_p4->w1 + layer)->q; const int8_t *wq_base5 = (w_p5->w1 + layer)->q;
    const int8_t *wq_base6 = (w_p6->w1 + layer)->q; const int8_t *wq_base7 = (w_p7->w1 + layer)->q;
    const int8_t *wq_base8 = (w_p8->w1 + layer)->q; const int8_t *wq_base9 = (w_p9->w1 + layer)->q;
    const int8_t *wq_base10 = (w_p10->w1 + layer)->q; const int8_t *wq_base11 = (w_p11->w1 + layer)->q;
    const int8_t *wq_base12 = (w_p12->w1 + layer)->q; const int8_t *wq_base13 = (w_p13->w1 + layer)->q;
    const int8_t *wq_base14 = (w_p14->w1 + layer)->q; const int8_t *wq_base15 = (w_p15->w1 + layer)->q;
    const int8_t *wq_base16 = (w_p16->w1 + layer)->q; const int8_t *wq_base17 = (w_p17->w1 + layer)->q;
    const int8_t *wq_base18 = (w_p18->w1 + layer)->q; const int8_t *wq_base19 = (w_p19->w1 + layer)->q;
    const int8_t *wq_base20 = (w_p20->w1 + layer)->q; const int8_t *wq_base21 = (w_p21->w1 + layer)->q;
    const int8_t *wq_base22 = (w_p22->w1 + layer)->q; const int8_t *wq_base23 = (w_p23->w1 + layer)->q;
    const int8_t *wq_base24 = (w_p24->w1 + layer)->q; const int8_t *wq_base25 = (w_p25->w1 + layer)->q;
    const int8_t *wq_base26 = (w_p26->w1 + layer)->q; const int8_t *wq_base27 = (w_p27->w1 + layer)->q;
    const int8_t *wq_base28 = (w_p28->w1 + layer)->q; const int8_t *wq_base29 = (w_p29->w1 + layer)->q;
    const int8_t *wq_base30 = (w_p30->w1 + layer)->q; const int8_t *wq_base31 = (w_p31->w1 + layer)->q;

    const int8_t *sg_base0 = (w_p0->w1 + layer)->sg; const int8_t *sg_base1 = (w_p1->w1 + layer)->sg;
    const int8_t *sg_base2 = (w_p2->w1 + layer)->sg; const int8_t *sg_base3 = (w_p3->w1 + layer)->sg;
    const int8_t *sg_base4 = (w_p4->w1 + layer)->sg; const int8_t *sg_base5 = (w_p5->w1 + layer)->sg;
    const int8_t *sg_base6 = (w_p6->w1 + layer)->sg; const int8_t *sg_base7 = (w_p7->w1 + layer)->sg;
    const int8_t *sg_base8 = (w_p8->w1 + layer)->sg; const int8_t *sg_base9 = (w_p9->w1 + layer)->sg;
    const int8_t *sg_base10 = (w_p10->w1 + layer)->sg; const int8_t *sg_base11 = (w_p11->w1 + layer)->sg;
    const int8_t *sg_base12 = (w_p12->w1 + layer)->sg; const int8_t *sg_base13 = (w_p13->w1 + layer)->sg;
    const int8_t *sg_base14 = (w_p14->w1 + layer)->sg; const int8_t *sg_base15 = (w_p15->w1 + layer)->sg;
    const int8_t *sg_base16 = (w_p16->w1 + layer)->sg; const int8_t *sg_base17 = (w_p17->w1 + layer)->sg;
    const int8_t *sg_base18 = (w_p18->w1 + layer)->sg; const int8_t *sg_base19 = (w_p19->w1 + layer)->sg;
    const int8_t *sg_base20 = (w_p20->w1 + layer)->sg; const int8_t *sg_base21 = (w_p21->w1 + layer)->sg;
    const int8_t *sg_base22 = (w_p22->w1 + layer)->sg; const int8_t *sg_base23 = (w_p23->w1 + layer)->sg;
    const int8_t *sg_base24 = (w_p24->w1 + layer)->sg; const int8_t *sg_base25 = (w_p25->w1 + layer)->sg;
    const int8_t *sg_base26 = (w_p26->w1 + layer)->sg; const int8_t *sg_base27 = (w_p27->w1 + layer)->sg;
    const int8_t *sg_base28 = (w_p28->w1 + layer)->sg; const int8_t *sg_base29 = (w_p29->w1 + layer)->sg;
    const int8_t *sg_base30 = (w_p30->w1 + layer)->sg; const int8_t *sg_base31 = (w_p31->w1 + layer)->sg;
    
    const float  *sc_base0 = (w_p0->w1 + layer)->sc; const float  *sc_base1 = (w_p1->w1 + layer)->sc;
    const float  *sc_base2 = (w_p2->w1 + layer)->sc; const float  *sc_base3 = (w_p3->w1 + layer)->sc;
    const float  *sc_base4 = (w_p4->w1 + layer)->sc; const float  *sc_base5 = (w_p5->w1 + layer)->sc;
    const float  *sc_base6 = (w_p6->w1 + layer)->sc; const float  *sc_base7 = (w_p7->w1 + layer)->sc;
    const float  *sc_base8 = (w_p8->w1 + layer)->sc; const float  *sc_base9 = (w_p9->w1 + layer)->sc;
    const float  *sc_base10 = (w_p10->w1 + layer)->sc; const float  *sc_base11 = (w_p11->w1 + layer)->sc;
    const float  *sc_base12 = (w_p12->w1 + layer)->sc; const float  *sc_base13 = (w_p13->w1 + layer)->sc;
    const float  *sc_base14 = (w_p14->w1 + layer)->sc; const float  *sc_base15 = (w_p15->w1 + layer)->sc;
    const float  *sc_base16 = (w_p16->w1 + layer)->sc; const float  *sc_base17 = (w_p17->w1 + layer)->sc;
    const float  *sc_base18 = (w_p18->w1 + layer)->sc; const float  *sc_base19 = (w_p19->w1 + layer)->sc;
    const float  *sc_base20 = (w_p20->w1 + layer)->sc; const float  *sc_base21 = (w_p21->w1 + layer)->sc;
    const float  *sc_base22 = (w_p22->w1 + layer)->sc; const float  *sc_base23 = (w_p23->w1 + layer)->sc;
    const float  *sc_base24 = (w_p24->w1 + layer)->sc; const float  *sc_base25 = (w_p25->w1 + layer)->sc;
    const float  *sc_base26 = (w_p26->w1 + layer)->sc; const float  *sc_base27 = (w_p27->w1 + layer)->sc;
    const float  *sc_base28 = (w_p28->w1 + layer)->sc; const float  *sc_base29 = (w_p29->w1 + layer)->sc;
    const float  *sc_base30 = (w_p30->w1 + layer)->sc; const float  *sc_base31 = (w_p31->w1 + layer)->sc;

    matmul_engine_w4a8_32port(xout, xq, xs,
                             wq_base0, wq_base1, wq_base2, wq_base3, wq_base4, wq_base5, wq_base6, wq_base7,
                             wq_base8, wq_base9, wq_base10, wq_base11, wq_base12, wq_base13, wq_base14, wq_base15,
                             wq_base16, wq_base17, wq_base18, wq_base19, wq_base20, wq_base21, wq_base22, wq_base23,
                             wq_base24, wq_base25, wq_base26, wq_base27, wq_base28, wq_base29, wq_base30, wq_base31,
                             sg_base0, sg_base1, sg_base2, sg_base3, sg_base4, sg_base5, sg_base6, sg_base7,
                             sg_base8, sg_base9, sg_base10, sg_base11, sg_base12, sg_base13, sg_base14, sg_base15,
                             sg_base16, sg_base17, sg_base18, sg_base19, sg_base20, sg_base21, sg_base22, sg_base23,
                             sg_base24, sg_base25, sg_base26, sg_base27, sg_base28, sg_base29, sg_base30, sg_base31,
                             sc_base0, sc_base1, sc_base2, sc_base3, sc_base4, sc_base5, sc_base6, sc_base7,
                             sc_base8, sc_base9, sc_base10, sc_base11, sc_base12, sc_base13, sc_base14, sc_base15,
                             sc_base16, sc_base17, sc_base18, sc_base19, sc_base20, sc_base21, sc_base22, sc_base23,
                             sc_base24, sc_base25, sc_base26, sc_base27, sc_base28, sc_base29, sc_base30, sc_base31,
                             M, N, K);
}

static void matmul_engine_w3(
    float *xout,
    const int8_t *xq,
    const float *xs,
    const ShardWeights_t* w_p0, const ShardWeights_t* w_p1, const ShardWeights_t* w_p2, const ShardWeights_t* w_p3,
    const ShardWeights_t* w_p4, const ShardWeights_t* w_p5, const ShardWeights_t* w_p6, const ShardWeights_t* w_p7,
    const ShardWeights_t* w_p8, const ShardWeights_t* w_p9, const ShardWeights_t* w_p10, const ShardWeights_t* w_p11,
    const ShardWeights_t* w_p12, const ShardWeights_t* w_p13, const ShardWeights_t* w_p14, const ShardWeights_t* w_p15,
    const ShardWeights_t* w_p16, const ShardWeights_t* w_p17, const ShardWeights_t* w_p18, const ShardWeights_t* w_p19,
    const ShardWeights_t* w_p20, const ShardWeights_t* w_p21, const ShardWeights_t* w_p22, const ShardWeights_t* w_p23,
    const ShardWeights_t* w_p24, const ShardWeights_t* w_p25, const ShardWeights_t* w_p26, const ShardWeights_t* w_p27,
    const ShardWeights_t* w_p28, const ShardWeights_t* w_p29, const ShardWeights_t* w_p30, const ShardWeights_t* w_p31,
    int layer, int M, int N, int K) {
    #pragma HLS INLINE

    const int8_t *wq_base0 = (w_p0->w3 + layer)->q; const int8_t *wq_base1 = (w_p1->w3 + layer)->q;
    const int8_t *wq_base2 = (w_p2->w3 + layer)->q; const int8_t *wq_base3 = (w_p3->w3 + layer)->q;
    const int8_t *wq_base4 = (w_p4->w3 + layer)->q; const int8_t *wq_base5 = (w_p5->w3 + layer)->q;
    const int8_t *wq_base6 = (w_p6->w3 + layer)->q; const int8_t *wq_base7 = (w_p7->w3 + layer)->q;
    const int8_t *wq_base8 = (w_p8->w3 + layer)->q; const int8_t *wq_base9 = (w_p9->w3 + layer)->q;
    const int8_t *wq_base10 = (w_p10->w3 + layer)->q; const int8_t *wq_base11 = (w_p11->w3 + layer)->q;
    const int8_t *wq_base12 = (w_p12->w3 + layer)->q; const int8_t *wq_base13 = (w_p13->w3 + layer)->q;
    const int8_t *wq_base14 = (w_p14->w3 + layer)->q; const int8_t *wq_base15 = (w_p15->w3 + layer)->q;
    const int8_t *wq_base16 = (w_p16->w3 + layer)->q; const int8_t *wq_base17 = (w_p17->w3 + layer)->q;
    const int8_t *wq_base18 = (w_p18->w3 + layer)->q; const int8_t *wq_base19 = (w_p19->w3 + layer)->q;
    const int8_t *wq_base20 = (w_p20->w3 + layer)->q; const int8_t *wq_base21 = (w_p21->w3 + layer)->q;
    const int8_t *wq_base22 = (w_p22->w3 + layer)->q; const int8_t *wq_base23 = (w_p23->w3 + layer)->q;
    const int8_t *wq_base24 = (w_p24->w3 + layer)->q; const int8_t *wq_base25 = (w_p25->w3 + layer)->q;
    const int8_t *wq_base26 = (w_p26->w3 + layer)->q; const int8_t *wq_base27 = (w_p27->w3 + layer)->q;
    const int8_t *wq_base28 = (w_p28->w3 + layer)->q; const int8_t *wq_base29 = (w_p29->w3 + layer)->q;
    const int8_t *wq_base30 = (w_p30->w3 + layer)->q; const int8_t *wq_base31 = (w_p31->w3 + layer)->q;
    
    const int8_t *sg_base0 = (w_p0->w3 + layer)->sg; const int8_t *sg_base1 = (w_p1->w3 + layer)->sg;
    const int8_t *sg_base2 = (w_p2->w3 + layer)->sg; const int8_t *sg_base3 = (w_p3->w3 + layer)->sg;
    const int8_t *sg_base4 = (w_p4->w3 + layer)->sg; const int8_t *sg_base5 = (w_p5->w3 + layer)->sg;
    const int8_t *sg_base6 = (w_p6->w3 + layer)->sg; const int8_t *sg_base7 = (w_p7->w3 + layer)->sg;
    const int8_t *sg_base8 = (w_p8->w3 + layer)->sg; const int8_t *sg_base9 = (w_p9->w3 + layer)->sg;
    const int8_t *sg_base10 = (w_p10->w3 + layer)->sg; const int8_t *sg_base11 = (w_p11->w3 + layer)->sg;
    const int8_t *sg_base12 = (w_p12->w3 + layer)->sg; const int8_t *sg_base13 = (w_p13->w3 + layer)->sg;
    const int8_t *sg_base14 = (w_p14->w3 + layer)->sg; const int8_t *sg_base15 = (w_p15->w3 + layer)->sg;
    const int8_t *sg_base16 = (w_p16->w3 + layer)->sg; const int8_t *sg_base17 = (w_p17->w3 + layer)->sg;
    const int8_t *sg_base18 = (w_p18->w3 + layer)->sg; const int8_t *sg_base19 = (w_p19->w3 + layer)->sg;
    const int8_t *sg_base20 = (w_p20->w3 + layer)->sg; const int8_t *sg_base21 = (w_p21->w3 + layer)->sg;
    const int8_t *sg_base22 = (w_p22->w3 + layer)->sg; const int8_t *sg_base23 = (w_p23->w3 + layer)->sg;
    const int8_t *sg_base24 = (w_p24->w3 + layer)->sg; const int8_t *sg_base25 = (w_p25->w3 + layer)->sg;
    const int8_t *sg_base26 = (w_p26->w3 + layer)->sg; const int8_t *sg_base27 = (w_p27->w3 + layer)->sg;
    const int8_t *sg_base28 = (w_p28->w3 + layer)->sg; const int8_t *sg_base29 = (w_p29->w3 + layer)->sg;
    const int8_t *sg_base30 = (w_p30->w3 + layer)->sg; const int8_t *sg_base31 = (w_p31->w3 + layer)->sg;
    
    const float  *sc_base0 = (w_p0->w3 + layer)->sc; const float  *sc_base1 = (w_p1->w3 + layer)->sc;
    const float  *sc_base2 = (w_p2->w3 + layer)->sc; const float  *sc_base3 = (w_p3->w3 + layer)->sc;
    const float  *sc_base4 = (w_p4->w3 + layer)->sc; const float  *sc_base5 = (w_p5->w3 + layer)->sc;
    const float  *sc_base6 = (w_p6->w3 + layer)->sc; const float  *sc_base7 = (w_p7->w3 + layer)->sc;
    const float  *sc_base8 = (w_p8->w3 + layer)->sc; const float  *sc_base9 = (w_p9->w3 + layer)->sc;
    const float  *sc_base10 = (w_p10->w3 + layer)->sc; const float  *sc_base11 = (w_p11->w3 + layer)->sc;
    const float  *sc_base12 = (w_p12->w3 + layer)->sc; const float  *sc_base13 = (w_p13->w3 + layer)->sc;
    const float  *sc_base14 = (w_p14->w3 + layer)->sc; const float  *sc_base15 = (w_p15->w3 + layer)->sc;
    const float  *sc_base16 = (w_p16->w3 + layer)->sc; const float  *sc_base17 = (w_p17->w3 + layer)->sc;
    const float  *sc_base18 = (w_p18->w3 + layer)->sc; const float  *sc_base19 = (w_p19->w3 + layer)->sc;
    const float  *sc_base20 = (w_p20->w3 + layer)->sc; const float  *sc_base21 = (w_p21->w3 + layer)->sc;
    const float  *sc_base22 = (w_p22->w3 + layer)->sc; const float  *sc_base23 = (w_p23->w3 + layer)->sc;
    const float  *sc_base24 = (w_p24->w3 + layer)->sc; const float  *sc_base25 = (w_p25->w3 + layer)->sc;
    const float  *sc_base26 = (w_p26->w3 + layer)->sc; const float  *sc_base27 = (w_p27->w3 + layer)->sc;
    const float  *sc_base28 = (w_p28->w3 + layer)->sc; const float  *sc_base29 = (w_p29->w3 + layer)->sc;
    const float  *sc_base30 = (w_p30->w3 + layer)->sc; const float  *sc_base31 = (w_p31->w3 + layer)->sc;

    matmul_engine_w4a8_32port(xout, xq, xs,
                             wq_base0, wq_base1, wq_base2, wq_base3, wq_base4, wq_base5, wq_base6, wq_base7,
                             wq_base8, wq_base9, wq_base10, wq_base11, wq_base12, wq_base13, wq_base14, wq_base15,
                             wq_base16, wq_base17, wq_base18, wq_base19, wq_base20, wq_base21, wq_base22, wq_base23,
                             wq_base24, wq_base25, wq_base26, wq_base27, wq_base28, wq_base29, wq_base30, wq_base31,
                             sg_base0, sg_base1, sg_base2, sg_base3, sg_base4, sg_base5, sg_base6, sg_base7,
                             sg_base8, sg_base9, sg_base10, sg_base11, sg_base12, sg_base13, sg_base14, sg_base15,
                             sg_base16, sg_base17, sg_base18, sg_base19, sg_base20, sg_base21, sg_base22, sg_base23,
                             sg_base24, sg_base25, sg_base26, sg_base27, sg_base28, sg_base29, sg_base30, sg_base31,
                             sc_base0, sc_base1, sc_base2, sc_base3, sc_base4, sc_base5, sc_base6, sc_base7,
                             sc_base8, sc_base9, sc_base10, sc_base11, sc_base12, sc_base13, sc_base14, sc_base15,
                             sc_base16, sc_base17, sc_base18, sc_base19, sc_base20, sc_base21, sc_base22, sc_base23,
                             sc_base24, sc_base25, sc_base26, sc_base27, sc_base28, sc_base29, sc_base30, sc_base31,
                             M, N, K);
}

static void matmul_engine_w2(
    float *xout,
    const int8_t *xq,
    const float *xs,
    const ShardWeights_t* w_p0, const ShardWeights_t* w_p1, const ShardWeights_t* w_p2, const ShardWeights_t* w_p3,
    const ShardWeights_t* w_p4, const ShardWeights_t* w_p5, const ShardWeights_t* w_p6, const ShardWeights_t* w_p7,
    const ShardWeights_t* w_p8, const ShardWeights_t* w_p9, const ShardWeights_t* w_p10, const ShardWeights_t* w_p11,
    const ShardWeights_t* w_p12, const ShardWeights_t* w_p13, const ShardWeights_t* w_p14, const ShardWeights_t* w_p15,
    const ShardWeights_t* w_p16, const ShardWeights_t* w_p17, const ShardWeights_t* w_p18, const ShardWeights_t* w_p19,
    const ShardWeights_t* w_p20, const ShardWeights_t* w_p21, const ShardWeights_t* w_p22, const ShardWeights_t* w_p23,
    const ShardWeights_t* w_p24, const ShardWeights_t* w_p25, const ShardWeights_t* w_p26, const ShardWeights_t* w_p27,
    const ShardWeights_t* w_p28, const ShardWeights_t* w_p29, const ShardWeights_t* w_p30, const ShardWeights_t* w_p31,
    int layer, int M, int N, int K) {
    #pragma HLS INLINE

    const int8_t *wq_base0 = (w_p0->w2 + layer)->q; const int8_t *wq_base1 = (w_p1->w2 + layer)->q;
    const int8_t *wq_base2 = (w_p2->w2 + layer)->q; const int8_t *wq_base3 = (w_p3->w2 + layer)->q;
    const int8_t *wq_base4 = (w_p4->w2 + layer)->q; const int8_t *wq_base5 = (w_p5->w2 + layer)->q;
    const int8_t *wq_base6 = (w_p6->w2 + layer)->q; const int8_t *wq_base7 = (w_p7->w2 + layer)->q;
    const int8_t *wq_base8 = (w_p8->w2 + layer)->q; const int8_t *wq_base9 = (w_p9->w2 + layer)->q;
    const int8_t *wq_base10 = (w_p10->w2 + layer)->q; const int8_t *wq_base11 = (w_p11->w2 + layer)->q;
    const int8_t *wq_base12 = (w_p12->w2 + layer)->q; const int8_t *wq_base13 = (w_p13->w2 + layer)->q;
    const int8_t *wq_base14 = (w_p14->w2 + layer)->q; const int8_t *wq_base15 = (w_p15->w2 + layer)->q;
    const int8_t *wq_base16 = (w_p16->w2 + layer)->q; const int8_t *wq_base17 = (w_p17->w2 + layer)->q;
    const int8_t *wq_base18 = (w_p18->w2 + layer)->q; const int8_t *wq_base19 = (w_p19->w2 + layer)->q;
    const int8_t *wq_base20 = (w_p20->w2 + layer)->q; const int8_t *wq_base21 = (w_p21->w2 + layer)->q;
    const int8_t *wq_base22 = (w_p22->w2 + layer)->q; const int8_t *wq_base23 = (w_p23->w2 + layer)->q;
    const int8_t *wq_base24 = (w_p24->w2 + layer)->q; const int8_t *wq_base25 = (w_p25->w2 + layer)->q;
    const int8_t *wq_base26 = (w_p26->w2 + layer)->q; const int8_t *wq_base27 = (w_p27->w2 + layer)->q;
    const int8_t *wq_base28 = (w_p28->w2 + layer)->q; const int8_t *wq_base29 = (w_p29->w2 + layer)->q;
    const int8_t *wq_base30 = (w_p30->w2 + layer)->q; const int8_t *wq_base31 = (w_p31->w2 + layer)->q;
    
    const int8_t *sg_base0 = (w_p0->w2 + layer)->sg; const int8_t *sg_base1 = (w_p1->w2 + layer)->sg;
    const int8_t *sg_base2 = (w_p2->w2 + layer)->sg; const int8_t *sg_base3 = (w_p3->w2 + layer)->sg;
    const int8_t *sg_base4 = (w_p4->w2 + layer)->sg; const int8_t *sg_base5 = (w_p5->w2 + layer)->sg;
    const int8_t *sg_base6 = (w_p6->w2 + layer)->sg; const int8_t *sg_base7 = (w_p7->w2 + layer)->sg;
    const int8_t *sg_base8 = (w_p8->w2 + layer)->sg; const int8_t *sg_base9 = (w_p9->w2 + layer)->sg;
    const int8_t *sg_base10 = (w_p10->w2 + layer)->sg; const int8_t *sg_base11 = (w_p11->w2 + layer)->sg;
    const int8_t *sg_base12 = (w_p12->w2 + layer)->sg; const int8_t *sg_base13 = (w_p13->w2 + layer)->sg;
    const int8_t *sg_base14 = (w_p14->w2 + layer)->sg; const int8_t *sg_base15 = (w_p15->w2 + layer)->sg;
    const int8_t *sg_base16 = (w_p16->w2 + layer)->sg; const int8_t *sg_base17 = (w_p17->w2 + layer)->sg;
    const int8_t *sg_base18 = (w_p18->w2 + layer)->sg; const int8_t *sg_base19 = (w_p19->w2 + layer)->sg;
    const int8_t *sg_base20 = (w_p20->w2 + layer)->sg; const int8_t *sg_base21 = (w_p21->w2 + layer)->sg;
    const int8_t *sg_base22 = (w_p22->w2 + layer)->sg; const int8_t *sg_base23 = (w_p23->w2 + layer)->sg;
    const int8_t *sg_base24 = (w_p24->w2 + layer)->sg; const int8_t *sg_base25 = (w_p25->w2 + layer)->sg;
    const int8_t *sg_base26 = (w_p26->w2 + layer)->sg; const int8_t *sg_base27 = (w_p27->w2 + layer)->sg;
    const int8_t *sg_base28 = (w_p28->w2 + layer)->sg; const int8_t *sg_base29 = (w_p29->w2 + layer)->sg;
    const int8_t *sg_base30 = (w_p30->w2 + layer)->sg; const int8_t *sg_base31 = (w_p31->w2 + layer)->sg;
    
    const float  *sc_base0 = (w_p0->w2 + layer)->sc; const float  *sc_base1 = (w_p1->w2 + layer)->sc;
    const float  *sc_base2 = (w_p2->w2 + layer)->sc; const float  *sc_base3 = (w_p3->w2 + layer)->sc;
    const float  *sc_base4 = (w_p4->w2 + layer)->sc; const float  *sc_base5 = (w_p5->w2 + layer)->sc;
    const float  *sc_base6 = (w_p6->w2 + layer)->sc; const float  *sc_base7 = (w_p7->w2 + layer)->sc;
    const float  *sc_base8 = (w_p8->w2 + layer)->sc; const float  *sc_base9 = (w_p9->w2 + layer)->sc;
    const float  *sc_base10 = (w_p10->w2 + layer)->sc; const float  *sc_base11 = (w_p11->w2 + layer)->sc;
    const float  *sc_base12 = (w_p12->w2 + layer)->sc; const float  *sc_base13 = (w_p13->w2 + layer)->sc;
    const float  *sc_base14 = (w_p14->w2 + layer)->sc; const float  *sc_base15 = (w_p15->w2 + layer)->sc;
    const float  *sc_base16 = (w_p16->w2 + layer)->sc; const float  *sc_base17 = (w_p17->w2 + layer)->sc;
    const float  *sc_base18 = (w_p18->w2 + layer)->sc; const float  *sc_base19 = (w_p19->w2 + layer)->sc;
    const float  *sc_base20 = (w_p20->w2 + layer)->sc; const float  *sc_base21 = (w_p21->w2 + layer)->sc;
    const float  *sc_base22 = (w_p22->w2 + layer)->sc; const float  *sc_base23 = (w_p23->w2 + layer)->sc;
    const float  *sc_base24 = (w_p24->w2 + layer)->sc; const float  *sc_base25 = (w_p25->w2 + layer)->sc;
    const float  *sc_base26 = (w_p26->w2 + layer)->sc; const float  *sc_base27 = (w_p27->w2 + layer)->sc;
    const float  *sc_base28 = (w_p28->w2 + layer)->sc; const float  *sc_base29 = (w_p29->w2 + layer)->sc;
    const float  *sc_base30 = (w_p30->w2 + layer)->sc; const float  *sc_base31 = (w_p31->w2 + layer)->sc;

    matmul_engine_w4a8_32port(xout, xq, xs,
                             wq_base0, wq_base1, wq_base2, wq_base3, wq_base4, wq_base5, wq_base6, wq_base7,
                             wq_base8, wq_base9, wq_base10, wq_base11, wq_base12, wq_base13, wq_base14, wq_base15,
                             wq_base16, wq_base17, wq_base18, wq_base19, wq_base20, wq_base21, wq_base22, wq_base23,
                             wq_base24, wq_base25, wq_base26, wq_base27, wq_base28, wq_base29, wq_base30, wq_base31,
                             sg_base0, sg_base1, sg_base2, sg_base3, sg_base4, sg_base5, sg_base6, sg_base7,
                             sg_base8, sg_base9, sg_base10, sg_base11, sg_base12, sg_base13, sg_base14, sg_base15,
                             sg_base16, sg_base17, sg_base18, sg_base19, sg_base20, sg_base21, sg_base22, sg_base23,
                             sg_base24, sg_base25, sg_base26, sg_base27, sg_base28, sg_base29, sg_base30, sg_base31,
                             sc_base0, sc_base1, sc_base2, sc_base3, sc_base4, sc_base5, sc_base6, sc_base7,
                             sc_base8, sc_base9, sc_base10, sc_base11, sc_base12, sc_base13, sc_base14, sc_base15,
                             sc_base16, sc_base17, sc_base18, sc_base19, sc_base20, sc_base21, sc_base22, sc_base23,
                             sc_base24, sc_base25, sc_base26, sc_base27, sc_base28, sc_base29, sc_base30, sc_base31,
                             M, N, K);
}


static const int  EXPF_LUT_SIZE    = 16;
static const half EXPF_MIN_X_HF    = -10.0f;  // exp(x) 很小了，可以近似为 0
static const half EXPF_MAX_X_HF    =  0.0f;   // softmax 里 (x - max) 通常 <= 0
static const half EXPF_STEP_HF     = (EXPF_MAX_X_HF - EXPF_MIN_X_HF) / (EXPF_LUT_SIZE - 1);
static const half EXPF_INV_STEP_HF = 1.0f / EXPF_STEP_HF;

static const half EXPF_MIN_X_HF_OF = -8.8f;
static const half EXPF_MAX_X_HF_OF = 1.2f;

static const half EXPF_LUT_HALF[16] = {
    (half)4.5399929762484854e-05f, (half)8.8426988659883016e-05f, (half)1.7223225596081014e-04f, (half)3.3546262790251185e-04f, 
    (half)6.5339197986738007e-04f, (half)1.2726338013398079e-03f, (half)2.4787521766663585e-03f, (half)4.8279499938314371e-03f, 
    (half)9.4035625514952061e-03f, (half)1.8315638888734179e-02f, (half)3.5673993347252374e-02f, (half)6.9483451222801515e-02f, 
    (half)1.3533528323661270e-01f, (half)2.6359713811572660e-01f, (half)5.1341711903259146e-01f, (half)1.0000000000000000e+00f
};


static const float EXPF_LUT[16] = {
    4.5399929762484854e-05f, 8.8426988659883016e-05f, 1.7223225596081014e-04f, 3.3546262790251185e-04f, 
    6.5339197986738007e-04f, 1.2726338013398079e-03f, 2.4787521766663585e-03f, 4.8279499938314371e-03f, 
    9.4035625514952061e-03f, 1.8315638888734179e-02f, 3.5673993347252374e-02f, 6.9483451222801515e-02f, 
    1.3533528323661270e-01f, 2.6359713811572660e-01f, 5.1341711903259146e-01f, 1.0000000000000000e+00f
};


inline half fast_expf_half(half x) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=EXPF_LUT_HALF complete

    if (x <= EXPF_MIN_X_HF_OF) return (half)0.0f;
    if (x >= EXPF_MAX_X_HF_OF) return (half)1.0f;

    half u = (x - EXPF_MIN_X_HF_OF) * EXPF_INV_STEP_HF;
    int idx = (int)u;
    if (idx >= EXPF_LUT_SIZE - 1) idx = EXPF_LUT_SIZE - 1;

    return EXPF_LUT_HALF[idx];  // 直接返回 half
}

static const float EXPF_MIN_X    = -10.0f;  // exp(x) 很小了，可以近似为 0
static const float EXPF_MAX_X    =  0.0f;   // softmax 里 (x - max) 通常 <= 0
static const float EXPF_STEP     = (EXPF_MAX_X - EXPF_MIN_X) / (EXPF_LUT_SIZE - 1);
static const float EXPF_INV_STEP = 1.0f / EXPF_STEP;

inline float fast_expf_float(float x) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=EXPF_LUT complete

    if (x <= EXPF_MIN_X) return (float)0.0f;
    if (x >= EXPF_MAX_X) return (float)1.0f;

    float u = (x - EXPF_MIN_X) * EXPF_INV_STEP;
    int idx = (int)u;
    if (idx >= EXPF_LUT_SIZE - 1) idx = EXPF_LUT_SIZE - 1;

    return EXPF_LUT[idx];  // 直接返回 float
}


static inline half fast_sigmoid_half(half x) {
#pragma HLS INLINE
    // 使用多项式近似: sigmoid(x) ≈ 0.5 + 0.25*x - 0.03125*x^3 (|x| < 4)
    // 或直接 clamp: 1/(1+exp(-x)) ≈ max(0, min(1, 0.5 + 0.25*x))
    half clamped = (x > (half)4.0f) ? (half)1.0f : 
                   (x < (half)-4.0f) ? (half)0.0f :
                   (half)0.5f + (half)0.25f * x;
    return clamped;
}

#endif // FORWARD_H
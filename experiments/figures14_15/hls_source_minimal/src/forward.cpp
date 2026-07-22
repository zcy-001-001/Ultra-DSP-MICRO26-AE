#include "forward.h"
#include <cstring>
#include <cmath>
#include <ap_int.h>
#include <hls_math.h>

// =====================
// Dataflow helper types for MHA
// =====================

static const int HEAD_SIZE_CFG = dim / n_heads;  // 64

// 使用 1024-bit 总线：64 halves * 16-bit = 1024-bit，一次读取整个 head
typedef ap_uint<1024> KVBusVec1024;

static inline half unpack_half_1024(KVBusVec1024 packet, int idx) {
#pragma HLS INLINE
    ap_uint<16> bits = packet.range((idx + 1) * 16 - 1, idx * 16);
    return *(half*)&bits;
}

struct KVHeadVec {
    half data[HEAD_SIZE_CFG];
};

// 用于跨 head 流水的输出结构
struct OutLocalPacket {
    half data[HEAD_SIZE_CFG];
};

// =====================================================
// 跨 Head 流水化的 Dataflow Stages
// =====================================================

// Stage 1: 加载所有 heads 的 K/V 数据
static void load_kv_all_heads(
    const half* key_cache,
    const half* value_cache,
    int loff,
    int pos,
    hls::stream<KVHeadVec>& k_stream,
    hls::stream<KVHeadVec>& v_stream) {

    const KVBusVec1024* key_cache_vec   = reinterpret_cast<const KVBusVec1024*>(key_cache);
    const KVBusVec1024* value_cache_vec = reinterpret_cast<const KVBusVec1024*>(value_cache);
    const int HALVES_PER_VEC = 64;

load_kv_heads:
    for (int h = 0; h < n_heads; ++h) {
        int kv_head_offset = h * HEAD_SIZE_CFG;
    load_kv_t:
        for (int t = 0; t <= pos; ++t) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=seq_len avg=seq_len/2
            KVHeadVec k_vec;
            KVHeadVec v_vec;

            int base_half_offset = loff + t * kv_dim + kv_head_offset;
            int vec_offset = base_half_offset / HALVES_PER_VEC;

            KVBusVec1024 k_pack = key_cache_vec[vec_offset];
            KVBusVec1024 v_pack = value_cache_vec[vec_offset];

        unpack_vec:
            for (int i = 0; i < HEAD_SIZE_CFG; ++i) {
            #pragma HLS UNROLL
                k_vec.data[i] = unpack_half_1024(k_pack, i);
                v_vec.data[i] = unpack_half_1024(v_pack, i);
            }

            k_stream.write(k_vec);
            v_stream.write(v_vec);
        }
    }
}

// Stage 2: 计算所有 heads 的 Q·K scores
static void compute_qk_all_heads(
    const float* q,
    int pos,
    hls::stream<KVHeadVec>& k_stream,
    hls::stream<half>& score_stream) {

compute_qk_heads:
    for (int h = 0; h < n_heads; ++h) {
        int q_offset = h * HEAD_SIZE_CFG;
    compute_qk_t:
        for (int t = 0; t <= pos; ++t) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=seq_len avg=seq_len/2
            KVHeadVec k_vec = k_stream.read();
            const half* key_ptr = k_vec.data;

            // 4 路并行点积
            half partial0 = (half)0.0f;
            half partial1 = (half)0.0f;
            half partial2 = (half)0.0f;
            half partial3 = (half)0.0f;

        dot_product_qk:
            for (int i = 0; i < HEAD_SIZE_CFG; i += 4) {
            #pragma HLS UNROLL
                half q0 = (half)q[q_offset + i];
                half q1 = (half)q[q_offset + i + 1];
                half q2 = (half)q[q_offset + i + 2];
                half q3 = (half)q[q_offset + i + 3];
                partial0 += q0 * key_ptr[i];
                partial1 += q1 * key_ptr[i + 1];
                partial2 += q2 * key_ptr[i + 2];
                partial3 += q3 * key_ptr[i + 3];
            }

            half score = (partial0 + partial1) + (partial2 + partial3);
            score_stream.write(score);
        }
    }
}

// Stage 3: 所有 heads 的 Softmax
static void compute_softmax_all_heads(
    int pos,
    hls::stream<half>& score_stream,
    hls::stream<half>& beta_stream_main,
    hls::stream<half>& beta_stream_lrun) {

    half scale = (half)(1.0f / sqrtf((float)HEAD_SIZE_CFG));

softmax_heads:
    for (int h = 0; h < n_heads; ++h) {
    softmax_t:
        for (int t = 0; t <= pos; ++t) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=seq_len avg=seq_len/2
            half score = score_stream.read();
            half score_scaled = score * scale;
            half beta = fast_expf_half(score_scaled);

            beta_stream_main.write(beta);
            beta_stream_lrun.write(beta);
        }
    }
}

// Stage 4: 所有 heads 的 V 累加，输出未归一化结果到流（4路并行 + II=2）
static void update_output_all_heads(
    int pos,
    hls::stream<KVHeadVec>& v_stream,
    hls::stream<half>& beta_stream,
    hls::stream<OutLocalPacket>& out_unnorm_stream) {

update_out_heads:
    for (int h = 0; h < n_heads; ++h) {
        // 4 组独立累加器
        half out_0[HEAD_SIZE_CFG];
        half out_1[HEAD_SIZE_CFG];
        half out_2[HEAD_SIZE_CFG];
        half out_3[HEAD_SIZE_CFG];
        #pragma HLS ARRAY_PARTITION variable=out_0 complete
        #pragma HLS ARRAY_PARTITION variable=out_1 complete
        #pragma HLS ARRAY_PARTITION variable=out_2 complete
        #pragma HLS ARRAY_PARTITION variable=out_3 complete

    init_out_accum:
        for (int i = 0; i < HEAD_SIZE_CFG; ++i) {
        #pragma HLS UNROLL
            out_0[i] = (half)0.0f;
            out_1[i] = (half)0.0f;
            out_2[i] = (half)0.0f;
            out_3[i] = (half)0.0f;
        }

        int total = pos + 1;
        int chunks = total >> 2;  // /4
        int rem = total & 3;      // %4

    update_out_main:
        for (int c = 0; c < chunks; ++c) {
        #pragma HLS PIPELINE II=2
        #pragma HLS LOOP_TRIPCOUNT min=1 max=seq_len/4 avg=seq_len/8
            KVHeadVec v0 = v_stream.read(); half beta0 = beta_stream.read();
            KVHeadVec v1 = v_stream.read(); half beta1 = beta_stream.read();
            KVHeadVec v2 = v_stream.read(); half beta2 = beta_stream.read();
            KVHeadVec v3 = v_stream.read(); half beta3 = beta_stream.read();

        update_out_chunk:
            for (int i = 0; i < HEAD_SIZE_CFG; ++i) {
            #pragma HLS UNROLL
                out_0[i] = out_0[i] + beta0 * v0.data[i];
                out_1[i] = out_1[i] + beta1 * v1.data[i];
                out_2[i] = out_2[i] + beta2 * v2.data[i];
                out_3[i] = out_3[i] + beta3 * v3.data[i];
            }
        }

    update_out_rem:
        for (int r = 0; r < rem; ++r) {
        #pragma HLS PIPELINE II=1
            KVHeadVec v_r = v_stream.read();
            half beta_r = beta_stream.read();
        update_out_rem_i:
            for (int i = 0; i < HEAD_SIZE_CFG; ++i) {
            #pragma HLS UNROLL
                if (r == 0) out_0[i] = out_0[i] + beta_r * v_r.data[i];
                else if (r == 1) out_1[i] = out_1[i] + beta_r * v_r.data[i];
                else if (r == 2) out_2[i] = out_2[i] + beta_r * v_r.data[i];
            }
        }

        // 归约并输出到流
        OutLocalPacket pkt;
    reduce_out_accum:
        for (int i = 0; i < HEAD_SIZE_CFG; ++i) {
        #pragma HLS UNROLL
            half sum01 = out_0[i] + out_1[i];
            half sum23 = out_2[i] + out_3[i];
            pkt.data[i] = sum01 + sum23;
        }
        out_unnorm_stream.write(pkt);
    }
}

// Stage 5: 所有 heads 的 l_run 归约，输出到流（4路并行 + II=2）
static void reduce_lrun_all_heads(
    int pos,
    hls::stream<half>& beta_stream_lrun,
    hls::stream<half>& l_run_stream) {

reduce_lrun_heads:
    for (int h = 0; h < n_heads; ++h) {
        half p0 = (half)0.0f;
        half p1 = (half)0.0f;
        half p2 = (half)0.0f;
        half p3 = (half)0.0f;

        int total = pos + 1;
        int chunks = total >> 2;  // /4
        int rem = total & 3;      // %4

    reduce_lrun_main:
        for (int i = 0; i < chunks; ++i) {
        #pragma HLS PIPELINE II=2
        #pragma HLS LOOP_TRIPCOUNT min=1 max=seq_len/4 avg=seq_len/8
            half b0 = beta_stream_lrun.read();
            half b1 = beta_stream_lrun.read();
            half b2 = beta_stream_lrun.read();
            half b3 = beta_stream_lrun.read();
            p0 = p0 + b0;
            p1 = p1 + b1;
            p2 = p2 + b2;
            p3 = p3 + b3;
        }

    reduce_lrun_rem:
        for (int r = 0; r < rem; ++r) {
        #pragma HLS PIPELINE II=1
            half b = beta_stream_lrun.read();
            if (r == 0) p0 = p0 + b;
            else if (r == 1) p1 = p1 + b;
            else p2 = p2 + b;
        }

        half sum01 = p0 + p1;
        half sum23 = p2 + p3;
        half l_run = sum01 + sum23;
        l_run_stream.write(l_run);
    }
}

// Stage 6: 归一化并输出到流（不直接写数组，避免 dataflow 读写冲突）
static void normalize_all_heads(
    hls::stream<OutLocalPacket>& out_unnorm_stream,
    hls::stream<half>& l_run_stream,
    hls::stream<OutLocalPacket>& out_norm_stream) {

normalize_heads:
    for (int h = 0; h < n_heads; ++h) {
        OutLocalPacket pkt = out_unnorm_stream.read();
        half l_run = l_run_stream.read();
        half inv_l_run = (l_run == (half)0.0f) ? (half)0.0f : (half)1.0f / l_run;

        OutLocalPacket out_pkt;
    normalize_out:
        for (int i = 0; i < HEAD_SIZE_CFG; ++i) {
        #pragma HLS UNROLL
            out_pkt.data[i] = pkt.data[i] * inv_l_run;
        }
        out_norm_stream.write(out_pkt);
    }
}

// ============================================================================
// MHA Dataflow 核心函数 - 纯 DATAFLOW 区域
// ============================================================================
static void mha_dataflow_core(
    const half* key_cache,
    const half* value_cache,
    int loff,
    int pos,
    const float q_local[dim],
    hls::stream<OutLocalPacket>& out_norm_stream) {

#pragma HLS DATAFLOW

    // 定义跨 head 流水的所有流
    hls::stream<KVHeadVec>      k_stream("k_stream");
    hls::stream<KVHeadVec>      v_stream("v_stream");
    hls::stream<half>           score_stream("score_stream");
    hls::stream<half>           beta_stream("beta_stream");
    hls::stream<half>           beta_stream_lrun("beta_stream_lrun");
    hls::stream<OutLocalPacket> out_unnorm_stream("out_unnorm_stream");
    hls::stream<half>           l_run_stream("l_run_stream");

    // 流深度设置
    #pragma HLS STREAM variable=k_stream         depth=16
    #pragma HLS STREAM variable=v_stream         depth=16
    #pragma HLS STREAM variable=score_stream     depth=16
    #pragma HLS STREAM variable=beta_stream      depth=16
    #pragma HLS STREAM variable=beta_stream_lrun depth=16
    #pragma HLS STREAM variable=out_unnorm_stream depth=4
    #pragma HLS STREAM variable=l_run_stream     depth=4

    // Stage 1: 加载所有 heads 的 K/V
    load_kv_all_heads(
        key_cache,
        value_cache,
        loff,
        pos,
        k_stream,
        v_stream);

    // Stage 2: 计算所有 heads 的 Q·K scores
    compute_qk_all_heads(
        q_local,
        pos,
        k_stream,
        score_stream);

    // Stage 3: 所有 heads 的 Softmax
    compute_softmax_all_heads(
        pos,
        score_stream,
        beta_stream,
        beta_stream_lrun);

    // Stage 4: 所有 heads 的 V 加权累加
    update_output_all_heads(
        pos,
        v_stream,
        beta_stream,
        out_unnorm_stream);

    // Stage 5: 所有 heads 的 l_run 归约
    reduce_lrun_all_heads(
        pos,
        beta_stream_lrun,
        l_run_stream);

    // Stage 6: 归一化并输出到流
    normalize_all_heads(
        out_unnorm_stream,
        l_run_stream,
        out_norm_stream);
}

// ============================================================================
// Multi-Head Attention 包装函数
// ============================================================================
static void multihead_attention_dataflow_wrapper(
    const half* key_cache,
    const half* value_cache,
    int loff,
    int pos,
    const float q_in[dim],
    float attn_out[dim]) {

    // 复制 q 到本地缓冲区，避免 dataflow 区域内的别名问题
    float q_local[dim];
    #pragma HLS ARRAY_PARTITION variable=q_local cyclic factor=16

    // 输出流：用于从 dataflow 区域传出归一化结果
    hls::stream<OutLocalPacket> out_norm_stream("out_norm_stream");
    #pragma HLS STREAM variable=out_norm_stream depth=4

    // 批量读取 q（每次读 16 个 float，使用循环展开）
    constexpr int FLOATS_PER_VEC = 16;
    constexpr int NUM_VECS = dim / FLOATS_PER_VEC;  // 768 / 16 = 48

copy_q:
    for (int v = 0; v < NUM_VECS; ++v) {
        #pragma HLS PIPELINE II=1
        int base = v * FLOATS_PER_VEC;
    copy_q_inner:
        for (int i = 0; i < FLOATS_PER_VEC; ++i) {
            #pragma HLS UNROLL
            q_local[base + i] = q_in[base + i];
        }
    }

    // 调用 DATAFLOW 核心函数
    mha_dataflow_core(
        key_cache,
        value_cache,
        loff,
        pos,
        q_local,
        out_norm_stream);

    // 在 dataflow 区域外将流数据写入输出数组（批量写入）
    constexpr int FLOATS_PER_WRITE = 16;
    constexpr int WRITES_PER_HEAD = HEAD_SIZE_CFG / FLOATS_PER_WRITE;  // 64 / 16 = 4

write_attn_out:
    for (int h = 0; h < n_heads; ++h) {
        OutLocalPacket pkt = out_norm_stream.read();
        int out_offset = h * HEAD_SIZE_CFG;
    write_head:
        for (int w = 0; w < WRITES_PER_HEAD; ++w) {
            #pragma HLS PIPELINE II=1
            int base = w * FLOATS_PER_WRITE;
        write_floats:
            for (int i = 0; i < FLOATS_PER_WRITE; ++i) {
                #pragma HLS UNROLL
                attn_out[out_offset + base + i] = (float)pkt.data[base + i];
            }
        }
    }
}

// ============================================================================
// Kernel 1: Initial Embedding Lookup
// ============================================================================
extern "C" void
initial_embedding_lookup(
    const float* token_embedding_table,
    int token,
    hls::stream<float>& stream_out_x
) {
#pragma HLS INTERFACE m_axi port=token_embedding_table offset=slave bundle=gmem0 depth=vocab_size*dim latency=100 num_read_outstanding=32
#pragma HLS INTERFACE axis port=stream_out_x
#pragma HLS INTERFACE s_axilite port=token
#pragma HLS INTERFACE s_axilite port=return

    float embedding_buffer[dim];
#pragma HLS ARRAY_PARTITION variable=embedding_buffer type=cyclic factor=2

    if (token < 0 || token >= vocab_size) {
        for (int i = 0; i < dim; ++i) {
        #pragma HLS PIPELINE II=1
            stream_out_x.write(0.0f);
        }
        return;
    }
    const float* embedding_start_addr = token_embedding_table + token * dim;
read_embedding:
    std::memcpy(embedding_buffer, embedding_start_addr, dim * sizeof(float));
write_embedding_stream:
    for (int i = 0; i < dim; ++i) {
#pragma HLS PIPELINE II=1
        stream_out_x.write(embedding_buffer[i]);
    }
}



// ============================================================================
// Helper Functions for Scheduler (Using Pointers to Unified Pools)
// ============================================================================

static void prepare_attn_input(
    int l,
    float* current_x,
    int8_t* xq_ptr,   // Output to Int8 Pool
    float* xs_ptr,    // Output to Scale Pool
    const ShardWeights_t* w_p0
) {
#pragma HLS INLINE off
    float xb[dim];
    // Use w_p0 as the sole source for RMSNorm weights
    rmsnorm<dim>(xb, current_x, w_p0->rms_att_weight + l * dim);
    quantize_ptr<dim, 1>(xq_ptr, xs_ptr, xb);
}

static void compute_mha(
    int l, int pos,
    float* q, float* k, float* v,
    float* attn_out, 
    half* key_cache, half* value_cache
) {
#pragma HLS INLINE off
    constexpr int kv_dim_local = kv_dim;
    constexpr int head_size_local = dim / n_heads;

    // RoPE rotation on Q and K
rope_rotation:
    for (int i = 0; i < kv_dim_local; i += 2) {
    #pragma HLS PIPELINE II=1
        int head_dim_rot = i % head_size_local;
        if (head_size_local == 0) continue;
        float freq = 1.0f / powf(10000.0f, (float)head_dim_rot / (float)head_size_local);
        float val = pos * freq;
        float fcr = cosf(val); float fci = sinf(val);
        float v0_q = q[i]; float v1_q = q[i + 1]; 
        q[i] = v0_q * fcr - v1_q * fci; 
        q[i+1] = v0_q * fci + v1_q * fcr;
        float v0_k = k[i]; float v1_k = k[i + 1]; 
        k[i] = v0_k * fcr - v1_k * fci; 
        k[i+1] = v0_k * fci + v1_k * fcr;
    }

    // Update KV Cache with current position's K and V
    int loff = l * seq_len * kv_dim_local;
    half *key_cache_row = key_cache + loff + pos * kv_dim_local;
    half *value_cache_row = value_cache + loff + pos * kv_dim_local;
write_kv_cache:
    for (int i = 0; i < kv_dim_local; i++) {
    #pragma HLS PIPELINE II=1
        key_cache_row[i]   = (half)k[i];
        value_cache_row[i] = (half)v[i];
    }

    // Multi-Head Attention using DATAFLOW pipeline
    multihead_attention_dataflow_wrapper(
        key_cache,
        value_cache,
        loff,
        pos,
        q,
        attn_out);
}

static void prepare_wo_input(
    float* attn_out,
    int8_t* xq_ptr,
    float* xs_ptr
) {
#pragma HLS INLINE off
    quantize_ptr<dim, 1>(xq_ptr, xs_ptr, attn_out);
}

static void update_residual(
    float* current_x,
    float* residual
) {
#pragma HLS INLINE off
    for (int i = 0; i < dim; i++) {
    #pragma HLS PIPELINE II=1
        current_x[i] += residual[i];
    }
}

static void prepare_ffn_input(
    int l,
    float* current_x,
    int8_t* xq_ptr,
    float* xs_ptr,
    const ShardWeights_t* w_p0
) {
#pragma HLS INLINE off
    float xb[dim];
    // Use w_p0 as the sole source for RMSNorm weights
    rmsnorm<dim>(xb, current_x, w_p0->rms_ffn_weight + l * dim);
    quantize_ptr<dim, 1>(xq_ptr, xs_ptr, xb);
}

static void compute_swiglu_prepare_w2(
    float* hb, float* hb2,
    int8_t* xq_ptr,
    float* xs_ptr
) {
#pragma HLS INLINE off

    float hb_buff[hidden_dim];
    float hb2_buff[hidden_dim];
    #pragma HLS ARRAY_PARTITION variable=hb_buff cyclic factor=32
    #pragma HLS ARRAY_PARTITION variable=hb2_buff cyclic factor=32

    load_hb_hb2:
    for (int j = 0; j < hidden_dim; j += 16) {
    #pragma HLS PIPELINE II=1
        for (int k = 0; k < 16; k++) {
        #pragma HLS UNROLL
            hb_buff[j + k] = hb[j + k];
            hb2_buff[j + k] = hb2[j + k];
        }
    }

    const int BLOCK = 32;
    half prev_x[BLOCK];
    half prev_gate[BLOCK];
    half prev_sig[BLOCK];
    #pragma HLS ARRAY_PARTITION variable=prev_x complete
    #pragma HLS ARRAY_PARTITION variable=prev_gate complete
    #pragma HLS ARRAY_PARTITION variable=prev_sig complete
    bool prev_valid = false;

    swiglu_compute:
    for (int j = 0; j < hidden_dim; j += BLOCK) {
    #pragma HLS PIPELINE II=2
        int base = j;
        for (int k = 0; k < BLOCK; k++) {
        #pragma HLS UNROLL
            int idx = base + k;
            float x = hb_buff[idx];
            float gate = hb2_buff[idx];
            half hx = (half)x;
            half sig_h = (half(1.0f) / (half(1.0f) + exp(-hx)));
            half sig = sig_h;

            if (prev_valid) {
                int prev_idx = base - BLOCK + k;
                half val_prev = prev_x[k] * prev_sig[k] * prev_gate[k];
                hb_buff[prev_idx] = (float)val_prev;
            }
            prev_x[k] = x;
            prev_gate[k] = gate;
            prev_sig[k] = sig;
        }
        prev_valid = true;
    }

    if (prev_valid) {
        for (int k = 0; k < BLOCK; k++) {
        #pragma HLS UNROLL
            int idx = hidden_dim - BLOCK + k;
            half val = prev_x[k] * prev_sig[k] * prev_gate[k];
            hb_buff[idx] = (float)val;
        }
    }

    quantize_ptr<hidden_dim, 1>(xq_ptr, xs_ptr, hb_buff);
}

// ============================================================================
// Kernel 2 & 3: Main Scheduler (Unified Memory Architecture)
// ============================================================================
static void transformer_layer_scheduler(
    hls::stream<float>& stream_initial_in,
    hls::stream<float>& stream_final_out,
    const ShardWeights_t* w_p0, const ShardWeights_t* w_p1, const ShardWeights_t* w_p2, const ShardWeights_t* w_p3,
    const ShardWeights_t* w_p4, const ShardWeights_t* w_p5, const ShardWeights_t* w_p6, const ShardWeights_t* w_p7,
    const ShardWeights_t* w_p8, const ShardWeights_t* w_p9, const ShardWeights_t* w_p10, const ShardWeights_t* w_p11,
    const ShardWeights_t* w_p12, const ShardWeights_t* w_p13, const ShardWeights_t* w_p14, const ShardWeights_t* w_p15,
    const ShardWeights_t* w_p16, const ShardWeights_t* w_p17, const ShardWeights_t* w_p18, const ShardWeights_t* w_p19,
    const ShardWeights_t* w_p20, const ShardWeights_t* w_p21, const ShardWeights_t* w_p22, const ShardWeights_t* w_p23,
    const ShardWeights_t* w_p24, const ShardWeights_t* w_p25, const ShardWeights_t* w_p26, const ShardWeights_t* w_p27,
    const ShardWeights_t* w_p28, const ShardWeights_t* w_p29, const ShardWeights_t* w_p30, const ShardWeights_t* w_p31,
    int pos,   
    half* key_cache,
    half* value_cache
) {
    // 限制函数实例数，复用相同 dim 的 rmsnorm 和 quantize_ptr
// #pragma HLS ALLOCATION function instances=rmsnorm<768> limit=1
// #pragma HLS ALLOCATION function instances=quantize_ptr<768,1> limit=1

    static float current_x[dim];
#pragma HLS ARRAY_PARTITION variable=current_x type=cyclic factor=32

    // Unified Workspace Buffers (TPU Memory)
    // Size breakdown:
    // Float: Max(Attn usage, FFN usage)
    // Attn: Q(dim) + K(kv) + V(kv) + AttnOut(dim) = 2*dim + 2*kv
    // FFN: HB(hidden) + HB2(hidden) + FFNOut(dim) = 2*hidden + dim
    // Hidden is approx 4*dim, so FFN dominates.
    const int POOL_SIZE = 3 * hidden_dim; // Safe upper bound
    float pool[POOL_SIZE];
    
    // Int8: Max(dim, hidden_dim) = hidden_dim
    const int INT8_POOL_SIZE = hidden_dim;
    int8_t int8_pool[INT8_POOL_SIZE];
    
    // Scales: 1 per row (row=1) -> size 2 (safe)
    float scale_pool[2];

    #pragma HLS ARRAY_PARTITION variable=pool cyclic factor=32
    #pragma HLS ARRAY_PARTITION variable=int8_pool cyclic factor=64

    ReadInitialInput:
    for (int i = 0; i < dim; ++i) {
    #pragma HLS PIPELINE II=1
        current_x[i] = stream_initial_in.read();
    }

    LayerLoop:
    for (int l = 0; l < n_layers; ++l) {
        // --- Attention Phase ---
        prepare_attn_input(l, current_x, int8_pool, scale_pool, w_p0);
        
        // Dispatch QKV Matmuls (All using Unified Pools)
        // Pointers into pool
        float* q_ptr = &pool[0];
        float* k_ptr = &pool[dim];
        float* v_ptr = &pool[dim + kv_dim];
        
        matmul_engine_q(q_ptr, int8_pool, scale_pool,
                        w_p0, w_p1, w_p2, w_p3, w_p4, w_p5, w_p6, w_p7,
                        w_p8, w_p9, w_p10, w_p11, w_p12, w_p13, w_p14, w_p15,
                        w_p16, w_p17, w_p18, w_p19, w_p20, w_p21, w_p22, w_p23,
                        w_p24, w_p25, w_p26, w_p27, w_p28, w_p29, w_p30, w_p31,
                        l, 1, dim, dim);
        matmul_engine_k(k_ptr, int8_pool, scale_pool,
                        w_p0, w_p1, w_p2, w_p3, w_p4, w_p5, w_p6, w_p7,
                        w_p8, w_p9, w_p10, w_p11, w_p12, w_p13, w_p14, w_p15,
                        w_p16, w_p17, w_p18, w_p19, w_p20, w_p21, w_p22, w_p23,
                        w_p24, w_p25, w_p26, w_p27, w_p28, w_p29, w_p30, w_p31,
                        l, 1, kv_dim, dim);
        matmul_engine_v(v_ptr, int8_pool, scale_pool,
                        w_p0, w_p1, w_p2, w_p3, w_p4, w_p5, w_p6, w_p7,
                        w_p8, w_p9, w_p10, w_p11, w_p12, w_p13, w_p14, w_p15,
                        w_p16, w_p17, w_p18, w_p19, w_p20, w_p21, w_p22, w_p23,
                        w_p24, w_p25, w_p26, w_p27, w_p28, w_p29, w_p30, w_p31,
                        l, 1, kv_dim, dim);
        
        float* attn_out_ptr = &pool[dim + 2*kv_dim];
        compute_mha(l, pos, q_ptr, k_ptr, v_ptr, attn_out_ptr, key_cache, value_cache);
        
        prepare_wo_input(attn_out_ptr, int8_pool, scale_pool);
        
        float* wo_out_ptr = &pool[0]; // Reuse beginning of pool
        matmul_engine_o(wo_out_ptr, int8_pool, scale_pool,
                        w_p0, w_p1, w_p2, w_p3, w_p4, w_p5, w_p6, w_p7,
                        w_p8, w_p9, w_p10, w_p11, w_p12, w_p13, w_p14, w_p15,
                        w_p16, w_p17, w_p18, w_p19, w_p20, w_p21, w_p22, w_p23,
                        w_p24, w_p25, w_p26, w_p27, w_p28, w_p29, w_p30, w_p31,
                        l, 1, dim, dim);
        
        update_residual(current_x, wo_out_ptr);

        // --- FFN Phase ---
        prepare_ffn_input(l, current_x, int8_pool, scale_pool, w_p0);
        
        float* hb_ptr = &pool[0];
        float* hb2_ptr = &pool[hidden_dim];
        
        matmul_engine_w1(hb_ptr, int8_pool, scale_pool,
                        w_p0, w_p1, w_p2, w_p3, w_p4, w_p5, w_p6, w_p7,
                        w_p8, w_p9, w_p10, w_p11, w_p12, w_p13, w_p14, w_p15,
                        w_p16, w_p17, w_p18, w_p19, w_p20, w_p21, w_p22, w_p23,
                        w_p24, w_p25, w_p26, w_p27, w_p28, w_p29, w_p30, w_p31,
                        l, 1, hidden_dim, dim);
        matmul_engine_w3(hb2_ptr, int8_pool, scale_pool,
                        w_p0, w_p1, w_p2, w_p3, w_p4, w_p5, w_p6, w_p7,
                        w_p8, w_p9, w_p10, w_p11, w_p12, w_p13, w_p14, w_p15,
                        w_p16, w_p17, w_p18, w_p19, w_p20, w_p21, w_p22, w_p23,
                        w_p24, w_p25, w_p26, w_p27, w_p28, w_p29, w_p30, w_p31,
                        l, 1, hidden_dim, dim);
        
        compute_swiglu_prepare_w2(hb_ptr, hb2_ptr, int8_pool, scale_pool);
        
        float* ffn_out_ptr = &pool[2*hidden_dim];
        matmul_engine_w2(ffn_out_ptr, int8_pool, scale_pool,
                        w_p0, w_p1, w_p2, w_p3, w_p4, w_p5, w_p6, w_p7,
                        w_p8, w_p9, w_p10, w_p11, w_p12, w_p13, w_p14, w_p15,
                        w_p16, w_p17, w_p18, w_p19, w_p20, w_p21, w_p22, w_p23,
                        w_p24, w_p25, w_p26, w_p27, w_p28, w_p29, w_p30, w_p31,
                        l, 1, dim, hidden_dim);
        
        update_residual(current_x, ffn_out_ptr);
    }

    WriteFinalOutput:
    for (int i = 0; i < dim; ++i) {
    #pragma HLS PIPELINE II=1
        stream_final_out.write(current_x[i]);
    }
}

extern "C" void transformer_layer_pipeline(
    hls::stream<float>& stream_initial_in,
    hls::stream<float>& stream_final_out,
    const ShardWeights_t* w_p0, const ShardWeights_t* w_p1, const ShardWeights_t* w_p2, const ShardWeights_t* w_p3,
    const ShardWeights_t* w_p4, const ShardWeights_t* w_p5, const ShardWeights_t* w_p6, const ShardWeights_t* w_p7,
    const ShardWeights_t* w_p8, const ShardWeights_t* w_p9, const ShardWeights_t* w_p10, const ShardWeights_t* w_p11,
    const ShardWeights_t* w_p12, const ShardWeights_t* w_p13, const ShardWeights_t* w_p14, const ShardWeights_t* w_p15,
    const ShardWeights_t* w_p16, const ShardWeights_t* w_p17, const ShardWeights_t* w_p18, const ShardWeights_t* w_p19,
    const ShardWeights_t* w_p20, const ShardWeights_t* w_p21, const ShardWeights_t* w_p22, const ShardWeights_t* w_p23,
    const ShardWeights_t* w_p24, const ShardWeights_t* w_p25, const ShardWeights_t* w_p26, const ShardWeights_t* w_p27,
    const ShardWeights_t* w_p28, const ShardWeights_t* w_p29, const ShardWeights_t* w_p30, const ShardWeights_t* w_p31,
    int pos,   
    half* key_cache,
    half* value_cache
) {
#pragma HLS INTERFACE axis port=stream_initial_in name=s_initial_in
#pragma HLS INTERFACE axis port=stream_final_out name=s_final_out
#pragma HLS INTERFACE m_axi port=w_p0 offset=slave bundle=gmem_wq0
#pragma HLS INTERFACE m_axi port=w_p1 offset=slave bundle=gmem_wq1
#pragma HLS INTERFACE m_axi port=w_p2 offset=slave bundle=gmem_wq2
#pragma HLS INTERFACE m_axi port=w_p3 offset=slave bundle=gmem_wq3
#pragma HLS INTERFACE m_axi port=w_p4 offset=slave bundle=gmem_wq4
#pragma HLS INTERFACE m_axi port=w_p5 offset=slave bundle=gmem_wq5
#pragma HLS INTERFACE m_axi port=w_p6 offset=slave bundle=gmem_wq6
#pragma HLS INTERFACE m_axi port=w_p7 offset=slave bundle=gmem_wq7
#pragma HLS INTERFACE m_axi port=w_p8 offset=slave bundle=gmem_wq8
#pragma HLS INTERFACE m_axi port=w_p9 offset=slave bundle=gmem_wq9
#pragma HLS INTERFACE m_axi port=w_p10 offset=slave bundle=gmem_wq10
#pragma HLS INTERFACE m_axi port=w_p11 offset=slave bundle=gmem_wq11
#pragma HLS INTERFACE m_axi port=w_p12 offset=slave bundle=gmem_wq12
#pragma HLS INTERFACE m_axi port=w_p13 offset=slave bundle=gmem_wq13
#pragma HLS INTERFACE m_axi port=w_p14 offset=slave bundle=gmem_wq14
#pragma HLS INTERFACE m_axi port=w_p15 offset=slave bundle=gmem_wq15
#pragma HLS INTERFACE m_axi port=w_p16 offset=slave bundle=gmem_wq16
#pragma HLS INTERFACE m_axi port=w_p17 offset=slave bundle=gmem_wq17
#pragma HLS INTERFACE m_axi port=w_p18 offset=slave bundle=gmem_wq18
#pragma HLS INTERFACE m_axi port=w_p19 offset=slave bundle=gmem_wq19
#pragma HLS INTERFACE m_axi port=w_p20 offset=slave bundle=gmem_wq20
#pragma HLS INTERFACE m_axi port=w_p21 offset=slave bundle=gmem_wq21
#pragma HLS INTERFACE m_axi port=w_p22 offset=slave bundle=gmem_wq22
#pragma HLS INTERFACE m_axi port=w_p23 offset=slave bundle=gmem_wq23
#pragma HLS INTERFACE m_axi port=w_p24 offset=slave bundle=gmem_wq24
#pragma HLS INTERFACE m_axi port=w_p25 offset=slave bundle=gmem_wq25
#pragma HLS INTERFACE m_axi port=w_p26 offset=slave bundle=gmem_wq26
#pragma HLS INTERFACE m_axi port=w_p27 offset=slave bundle=gmem_wq27
#pragma HLS INTERFACE m_axi port=w_p28 offset=slave bundle=gmem_wq28
#pragma HLS INTERFACE m_axi port=w_p29 offset=slave bundle=gmem_wq29
#pragma HLS INTERFACE m_axi port=w_p30 offset=slave bundle=gmem_wq30
#pragma HLS INTERFACE m_axi port=w_p31 offset=slave bundle=gmem_wq31
#pragma HLS INTERFACE m_axi port=key_cache offset=slave bundle=gmem_kvc depth=n_layers*seq_len*kv_dim latency=100 num_read_outstanding=32 num_write_outstanding=32
#pragma HLS INTERFACE m_axi port=value_cache offset=slave bundle=gmem_kvc depth=n_layers*seq_len*kv_dim latency=100 num_read_outstanding=32 num_write_outstanding=32
#pragma HLS INTERFACE s_axilite port=pos       
#pragma HLS INTERFACE s_axilite port=return 

    #pragma HLS ALLOCATION function instances=matmul_engine       limit=1
    #pragma HLS ALLOCATION function instances=matmul_core         limit=1
    #pragma HLS ALLOCATION function instances=matmul_engine_w4a8_32port limit=1

    transformer_layer_scheduler(
        stream_initial_in,
        stream_final_out,
        w_p0, w_p1, w_p2, w_p3, w_p4, w_p5, w_p6, w_p7,
        w_p8, w_p9, w_p10, w_p11, w_p12, w_p13, w_p14, w_p15,
        w_p16, w_p17, w_p18, w_p19, w_p20, w_p21, w_p22, w_p23,
        w_p24, w_p25, w_p26, w_p27, w_p28, w_p29, w_p30, w_p31,
        pos,
        key_cache,
        value_cache);
}

// ============================================================================
// Kernel 4: Final Norm and Classifier
// ============================================================================
extern "C" void
final_norm_classifier(
    hls::stream<float>& stream_in_x,
    float* logits_out,
    const ModelWeights_t* w,
    float GS_val
) {
#pragma HLS INTERFACE axis port=stream_in_x
#pragma HLS INTERFACE m_axi port=logits_out offset=slave bundle=gmem_out depth=vocab_size latency=100 num_write_outstanding=32
#pragma HLS INTERFACE m_axi port=w offset=slave bundle=gmem_w latency=100 num_read_outstanding=32
#pragma HLS INTERFACE s_axilite port=GS_val
#pragma HLS INTERFACE s_axilite port=return

    constexpr int UNROLL_FACTOR = 2;

    float x_local[dim];
    float x_norm[dim];
    // Local buffers for final classifier (small)
    int8_t xq_q[dim];
    float  xq_s[1]; 

#pragma HLS ARRAY_PARTITION variable=x_local type=cyclic factor=UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable=x_norm type=cyclic factor=UNROLL_FACTOR

read_x_stream_final:
    for (int i = 0; i < dim; ++i) {
#pragma HLS PIPELINE II=1
        x_local[i] = stream_in_x.read();
    }
    rmsnorm<dim>(x_norm, x_local, w->rms_final_weight);
    quantize_ptr<dim, 1>(xq_q, xq_s, x_norm);
    
    matmul_engine(
        logits_out,
        xq_q,
        xq_s,
        w->wcls[0].q,
        w->wcls[0].s,
        1,
        vocab_size,
        dim);
}

#ifndef TYPEDEFS_H // It's good practice to ensure header guards match filename
#define TYPEDEFS_H

#include <stdint.h>
#include <stdio.h>
// Include config.h here if GS is needed globally, or ensure it's included before typedefs.h
#include "config.h" // Provides dim, hidden_dim, ..., GS

// Use Vitis HLS half-precision floating type in both synthesis and C simulation
// The header defines the 'half' type directly.
#include "hls_half.h"

// TODO: replace with HLS types if necessary later

//===========================================================================
//  typedefs.h
//===========================================================================
//  @brief: Defines core data structures for the transformer model.

// Configuration structure remains the same
struct Config
{
    int dim;        // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers;   // number of layers
    int n_heads;    // number of query heads
    int n_kv_heads; // number of key/value heads
    int vocab_size; // vocabulary size
    int seq_len;    // max sequence length
    int GS;         // group size for quantization
    int kv_dim;     // Dimension of key/value vectors
};

// --- QuantizedTensor definition ---
template <int col, int row>
struct ActivationQuantizedTensor
{
    static_assert(row * col > 0, "QuantizedTensor SIZE must be positive");

    int8_t q[row * col];             // quantized values
    float s[row]; // scaling factors (one per token)
};

template <int col, int row>
struct WeightQuantizedTensor
{
    static_assert(row * col > 0, "QuantizedTensor SIZE must be positive");

    int8_t q[row * col];             // quantized values
    float s[col]; // scaling factors (one per channel)
};

template <int col, int row, int GS>
struct W4A8Tensor
{
    static_assert(row * col > 0, "W4A8Tensor SIZE must be positive");
    static_assert(row % GS == 0, "Input dimension must be divisible by group size");
    static constexpr int groups_per_row = row / GS;

    int8_t q[row * col / 2];
    int8_t sg[col * groups_per_row];
    float sc[col];
};

// --- UPDATED RunState to use corrected QuantizedTensor ---
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
struct RunState
{
    // current wave of activations
    float x[dim];         // activation at current time stamp (dim,)
    float xb[dim];        // same, but inside a residual branch (dim,)
    float xb2[dim];       // an additional buffer just for convenience (dim,)
    float hb[hidden_dim]; // buffer for hidden dimension in the ffn (hidden_dim,)
    float hb2[hidden_dim];// buffer for hidden dimension in the ffn (hidden_dim,)

    // Pass GS as the GROUP_SIZE template parameter
    ActivationQuantizedTensor<dim, 1> xq;        // quantized x (dim,)
    ActivationQuantizedTensor<hidden_dim, 1> hq; // quantized hb (hidden_dim,)

    float q[dim];                     // query (dim,)
    // Use kv_dim directly from config (assuming it's pre-calculated)
    static constexpr int kv_dim_calc = (dim * n_kv_heads) / n_heads; // Ensure consistency
    float k[kv_dim_calc];             // key (kv_dim,)
    float v[kv_dim_calc];             // value (kv_dim,)

    float att[n_heads * seq_len];     // buffer for scores/attention values (n_heads, seq_len)

    // kv cache
    half key_cache[n_layers * seq_len * kv_dim_calc];   // (layer, seq_len, kv_dim)
    half value_cache[n_layers * seq_len * kv_dim_calc]; // (layer, seq_len, kv_dim)
};

// --- UPDATED TransformerWeights to use corrected QuantizedTensor ---
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
struct TransformerWeights
{
    // Calculate kv_dim locally for template instantiation if needed, or assume it's passed/global
    static constexpr int kv_dim_calc = (dim * n_kv_heads) / n_heads;

    // token embedding table
    // Pass GS as the GROUP_SIZE template parameter
    WeightQuantizedTensor<vocab_size, dim> q_tokens[1]; // (vocab_size, dim) - Size 1 array? Check usage.
    float token_embedding_table[vocab_size * dim];     // same, but dequantized

    // weights for rmsnorms (float, no change needed here)
    float rms_att_weight[n_layers * dim]; // (layer, dim) rmsnorm weights
    float rms_ffn_weight[n_layers * dim]; // (layer, dim)

    // weights for matmuls. Pass GS as GROUP_SIZE template parameter
    // Note: SIZE calculation based on OutputDim * InputDim appears correct.
    W4A8Tensor<dim, dim, GS>                 wq[n_layers]; // (layer, dim, dim)
    W4A8Tensor<kv_dim_calc, dim, GS>         wk[n_layers]; // (layer, kv_dim, dim)
    W4A8Tensor<kv_dim_calc, dim, GS>         wv[n_layers]; // (layer, kv_dim, dim)
    W4A8Tensor<dim, dim, GS>                 wo[n_layers]; // (layer, dim, dim)

    // weights for ffn. Pass GS as GROUP_SIZE template parameter
    W4A8Tensor<hidden_dim, dim, GS>          w1[n_layers]; // (layer, hidden_dim, dim)
    W4A8Tensor<dim, hidden_dim, GS>          w2[n_layers]; // (layer, dim, hidden_dim)
    W4A8Tensor<hidden_dim, dim, GS>          w3[n_layers]; // (layer, hidden_dim, dim)

    // final rmsnorm (float, no change needed here)
    float rms_final_weight[dim]; // (dim,)

    // classifier weights. Pass GS as GROUP_SIZE template parameter
    WeightQuantizedTensor<vocab_size, dim>   wcls[1]; // Size 1 array? Check usage.
};

// Sharded weights structure for distributing matmul weights across multiple HBM channels
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads,
          int vocab_size, int seq_len, int GS, int SHARDS>
struct MatmulShardWeights
{
    static constexpr int kv_dim_calc = (dim * n_kv_heads) / n_heads;

    static_assert(SHARDS > 0, "SHARDS must be positive");
    static_assert(dim % SHARDS == 0, "dim must be divisible by SHARDS");
    static_assert(hidden_dim % SHARDS == 0, "hidden_dim must be divisible by SHARDS");
    static_assert(kv_dim_calc % SHARDS == 0, "kv_dim must be divisible by SHARDS");

    static constexpr int dim_shard        = dim / SHARDS;
    static constexpr int hidden_dim_shard = hidden_dim / SHARDS;
    static constexpr int kv_dim_shard     = kv_dim_calc / SHARDS;

    float rms_att_weight[n_layers * dim];
    float rms_ffn_weight[n_layers * dim];

    W4A8Tensor<dim_shard,        dim,        GS> wq[n_layers];
    W4A8Tensor<kv_dim_shard,     dim,        GS> wk[n_layers];
    W4A8Tensor<kv_dim_shard,     dim,        GS> wv[n_layers];
    W4A8Tensor<dim_shard,        dim,        GS> wo[n_layers];

    W4A8Tensor<hidden_dim_shard, dim,        GS> w1[n_layers];
    W4A8Tensor<dim_shard,        hidden_dim, GS> w2[n_layers];
    W4A8Tensor<hidden_dim_shard, dim,        GS> w3[n_layers];
};

// ----------------------------------------------------------------------------
// Transformer model structure (no changes needed here)
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
struct Transformer
{
    Config config;                                                                                      // hyperparameters
    TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> weights; // weights
    RunState<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> state;           // activations
    // Removed memory mapping related fields (fd, data, file_size) as they are likely C specific
};

#endif // TYPEDEFS_H
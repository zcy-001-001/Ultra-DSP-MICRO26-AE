/* Inference for Llama-2 Transformer model in pure C, int8 quantized forward pass. */
#include "config.h"   // Defines model parameters (dim, hidden_dim, etc.) - Provides global GS
#include "forward.h"  // Declares kernel functions, helper templates (quantize/dequantize), and types
#include "typedefs.h" // Contains corrected type definitions (QuantizedTensor<SIZE, GROUP_SIZE>, etc.)
#include <cstring>
#include <ctype.h>
#include <fcntl.h>
#include <iostream>
#include <math.h>
#include <hls_math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include "hls_stream.h" // HLS Stream header
#include <string>
#include <cstdlib> // 关键：包含这个头文件以使用 getenv
// No XRT includes for csim
#if defined _WIN32
#include "win.h"
#else
#include <sys/mman.h>
#include <unistd.h>
#endif
// ----------------------------------------------------------------------------
// Globals and Helper Functions

// Softmax (CPU version for sampling - unchanged)
void softmax(float *x, int size) {
    if (size <= 0) return; // Handle empty case
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // Avoid division by zero
    float inv_sum = (sum == 0.0f) ? 0.0f : (1.0f / sum);
    for (int i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}

// --- Host-side loader for per-channel quantized weights ---
// Layout in checkpoint (per tensor): [int8 q[out_dim * in_dim]] then [float s[out_dim]]
// This matches WeightQuantizedTensor<col=out_dim, row=in_dim> where
//   q[row * col] is stored row-major as q[out * in_dim + in]
template <int col, int row>
void init_weight_tensors(void **ptr,
                         WeightQuantizedTensor<col, row> *tensor,
                         int n,          // Number of tensors in the array
                         int size_each,  // Total q elements per tensor (should be col*row)
                         int n_scales)   // Number of scales per tensor (should be col)
{
    void *p = *ptr;
    const int expected_size_each = col * row;
    const int expected_scales    = col;

    if (size_each != expected_size_each || n_scales != expected_scales) {
        fprintf(stderr,
                "Error in init_weight_tensors: mismatched sizes (got size_each=%d,n_scales=%d, expected %d,%d)\n",
                size_each, n_scales, expected_size_each, expected_scales);
        return;
    }

    for (int i = 0; i < n; i++) {
        // Copy quantized values (q)
        std::memcpy(tensor[i].q, p, size_each * sizeof(int8_t));
        p = (int8_t *)p + size_each; // Advance pointer past q data

        // Copy per-channel scale factors (s)
        std::memcpy(tensor[i].s, p, n_scales * sizeof(float));
        p = (float *)p + n_scales; // Advance pointer past s data
    }
    *ptr = p; // Update the original pointer
}

template <int col, int row, int group_size_gs>
void init_w4a8_tensors(void **ptr,
                       W4A8Tensor<col, row, group_size_gs> *tensor,
                       int n)
{
    void *p = *ptr;
    static_assert(row % group_size_gs == 0, "Input dimension must be divisible by group size");
    constexpr int groups_per_row = row / group_size_gs;
    constexpr int n_q4 = row * col / 2;
    constexpr int n_sg = col * groups_per_row;
    constexpr int n_sc = col;

    for (int i = 0; i < n; i++) {
        std::memcpy(tensor[i].q, p, n_q4 * sizeof(int8_t));
        p = (int8_t *)p + n_q4;
        std::memcpy(tensor[i].sg, p, n_sg * sizeof(int8_t));
        p = (int8_t *)p + n_sg;
        std::memcpy(tensor[i].sc, p, n_sc * sizeof(float));
        p = (float *)p + n_sc;
    }

    *ptr = p;
}


template <int col, int row, int group_size_gs>
void copy_w4a8_shard(const W4A8Tensor<col, row, group_size_gs> &src,
                     int shard_id,
                     W4A8Tensor<col / SHARDS, row, group_size_gs> &dst)
{
    static_assert(col % SHARDS == 0, "col must be divisible by SHARDS");
    constexpr int COL_SHARD = col / SHARDS;
    constexpr int bytes_per_row = row / 2;
    constexpr int groups_per_row = row / group_size_gs;

    int col_start = shard_id * COL_SHARD;

    const int8_t *src_q_base = src.q + col_start * bytes_per_row;
    std::memcpy(dst.q, src_q_base, COL_SHARD * bytes_per_row * sizeof(int8_t));

    const int8_t *src_sg_base = src.sg + col_start * groups_per_row;
    std::memcpy(dst.sg, src_sg_base, COL_SHARD * groups_per_row * sizeof(int8_t));

    const float *src_sc_base = src.sc + col_start;
    std::memcpy(dst.sc, src_sc_base, COL_SHARD * sizeof(float));
}


template <int dim_t, int hidden_dim_t, int n_layers_t, int n_heads_t, int n_kv_heads_t,
          int vocab_size_t, int seq_len_t, int group_size_gs>
void build_sharded_weights(
    const TransformerWeights<dim_t, hidden_dim_t, n_layers_t, n_heads_t, n_kv_heads_t,
                             vocab_size_t, seq_len_t, group_size_gs> &full,
    MatmulShardWeights<dim_t, hidden_dim_t, n_layers_t, n_heads_t, n_kv_heads_t,
                       vocab_size_t, seq_len_t, group_size_gs, SHARDS> shards[SHARDS])
{
    static constexpr int kv_dim_calc_t = (dim_t * n_kv_heads_t) / n_heads_t;

    for (int s = 0; s < SHARDS; ++s) {
        std::memcpy(shards[s].rms_att_weight, full.rms_att_weight,
                    n_layers_t * dim_t * sizeof(float));
        std::memcpy(shards[s].rms_ffn_weight, full.rms_ffn_weight,
                    n_layers_t * dim_t * sizeof(float));
    }

    for (int l = 0; l < n_layers_t; ++l) {
        for (int s = 0; s < SHARDS; ++s) {
            copy_w4a8_shard<dim_t,       dim_t,       group_size_gs>(full.wq[l], s, shards[s].wq[l]);
            copy_w4a8_shard<kv_dim_calc_t, dim_t,     group_size_gs>(full.wk[l], s, shards[s].wk[l]);
            copy_w4a8_shard<kv_dim_calc_t, dim_t,     group_size_gs>(full.wv[l], s, shards[s].wv[l]);
            copy_w4a8_shard<dim_t,       dim_t,       group_size_gs>(full.wo[l], s, shards[s].wo[l]);

            copy_w4a8_shard<hidden_dim_t, dim_t,      group_size_gs>(full.w1[l], s, shards[s].w1[l]);
            copy_w4a8_shard<dim_t,        hidden_dim_t, group_size_gs>(full.w2[l], s, shards[s].w2[l]);
            copy_w4a8_shard<hidden_dim_t, dim_t,      group_size_gs>(full.w3[l], s, shards[s].w3[l]);
        }
    }
}


// --- UPDATED memory_map_weights for per-channel weights ---
// Uses distinct template name group_size_gs but does not rely on it for weight layout
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads,
          int vocab_size, int seq_len, int group_size_gs> // group_size_gs kept for consistency
bool memory_map_weights(
    // Use the updated TransformerWeights type which expects GS
    TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                       vocab_size, seq_len, group_size_gs> *w,
    void *ptr, uint8_t shared_classifier)
{
    int head_size = dim / n_heads;
    // Ensure kv_dim calculation matches typedefs.h if used implicitly there
    constexpr int kv_dim_calc = (dim * n_kv_heads) / n_heads;

    // Map float weights (RMSNorm)
    // Export format: all attention norms, then all ffn norms, then final norm (contiguous)
    float *fptr = (float *)ptr;
    std::memcpy(w->rms_att_weight, fptr, n_layers * dim * sizeof(float)); fptr += n_layers * dim;
    std::memcpy(w->rms_ffn_weight, fptr, n_layers * dim * sizeof(float)); fptr += n_layers * dim;
    std::memcpy(w->rms_final_weight, fptr, dim * sizeof(float));          fptr += dim;

    ptr = (void *)fptr; // Update base pointer past float weights

    // Map per-channel quantized token embeddings: q_tokens[1] as WeightQuantizedTensor<vocab_size, dim>
    init_weight_tensors<vocab_size, dim>(&ptr, w->q_tokens, 1, vocab_size * dim, vocab_size);

    // Dequantize token embeddings into the float table used by the embedding kernel
    dequantize<vocab_size, dim>(&(w->q_tokens[0]), w->token_embedding_table);

    // Map Attention weights: W4A8Tensor<out_dim, in_dim, GS>
    init_w4a8_tensors<dim,        dim,        group_size_gs>(&ptr, w->wq, n_layers);
    init_w4a8_tensors<kv_dim_calc, dim,       group_size_gs>(&ptr, w->wk, n_layers);
    init_w4a8_tensors<kv_dim_calc, dim,       group_size_gs>(&ptr, w->wv, n_layers);
    init_w4a8_tensors<dim,        dim,        group_size_gs>(&ptr, w->wo, n_layers);

    // Map FFN weights
    init_w4a8_tensors<hidden_dim, dim,        group_size_gs>(&ptr, w->w1, n_layers);
    init_w4a8_tensors<dim,       hidden_dim,  group_size_gs>(&ptr, w->w2, n_layers);
    init_w4a8_tensors<hidden_dim, dim,        group_size_gs>(&ptr, w->w3, n_layers);

    // Map Classifier weights
    if (shared_classifier) {
        // Share quantized embedding weights with classifier when requested
        w->wcls[0] = w->q_tokens[0];
    } else {
        init_weight_tensors<vocab_size, dim>(&ptr, w->wcls, 1, vocab_size * dim, vocab_size);
    }
    return true;
}

// --- UPDATED read_checkpoint ---
// Uses distinct template name group_size_gs
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads,
          int vocab_size, int seq_len, int group_size_gs> // Use distinct template name for GS
bool read_checkpoint(
    std::string checkpoint, Config *config_out,
    // Use the updated TransformerWeights type which expects GS
    TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                       vocab_size, seq_len, group_size_gs> *weights)
{
    FILE *file = fopen(checkpoint.c_str(), "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint.c_str()); return false; }

    uint32_t magic_number;
    if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) { fclose(file); return false; }
    if (magic_number != 0x616b3432) { fprintf(stderr, "Bad magic number\n"); fclose(file); return false; }

    int version;
    if (fread(&version, sizeof(int), 1, file) != 1) { fclose(file); return false; }
    if (version != 3) { fprintf(stderr, "Bad version %d, need version 3\n", version); fclose(file); return false; }

    // Read config fields individually to avoid potential struct packing issues
    // Assuming Config struct layout matches file header:
    if (fread(&config_out->dim, sizeof(int), 1, file) != 1) return false;
    if (fread(&config_out->hidden_dim, sizeof(int), 1, file) != 1) return false;
    if (fread(&config_out->n_layers, sizeof(int), 1, file) != 1) return false;
    if (fread(&config_out->n_heads, sizeof(int), 1, file) != 1) return false;
    if (fread(&config_out->n_kv_heads, sizeof(int), 1, file) != 1) return false;
    if (fread(&config_out->vocab_size, sizeof(int), 1, file) != 1) return false;
    if (fread(&config_out->seq_len, sizeof(int), 1, file) != 1) return false;
    // Skip kv_dim, as it should be calculated

    uint8_t shared_classifier;
    if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) { fclose(file); return false; }

    int group_size;
    if (fread(&group_size, sizeof(int), 1, file) != 1) { fclose(file); return false; }

    if (group_size <= 0) {
        fprintf(stderr,
                "ERROR: Checkpoint must use W4A8 group quantization (group_size>0). Got group_size=%d.\n",
                group_size);
        fclose(file);
        return false;
    }

    // Store runtime group size from checkpoint and validate against compile-time GS
    config_out->GS = group_size;
    if (group_size != group_size_gs) {
        fprintf(stderr,
                "ERROR: Checkpoint group size (%d) does not match compiled GS (%d). Please re-export or recompile.\n",
                group_size, group_size_gs);
        fclose(file);
        return false;
    }
    // Calculate kv_dim based on other config values
    config_out->kv_dim = (config_out->dim * config_out->n_kv_heads) / config_out->n_heads;

    // --- Validate checkpoint config vs compile-time constants ---
    if (config_out->dim != dim || config_out->hidden_dim != hidden_dim ||
        config_out->n_layers != n_layers || config_out->n_heads != n_heads ||
        config_out->n_kv_heads != n_kv_heads || config_out->vocab_size != vocab_size) {
        fprintf(stderr, "ERROR: Checkpoint config does not match compiled config!\n");
        fprintf(stderr, "  Checkpoint: dim=%d, hidden_dim=%d, n_layers=%d, n_heads=%d, n_kv_heads=%d, vocab_size=%d\n",
                config_out->dim, config_out->hidden_dim, config_out->n_layers,
                config_out->n_heads, config_out->n_kv_heads, config_out->vocab_size);
        fprintf(stderr, "  Compiled:   dim=%d, hidden_dim=%d, n_layers=%d, n_heads=%d, n_kv_heads=%d, vocab_size=%d\n",
                dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size);
        fclose(file);
        return false;
    }

    // Seek past the rest of the header (assuming header size is fixed or read previously)
    int header_size = 28; // Minimal size: magic, version, 7 ints, shared_flag, group_size = 4+4+7*4+1+4 = 41 bytes? Or is it fixed 256?
                          // Using fixed 256 based on original comment structure, adjust if needed.
    header_size = 256;
    fseek(file, header_size, SEEK_SET); // Seek to end of header

    // Memory map the rest of the file
    long current_pos = ftell(file);
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    long weights_size = file_size - current_pos;
    fclose(file); // Close file now, mmap uses fd

    // Use mmap (consider error checking)
    int fd = open(checkpoint.c_str(), O_RDONLY);
    if (fd == -1) { fprintf(stderr, "open failed for mmap!\n"); return false; }
    // Map only the weights part
    void *data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0); // Map whole file for simplicity
    if (data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); close(fd); return false; }
    close(fd); // Close fd after mmap

    // Point to the start of weights data in the mapped region
    void *weights_ptr = ((char *)data) + header_size;

    // Call memory_map_weights with the correct pointer and validated group size
    if (!memory_map_weights<dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                           vocab_size, seq_len, group_size_gs>(weights, weights_ptr, shared_classifier))
    {
        munmap(data, file_size); // Unmap on error
        return false;
    }

    // Unmap memory after weights are presumably copied or used directly if pointers are stored
    // Assuming memory_map_weights copies the data structure contents
    munmap(data, file_size);

    return true;
}

// --- UPDATED build_transformer ---
// Passes group_size_gs template parameter correctly
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads,
          int vocab_size, int seq_len, int group_size_gs> // Use distinct template name for GS
bool build_transformer(
    // Use updated Transformer type
    Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                vocab_size, seq_len, group_size_gs> *t,
    std::string checkpoint_path)
{
    // Call read_checkpoint which now also takes group_size_gs
    return read_checkpoint<dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                           vocab_size, seq_len, group_size_gs>(checkpoint_path, &t->config, &t->weights);
}


// Tokenizer struct and functions (assuming no changes needed)
// ... (Tokenizer code as provided before) ...
typedef struct { char *str; int id; } TokenIndex;
typedef struct { char **vocab; float *vocab_scores; TokenIndex *sorted_vocab; int vocab_size; unsigned int max_token_length; unsigned char byte_pieces[512]; } Tokenizer;
int compare_tokens(const void *a, const void *b) { return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str); }
bool build_tokenizer(Tokenizer *t, std::string tokenizer_path, int vocab_size) {/* ... implementation ... */
    t->vocab_size = vocab_size;
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // Initialize to NULL for safety
    FILE *file = fopen(tokenizer_path.c_str(), "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path.c_str()); free(t->vocab); free(t->vocab_scores); return false; }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); fclose(file); free(t->vocab); free(t->vocab_scores); return false; }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { /* error handling */ fclose(file); /* free */ return false; }
        if (fread(&len, sizeof(int), 1, file) != 1) { /* error handling */ fclose(file); /* free */ return false; }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { /* error handling */ fclose(file); /* free */ return false; }
        t->vocab[i][len] = '\0';
    }
    fclose(file);
    t->sorted_vocab = (TokenIndex *)malloc(vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < vocab_size; i++) { t->sorted_vocab[i].str = t->vocab[i]; t->sorted_vocab[i].id = i; }
    qsort(t->sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    for (int i = 0; i < 256; i++) { t->byte_pieces[i * 2] = (unsigned char)i; t->byte_pieces[i * 2 + 1] = '\0'; }
    return true;
 }
void free_tokenizer(Tokenizer *t) { if (!t) return; for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); } free(t->vocab); free(t->vocab_scores); free(t->sorted_vocab); }
char* decode(Tokenizer *t, int prev_token, int token) { if (token < 0 || token >= t->vocab_size) return NULL; char *piece = t->vocab[token]; if (prev_token == 1 && piece[0] == ' ') { piece++; } return piece; }
void safe_printf(char *piece) { if (piece == NULL || piece[0] == '\0') { return; } if (piece[1] == '\0') { unsigned char byte_val = piece[0]; if (!(isprint(byte_val) || byte_val == '\n')) { printf("?"); } else { printf("%c", byte_val); } } else { printf("%s", piece); } }
int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) { if (!str || !sorted_vocab) return -1; TokenIndex tok = {.str = str}; TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens); return res != NULL ? res->id : -1; }
void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) { /* ... implementation ... */
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }
    size_t text_len = strlen(text);
    int *str_buffer = (int *)malloc((text_len + 1) * sizeof(int)); // Use size_t for strlen result
    if (!str_buffer) { fprintf(stderr, "malloc failed\n"); exit(EXIT_FAILURE); }
    int str_len = 0;
    if (bos) tokens[(*n_tokens)++] = 1; // Assuming BOS token ID is 1
    for (size_t i = 0; i < text_len; ++i) { // Use size_t for loop
        str_buffer[str_len++] = (unsigned char)(text[i]); // Store bytes directly
    }
    while (1) {
        float best_score = -1e10; int best_id = -1; int best_idx = -1;
        for (int i=0; i < str_len - 1; i++) {
            char merge_candidate[t->max_token_length * 2 + 1]; // Ensure buffer is safe based on max_token_length
            char* piece1 = (str_buffer[i] < 256) ? (char*)t->byte_pieces + str_buffer[i] * 2 : t->vocab[str_buffer[i]];
            char* piece2 = (str_buffer[i+1] < 256) ? (char*)t->byte_pieces + str_buffer[i+1] * 2 : t->vocab[str_buffer[i+1]];
             // Check lengths before snprintf to prevent buffer overflow
            if (strlen(piece1) + strlen(piece2) < sizeof(merge_candidate)) {
               snprintf(merge_candidate, sizeof(merge_candidate), "%s%s", piece1, piece2);
                int id = str_lookup(merge_candidate, t->sorted_vocab, t->vocab_size);
                if (id != -1 && t->vocab_scores[id] > best_score) { best_score = t->vocab_scores[id]; best_id = id; best_idx = i; }
            } // else: handle case where merged token is too long (optional)
        }
        if (best_idx == -1) break;
        str_buffer[best_idx] = best_id;
        for (int i = best_idx+1; i < str_len-1; i++) { str_buffer[i] = str_buffer[i+1]; }
        str_len--;
    }
    for (int i=0; i < str_len; i++) { tokens[(*n_tokens)++] = str_buffer[i]; }
    free(str_buffer);
    if (eos) tokens[(*n_tokens)++] = 2; // Assuming EOS token ID is 2
}

// Sampler struct and functions (assuming no changes needed)
// ... (Sampler code as provided before) ...
typedef struct { float prob; int index; } ProbIndex;
typedef struct { int vocab_size; ProbIndex *probindex; float temperature; float topp; unsigned long long rng_state; } Sampler;
int sample_argmax(float *probabilities, int n) { int max_i = 0; float max_p = probabilities[0]; for (int i = 1; i < n; i++) { if (probabilities[i] > max_p) { max_i = i; max_p = probabilities[i]; } } return max_i; }
int sample_mult(float *probabilities, int n, float coin) { float cdf = 0.0f; for (int i = 0; i < n; i++) { cdf += probabilities[i]; if (coin < cdf) return i; } return n - 1; }
int compare(const void *a, const void *b) { ProbIndex *a_ = (ProbIndex *)a; ProbIndex *b_ = (ProbIndex *)b; if (a_->prob > b_->prob) return -1; if (a_->prob < b_->prob) return 1; return 0; }
int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin) { int n0 = 0; for (int i = 0; i < n; i++) { if (probabilities[i] > 0) { probindex[n0].index = i; probindex[n0].prob = probabilities[i]; n0++; } } if (n0 == 0) return 0; qsort(probindex, n0, sizeof(ProbIndex), compare); float cumulative_prob = 0.0f; int last_idx = n0 - 1; for (int i = 0; i < n0; i++) { cumulative_prob += probindex[i].prob; if (cumulative_prob > topp) { last_idx = i; break; } } float r = coin * cumulative_prob; float cdf = 0.0f; for (int i = 0; i <= last_idx; i++) { cdf += probindex[i].prob; if (r < cdf) return probindex[i].index; } return probindex[last_idx].index; }
void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) { sampler->vocab_size = vocab_size; sampler->temperature = temperature; sampler->topp = topp; sampler->rng_state = rng_seed; sampler->probindex = (ProbIndex *)malloc(vocab_size * sizeof(ProbIndex)); if (!sampler->probindex) { fprintf(stderr, "malloc failed\n"); exit(EXIT_FAILURE); } }
void free_sampler(Sampler *sampler) { if (sampler) free(sampler->probindex); }
unsigned int random_u32(unsigned long long *state) { *state ^= *state >> 12; *state ^= *state << 25; *state ^= *state >> 27; return (*state * 0x2545F4914F6CDD1Dull) >> 32; }
float random_f32(unsigned long long *state) { return (random_u32(state) >> 8) / 16777216.0f; }
int sample(Sampler *sampler, float *logits) { int next; if (sampler->temperature == 0.0f) { next = sample_argmax(logits, sampler->vocab_size); } else { for (int q = 0; q < sampler->vocab_size; q++) { logits[q] /= sampler->temperature; } softmax(logits, sampler->vocab_size); float coin = random_f32(&sampler->rng_state); if (sampler->topp <= 0 || sampler->topp >= 1) { next = sample_mult(logits, sampler->vocab_size, coin); } else { next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin); } } return next; }

// Utilities: time (assuming no changes needed)
long time_in_ms() { struct timespec time; clock_gettime(CLOCK_REALTIME, &time); return time.tv_sec * 1000 + time.tv_nsec / 1000000; }

// ----------------------------------------------------------------------------
// --- UPDATED generate function ---
// Passes group_size_gs template parameter correctly
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads,
          int vocab_size, int seq_len, int group_size_gs> // Use distinct template name for GS
void generate(
    // Use updated Transformer type
    Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                vocab_size, seq_len, group_size_gs> *transformer,
    Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps)
{
    const char *empty_prompt = "";
    if (prompt == NULL) { prompt = (char*)empty_prompt; }

    int num_prompt_tokens = 0;
    // Allocate slightly more space potentially needed for BOS/EOS
    int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int));
    if (!prompt_tokens) { fprintf(stderr, "malloc failed for prompt_tokens\n"); exit(EXIT_FAILURE); }

    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    std::cout << "Encoded prompt into " << num_prompt_tokens << " tokens." << std::endl;
    if (num_prompt_tokens < 1) { fprintf(stderr, "Error: encode() generated no tokens.\n"); free(prompt_tokens); exit(EXIT_FAILURE);} // Exit if encoding fails

    // Allocate token buffer for prompt + generation steps
    int* tokens = (int*)malloc((num_prompt_tokens + steps) * sizeof(int));
    if (!tokens) { fprintf(stderr, "malloc failed for tokens\n"); free(prompt_tokens); exit(EXIT_FAILURE); }
    memcpy(tokens, prompt_tokens, num_prompt_tokens * sizeof(int));
    free(prompt_tokens); // Free temporary prompt buffer

    // Allocate logits buffer
    float* logits = (float*)malloc(vocab_size * sizeof(float));
    if (!logits) { fprintf(stderr, "malloc failed for logits\n"); free(tokens); exit(EXIT_FAILURE); }

    // Calculate KV cache size and allocate (stored in half precision)
    constexpr int kv_dim = (dim * n_kv_heads) / n_heads;
    size_t kv_cache_size = (size_t)n_layers * seq_len * kv_dim; // Use size_t for large allocations
    half* key_cache = (half*)malloc(kv_cache_size * sizeof(half));
    half* value_cache = (half*)malloc(kv_cache_size * sizeof(half));
    if (!key_cache || !value_cache) {
        fprintf(stderr, "malloc failed for KV cache\n");
        free(logits); free(tokens); free(key_cache); /* free value_cache if key_cache succeeded */
        exit(EXIT_FAILURE);
    }
    memset(key_cache, 0, kv_cache_size * sizeof(half));
    memset(value_cache, 0, kv_cache_size * sizeof(half));

    typedef MatmulShardWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                               vocab_size, seq_len, group_size_gs, SHARDS> ShardWeights_t;
    static ShardWeights_t shard_weights[SHARDS];
    build_sharded_weights<dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                          vocab_size, seq_len, group_size_gs>(transformer->weights, shard_weights);

    std::cout << "Running inference for " << steps << " steps using split kernels..." << std::endl;

    hls::stream<float> stream_init_to_pipeline_in("stream_init_to_pipeline_in");
    hls::stream<float> stream_pipeline_out_to_final("stream_pipeline_out_to_final");

    int pos = 0; // Current position in the sequence

// --- Process Prompt Tokens ---
int next_token; // 用于存储第一个生成步骤所需的 token


for (int t = 0; t < num_prompt_tokens; ++t) {
    int current_token = tokens[t];
    std::cout << "Processing prompt token " << t << "/" << num_prompt_tokens << " (pos=" << pos << ")" << std::endl; // Debug

    // 1. Initial Embedding Lookup
    // 将结果写入 pipeline Kernel 的输入流
    initial_embedding_lookup(transformer->weights.token_embedding_table,
                               current_token,
                               stream_init_to_pipeline_in); // 使用连接 pipeline 的流

    // 2. 调用新的合并后的 Kernel，它内部会执行所有 n_layers 的计算
    // 它读取 embedding 的输出，写入 final_norm 的输入
    transformer_layer_pipeline(
        stream_init_to_pipeline_in,     // Kernel 读取此流
        stream_pipeline_out_to_final,   // Kernel 写入此流
        &shard_weights[0], &shard_weights[1], &shard_weights[2], &shard_weights[3], &shard_weights[4], &shard_weights[5], &shard_weights[6], &shard_weights[7],
        &shard_weights[8], &shard_weights[9], &shard_weights[10], &shard_weights[11], &shard_weights[12], &shard_weights[13], &shard_weights[14], &shard_weights[15],
        &shard_weights[16], &shard_weights[17], &shard_weights[18], &shard_weights[19], &shard_weights[20], &shard_weights[21], &shard_weights[22], &shard_weights[23],
        &shard_weights[24], &shard_weights[25], &shard_weights[26], &shard_weights[27], &shard_weights[28], &shard_weights[29], &shard_weights[30], &shard_weights[31],
        pos,                            // 传递当前位置
        transformer->state.key_cache,   // 传递 KV Cache 指针
        transformer->state.value_cache // 传递 KV Cache 指针
    );
    // 3. Final Norm and Classifier
    // 读取 pipeline Kernel 的输出流
    final_norm_classifier(stream_pipeline_out_to_final, // 使用连接 pipeline 的流
                          logits,
                          &transformer->weights,
                          (float)group_size_gs);

    // --- 采样逻辑保持不变 ---
    // 如果是最后一个 prompt token，为第一个生成步骤准备 next_token
    if (t == num_prompt_tokens - 1) {
         // Logits 现在包含了处理完最后一个 prompt token 后对下一个 token 的预测
         next_token = sample(sampler, logits);
    }

    pos++; // 处理完一个 token，位置加一
} // End of prompt processing loop

    std::cout << "\nStarting generation..." << std::endl;
    long start = time_in_ms(); // Start timer after prompt processing

// --- Generate New Tokens (同样使用新的 Kernel 调用流程) ---
for (int t = 0; t < steps; ++t) {

    // 使用上一步采样的 token 作为当前输入
    int current_token = next_token;
    tokens[num_prompt_tokens + t] = current_token; // Store generated token

    // --- 解码和打印 ---
    int prev_token = (num_prompt_tokens + t > 0) ? tokens[num_prompt_tokens + t - 1] : 1; // BOS if first token
    char* piece = decode(tokenizer, prev_token, current_token);
    safe_printf(piece);
    fflush(stdout);

    // --- 如果生成 EOS token 则停止 ---
    if (current_token == 2) { // Assuming EOS token ID is 2
         printf("\n[EOS]");
         break;
    }

    // --- 检查序列长度是否超出限制 ---
    if (pos >= seq_len) {
        printf("\n[SEQUENCE LENGTH LIMIT REACHED]\n");
        break;
    }

    // 1. Initial Embedding Lookup
    // (输入: current_token, 输出到 stream_init_to_pipeline_in)
    initial_embedding_lookup(transformer->weights.token_embedding_table,
                               current_token,
                               stream_init_to_pipeline_in); // 使用之前声明的流

    // 2. 调用新的合并后的 Kernel (它内部执行所有 n_layers 计算)
    // (输入: stream_init_to_pipeline_in, 输出到 stream_pipeline_out_to_final)
    transformer_layer_pipeline(
        stream_init_to_pipeline_in,     // Kernel 读取此流
        stream_pipeline_out_to_final,   // Kernel 写入此流
        &shard_weights[0], &shard_weights[1], &shard_weights[2], &shard_weights[3], &shard_weights[4], &shard_weights[5], &shard_weights[6], &shard_weights[7],
        &shard_weights[8], &shard_weights[9], &shard_weights[10], &shard_weights[11], &shard_weights[12], &shard_weights[13], &shard_weights[14], &shard_weights[15],
        &shard_weights[16], &shard_weights[17], &shard_weights[18], &shard_weights[19], &shard_weights[20], &shard_weights[21], &shard_weights[22], &shard_weights[23],
        &shard_weights[24], &shard_weights[25], &shard_weights[26], &shard_weights[27], &shard_weights[28], &shard_weights[29], &shard_weights[30], &shard_weights[31],
        pos,                            // 传递当前位置
        key_cache,                      // 传递 KV Cache 指针
        value_cache                    // 传递 KV Cache 指针
    );

    // 3. Final Norm and Classifier
    // (输入: stream_pipeline_out_to_final)
    final_norm_classifier(stream_pipeline_out_to_final, // 使用之前声明的流
                          logits,
                          &transformer->weights,
                          (float)group_size_gs);

    // --- 采样下一个 token ---
    next_token = sample(sampler, logits); // 使用修正后的调用

    pos++; // 位置加一，为处理下一个生成的 token 做准备
} // End of generation loop
    printf("\n");

    long end = time_in_ms();
    if (pos > num_prompt_tokens) { // Only report if actual generation happened
        long elapsed_ms = end - start;
        int generated_tokens = pos - num_prompt_tokens;
         if (generated_tokens > 0) {
             float tokens_per_sec = (float)generated_tokens / (elapsed_ms / 1000.0f);
             std::cout << "--------------------------------\n";
             std::cout << "Generated " << generated_tokens << " tokens in " << elapsed_ms << " ms" << std::endl;
             std::cout << "Tokens per second: " << tokens_per_sec << std::endl;
         }
    } else {
         std::cout << "No new tokens generated.\n";
    }


    // Cleanup
    free(tokens);
    free(logits);
    free(key_cache);
    free(value_cache);
    std::cout << "Generation completed successfully." << std::endl;
}

// --- Utilities: read_stdin (assuming no changes needed) ---
void read_stdin(const char *guide, char *buffer, size_t bufsize) { printf("%s", guide); if (fgets(buffer, bufsize, stdin) != NULL) { size_t len = strlen(buffer); if (len > 0 && buffer[len - 1] == '\n') { buffer[len - 1] = '\0'; } } }

// ----------------------------------------------------------------------------
// --- UPDATED main function ---
// Uses global GS from config.h when instantiating/calling templates
// ----------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    std::cout << "Start - Testbench for Split Kernels (Quantized)" << std::endl;

    const char* basePathCStr = getenv("MODEL_BASE_PATH");
    std::string model_base_path(basePathCStr);
    std::string checkpoint_path =model_base_path+ "/model_s8g-1s4g128_w4w8_sg8.bin"; // Default for CSIM
    std::string tokenizer_path = model_base_path+"/tokenizer.bin"; // Default for CSIM
    
    float temperature = 0.0f; // Default to argmax (deterministic) for testing
    float topp = 0.9f;      // Top-p (not used if temp=0)
    int steps = seq_len;    // Default steps = max sequence length
    char *prompt = NULL;    // Default prompt is NULL (empty -> starts with BOS)
    unsigned long long rng_seed = 1234; // Fixed seed for deterministic testing
    const char *mode = "generate"; // Default mode
    char *system_prompt = "Once upon a time"; // Default system prompt

    // --- Argument Parsing (CSIM Friendly) ---
     if (argc >= 2) {
         checkpoint_path = argv[1];
         std::cout << "INFO: Using checkpoint path from argv[1]: " << checkpoint_path << std::endl;
         for (int i = 2; i < argc; i += 2) {
             if (i + 1 >= argc) { std::cerr << "WARN: Option " << argv[i] << " requires a value." << std::endl; break; }
             if (argv[i][0] != '-' || strlen(argv[i]) != 2) { std::cerr << "WARN: Invalid option format '" << argv[i] << "'. Skipping." << std::endl; continue; }
             switch (argv[i][1]) {
                 case 't': temperature = atof(argv[i + 1]); break;
                 case 'p': topp = atof(argv[i + 1]); break;
                 case 's': rng_seed = strtoull(argv[i + 1], NULL, 10); break;
                 case 'n': steps = atoi(argv[i + 1]); break;
                 case 'i': prompt = argv[i + 1]; break;
                 case 'z': tokenizer_path = argv[i + 1]; break;
                 case 'm': mode = argv[i + 1]; break;
                 case 'y': system_prompt = argv[i + 1]; break;
                 default: std::cerr << "WARN: Unknown option '" << argv[i] << "'. Ignoring." << std::endl; break;
             }
         }
     } else {
         std::cout << "INFO: Not enough command line arguments provided (argc=" << argc << "). Using default paths/params for CSIM." << std::endl;
     }


    // --- Parameter Validation/Setup ---
    if (rng_seed == 0) { rng_seed = (unsigned long long)time(NULL); }
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps <= 0 || steps > seq_len) steps = seq_len; // Ensure steps is valid

    // --- Build Transformer ---
    // Instantiate using constants from config.h (including GS)
    static Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                       vocab_size, seq_len, GS> // Use global GS from config.h
        transformer;

    // Call build_transformer which uses the global GS via template arg
    bool weights_loaded = build_transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                                            vocab_size, seq_len, GS>(&transformer, checkpoint_path);
    if (!weights_loaded) {
        fprintf(stderr, "ERROR: failed to load checkpoint weights from %s\n", checkpoint_path.c_str());
        return 1;
    }
     // Check if loaded config GS matches compile-time GS
     if (transformer.config.GS != GS) {
          fprintf(stderr, "ERROR: Config group size (%d) does not match compiled group size (%d)!\n", transformer.config.GS, GS);
          return 1;
     }


    // --- Build Tokenizer ---
    Tokenizer tokenizer;
    bool tokenizer_loaded = build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
    if (!tokenizer_loaded) {
        fprintf(stderr, "ERROR: failed to load tokenizer from %s\n", tokenizer_path.c_str());
        return 1;
    }

    // --- Build Sampler ---
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // --- Run Generation ---
    if (strcmp(mode, "generate") == 0) {
        // Call generate which uses the global GS via template arg
        generate<dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                 vocab_size, seq_len, GS>(&transformer, &tokenizer, &sampler, prompt, steps);
    } else {
        fprintf(stderr, "ERROR: unknown mode: %s\n", mode);
        free_sampler(&sampler);
        free_tokenizer(&tokenizer);
        return 1;
    }

    // --- Cleanup ---
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);

    std::cout << "INFO: Testbench main finished successfully." << std::endl;
    return 0;
}

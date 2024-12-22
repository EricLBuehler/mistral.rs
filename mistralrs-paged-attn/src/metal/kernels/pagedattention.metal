// Updated from MLX commit has f70764a

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

#if defined(__HAVE_BFLOAT__)

typedef bfloat bfloat16_t;

#else

/////////////////////////////////////////////////////////////////////////////
// Helpers
/////////////////////////////////////////////////////////////////////////////

constexpr METAL_FUNC uint16_t float_to_bfloat_bits(float x) {
  // Check for nan
  if ((as_type<uint32_t>(x) & ~_fp_encoding_traits<float>::sign_mask) >
      _fp_encoding_traits<float>::inf_mask) {
    return uint16_t(as_type<uint32_t>(0x7FC0));
  }
  // Take bits
  uint32_t float_bits = as_type<uint32_t>(x);

  // Round to nearest even
  float_bits += ((float_bits >> 16) & 1) + as_type<uint32_t>(0x7FFF);

  // Take upper 16 bits
  return float_bits >> 16;
}

constexpr METAL_FUNC float bfloat_bits_to_float(uint16_t x) {
  // Upper 16 bits are the data and lower 16 bits are 0s
  return as_type<float>((uint32_t)x << 16);
}

struct _MLX_BFloat16;

template <typename T>
static constexpr constant bool can_convert_to_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<T, float>;

template <typename T>
static constexpr constant bool can_convert_from_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<float, T>;

/////////////////////////////////////////////////////////////////////////////
// Bfloat struct
/////////////////////////////////////////////////////////////////////////////

struct _MLX_BFloat16 {
  /////////////////////////////////////////////////////////////////////////////
  // Constructors
  uint16_t bits_;
  _MLX_BFloat16() thread = default;
  _MLX_BFloat16() threadgroup = default;
  _MLX_BFloat16() device = default;
  _MLX_BFloat16() constant = default;

  struct bits_to_bfloat_struct {};
  static constexpr METAL_FUNC bits_to_bfloat_struct bits_to_bfloat() {
    return bits_to_bfloat_struct();
  }
  constexpr METAL_FUNC _MLX_BFloat16(uint16_t bits, bits_to_bfloat_struct)
      : bits_(bits) {}

  /////////////////////////////////////////////////////////////////////////////
  // Conversions to bfloat

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) thread
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) threadgroup
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) device
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) constant
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  /////////////////////////////////////////////////////////////////////////////
  // Conversions from bfloat

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const thread {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const threadgroup {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const device {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const constant {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }
};

/////////////////////////////////////////////////////////////////////////////
// Bfloat operators
/////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
// Unary ops
constexpr METAL_FUNC _MLX_BFloat16 operator-(_MLX_BFloat16 x) {
  return -static_cast<float>(x);
}

/////////////////////////////////////////////////////////////////////////////
// Binary operators
#define bfloat_binop_base(__op__, __operator__, otype, atype, btype, ctype) \
  constexpr METAL_FUNC otype __operator__(atype lhs, btype rhs) {           \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);          \
  }

#define bfloat_binop_helper(__op__, __operator__, otype, itype, ctype)    \
  constexpr METAL_FUNC otype __operator__(_MLX_BFloat16 lhs, itype rhs) { \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);        \
  }                                                                       \
  constexpr METAL_FUNC otype __operator__(itype lhs, _MLX_BFloat16 rhs) { \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);        \
  }

/////////////////////////////////////////////////////////////////////////////
// Arithmetic Operators
#define bfloat_binop(_op_, _operator_)                                       \
  bfloat_binop_base(                                                         \
      _op_, _operator_, _MLX_BFloat16, _MLX_BFloat16, _MLX_BFloat16, float); \
  bfloat_binop_helper(_op_, _operator_, float, float, float);                \
  bfloat_binop_helper(_op_, _operator_, float, half, float);                 \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int32_t, float);      \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint32_t, float);     \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int64_t, float);      \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint64_t, float);

bfloat_binop(+, operator+);
bfloat_binop(-, operator-);
bfloat_binop(*, operator*);
bfloat_binop(/, operator/);

/////////////////////////////////////////////////////////////////////////////
// Comparison ops
#define bfloat_compop(__op__, __operator__)                             \
  bfloat_binop_base(                                                    \
      __op__, __operator__, bool, _MLX_BFloat16, _MLX_BFloat16, float); \
  bfloat_binop_helper(__op__, __operator__, bool, float, float);        \
  bfloat_binop_helper(__op__, __operator__, bool, half, float);         \
  bfloat_binop_helper(__op__, __operator__, bool, int32_t, float);      \
  bfloat_binop_helper(__op__, __operator__, bool, uint32_t, float);     \
  bfloat_binop_helper(__op__, __operator__, bool, int64_t, float);      \
  bfloat_binop_helper(__op__, __operator__, bool, uint64_t, float);

bfloat_compop(>, operator>);
bfloat_compop(<, operator<);
bfloat_compop(>=, operator>=);
bfloat_compop(<=, operator<=);
bfloat_compop(==, operator==);
bfloat_compop(!=, operator!=);

#undef bfloat_compop
#undef bfloat_binop_base
#undef bfloat_binop_helper
#undef bfloat_binop

/////////////////////////////////////////////////////////////////////////////
// Inplace Operators
#define bfloat_inplace_op_helper(__op__, __operator__, itype, addr_space) \
  constexpr METAL_FUNC addr_space _MLX_BFloat16& __operator__(            \
      addr_space _MLX_BFloat16& lhs, itype rhs) {                         \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);         \
    return lhs;                                                           \
  }                                                                       \
  constexpr METAL_FUNC addr_space itype& __operator__(                    \
      addr_space itype& lhs, _MLX_BFloat16 rhs) {                         \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);         \
    return lhs;                                                           \
  }

#define bfloat_inplace_op_addr_space_helper(__op__, __operator__, itype) \
  bfloat_inplace_op_helper(__op__, __operator__, itype, device);         \
  bfloat_inplace_op_helper(__op__, __operator__, itype, thread);         \
  bfloat_inplace_op_helper(__op__, __operator__, itype, threadgroup);

#define bfloat_inplace_op(itype)                             \
  bfloat_inplace_op_addr_space_helper(+, operator+=, itype); \
  bfloat_inplace_op_addr_space_helper(-, operator-=, itype); \
  bfloat_inplace_op_addr_space_helper(*, operator*=, itype); \
  bfloat_inplace_op_addr_space_helper(/, operator/=, itype);

bfloat_inplace_op(float);
bfloat_inplace_op(half);
bfloat_inplace_op(int16_t);
bfloat_inplace_op(int32_t);
bfloat_inplace_op(int64_t);
bfloat_inplace_op(uint16_t);
bfloat_inplace_op(uint32_t);
bfloat_inplace_op(uint64_t);

#undef bfloat_inplace_op_helper
#undef bfloat_inplace_op_addr_space_helper
#undef bfloat_inplace_op

#define bfloat_inplace_op_helper(__op__, __operator__, addr_space) \
  constexpr METAL_FUNC addr_space _MLX_BFloat16& __operator__(     \
      addr_space _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) {          \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);  \
    return lhs;                                                    \
  }

#define bfloat_inplace_op_addr_space_helper(__op__, __operator__) \
  bfloat_inplace_op_helper(__op__, __operator__, device);         \
  bfloat_inplace_op_helper(__op__, __operator__, thread);         \
  bfloat_inplace_op_helper(__op__, __operator__, threadgroup);

bfloat_inplace_op_addr_space_helper(+, operator+=);
bfloat_inplace_op_addr_space_helper(-, operator-=);
bfloat_inplace_op_addr_space_helper(*, operator*=);
bfloat_inplace_op_addr_space_helper(/, operator/=);

#undef bfloat_inplace_op_helper
#undef bfloat_inplace_op_addr_space_helper

/////////////////////////////////////////////////////////////////////////////
// Bfloat typedef
/////////////////////////////////////////////////////////////////////////////

typedef struct _MLX_BFloat16 bfloat16_t;

#endif

// ========================================== Dot product utilities

// TODO(EricLBuehler): optimize with vectorization
template<int THREAD_GROUP_SIZE, int VEC_SIZE, typename T, int N>
inline float qk_dot(device T* q[N], device T* k[N]) {
  // Compute the parallel products then sum for Q*K^T (treat vector lanes separately).
  float qk = 0;
#pragma unroll
  for (int ii = 1; ii < N; ++ii) {
#pragma unroll
    for (int vi = 0; vi < VEC_SIZE; ++vi) {
      qk = fma(float(q[ii][vi]), float(k[ii][vi]), qk);
    }
  }
  // Finalize the reduction across lanes.
#pragma unroll
  for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
    qk += simd_shuffle_xor(qk, mask);
  }
  return qk;
}

template<int VEC_SIZE, typename T>
inline float dot(const threadgroup float* x, device T* y) {
  // Compute the parallel products then sum for Q*K^T (treat vector lanes separately).
  float res = 0;
#pragma unroll
  for (int vi = 0; vi < VEC_SIZE; ++vi) {
    res = fma(x[vi], float(y[vi]), res);
  }
  return res;
}

// ========================================== Block sum utility

// Utility function for attention softmax.
template<int NUM_WARPS, int NUM_SIMD_LANES>
inline float block_sum(threadgroup float* red_smem, float sum, uint simd_tid, uint simd_lid) {
  // Compute the sum per simdgroup.
#pragma unroll
  for (int mask = NUM_SIMD_LANES / 2; mask >= 1; mask /= 2) {
    sum += simd_shuffle_xor(sum, mask);
  }

  // Simd leaders store the data to shared memory.
  if (simd_lid == 0) {
    red_smem[simd_tid] = sum;
  }

  // Make sure the data is in shared memory.
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // The warps compute the final sums.
  if (simd_lid < NUM_WARPS) {
    sum = red_smem[simd_lid];
  }

  // Parallel reduction inside the simd group.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += simd_shuffle_xor(sum, mask);
  }

  // Broadcast to other threads.
  return simd_shuffle(sum, 0);
}

// ========================================== Paged Attention kernel


#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

constant bool use_partitioning [[function_constant(10)]];
constant bool use_alibi [[function_constant(20)]];

template <typename T, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS, int NUM_SIMD_LANES, int PARTITION_SIZE = 0>
[[kernel]] void paged_attention(
    device float* exp_sums [[buffer(0), function_constant(use_partitioning)]],         // [num_seqs, num_heads, max_num_partitions]
    device float* max_logits [[buffer(1), function_constant(use_partitioning)]],       // [num_seqs, num_heads, max_num_partitions]
    device T* out [[buffer(2)]],              // [num_seqs, num_heads, max_num_partitions, head_size]
    device const T* q [[buffer(3)]],          // [num_seqs, num_heads, head_size]
    device const T* k_cache [[buffer(4)]],    // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    device const T* v_cache [[buffer(5)]],    // [num_blocks, num_kv_heads, head_size, block_size]
    const constant int& num_kv_heads [[buffer(6)]],     // [num_heads]
    const constant float& scale [[buffer(7)]],
    const constant float& softcapping [[buffer(8)]],
    device const uint32_t* block_tables [[buffer(9)]],   // [num_seqs, max_num_blocks_per_seq]
    device const uint32_t* context_lens [[buffer(10)]],  // [num_seqs]
    const constant int& max_num_blocks_per_seq [[buffer(11)]],
    device const float* alibi_slopes [[buffer(12), function_constant(use_alibi)]],     // [num_heads]
    const constant int& q_stride [[buffer(13)]],
    const constant int& kv_block_stride [[buffer(14)]],
    const constant int& kv_head_stride [[buffer(15)]],
    threadgroup char* shared_mem [[threadgroup(0)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint simd_tid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
  const int seq_idx = threadgroup_position_in_grid.y;
  const int partition_idx = threadgroup_position_in_grid.z;
  const int max_num_partitions = threadgroups_per_grid.z;
  const int thread_idx = thread_position_in_threadgroup.x;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  static_assert(!USE_PARTITIONING);
  const uint32_t context_len = context_lens[seq_idx];
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= context_len) {
    // No work to do. Terminate the thread block.
    return;
  }

  const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
  const int num_blocks_per_partition = USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_context_blocks;

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  const int start_block_idx = USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx = MIN(start_block_idx + num_blocks_per_partition, num_context_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE, context_len);
  const int num_tokens = end_token_idx - start_token_idx;

  constexpr int THREAD_GROUP_SIZE = MAX(NUM_SIMD_LANES / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS = NUM_THREADS / THREAD_GROUP_SIZE; // Note: This assumes THREAD_GROUP_SIZE divides NUM_THREADS
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP = DIVIDE_ROUND_UP(BLOCK_SIZE, NUM_SIMD_LANES);
  constexpr int NUM_WARPS = NUM_THREADS / NUM_SIMD_LANES;
  const int warp_idx = simd_tid;
  const int lane = simd_lid;

  const int head_idx = threadgroup_position_in_grid.x;
  const int num_heads = threadgroups_per_grid.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float alibi_slope = use_alibi ? 0.f : alibi_slopes[head_idx];

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread group
  // fetch or compute 16 bytes at a time.
  // For example, if the size of a thread group is 4 and the data type is half,
  // then the vector size is 16 / (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(T)), 1);

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the thread group size is 4, then the first thread in the group
  // has 0, 4, 8, ... th vectors of the query, and the second thread has 1, 5, 9, ...
  // th vectors of the query, and so on.
  const device T* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  device T* q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD; i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] = const_cast<device T*>(q_ptr) + vec_idx * VEC_SIZE;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Use fp32 on softmax logits for better accuracy
  threadgroup float* logits = reinterpret_cast<threadgroup float*>(shared_mem);
  // Workspace for reduction
  threadgroup float red_smem[2 * NUM_WARPS];

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16 / sizeof(T);
  float qk_max = -FLT_MAX;

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  const device uint32_t* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx; block_idx += NUM_WARPS) {
    // NOTE(woosuk): The block number is stored in int32. However, we cast it to int64
    // because int32 can lead to overflow when this variable is multiplied by large numbers
    // (e.g., kv_block_stride).
    const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);

    // Load a key to registers.
    // Each thread in a thread group has a different part of the key.
    // For example, if the thread group size is 4, then the first thread in the group
    // has 0, 4, 8, ... th vectors of the key, and the second thread has 1, 5, 9, ... th
    // vectors of the key, and so on.
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset = (thread_group_idx + i * NUM_SIMD_LANES) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      device T* k_vecs[NUM_VECS_PER_THREAD];

#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const device T* k_ptr = k_cache + physical_block_number * kv_block_stride
                                        + kv_head_idx * kv_head_stride
                                        + physical_block_offset * x;
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;
        k_vecs[j] = const_cast<device T*>(k_ptr) + offset1 * BLOCK_SIZE * x + offset2;
      }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      float qk = scale * qk_dot<THREAD_GROUP_SIZE, VEC_SIZE, T, NUM_VECS_PER_THREAD>(q_vecs[thread_group_offset], k_vecs);
      
      // Apply softcapping
      if (softcapping != 1.0) {
        qk = precise::tanh(qk / softcapping) * softcapping;
      }

      // Add the ALiBi bias if slopes are given.
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - context_len + 1) : 0;

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= context_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : max(qk_max, qk);
      }
    }
  }

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = NUM_SIMD_LANES / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = max(qk_max, simd_shuffle_xor(qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = max(qk_max, simd_shuffle_xor(qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = simd_shuffle(qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = exp(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS, NUM_SIMD_LANES>(&red_smem[NUM_WARPS], exp_sum, simd_tid, simd_lid);

  // Compute softmax.
  const float inv_sum = divide(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // If partitioning is enabled, store the max logit and exp_sum.
  if (USE_PARTITIONING && thread_idx == 0 && use_partitioning) {
    device float* max_logits_ptr = max_logits + seq_idx * num_heads * max_num_partitions
                                       + head_idx * max_num_partitions
                                       + partition_idx;
    *max_logits_ptr = qk_max;
    device float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions
                                   + head_idx * max_num_partitions
                                   + partition_idx;
    *exp_sums_ptr = exp_sum;
  }

  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(T), BLOCK_SIZE);

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = NUM_SIMD_LANES / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD = DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

  // NOTE: We use FP32 for the accumulator for better accuracy.
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  T zero_value = 0;
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx; block_idx += NUM_WARPS) {
    // NOTE: The block number is stored in int32. However, we cast it to int64
    // because int32 can lead to overflow when this variable is multiplied by large numbers
    // (e.g., kv_block_stride).
    const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    threadgroup float* logits_vec = logits + token_idx - start_token_idx;

    const device T* v_ptr = v_cache + physical_block_number * kv_block_stride
                                    + kv_head_idx * kv_head_stride;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        // NOTE: When v_vec contains the tokens that are out of the context,
        // we should explicitly zero out the values since they may contain NaNs.
        // See https://github.com/vllm-project/vllm/issues/641#issuecomment-1682544472
        device T* v_vec = const_cast<device T*>(v_ptr) + offset;
        if (block_idx == num_context_blocks - 1) {
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            v_vec[j] = token_idx + j < context_len ? v_vec[j] : zero_value;
          }
        }
        accs[i] += dot<V_VEC_SIZE, T>(logits_vec, v_vec);
      }
    }
  }

  // Perform reduction within each warp.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += simd_shuffle_xor(acc, mask);
    }
    accs[i] = acc;
  }

  // NOTE: A barrier is required because the shared memory space for logits
  // is reused for the output.
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Perform reduction across warps.
  threadgroup float* out_smem = reinterpret_cast<threadgroup float*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
      threadgroup float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Lower warps update the output.
    if (warp_idx < mid) {
      const threadgroup float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write the final output.
  if (warp_idx == 0) {
    device T* out_ptr = out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                            + head_idx * max_num_partitions * HEAD_SIZE
                            + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        *(out_ptr + row_idx) = float(accs[i]);
      }
    }
  }
}

#define instantiate_paged_attention_inner(type, head_size, block_size, num_threads, num_simd_lanes, partition_size)                                               \
  template [[host_name("paged_attention_" #type "_hs" #head_size "_bs" #block_size "_nt" #num_threads "_nsl" #num_simd_lanes "_ps" #partition_size)]]  \
  [[kernel]] void paged_attention<type, head_size, block_size, num_threads, num_simd_lanes, partition_size>(                                            \
      device float* exp_sums [[buffer(0), function_constant(use_partitioning)]],                                             \
      device float* max_logits [[buffer(1), function_constant(use_partitioning)]],                                            \
      device type* out [[buffer(2)]],                                            \
      device const type* q [[buffer(3)]],                                            \
      device const type* k_cache [[buffer(4)]],                                            \
      device const type* v_cache [[buffer(5)]],                                            \
      const constant int& num_kv_heads [[buffer(6)]],                                          \
      const constant float& scale [[buffer(7)]],                                            \
      const constant float& softcapping [[buffer(8)]],                                            \
      device const uint32_t* block_tables [[buffer(9)]],                                            \
      device const uint32_t* context_lens [[buffer(10)]],                                            \
      const constant int& max_num_blocks_per_seq [[buffer(11)]],                                            \
      device const float* alibi_slopes [[buffer(12), function_constant(use_alibi)]],                                            \
      const constant int& q_stride [[buffer(13)]],                                            \
      const constant int& kv_block_stride [[buffer(14)]],                                            \
      const constant int& kv_head_stride [[buffer(15)]],                                            \
      threadgroup char* shared_mem [[threadgroup(0)]],                                            \
      uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],                                            \
      uint3 threadgroups_per_grid [[threadgroups_per_grid]],                                            \
      uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],                                            \
      uint simd_tid [[simdgroup_index_in_threadgroup]],                                            \
      uint simd_lid [[thread_index_in_simdgroup]]);                                            \


#define instantiate_paged_attention_heads(type, block_size, num_threads, num_simd_lanes, partition_size) \
  instantiate_paged_attention_inner(type, 64, block_size, num_threads, num_simd_lanes, partition_size)         \
  instantiate_paged_attention_inner(type, 80, block_size, num_threads, num_simd_lanes, partition_size)         \
  instantiate_paged_attention_inner(type, 96, block_size, num_threads, num_simd_lanes, partition_size)         \
  instantiate_paged_attention_inner(type, 112, block_size, num_threads, num_simd_lanes, partition_size)         \
  instantiate_paged_attention_inner(type, 128, block_size, num_threads, num_simd_lanes, partition_size)         \
  instantiate_paged_attention_inner(type, 256, block_size, num_threads, num_simd_lanes, partition_size)

#define instantiate_paged_attention_block_size(type, num_threads, num_simd_lanes, partition_size) \
  instantiate_paged_attention_heads(type, 8, num_threads, num_simd_lanes, partition_size)         \
  instantiate_paged_attention_heads(type, 16, num_threads, num_simd_lanes, partition_size)         \
  instantiate_paged_attention_heads(type, 32, num_threads, num_simd_lanes, partition_size)

// TODO: tune num_threads = 128
// NOTE: partition_size = 0
#define instantiate_paged_attention_no_partitioning(type, num_simd_lanes) \
  instantiate_paged_attention_block_size(type, 128, num_simd_lanes, 0)

instantiate_paged_attention_no_partitioning(float, 32)
instantiate_paged_attention_no_partitioning(bfloat16_t, 32)
instantiate_paged_attention_no_partitioning(half, 32)

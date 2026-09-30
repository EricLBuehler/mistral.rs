constexpr int MMQ_COMPACT_MAX_EXPERTS = 1024;
constexpr int MMQ_COMPACT_MAX_ASSIGNMENTS = 640;
constexpr int MMQ_COMPACT_BLOCKS_PER_SM = 2;
constexpr int MMQ_COMPACT_SHARED_INTS = 4;

template <int mmq_x>
static __global__ void mmq_compact_prefix(const int32_t *expert_bounds,
                                         int32_t *tile_prefix,
                                         const int num_experts) {
    int count = 0;
    tile_prefix[0] = count;
    for (int expert = 0; expert < num_experts; ++expert) {
        const int rows = expert_bounds[expert + 1] - expert_bounds[expert];
        count += (rows + mmq_x - 1) / mmq_x;
        tile_prefix[expert + 1] = count;
    }
}

template <ggml_type type, int mmq_x, bool need_check>
static __global__ void mul_mat_q_compact(const mmq_args args,
                                       const int32_t *tile_prefix,
                                       const uint3 blocks_per_ne00) {
    if (mmq_x > get_mmq_x_max_device() || mmq_x % mmq_get_granularity_device(mmq_x) != 0) {
        NO_DEVICE_CODE;
        return;
    }

    constexpr int mmq_y = get_mmq_y_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int activation_ints = sizeof(block_q8_1_mmq) / sizeof(int);
    const int thread = threadIdx.y * warp_size + threadIdx.x;
    const int tile_count = tile_prefix[args.nchannels_y];
    const int nty = (args.nrows_x + mmq_y - 1) / mmq_y;
    const int job_count = tile_count * nty;
    extern __shared__ int ids_dst_shared[];
    __shared__ int job_info[MMQ_COMPACT_SHARED_INTS];

    for (int job = blockIdx.x; job < job_count; job += gridDim.x) {
        if (thread == 0) {
            const int tile = job % tile_count;
            int low = 0;
            int high = args.nchannels_y;
            while (low < high) {
                const int middle = low + (high - low) / 2;
                if (tile_prefix[middle + 1] <= tile) {
                    low = middle + 1;
                } else {
                    high = middle;
                }
            }
            job_info[0] = low;
            job_info[1] = (tile - tile_prefix[low]) * mmq_x;
            job_info[2] = args.expert_bounds[low + 1] - args.expert_bounds[low];
            job_info[3] = job / tile_count;
        }
        __syncthreads();

        const int expert = job_info[0];
        const int column = job_info[1];
        const int rows = job_info[2];
        const int it = job_info[3];
        const int source_row = args.expert_bounds[expert] + column;
        for (int j = thread; j < mmq_x; j += nwarps * warp_size) {
            ids_dst_shared[j] = column + j < rows ? args.ids_dst[source_row + j] : 0;
        }
        __syncthreads();

        const int offset_x = expert * args.stride_channel_x + it * mmq_y * args.stride_row_x;
        const int offset_y = source_row * activation_ints;
        const int64_t offset_dst = int64_t(it) * mmq_y * mmq_type_size(args.type_dst);
        mul_mat_q_process_tile<type, mmq_x, need_check, false>(
            args.x, offset_x, args.y + offset_y, ids_dst_shared,
            (char *)args.dst + offset_dst, args.type_dst, nullptr,
            args.stride_row_x, args.ncols_y, args.nrows_dst,
            args.nrows_x - it * mmq_y - 1, rows - column - 1,
            0, blocks_per_ne00.z, true);
        // Some writeback threads finish early; all readers must leave shared storage before reuse.
        __syncthreads();
    }
}

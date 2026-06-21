// MIT, Copyright (c) ggml-org. Specialized to bf16 / DK=DV=512.

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define N_SIMDWIDTH 32

#define OP_FLASH_ATTN_EXT_NQPSG 8
#define OP_FLASH_ATTN_EXT_NCPSG 64

#define FC_FLASH_ATTN_EXT_PAD 100
#define FC_FLASH_ATTN_EXT_BLK 200
#define FC_FLASH_ATTN_EXT 300

// kargs layouts here must match the Rust #[repr(C)] structs byte-for-byte.
struct flash_attn_ext_pad_kargs {
  int32_t ne11;
  int32_t ne_12_2;
  int32_t ne_12_3;
  uint64_t nb11;
  uint64_t nb12;
  uint64_t nb13;
  uint64_t nb21;
  uint64_t nb22;
  uint64_t nb23;
  int32_t ne31;
  int32_t ne32;
  int32_t ne33;
  uint64_t nb31;
  uint64_t nb32;
  uint64_t nb33;
};

struct flash_attn_ext_blk_kargs {
  int32_t ne01;
  int32_t ne30;
  int32_t ne31;
  int32_t ne32;
  int32_t ne33;
  uint64_t nb31;
  uint64_t nb32;
  uint64_t nb33;
};

constant bool flash_attn_ext_pad_has_mask
    [[function_constant(FC_FLASH_ATTN_EXT_PAD + 0)]];
#define flash_attn_ext_pad_ncpsg OP_FLASH_ATTN_EXT_NCPSG

kernel void kernel_flash_attn_ext_pad(
    constant flash_attn_ext_pad_kargs &args [[buffer(0)]],
    device const char *k [[buffer(1)]], device const char *v [[buffer(2)]],
    device const char *mask [[buffer(3)]], device char *dst [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort3 ntg [[threads_per_threadgroup]]) {
  const int32_t C = flash_attn_ext_pad_ncpsg;

  device char *k_pad = dst;
  device char *v_pad = k_pad + args.nb11 * C * args.ne_12_2 * args.ne_12_3;
  device char *mask_pad = v_pad + args.nb21 * C * args.ne_12_2 * args.ne_12_3;

  const int32_t icp = args.ne11 % C;
  const int32_t ic0 = args.ne11 - icp;

  const int32_t i1 = tgpig[0];
  const int32_t i2 = tgpig[1];
  const int32_t i3 = tgpig[2];

  if (i2 < args.ne_12_2 && i3 < args.ne_12_3) {
    device const char *k_src =
        k + args.nb11 * (ic0 + i1) + args.nb12 * i2 + args.nb13 * i3;
    device const char *v_src =
        v + args.nb21 * (ic0 + i1) + args.nb22 * i2 + args.nb23 * i3;

    device char *k_dst = k_pad + args.nb11 * i1 + args.nb11 * C * i2 +
                         args.nb11 * C * args.ne_12_2 * i3;
    device char *v_dst = v_pad + args.nb21 * i1 + args.nb21 * C * i2 +
                         args.nb21 * C * args.ne_12_2 * i3;

    if (i1 >= icp) {
      // padded slots are masked out in attention; zero for determinism
      for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
        k_dst[i] = 0;
      }
      for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
        v_dst[i] = 0;
      }
    } else {
      for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
        k_dst[i] = k_src[i];
      }
      for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
        v_dst[i] = v_src[i];
      }
    }
  }

  if (flash_attn_ext_pad_has_mask) {
    if (i2 < args.ne32 && i3 < args.ne33) {
      for (int ib = i1; ib < args.ne31; ib += C) {
        device const half *mask_src =
            (device const half *)(mask + args.nb31 * ib + args.nb32 * i2 +
                                  args.nb33 * i3) +
            ic0;
        device half *mask_dst = (device half *)(mask_pad) + C * ib +
                                C * args.ne31 * i2 +
                                C * args.ne31 * args.ne32 * i3;

        for (int i = tiitg; i < C; i += ntg.x) {
          if (i >= icp) {
            mask_dst[i] = -MAXHALF;
          } else {
            mask_dst[i] = mask_src[i];
          }
        }
      }
    }
  }
}

// Tags each QxK block: 0=fully-masked (main kernel skips), 1=needs-compute,
// 2=all-zero.
#define flash_attn_ext_blk_nqptg OP_FLASH_ATTN_EXT_NQPSG
#define flash_attn_ext_blk_ncpsg OP_FLASH_ATTN_EXT_NCPSG

kernel void
kernel_flash_attn_ext_blk(constant flash_attn_ext_blk_kargs &args [[buffer(0)]],
                          device const char *mask [[buffer(1)]],
                          device char *dst [[buffer(2)]],
                          uint3 tgpig [[threadgroup_position_in_grid]],
                          ushort tiisg [[thread_index_in_simdgroup]]) {
  const int32_t Q = flash_attn_ext_blk_nqptg;
  const int32_t C = flash_attn_ext_blk_ncpsg;

  constexpr short NW = N_SIMDWIDTH;

  const int32_t i3 = tgpig[2] / args.ne32;
  const int32_t i2 = tgpig[2] % args.ne32;
  const int32_t i1 = tgpig[1];
  const int32_t i0 = tgpig[0];

  char res = i0 * C + C > args.ne30 ? 1 : 0;

  threadgroup int s_any_unmasked;
  if (tiisg == 0)
    s_any_unmasked = 0;
  simdgroup_barrier(mem_flags::mem_threadgroup);

  for (int jq = 0; jq < Q; ++jq) {
    const int iq = i1 * Q + jq;
    if (iq >= args.ne31)
      break;
    device const half *mp =
        (device const half *)(mask + args.nb31 * iq + args.nb32 * i2 +
                              args.nb33 * i3) +
        i0 * C;
    for (int ic = tiisg; ic < C; ic += NW) {
      if (mp[ic] > -MAXHALF / 2) {
        s_any_unmasked = 1;
      }
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);

  if (tiisg == 0) {
    if (s_any_unmasked) {
      res = 1;
    }
    const int32_t nblk1 = (args.ne01 + Q - 1) / Q;
    const int32_t nblk0 = (args.ne30 + C - 1) / C;
    dst[((i3 * args.ne32 + i2) * nblk1 + i1) * nblk0 + i0] = res;
  }
}

#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))

struct flash_attn_ext_kargs {
  int32_t ne01;
  int32_t ne02;
  int32_t ne03;
  uint64_t nb01;
  uint64_t nb02;
  uint64_t nb03;
  int32_t ne11;
  int32_t ne_12_2;
  int32_t ne_12_3;
  int32_t ns10;
  uint64_t nb11;
  uint64_t nb12;
  uint64_t nb13;
  int32_t ns20;
  uint64_t nb21;
  uint64_t nb22;
  uint64_t nb23;
  int32_t ne31;
  int32_t ne32;
  int32_t ne33;
  uint64_t nb31;
  uint64_t nb32;
  uint64_t nb33;
  int32_t ne1;
  int32_t ne2;
  int32_t ne3;
  float scale;
  float max_bias;
  float m0;
  float m1;
  int32_t n_head_log2;
  float logit_softcap;
};

constant bool flash_attn_ext_has_mask
    [[function_constant(FC_FLASH_ATTN_EXT + 0)]];
constant bool flash_attn_ext_has_kvpad
    [[function_constant(FC_FLASH_ATTN_EXT + 4)]];

#define flash_attn_ext_has_sinks false
#define flash_attn_ext_has_bias false
#define flash_attn_ext_has_softcap false
#define flash_attn_ext_bc_mask false

// K/V are contiguous bf16 [batch, heads, seq, 512]: next-row stride = 512
#define FA_NS10 512
#define FA_NS20 512

// simdgroups per threadgroup
#define FA_NSG 8

// Q/K/V: bfloat. QK accumulator + softmax: float. O accumulator: half.
// Output: bfloat.
kernel void kernel_flash_attn_ext_bf16_dk512_dv512(
    constant flash_attn_ext_kargs &args [[buffer(0)]],
    device const char *q [[buffer(1)]], device const char *k [[buffer(2)]],
    device const char *v [[buffer(3)]], device const char *mask [[buffer(4)]],
    device const char *sinks [[buffer(5)]],
    device const char *pad [[buffer(6)]], device const char *blk [[buffer(7)]],
    device char *dst [[buffer(8)]],
    threadgroup half *shmem_f16 [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  constexpr short NSG = FA_NSG;
  constexpr short DK = 512;
  constexpr short DV = 512;
  constexpr short Q = OP_FLASH_ATTN_EXT_NQPSG; // 8
  constexpr short C = OP_FLASH_ATTN_EXT_NCPSG; // 64

  const ushort iq3 = tgpig[2];
  const ushort iq2 = tgpig[1];
  const ushort iq1 = tgpig[0] * Q;

#define NS10 FA_NS10
#define NS20 FA_NS20

  constexpr short KV = 8;

  constexpr short DK4 = DK / 4;
  constexpr short DK8 = DK / 8;
  constexpr short DV4 = DV / 4;

  constexpr short PV = PAD2(DV, 64);
  constexpr short PV4 = PV / 4;
  constexpr short PV8 = PV / 8;

  constexpr short NW = N_SIMDWIDTH;
  constexpr short NQ = Q / NSG;
  constexpr short SH = 2 * C;
  constexpr short TS = 2 * SH;
  constexpr short T = DK + 2 * PV;

  threadgroup bfloat *sq = (threadgroup bfloat *)(shmem_f16 + 0 * T);
  threadgroup bfloat4 *sq4 = (threadgroup bfloat4 *)(shmem_f16 + 0 * T);
  threadgroup half *so = (threadgroup half *)(shmem_f16 + 0 * T + Q * DK);
  threadgroup half4 *so4 = (threadgroup half4 *)(shmem_f16 + 0 * T + Q * DK);
  threadgroup float *ss = (threadgroup float *)(shmem_f16 + Q * T);
  threadgroup float2 *ss2 = (threadgroup float2 *)(shmem_f16 + Q * T);

  threadgroup bfloat *sk =
      (threadgroup bfloat *)(shmem_f16 + sgitg * (4 * 16 * KV) + Q * T +
                             Q * TS);
  threadgroup bfloat *sv =
      (threadgroup bfloat *)(shmem_f16 + sgitg * (4 * 16 * KV) + Q * T +
                             Q * TS);

  threadgroup half2 *sm2 = (threadgroup half2 *)(shmem_f16 + Q * T + 2 * C);

  device const half2 *pm2[NQ];

  FOR_UNROLL(short jj = 0; jj < NQ; ++jj) {
    const short j = jj * NSG + sgitg;
    pm2[jj] = (device const half2 *)((device const char *)mask +
                                     (iq1 + j) * args.nb31 +
                                     (iq2 % args.ne32) * args.nb32 +
                                     (iq3 % args.ne33) * args.nb33);
  }

  {
    const int32_t nblk1 = ((args.ne01 + Q - 1) / Q);
    const int32_t nblk0 = ((args.ne11 + C - 1) / C);
    blk += (((iq3 % args.ne33) * args.ne32 + (iq2 % args.ne32)) * nblk1 +
            iq1 / Q) *
           nblk0;
  }

  {
    q += iq1 * args.nb01 + iq2 * args.nb02 + iq3 * args.nb03;
    const short ikv2 = iq2 / (args.ne02 / args.ne_12_2);
    const short ikv3 = iq3 / (args.ne03 / args.ne_12_3);
    k += ikv2 * args.nb12 + ikv3 * args.nb13;
    v += ikv2 * args.nb22 + ikv3 * args.nb23;
  }

  FOR_UNROLL(short jj = 0; jj < NQ; ++jj) {
    const short j = jj * NSG + sgitg;
    device const bfloat4 *q4 =
        (device const bfloat4 *)((device const char *)q + j * args.nb01);
    for (short i = tiisg; i < DK4; i += NW) {
      if (iq1 + j < args.ne01) {
        sq4[j * DK4 + i] = q4[i];
      } else {
        sq4[j * DK4 + i] = bfloat4(0);
      }
    }
  }

  FOR_UNROLL(short jj = 0; jj < NQ; ++jj) {
    const short j = jj * NSG + sgitg;
    for (short i = tiisg; i < DV4; i += NW) {
      so4[j * PV4 + i] = half4(0);
    }
    for (short i = tiisg; i < SH; i += NW) {
      ss[j * SH + i] = 0.0f;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  float S[NQ] = {[0 ... NQ - 1] = 0.0f};

  {
    float M[NQ] = {[0 ... NQ - 1] = -FLT_MAX / 2};

    float slope = 1.0f;
    if (flash_attn_ext_has_bias) {
      const short h = iq2;
      const float base = h < args.n_head_log2 ? args.m0 : args.m1;
      const short exph =
          h < args.n_head_log2 ? h + 1 : 2 * (h - args.n_head_log2) + 1;
      slope = pow(base, exph);
    }

    for (int ic0 = 0;; ++ic0) {
      int ic = ic0 * C;
      if (ic >= args.ne11)
        break;

      if (flash_attn_ext_has_kvpad && ic + C > args.ne11) {
        k = pad;
        v = k + args.nb11 * C * args.ne_12_2 * args.ne_12_3;
        mask = v + args.nb21 * C * args.ne_12_2 * args.ne_12_3;

        const short ikv2 = iq2 / (args.ne02 / args.ne_12_2);
        const short ikv3 = iq3 / (args.ne03 / args.ne_12_3);
        k += (ikv2 + ikv3 * args.ne_12_2) * args.nb11 * C;
        v += (ikv2 + ikv3 * args.ne_12_2) * args.nb21 * C;

        if (!flash_attn_ext_has_mask) {
          threadgroup half *sm = (threadgroup half *)(sm2);
          FOR_UNROLL(short jj = 0; jj < NQ; ++jj) {
            const short j = jj * NSG + sgitg;
            for (short i = tiisg; i < C; i += NW) {
              if (ic + i >= args.ne11) {
                sm[2 * j * SH + i] = -MAXHALF;
              }
            }
          }
        } else {
          FOR_UNROLL(short jj = 0; jj < NQ; ++jj) {
            const short j = jj * NSG + sgitg;
            pm2[jj] =
                (device const half2 *)((device const half *)mask +
                                       (iq1 + j) * C +
                                       (iq2 % args.ne32) * (C * args.ne31) +
                                       (iq3 % args.ne33) *
                                           (C * args.ne31 * args.ne32));
          }
        }
        ic = 0;
      }

      char blk_cur = 1;

      if (flash_attn_ext_has_mask) {
        blk_cur = blk[ic0];

        if (blk_cur == 0) {
          FOR_UNROLL(short jj = 0; jj < NQ; ++jj) pm2[jj] += NW;
          continue;
        }

        if (blk_cur == 1) {
          FOR_UNROLL(short jj = 0; jj < NQ; ++jj) {
            const short j = jj * NSG + sgitg;
            if (flash_attn_ext_bc_mask) {
              sm2[j * SH + tiisg] = (iq1 + j) < args.ne31
                                        ? pm2[jj][tiisg]
                                        : half2(-MAXHALF, -MAXHALF);
            } else {
              sm2[j * SH + tiisg] = pm2[jj][tiisg];
            }
            pm2[jj] += NW;
          }
        } else if (blk_cur == 2) {
          FOR_UNROLL(short jj = 0; jj < NQ; ++jj) pm2[jj] += NW;
        }
      }

      {
        device const bfloat *pk = (device const bfloat *)(k + ic * args.nb11);
        threadgroup const bfloat *pq = sq;
        threadgroup float *ps = ss;

        pk += sgitg * (8 * NS10);
        ps += sgitg * 8;

        static_assert((C / 8) % NSG == 0, "");

        constexpr short NC = (C / 8) / NSG;

        FOR_UNROLL(short cc = 0; cc < NC; ++cc) {
          simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);

          simdgroup_bfloat8x8 mk[2];
          simdgroup_bfloat8x8 mq[2];

#pragma unroll(4)
          for (short i = 0; i < DK8 / 2; ++i) {
            simdgroup_barrier(mem_flags::mem_none);

            simdgroup_load(mq[0], pq + 0 * 8 + 16 * i, DK);
            simdgroup_load(mq[1], pq + 1 * 8 + 16 * i, DK);

            simdgroup_load(mk[0], pk + 0 * 8 + 16 * i, NS10, 0, true);
            simdgroup_load(mk[1], pk + 1 * 8 + 16 * i, NS10, 0, true);

            simdgroup_barrier(mem_flags::mem_none);

            simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
            simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
          }

          simdgroup_store(mqk, ps, SH, 0, false);

          pk += 8 * (NSG * NS10);
          ps += 8 * NSG;
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      FOR_UNROLL(short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        const float m = M[jj];

        float2 s2 = ss2[j * SH / 2 + tiisg] * args.scale;

        if (flash_attn_ext_has_softcap) {
          s2 = args.logit_softcap * precise::tanh(s2);
        }

        if (blk_cur != 2) {
          if (flash_attn_ext_has_bias) {
            s2 += float2(sm2[j * SH + tiisg]) * slope;
          } else {
            s2 += float2(sm2[j * SH + tiisg]);
          }
        }

        M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));

        const float ms = exp(m - M[jj]);
        const float2 vs2 = exp(s2 - M[jj]);

        S[jj] = S[jj] * ms + simd_sum(vs2[0] + vs2[1]);
        ss2[j * SH / 2 + tiisg] = vs2;

        if (DV4 % NW == 0) {
          FOR_UNROLL(short ii = 0; ii < DV4 / NW; ++ii) {
            const short i = ii * NW + tiisg;
            so4[j * PV4 + i] *= ms;
          }
        } else {
          for (short i = tiisg; i < DV4; i += NW) {
            so4[j * PV4 + i] *= ms;
          }
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      {
        static_assert(PV8 % NSG == 0, "");
        constexpr short NO = PV8 / NSG;

        simdgroup_half8x8 lo[NO];

        {
          auto sot = so + 8 * sgitg;
          FOR_UNROLL(short ii = 0; ii < NO; ++ii) {
            simdgroup_load(lo[ii], sot, PV, 0, false);
            sot += 8 * NSG;
          }
        }

        {
          device const bfloat *pv = (device const bfloat *)(v + ic * args.nb21);
          pv += 8 * sgitg;

          // DV>64 path: 2x simdgroup-load variant
          constexpr short NC = (C / 8) / 2;

          FOR_UNROLL(short cc = 0; cc < NC; ++cc) {
            simdgroup_float8x8 vs[2];
            simdgroup_load(vs[0], ss + 16 * cc + 0, SH, 0, false);
            simdgroup_load(vs[1], ss + 16 * cc + 8, SH, 0, false);

            FOR_UNROLL(short ii = 0; ii < NO / 2; ++ii) {
              simdgroup_bfloat8x8 mv[4];

              simdgroup_load(mv[0], pv + 0 * NSG + 16 * ii * NSG + 0 * 8 * NS20,
                             NS20, 0, false);
              simdgroup_load(mv[1], pv + 8 * NSG + 16 * ii * NSG + 0 * 8 * NS20,
                             NS20, 0, false);
              simdgroup_load(mv[2], pv + 0 * NSG + 16 * ii * NSG + 1 * 8 * NS20,
                             NS20, 0, false);
              simdgroup_load(mv[3], pv + 8 * NSG + 16 * ii * NSG + 1 * 8 * NS20,
                             NS20, 0, false);

              simdgroup_multiply_accumulate(lo[2 * ii + 0], vs[0], mv[0],
                                            lo[2 * ii + 0]);
              simdgroup_multiply_accumulate(lo[2 * ii + 1], vs[0], mv[1],
                                            lo[2 * ii + 1]);
              simdgroup_multiply_accumulate(lo[2 * ii + 0], vs[1], mv[2],
                                            lo[2 * ii + 0]);
              simdgroup_multiply_accumulate(lo[2 * ii + 1], vs[1], mv[3],
                                            lo[2 * ii + 1]);
            }

            pv += 2 * 8 * NS20;
          }
        }

        {
          auto sot = so + 8 * sgitg;
          FOR_UNROLL(short ii = 0; ii < NO; ++ii) {
            simdgroup_store(lo[ii], sot, PV, 0, false);
            sot += 8 * NSG;
          }
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (flash_attn_ext_has_sinks) {
      FOR_UNROLL(short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        const float m = M[jj];
        const float s =
            tiisg == 0 ? ((device const float *)sinks)[iq2] : -FLT_MAX / 2;
        M[jj] = simd_max(max(M[jj], s));
        const float ms = exp(m - M[jj]);
        const float vs = exp(s - M[jj]);
        S[jj] = S[jj] * ms + simd_sum(vs);
        for (short i = tiisg; i < DV4; i += NW) {
          so4[j * PV4 + i] *= ms;
        }
      }
    }
  }

  for (short jj = 0; jj < NQ; ++jj) {
    const short j = jj * NSG + sgitg;
    if (iq1 + j >= args.ne01)
      break;

    // mistralrs output layout is [b, n_heads, q_seq, DV] (heads before
    // tokens). ggml's FA writes [b, tokens, heads, DV]; swap the iq2 and
    // (iq1+j) factors to land in the right slot here.
    device bfloat4 *dst4 =
        (device bfloat4 *)dst + ((uint64_t)iq3 * args.ne2 * args.ne1 +
                                 (uint64_t)iq2 * args.ne1 + (iq1 + j)) *
                                    DV4;

    const float scale = S[jj] == 0.0f ? 0.0f : 1.0f / S[jj];

    if (DV4 % NW == 0) {
      FOR_UNROLL(short ii = 0; ii < DV4 / NW; ++ii) {
        const short i = ii * NW + tiisg;
        dst4[i] = bfloat4(float4(so4[j * PV4 + i]) * scale);
      }
    } else {
      for (short i = tiisg; i < DV4; i += NW) {
        dst4[i] = bfloat4(float4(so4[j * PV4 + i]) * scale);
      }
    }
  }

#undef NS10
#undef NS20
}

// Vector flash-attn for decode: q_seq=1, DK=DV=512, bf16. Ported from
// llama.cpp's kernel_flash_attn_ext_vec
// (kernel_flash_attn_ext_vec_bf16_hk576_hv512 shape, but with DK=512=DV).
// Specialization choices:
//   NE = 2: each simdgroup handles 2 K cols per pass; threads-per-K = 16.
//   Per-thread O accumulator holds DV4/NL = 128/16 = 8 bfloat4 lanes.
// NSG (simdgroups per threadgroup) is set via threads_per_threadgroup.y at
// dispatch time. Output layout: [b, n_heads, q_seq, DV] (mistralrs convention;
// llama's iq2 vs (iq1+j) factors are swapped here, same fix as the prefill
// kernel above).
kernel void kernel_flash_attn_ext_vec_bf16_dk512_dv512(
    constant flash_attn_ext_kargs &args [[buffer(0)]],
    device const char *q [[buffer(1)]], device const char *k [[buffer(2)]],
    device const char *v [[buffer(3)]], device const char *mask [[buffer(4)]],
    device char *dst [[buffer(8)]],
    threadgroup half *shmem_f16 [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 ntg [[threads_per_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  constexpr short DK = 512;
  constexpr short DV = 512;
  constexpr short NE = 2;
  constexpr short C = 32;
  constexpr short Q = 1;

  constexpr short DK4 = DK / 4;
  constexpr short DV4 = DV / 4;
  constexpr short NW = N_SIMDWIDTH;
  constexpr short NL = NW / NE;
  constexpr short SH = 4 * C;

  const short nsg = ntg.y;
  const short T = DK + nsg * SH;

  const int iq3 = tgpig[2];
  const int iq2 = tgpig[1];
  const int iq1 = tgpig[0];

  threadgroup bfloat4 *sq4 = (threadgroup bfloat4 *)(shmem_f16 + 0 * DK);
  threadgroup float *ss =
      (threadgroup float *)(shmem_f16 + sgitg * SH + Q * DK);
  threadgroup float4 *ss4 =
      (threadgroup float4 *)(shmem_f16 + sgitg * SH + Q * DK);
  threadgroup float *sm =
      (threadgroup float *)(shmem_f16 + sgitg * SH + 2 * C + Q * DK);
  threadgroup bfloat4 *sr4 =
      (threadgroup bfloat4 *)(shmem_f16 + sgitg * DV + Q * T);

  float4 lo[DV4 / NL];

  device const float4 *q4 =
      (device const float4 *)((device const char *)q +
                              (iq1 * args.nb01 + iq2 * args.nb02 +
                               iq3 * args.nb03));

  for (short i = tiisg; i < DK4; i += NW) {
    if (iq1 < args.ne01) {
      sq4[i] = bfloat4(q4[i]);
    } else {
      sq4[i] = bfloat4(0);
    }
  }

  for (short i = 0; i < DV4 / NL; ++i) {
    lo[i] = float4(0.0f);
  }
  for (short i = tiisg; i < SH / 4; i += NW) {
    ss4[i] = float4(0.0f);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  {
    float S = 0.0f;
    float M = -INFINITY / 2.0f;

    const short tx = tiisg % NL;
    const short ty = tiisg / NL;

    const short ikv2 = iq2 / (args.ne02 / args.ne_12_2);
    const short ikv3 = iq3 / (args.ne03 / args.ne_12_3);

    const bool has_mask = mask != q;

    device const half *pm = (device const half *)(mask + iq1 * args.nb31);

    for (int ic0 = 0; ic0 < args.ne11; ic0 += C * nsg) {
      const int ic = ic0 + C * sgitg;
      if (ic >= args.ne11) {
        break;
      }

      if (has_mask) {
        sm[tiisg] = pm[ic + tiisg];
      } else {
        // No causal mask: synthesize bounds-mask for k >= ne11.
        sm[tiisg] = (ic + tiisg < args.ne11) ? 0.0f : -INFINITY;
      }

      if (simd_max(sm[tiisg]) == -INFINITY) {
        continue;
      }

      for (short cc = 0; cc < C / NE; ++cc) {
        float mqk = 0.0f;

        device const bfloat4 *pk =
            (device const bfloat4 *)((device const char *)k +
                                     ((ic + NE * cc + ty) * args.nb11 +
                                      ikv2 * args.nb12 + ikv3 * args.nb13));

#pragma unroll(DK4 / NL)
        for (short ii = 0; ii < DK4; ii += NL) {
          const short i = ii + tx;
          mqk += dot(float4(pk[i]), float4(sq4[i]));
        }

        mqk += simd_shuffle_down(mqk, 8);
        mqk += simd_shuffle_down(mqk, 4);
        mqk += simd_shuffle_down(mqk, 2);
        mqk += simd_shuffle_down(mqk, 1);

        if (tx == 0) {
          mqk *= args.scale;
          // sm holds the actual mask (when present) or the synthesized
          // bounds mask (0 for valid k, -INF for k >= ne11).
          mqk += sm[NE * cc + ty];
          ss[NE * cc + ty] = mqk;
        }
      }

      simdgroup_barrier(mem_flags::mem_threadgroup);

      {
        const float m = M;
        const float s = ss[tiisg];

        M = simd_max(max(M, s));
        const float ms = metal::exp(m - M);
        const float vs = metal::exp(s - M);

        S = S * ms + simd_sum(vs);
        ss[tiisg] = vs;

#pragma unroll(DV4 / NL)
        for (short ii = 0; ii < DV4; ii += NL) {
          lo[ii / NL] *= ms;
        }
      }

      simdgroup_barrier(mem_flags::mem_threadgroup);

      for (short cc = 0; cc < C / NE; ++cc) {
        device const bfloat4 *pv4 =
            (device const bfloat4 *)((device const char *)v +
                                     ((ic + NE * cc + ty) * args.nb21 +
                                      ikv2 * args.nb22 + ikv3 * args.nb23));

        const float4 ms_p = float4(ss[NE * cc + ty]);

#pragma unroll(DV4 / NL)
        for (short ii = 0; ii < DV4; ii += NL) {
          const short i = ii + tx;
          lo[ii / NL] += float4(pv4[i]) * ms_p;
        }
      }
    }

    if (tiisg == 0) {
      ss[0] = S;
      ss[1] = M;
    }
  }

  // simdgroup reduce across NE=2 lanes (offset 16)
  for (short ii = 0; ii < DV4; ii += NL) {
    lo[ii / NL][0] += simd_shuffle_down(lo[ii / NL][0], 16);
    lo[ii / NL][1] += simd_shuffle_down(lo[ii / NL][1], 16);
    lo[ii / NL][2] += simd_shuffle_down(lo[ii / NL][2], 16);
    lo[ii / NL][3] += simd_shuffle_down(lo[ii / NL][3], 16);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (short i = tiisg; i < DV4; i += NL) {
    sr4[i] = bfloat4(lo[i / NL]);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (short r = nsg / 2; r > 0; r >>= 1) {
    if (sgitg < r) {
      const float S0 = ss[0];
      const float S1 =
          ((threadgroup float *)(shmem_f16 + (sgitg + r) * SH + Q * DK))[0];
      const float M0 = ss[1];
      const float M1 =
          ((threadgroup float *)(shmem_f16 + (sgitg + r) * SH + Q * DK))[1];

      const float Mn = max(M0, M1);
      const float ms0 = metal::exp(M0 - Mn);
      const float ms1 = metal::exp(M1 - Mn);
      const float Sn = S0 * ms0 + S1 * ms1;

      if (tiisg == 0) {
        ss[0] = Sn;
        ss[1] = Mn;
      }

      threadgroup bfloat4 *sr_other =
          (threadgroup bfloat4 *)(shmem_f16 + (sgitg + r) * DV + Q * T);
      for (short i = tiisg; i < DV4; i += NW) {
        sr4[i] = bfloat4(float4(sr4[i]) * ms0 + float4(sr_other[i]) * ms1);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Final write. Output layout [b, n_heads, q_seq, DV] (mistralrs convention).
  if (sgitg == 0) {
    const float S = ss[0];
    const float inv = (S == 0.0f) ? 0.0f : (1.0f / S);

    device bfloat4 *dst4 =
        (device bfloat4 *)dst +
        ((uint64_t)iq3 * args.ne2 * args.ne1 + (uint64_t)iq2 * args.ne1 + iq1) *
            DV4;

    for (short i = tiisg; i < DV4; i += NW) {
      dst4[i] = bfloat4(float4(sr4[i]) * inv);
    }
  }
}

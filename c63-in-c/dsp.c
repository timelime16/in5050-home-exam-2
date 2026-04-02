#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <arm_neon.h>

#include "dsp.h"
#include "tables.h"

// neon instructions
static inline float16x8_t row_mat_mul(float16x8_t row, float16x8x4_t mat1, float16x8x4_t mat2) 
{
  float16x8_t buf1, buf2;

  buf1 = vdupq_n_f16(0.0f);
  buf2 = vdupq_n_f16(0.0f);

  buf1 = vfmaq_laneq_f16(buf1, mat1.val[0], row, 0);
  buf2 = vfmaq_laneq_f16(buf2, mat2.val[0], row, 4);

  buf1 = vfmaq_laneq_f16(buf1, mat1.val[1], row, 1);
  buf2 = vfmaq_laneq_f16(buf2, mat2.val[1], row, 5);

  buf1 = vfmaq_laneq_f16(buf1, mat1.val[2], row, 2);
  buf2 = vfmaq_laneq_f16(buf2, mat2.val[2], row, 6);

  buf1 = vfmaq_laneq_f16(buf1, mat1.val[3], row, 3);
  buf2 = vfmaq_laneq_f16(buf2, mat2.val[3], row, 7);

  return vaddq_f16(buf1, buf2);
}

static inline float16_t get_dct_val_from_row(float16x8_t row, uint8_t u) 
{
  switch (u) 
  {
  case 0:
    return vgetq_lane_f16(row, 0);
  case 1:
    return vgetq_lane_f16(row, 1);
  case 2:
    return vgetq_lane_f16(row, 2);
  case 3:
    return vgetq_lane_f16(row, 3);
  case 4:
    return vgetq_lane_f16(row, 4);
  case 5:
    return vgetq_lane_f16(row, 5);
  case 6:
    return vgetq_lane_f16(row, 6);
  default:
    return vgetq_lane_f16(row, 7);
  }
}

static inline float16_t get_dct_val(
  uint8_t u, uint8_t v,
  float16x8_t b0, float16x8_t b1, float16x8_t b2, float16x8_t b3, 
  float16x8_t b4, float16x8_t b5, float16x8_t b6, float16x8_t b7
)
{
  switch (v)
  {
  case 0:
    return get_dct_val_from_row(b0, u);
  case 1:
    return get_dct_val_from_row(b1, u);
  case 2:
    return get_dct_val_from_row(b2, u);
  case 3:
    return get_dct_val_from_row(b3, u);
  case 4:
    return get_dct_val_from_row(b4, u);
  case 5:
    return get_dct_val_from_row(b5, u);
  case 6:
    return get_dct_val_from_row(b6, u);
  default:
    return get_dct_val_from_row(b7, u);
  }
}

static inline float16x8_t set_dct_val(float16x8_t row, float16_t dct, int j) 
{
  switch (j) 
  {
  case 0:
    return vsetq_lane_f16(dct, row, 0);
  case 1:
    return vsetq_lane_f16(dct, row, 1);
  case 2:
    return vsetq_lane_f16(dct, row, 2);
  case 3:
    return vsetq_lane_f16(dct, row, 3);
  case 4:
    return vsetq_lane_f16(dct, row, 4);
  case 5:
    return vsetq_lane_f16(dct, row, 5);
  case 6:
    return vsetq_lane_f16(dct, row, 6);
  default:
    return vsetq_lane_f16(dct, row, 7);
  }
}

void dct_quant_block_8x8_neon(
  float16x8_t b0, float16x8_t b1, float16x8_t b2, float16x8_t b3, 
  float16x8_t b4, float16x8_t b5, float16x8_t b6, float16x8_t b7, 
  float16x8_t q0, float16x8_t q1, float16x8_t q2, float16x8_t q3, 
  float16x8_t q4, float16x8_t q5, float16x8_t q6, float16x8_t q7, 
  float16x8x4_t dct1, float16x8x4_t dct2,
  int16_t *out_data, uint8_t *quant_tbl
)
{
  float16x8x2_t tmp0, tmp1, tmp2, tmp3; // tanspose - 16bit
  float32x4x2_t tmp4, tmp5, tmp6, tmp7; // transpose - 32 bit
  
  // Matrix multiplcation using row-order traversals
  b0 = row_mat_mul(b0, dct1, dct2);
  b1 = row_mat_mul(b1, dct1, dct2);
  b2 = row_mat_mul(b2, dct1, dct2);
  b3 = row_mat_mul(b3, dct1, dct2);
  b4 = row_mat_mul(b4, dct1, dct2);
  b5 = row_mat_mul(b5, dct1, dct2);
  b6 = row_mat_mul(b6, dct1, dct2);
  b7 = row_mat_mul(b7, dct1, dct2);

  // In-place transpose
  tmp0 = vtrnq_f16(b0, b1);
  tmp1 = vtrnq_f16(b2, b3);
  tmp2 = vtrnq_f16(b4, b5);
  tmp3 = vtrnq_f16(b6, b7);
  tmp4 = vtrnq_f32(vreinterpretq_f32_f16(tmp0.val[0]), vreinterpretq_f32_f16(tmp1.val[0]));
  tmp5 = vtrnq_f32(vreinterpretq_f32_f16(tmp0.val[1]), vreinterpretq_f32_f16(tmp1.val[1]));
  tmp6 = vtrnq_f32(vreinterpretq_f32_f16(tmp2.val[0]), vreinterpretq_f32_f16(tmp3.val[0]));
  tmp7 = vtrnq_f32(vreinterpretq_f32_f16(tmp2.val[1]), vreinterpretq_f32_f16(tmp3.val[1]));
  b0 = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(tmp4.val[0]), vget_low_f32(tmp6.val[0])));
  b1 = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(tmp4.val[1]), vget_low_f32(tmp6.val[1])));
  b2 = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(tmp5.val[0]), vget_low_f32(tmp7.val[0])));
  b3 = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(tmp5.val[1]), vget_low_f32(tmp7.val[1])));
  b4 = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(tmp4.val[0]), vget_high_f32(tmp6.val[0])));
  b5 = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(tmp4.val[1]), vget_high_f32(tmp6.val[1])));
  b6 = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(tmp5.val[0]), vget_high_f32(tmp7.val[0])));
  b7 = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(tmp5.val[1]), vget_high_f32(tmp7.val[1])));

  // One more time
  b0 = row_mat_mul(b0, dct1, dct2);
  b1 = row_mat_mul(b1, dct1, dct2);
  b2 = row_mat_mul(b2, dct1, dct2);
  b3 = row_mat_mul(b3, dct1, dct2);
  b4 = row_mat_mul(b4, dct1, dct2);
  b5 = row_mat_mul(b5, dct1, dct2);
  b6 = row_mat_mul(b6, dct1, dct2);
  b7 = row_mat_mul(b7, dct1, dct2);

  // In-place transpose
  tmp0 = vtrnq_f16(b0, b1);
  tmp1 = vtrnq_f16(b2, b3);
  tmp2 = vtrnq_f16(b4, b5);
  tmp3 = vtrnq_f16(b6, b7);
  tmp4 = vtrnq_f32(vreinterpretq_f32_f16(tmp0.val[0]), vreinterpretq_f32_f16(tmp1.val[0]));
  tmp5 = vtrnq_f32(vreinterpretq_f32_f16(tmp0.val[1]), vreinterpretq_f32_f16(tmp1.val[1]));
  tmp6 = vtrnq_f32(vreinterpretq_f32_f16(tmp2.val[0]), vreinterpretq_f32_f16(tmp3.val[0]));
  tmp7 = vtrnq_f32(vreinterpretq_f32_f16(tmp2.val[1]), vreinterpretq_f32_f16(tmp3.val[1]));
  b0 = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(tmp4.val[0]), vget_low_f32(tmp6.val[0])));
  b1 = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(tmp4.val[1]), vget_low_f32(tmp6.val[1])));
  b2 = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(tmp5.val[0]), vget_low_f32(tmp7.val[0])));
  b3 = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(tmp5.val[1]), vget_low_f32(tmp7.val[1])));
  b4 = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(tmp4.val[0]), vget_high_f32(tmp6.val[0])));
  b5 = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(tmp4.val[1]), vget_high_f32(tmp6.val[1])));
  b6 = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(tmp5.val[0]), vget_high_f32(tmp7.val[0])));
  b7 = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(tmp5.val[1]), vget_high_f32(tmp7.val[1])));

  // Scale blocks
  float16x8_t scale_factors_row_0 = {ISQRT2 * ISQRT2, ISQRT2, ISQRT2, ISQRT2, ISQRT2, ISQRT2, ISQRT2, ISQRT2};
  float16x8_t scale_factors_norm = {ISQRT2, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  b0 = vmulq_f16(b0, scale_factors_row_0);
  b1 = vmulq_f16(b1, scale_factors_norm);
  b2 = vmulq_f16(b2, scale_factors_norm);
  b3 = vmulq_f16(b3, scale_factors_norm);
  b4 = vmulq_f16(b4, scale_factors_norm);
  b5 = vmulq_f16(b5, scale_factors_norm);
  b6 = vmulq_f16(b6, scale_factors_norm);
  b7 = vmulq_f16(b7, scale_factors_norm);

  // quantize
  float16x8x4_t out1, out2;
  #pragma unroll
  for (int i = 0; i < 4; ++i) 
  {
    out1.val[i] = vdupq_n_f16(0);
    out2.val[i] = vdupq_n_f16(0);
  }

  for (int i = 0; i < 4; ++i) 
  {
    int zigzag = i*8;
    
    #pragma unroll
    for (int j = 0; j < 8; ++j) 
    {
      uint8_t u = zigzag_U[zigzag + j];
      uint8_t v = zigzag_V[zigzag + j];
      float16_t dct = get_dct_val(u, v, b0, b1, b2, b3, b4, b5, b6, b7);
      out1.val[i] = set_dct_val(out1.val[i], dct, j);
    }
  }
  for (int i = 0; i < 4; ++i) 
  {
    int zigzag = (i+4)*8;
    
    #pragma unroll
    for (int j = 0; j < 8; ++j) 
    {
      uint8_t u = zigzag_U[zigzag + j];
      uint8_t v = zigzag_V[zigzag + j];
      float16_t dct = get_dct_val(u, v, b0, b1, b2, b3, b4, b5, b6, b7);
      out2.val[i] = set_dct_val(out2.val[i], dct, j);
    }
  }

  // Load tables
  float16x8_t quart;
  quart = vdupq_n_f16(0.25f);
  out1.val[0] = vmulq_f16(out1.val[0], vmulq_f16(quart, q0));
  out1.val[1] = vmulq_f16(out1.val[1], vmulq_f16(quart, q1));
  out1.val[2] = vmulq_f16(out1.val[2], vmulq_f16(quart, q2));
  out1.val[3] = vmulq_f16(out1.val[3], vmulq_f16(quart, q3));
  out2.val[0] = vmulq_f16(out2.val[0], vmulq_f16(quart, q4));
  out2.val[1] = vmulq_f16(out2.val[1], vmulq_f16(quart, q5));
  out2.val[2] = vmulq_f16(out2.val[2], vmulq_f16(quart, q6));
  out2.val[3] = vmulq_f16(out2.val[3], vmulq_f16(quart, q7));

  // Store back to memory
  vst1q_s16(out_data, vcvtq_s16_f16(vrndnq_f16(out1.val[0])));
  vst1q_s16(out_data + 8, vcvtq_s16_f16(vrndnq_f16(out1.val[1])));
  vst1q_s16(out_data + 2*8, vcvtq_s16_f16(vrndnq_f16(out1.val[2])));
  vst1q_s16(out_data + 3*8, vcvtq_s16_f16(vrndnq_f16(out1.val[3])));
  vst1q_s16(out_data + 4*8, vcvtq_s16_f16(vrndnq_f16(out2.val[0])));
  vst1q_s16(out_data + 5*8, vcvtq_s16_f16(vrndnq_f16(out2.val[1])));
  vst1q_s16(out_data + 6*8, vcvtq_s16_f16(vrndnq_f16(out2.val[2])));
  vst1q_s16(out_data + 7*8, vcvtq_s16_f16(vrndnq_f16(out2.val[3])));
}

static void dequantize_block_neon(float16_t *in_data, float16_t *out_data,
    uint8_t *quant_tbl)
{
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag)
  {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in_data[zigzag];

    /* Zig-zag and de-quantize */
    out_data[v*8+u] = (float16_t) round((dct * quant_tbl[zigzag]) / 4.0);
  }
}

void dequant_idct_block_8x8_neon(
  int16_t *in_data, uint8_t *out_data, uint8_t *quant_tbl, int w,
  int16x8_t p0, int16x8_t p1, int16x8_t p2, int16x8_t p3, 
  int16x8_t p4, int16x8_t p5, int16x8_t p6, int16x8_t p7
)
{
  float16_t mb[8*8] __attribute((aligned(16)));
  float16_t mb2[8*8] __attribute((aligned(16)));

  int i, v;

  for (i = 0; i < 64; ++i) { mb[i] = in_data[i]; }

  dequantize_block_neon(mb, mb2, quant_tbl);

  float16x8_t r0, r1, r2, r3, r4, r5, r6, r7;
  r0 = vld1q_f16(mb2);
  r1 = vld1q_f16(mb2 + 8);
  r2 = vld1q_f16(mb2 + 2*8);
  r3 = vld1q_f16(mb2 + 3*8);
  r4 = vld1q_f16(mb2 + 4*8);
  r5 = vld1q_f16(mb2 + 5*8);
  r6 = vld1q_f16(mb2 + 6*8);
  r7 = vld1q_f16(mb2 + 7*8);

  // scale
  float16x8_t scale_factors_row_0 = {ISQRT2 * ISQRT2, ISQRT2, ISQRT2, ISQRT2, ISQRT2, ISQRT2, ISQRT2, ISQRT2};
  float16x8_t scale_factors_norm = {ISQRT2, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  r0 = vmulq_f16(r0, scale_factors_row_0);
  r1 = vmulq_f16(r1, scale_factors_norm);
  r2 = vmulq_f16(r2, scale_factors_norm);
  r3 = vmulq_f16(r3, scale_factors_norm);
  r4 = vmulq_f16(r4, scale_factors_norm);
  r5 = vmulq_f16(r5, scale_factors_norm);
  r6 = vmulq_f16(r6, scale_factors_norm);
  r7 = vmulq_f16(r7, scale_factors_norm);

  // idct_1d
  float16x8x4_t dct1, dct2;
  #pragma unroll
  for (int i = 0; i < 4; ++i) 
  {
    dct1.val[i] = vld1q_f16(dctlookup_f16_T[i]);
    dct2.val[i] = vld1q_f16(dctlookup_f16_T[i+4]);
  }

  float16x8_t buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7;
  buf0 = row_mat_mul(r0, dct1, dct2);
  buf1 = row_mat_mul(r1, dct1, dct2);
  buf2 = row_mat_mul(r2, dct1, dct2);
  buf3 = row_mat_mul(r3, dct1, dct2);
  buf4 = row_mat_mul(r4, dct1, dct2);
  buf5 = row_mat_mul(r5, dct1, dct2);
  buf6 = row_mat_mul(r6, dct1, dct2);
  buf7 = row_mat_mul(r7, dct1, dct2);

  float16x8x2_t tmp0, tmp1, tmp2, tmp3; // tanspose - 16bit
  float32x4x2_t tmp4, tmp5, tmp6, tmp7; // transpose - 32 bit
  tmp0 = vtrnq_f16(buf0, buf1);
  tmp1 = vtrnq_f16(buf2, buf3);
  tmp2 = vtrnq_f16(buf4, buf5);
  tmp3 = vtrnq_f16(buf6, buf7);
  tmp4 = vtrnq_f32(vreinterpretq_f32_f16(tmp0.val[0]), vreinterpretq_f32_f16(tmp1.val[0]));
  tmp5 = vtrnq_f32(vreinterpretq_f32_f16(tmp0.val[1]), vreinterpretq_f32_f16(tmp1.val[1]));
  tmp6 = vtrnq_f32(vreinterpretq_f32_f16(tmp2.val[0]), vreinterpretq_f32_f16(tmp3.val[0]));
  tmp7 = vtrnq_f32(vreinterpretq_f32_f16(tmp2.val[1]), vreinterpretq_f32_f16(tmp3.val[1]));
  r0 = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(tmp4.val[0]), vget_low_f32(tmp6.val[0])));
  r1 = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(tmp4.val[1]), vget_low_f32(tmp6.val[1])));
  r2 = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(tmp5.val[0]), vget_low_f32(tmp7.val[0])));
  r3 = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(tmp5.val[1]), vget_low_f32(tmp7.val[1])));
  r4 = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(tmp4.val[0]), vget_high_f32(tmp6.val[0])));
  r5 = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(tmp4.val[1]), vget_high_f32(tmp6.val[1])));
  r6 = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(tmp5.val[0]), vget_high_f32(tmp7.val[0])));
  r7 = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(tmp5.val[1]), vget_high_f32(tmp7.val[1])));

  buf0 = row_mat_mul(r0, dct1, dct2);
  buf1 = row_mat_mul(r1, dct1, dct2);
  buf2 = row_mat_mul(r2, dct1, dct2);
  buf3 = row_mat_mul(r3, dct1, dct2);
  buf4 = row_mat_mul(r4, dct1, dct2);
  buf5 = row_mat_mul(r5, dct1, dct2);
  buf6 = row_mat_mul(r6, dct1, dct2);
  buf7 = row_mat_mul(r7, dct1, dct2);

  tmp0 = vtrnq_f16(buf0, buf1);
  tmp1 = vtrnq_f16(buf2, buf3);
  tmp2 = vtrnq_f16(buf4, buf5);
  tmp3 = vtrnq_f16(buf6, buf7);
  tmp4 = vtrnq_f32(vreinterpretq_f32_f16(tmp0.val[0]), vreinterpretq_f32_f16(tmp1.val[0]));
  tmp5 = vtrnq_f32(vreinterpretq_f32_f16(tmp0.val[1]), vreinterpretq_f32_f16(tmp1.val[1]));
  tmp6 = vtrnq_f32(vreinterpretq_f32_f16(tmp2.val[0]), vreinterpretq_f32_f16(tmp3.val[0]));
  tmp7 = vtrnq_f32(vreinterpretq_f32_f16(tmp2.val[1]), vreinterpretq_f32_f16(tmp3.val[1]));
  r0 = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(tmp4.val[0]), vget_low_f32(tmp6.val[0])));
  r1 = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(tmp4.val[1]), vget_low_f32(tmp6.val[1])));
  r2 = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(tmp5.val[0]), vget_low_f32(tmp7.val[0])));
  r3 = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(tmp5.val[1]), vget_low_f32(tmp7.val[1])));
  r4 = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(tmp4.val[0]), vget_high_f32(tmp6.val[0])));
  r5 = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(tmp4.val[1]), vget_high_f32(tmp6.val[1])));
  r6 = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(tmp5.val[0]), vget_high_f32(tmp7.val[0])));
  r7 = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(tmp5.val[1]), vget_high_f32(tmp7.val[1])));

  // int16_t tmp = block[i*8+j] + (int16_t)prediction[i*w+j+x];
  int16x8_t out0, out1, out2, out3, out4, out5, out6, out7;
  out0 = vaddq_s16(vcvtq_s16_f16(r0), p0);
  out1 = vaddq_s16(vcvtq_s16_f16(r1), p1);
  out2 = vaddq_s16(vcvtq_s16_f16(r2), p2);
  out3 = vaddq_s16(vcvtq_s16_f16(r3), p3);
  out4 = vaddq_s16(vcvtq_s16_f16(r4), p4);
  out5 = vaddq_s16(vcvtq_s16_f16(r5), p5);
  out6 = vaddq_s16(vcvtq_s16_f16(r6), p6);
  out7 = vaddq_s16(vcvtq_s16_f16(r7), p7);

  // if (tmp < 0) { tmp = 0; }
  // else if (tmp > 255) { tmp = 255; }

  // out_data[i*w+j+x] = tmp;
  // vqmovun_s16 -> automatically does min/max
  vst1_u8(out_data, vqmovun_s16(out0));
  vst1_u8(out_data + w, vqmovun_s16(out1));
  vst1_u8(out_data + 2*w, vqmovun_s16(out2));
  vst1_u8(out_data + 3*w, vqmovun_s16(out3));
  vst1_u8(out_data + 4*w, vqmovun_s16(out4));
  vst1_u8(out_data + 5*w, vqmovun_s16(out5));
  vst1_u8(out_data + 6*w, vqmovun_s16(out6));
  vst1_u8(out_data + 7*w, vqmovun_s16(out7));
}


// Keep original for dec
static void transpose_block(float *in_data, float *out_data)
{
  int i, j;

  for (i = 0; i < 8; ++i)
  {
    for (j = 0; j < 8; ++j)
    {
      out_data[i*8+j] = in_data[j*8+i];
    }
  }
}

static void scale_block(float *in_data, float *out_data)
{
  int u, v;

  for (v = 0; v < 8; ++v)
  {
    for (u = 0; u < 8; ++u)
    {
      float a1 = !u ? ISQRT2 : 1.0f;
      float a2 = !v ? ISQRT2 : 1.0f;

      /* Scale according to normalizing function */
      out_data[v*8+u] = in_data[v*8+u] * a1 * a2;
    }
  }
}

static void idct_1d(float *in_data, float *out_data)
{
  int i, j;

  for (i = 0; i < 8; ++i)
  {
    float idct = 0;

    for (j = 0; j < 8; ++j)
    {
      idct += in_data[j] * dctlookup[i][j];
    }

    out_data[i] = idct;
  }
}

static void dequantize_block(float *in_data, float *out_data,
    uint8_t *quant_tbl)
{
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag)
  {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in_data[zigzag];

    /* Zig-zag and de-quantize */
    out_data[v*8+u] = (float) round((dct * quant_tbl[zigzag]) / 4.0);
  }
}

void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
  float mb[8*8] __attribute((aligned(16)));
  float mb2[8*8] __attribute((aligned(16)));

  int i, v;

  for (i = 0; i < 64; ++i) { mb[i] = in_data[i]; }

  dequantize_block(mb, mb2, quant_tbl);
  scale_block(mb2, mb);

  /* Two 1D inverse DCT operations with transpose */
  for (v = 0; v < 8; ++v) { idct_1d(mb+v*8, mb2+v*8); }
  transpose_block(mb2, mb);
  for (v = 0; v < 8; ++v) { idct_1d(mb+v*8, mb2+v*8); }
  transpose_block(mb2, mb);

  for (i = 0; i < 64; ++i) { out_data[i] = mb[i]; }
}









static inline void dct_quant_block_4x8_neon(
  float16x8_t b0, float16x8_t b1,
  float16x8_t b2, float16x8_t b3,
  float16x8_t q0, float16x8_t q1,
  float16x8_t q2, float16x8_t q3,
  float16x8x4_t dct1, float16x8x4_t dct2,
  int16_t *out
)
{
  // ---- Row DCT ----
  b0 = row_mat_mul(b0, dct1, dct2);
  b1 = row_mat_mul(b1, dct1, dct2);
  b2 = row_mat_mul(b2, dct1, dct2);
  b3 = row_mat_mul(b3, dct1, dct2);

  // ---- Partial transpose (4x8) ----
  float16x8x2_t t0 = vtrnq_f16(b0, b1);
  float16x8x2_t t1 = vtrnq_f16(b2, b3);

  float32x4x2_t t2 = vtrnq_f32(
      vreinterpretq_f32_f16(t0.val[0]),
      vreinterpretq_f32_f16(t1.val[0]));

  float32x4x2_t t3 = vtrnq_f32(
      vreinterpretq_f32_f16(t0.val[1]),
      vreinterpretq_f32_f16(t1.val[1]));

  b0 = vreinterpretq_f16_f32(
      vcombine_f32(vget_low_f32(t2.val[0]), vget_low_f32(t3.val[0])));
  b1 = vreinterpretq_f16_f32(
      vcombine_f32(vget_low_f32(t2.val[1]), vget_low_f32(t3.val[1])));
  b2 = vreinterpretq_f16_f32(
      vcombine_f32(vget_high_f32(t2.val[0]), vget_high_f32(t3.val[0])));
  b3 = vreinterpretq_f16_f32(
      vcombine_f32(vget_high_f32(t2.val[1]), vget_high_f32(t3.val[1])));

  // ---- Column DCT ----
  b0 = row_mat_mul(b0, dct1, dct2);
  b1 = row_mat_mul(b1, dct1, dct2);
  b2 = row_mat_mul(b2, dct1, dct2);
  b3 = row_mat_mul(b3, dct1, dct2);

  // ---- Scale ----
  float16x8_t scale0 = {ISQRT2 * ISQRT2, ISQRT2, ISQRT2, ISQRT2,
                        ISQRT2, ISQRT2, ISQRT2, ISQRT2};
  float16x8_t scaleN = {ISQRT2, 1,1,1,1,1,1,1};

  b0 = vmulq_f16(b0, scale0);
  b1 = vmulq_f16(b1, scaleN);
  b2 = vmulq_f16(b2, scaleN);
  b3 = vmulq_f16(b3, scaleN);

  // ---- Quantize (NO zigzag) ----
  float16x8_t quart = vdupq_n_f16(0.25f);

  b0 = vmulq_f16(b0, vmulq_f16(q0, quart));
  b1 = vmulq_f16(b1, vmulq_f16(q1, quart));
  b2 = vmulq_f16(b2, vmulq_f16(q2, quart));
  b3 = vmulq_f16(b3, vmulq_f16(q3, quart));

  // ---- Store ----
  vst1q_s16(out + 0*8, vcvtq_s16_f16(vrndnq_f16(b0)));
  vst1q_s16(out + 1*8, vcvtq_s16_f16(vrndnq_f16(b1)));
  vst1q_s16(out + 2*8, vcvtq_s16_f16(vrndnq_f16(b2)));
  vst1q_s16(out + 3*8, vcvtq_s16_f16(vrndnq_f16(b3)));
}
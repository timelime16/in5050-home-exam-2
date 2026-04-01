#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <arm_neon.h>

#include "dsp.h"
#include "tables.h"

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

static void dct_1d(float *in_data, float *out_data)
{
  int i, j;

  for (i = 0; i < 8; ++i)
  {
    float dct = 0;

    for (j = 0; j < 8; ++j)
    {
      dct += in_data[j] * dctlookup[j][i];
    }

    out_data[i] = dct;
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

static void quantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag)
  {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in_data[v*8+u];

    /* Zig-zag and quantize */
    out_data[zigzag] = (float) round((dct / 4.0) / quant_tbl[zigzag]);
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

void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
  float mb[8*8] __attribute((aligned(16)));
  float mb2[8*8] __attribute((aligned(16)));

  int i, v;

  for (i = 0; i < 64; ++i) { mb2[i] = in_data[i]; }

  /* Two 1D DCT operations with transpose */
  for (v = 0; v < 8; ++v) { dct_1d(mb2+v*8, mb+v*8); }
  transpose_block(mb, mb2);
  for (v = 0; v < 8; ++v) { dct_1d(mb2+v*8, mb+v*8); }
  transpose_block(mb, mb2);

  scale_block(mb2, mb);
  quantize_block(mb, mb2, quant_tbl);

  for (i = 0; i < 64; ++i) { out_data[i] = mb2[i]; }
}


static inline float16x8_t row_mat_mul(float16x8_t row, float16x8x4_t mat1, float16x8x4_t mat2) 
{
  float16x8_t buf1, buf2;

  buf1 = vdupq_n_f16(0.0f);
  buf2 = vdupq_n_f16(0.0f);

  #pragma unroll
  for (int i = 0; i < 4; ++i) 
  {
    buf1 = vfmaq_laneq_f16(buf1, mat1.val[i], row, i);
    buf2 = vfmaq_laneq_f16(buf2, mat2.val[i], row, i+4);
  }
  return vaddq_f16(buf1, buf2);
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
    return vgetq_lane_f16(b0, u);
  case 1:
    return vgetq_lane_f16(b1, u);
  case 2:
    return vgetq_lane_f16(b2, u);
  case 3:
    return vgetq_lane_f16(b3, u);
  case 4:
    return vgetq_lane_f16(b4, u);
  case 5:
    return vgetq_lane_f16(b5, u);
  case 6:
    return vgetq_lane_f16(b6, u);
  default:
    return vgetq_lane_f16(b7, u);
  }
}

void dct_quant_block_8x8_neon(
  float16x8_t b0, float16x8_t b1, float16x8_t b2, float16x8_t b3, 
  float16x8_t b4, float16x8_t b5, float16x8_t b6, float16x8_t b7, 
  int16_t *out_data, uint8_t *quant_tbl
)
{
  // Variable declarations
  float16x8_t buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7; // intermediate buffers
  float16x8x4_t dct1, dct2; // lookup tables
  float16x8x2_t tmp0, tmp1, tmp2, tmp3; // tanspose - 16bit
  float32x4x2_t tmp4, tmp5, tmp6, tmp7; // transpose - 32 bit
  float16x8x4_t out1, out2; // quantize
  float16x8_t q0, q1, q2, q3, q4, q5, q6, q7;
  float16x8_t rp0, rp1, rp2, rp3, rp4, rp5, rp6, rp7;
  float16x8_t quart;

  #pragma unroll
  for (int i = 0; i < 4; ++i) 
  {
    dct1.val[i] = vld1q_f16((float16_t *)dctlookup[i]);
    dct2.val[i] = vld1q_f16((float16_t *)dctlookup[i+4]);
  }
  
  // Matrix multiplcation using row-order traversals
  buf0 = row_mat_mul(b0, dct1, dct2);
  buf1 = row_mat_mul(b1, dct1, dct2);
  buf2 = row_mat_mul(b2, dct1, dct2);
  buf3 = row_mat_mul(b3, dct1, dct2);
  buf4 = row_mat_mul(b4, dct1, dct2);
  buf5 = row_mat_mul(b5, dct1, dct2);
  buf6 = row_mat_mul(b6, dct1, dct2);
  buf7 = row_mat_mul(b7, dct1, dct2);

  // In-place transpose
  tmp0 = vtrnq_f16(buf0, buf1);
  tmp1 = vtrnq_f16(buf2, buf3);
  tmp2 = vtrnq_f16(buf4, buf5);
  tmp3 = vtrnq_f16(buf6, buf7);
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
  buf0 = row_mat_mul(b0, dct1, dct2);
  buf1 = row_mat_mul(b1, dct1, dct2);
  buf2 = row_mat_mul(b2, dct1, dct2);
  buf3 = row_mat_mul(b3, dct1, dct2);
  buf4 = row_mat_mul(b4, dct1, dct2);
  buf5 = row_mat_mul(b5, dct1, dct2);
  buf6 = row_mat_mul(b6, dct1, dct2);
  buf7 = row_mat_mul(b7, dct1, dct2);

  // In-place transpose
  tmp0 = vtrnq_f16(buf0, buf1);
  tmp1 = vtrnq_f16(buf2, buf3);
  tmp2 = vtrnq_f16(buf4, buf5);
  tmp3 = vtrnq_f16(buf6, buf7);
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
  for (int i = 0; i < 4; ++i) 
  {
    int zigzag = i*8;
    
    #pragma unroll
    for (int j = 0; j < 8; ++j) 
    {
      uint8_t u = zigzag_U[zigzag + j];
      uint8_t v = zigzag_V[zigzag + j];
      float16_t dct = get_dct_val(u, v, b0, b1, b2, b3, b4, b5, b6, b7);
      out1.val[i] = vsetq_lane_f16(dct, out1.val[i], j);
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
      out2.val[i] = vsetq_lane_f16(dct, out2.val[i], j);
    }
  }

  // Load tables
  quart = vdupq_n_f16(0.25f);
  q0 = vcvtq_f16_u16(vmovl_u8(vld1_u8(quant_tbl)));
  q1 = vcvtq_f16_u16(vmovl_u8(vld1_u8(quant_tbl + 8)));
  q2 = vcvtq_f16_u16(vmovl_u8(vld1_u8(quant_tbl + 2*8)));
  q3 = vcvtq_f16_u16(vmovl_u8(vld1_u8(quant_tbl + 3*8)));
  q4 = vcvtq_f16_u16(vmovl_u8(vld1_u8(quant_tbl + 4*8)));
  q5 = vcvtq_f16_u16(vmovl_u8(vld1_u8(quant_tbl + 5*8)));
  q6 = vcvtq_f16_u16(vmovl_u8(vld1_u8(quant_tbl + 6*8)));
  q7 = vcvtq_f16_u16(vmovl_u8(vld1_u8(quant_tbl + 7*8)));

  // recip
  rp0 = vrecpeq_f16(q0); rp0 = vmulq_f16(vrecpsq_f16(q0, rp0), rp0);
  rp1 = vrecpeq_f16(q1); rp1 = vmulq_f16(vrecpsq_f16(q1, rp1), rp1);
  rp2 = vrecpeq_f16(q2); rp2 = vmulq_f16(vrecpsq_f16(q2, rp2), rp2);
  rp3 = vrecpeq_f16(q3); rp3 = vmulq_f16(vrecpsq_f16(q3, rp3), rp3);
  rp4 = vrecpeq_f16(q4); rp4 = vmulq_f16(vrecpsq_f16(q4, rp4), rp4);
  rp5 = vrecpeq_f16(q5); rp5 = vmulq_f16(vrecpsq_f16(q5, rp5), rp5);
  rp6 = vrecpeq_f16(q6); rp6 = vmulq_f16(vrecpsq_f16(q6, rp6), rp6);
  rp7 = vrecpeq_f16(q7); rp7 = vmulq_f16(vrecpsq_f16(q7, rp7), rp7);

  out1.val[0] = vmulq_f16(out1.val[0], vmulq_f16(q0, rp0));
  out1.val[1] = vmulq_f16(out1.val[1], vmulq_f16(q1, rp1));
  out1.val[2] = vmulq_f16(out1.val[2], vmulq_f16(q2, rp2));
  out1.val[3] = vmulq_f16(out1.val[3], vmulq_f16(q3, rp3));
  out2.val[0] = vmulq_f16(out2.val[0], vmulq_f16(q4, rp4));
  out2.val[1] = vmulq_f16(out2.val[1], vmulq_f16(q5, rp5));
  out2.val[2] = vmulq_f16(out2.val[2], vmulq_f16(q6, rp6));
  out2.val[3] = vmulq_f16(out2.val[3], vmulq_f16(q7, rp7));

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

void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{
  int u, v;

  *result = 0;

  for (v = 0; v < 8; ++v)
  {
    for (u = 0; u < 8; ++u)
    {
      *result += abs(block2[v*stride+u] - block1[v*stride+u]);
    }
  }
}

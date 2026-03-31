#include <arm_neon.h>
#include <stdio.h>

static void dct_1d(float16_t *in_data, float16_t *out_data, float16_t *dctlookup)
{
  int i, j;

  for (i = 0; i < 8; ++i)
  {
    float dct = 0;

    for (j = 0; j < 8; ++j)
    {
      dct += in_data[j] * dctlookup[j*8+i];
    }

    out_data[i] = dct;
  }
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

static inline void transpose(float16_t *A) 
{

    float16x8_t b0, b1, b2, b3, b4, b5, b6, b7;
    float16x8x2_t tmp0, tmp1, tmp2, tmp3; // tanspose - 16bit
    float32x4x2_t tmp4, tmp5, tmp6, tmp7; // transpose - 32 bit
    b0 = vld1q_f16(A);
    b1 = vld1q_f16(A+8);
    b2 = vld1q_f16(A+2*8);
    b3 = vld1q_f16(A+3*8);
    b4 = vld1q_f16(A+4*8);
    b5 = vld1q_f16(A+5*8);
    b6 = vld1q_f16(A+6*8);
    b7 = vld1q_f16(A+7*8);
    tmp0 = vtrnq_f16(b0, b1);
    tmp1 = vtrnq_f16(b2, b3);
    tmp2 = vtrnq_f16(b4, b5);
    tmp3 = vtrnq_f16(b6, b7);
    tmp4 = vtrnq_f32(vreinterpretq_f32_f16(tmp0.val[0]), vreinterpretq_f32_f16(tmp1.val[0]));
    tmp5 = vtrnq_f32(vreinterpretq_f32_f16(tmp0.val[1]), vreinterpretq_f32_f16(tmp1.val[1]));
    tmp6 = vtrnq_f32(vreinterpretq_f32_f16(tmp2.val[0]), vreinterpretq_f32_f16(tmp3.val[0]));
    tmp7 = vtrnq_f32(vreinterpretq_f32_f16(tmp2.val[1]), vreinterpretq_f32_f16(tmp3.val[1]));
    b0 = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(tmp4.val[0]), vget_low_f32(tmp6.val[0])));
    b1 = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(tmp5.val[0]), vget_low_f32(tmp7.val[0])));
    b2 = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(tmp4.val[1]), vget_low_f32(tmp6.val[1])));
    b3 = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(tmp5.val[1]), vget_low_f32(tmp7.val[1])));
    b4 = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(tmp4.val[0]), vget_high_f32(tmp6.val[0])));
    b5 = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(tmp5.val[0]), vget_high_f32(tmp7.val[0])));
    b6 = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(tmp4.val[1]), vget_high_f32(tmp6.val[1])));
    b7 = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(tmp5.val[1]), vget_high_f32(tmp7.val[1])));
    vst1q_f16(A, b0);
    vst1q_f16(A+8, b1);
    vst1q_f16(A+2*8, b2);
    vst1q_f16(A+3*8, b3);
    vst1q_f16(A+4*8, b4);
    vst1q_f16(A+5*8, b5);
    vst1q_f16(A+6*8, b6);
    vst1q_f16(A+7*8, b7);
}


int main(void) 
{
    // float16_t A[8*8] =
    // {
    // 1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
    // 1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
    // 1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
    // 1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
    // 1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
    // 1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
    // 1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
    // 1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f
    // };

    // for (int i = 0; i < 64; ++i) 
    // {
    //     printf("%f ", (float)A[i]);
    //     if ((i % 8) == 7) printf("\n");
    // }
    // printf("\n");

    // transpose(A);

    // for (int i = 0; i < 64; ++i) 
    // {
    //     printf("%f ", (float)A[i]);
    //     if ((i % 8) == 7) printf("\n");
    // }
    // printf("\n");

    float16_t A[8][8], B[64], C[8][8];
    for (int i = 0; i < 8; ++i) 
    {
        for (int j = 0; j < 8; ++j) 
        {
            A[i][j] = i*8+j;
            B[i*8+j] = i*8+j;
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) 
    {
        dct_1d(A[i], C[i], B);
    }
    printf("Correct answer:\n");
    for (int i = 0; i < 8; ++i) 
    {
        for (int j = 0; j < 8; ++j) 
        {
            printf("%f ", (float)C[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    float16x8_t a0 = vld1q_f16(A[0]);
    float16x8_t a1 = vld1q_f16(A[1]);
    float16x8_t a2 = vld1q_f16(A[2]);
    float16x8_t a3 = vld1q_f16(A[3]);
    float16x8_t a4 = vld1q_f16(A[4]);
    float16x8_t a5 = vld1q_f16(A[5]);
    float16x8_t a6 = vld1q_f16(A[6]);
    float16x8_t a7 = vld1q_f16(A[7]);
    float16x8x4_t dct1, dct2;
    #pragma unroll
    for (int i = 0; i < 4; ++i) 
    {
        dct1.val[i] = vld1q_f16(B + i*8);
        dct2.val[i] = vld1q_f16(B + (i+4)*8);
    }
    float16x8_t c0 = row_mat_mul(a0, dct1, dct2);
    float16x8_t c1 = row_mat_mul(a1, dct1, dct2);
    float16x8_t c2 = row_mat_mul(a2, dct1, dct2);
    float16x8_t c3 = row_mat_mul(a3, dct1, dct2);
    float16x8_t c4 = row_mat_mul(a4, dct1, dct2);
    float16x8_t c5 = row_mat_mul(a5, dct1, dct2);
    float16x8_t c6 = row_mat_mul(a6, dct1, dct2);
    float16x8_t c7 = row_mat_mul(a7, dct1, dct2);
    vst1q_f16(C[0], c0);
    vst1q_f16(C[1], c1);
    vst1q_f16(C[2], c2);
    vst1q_f16(C[3], c3);
    vst1q_f16(C[4], c4);
    vst1q_f16(C[5], c5);
    vst1q_f16(C[6], c6);
    vst1q_f16(C[7], c7);
    printf("My answer:\n");
    for (int i = 0; i < 8; ++i) 
    {
        for (int j = 0; j < 8; ++j) 
        {
            printf("%f ", (float)C[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}
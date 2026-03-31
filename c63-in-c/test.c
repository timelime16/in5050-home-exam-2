#include <arm_neon.h>
#include <stdio.h>

static inline void transpose(float16_t **A) 
{

    float16x8_t b0, b1, b2, b3, b4, b5, b6, b7;
    float16x8x2_t tmp0, tmp1, tmp2, tmp3; // tanspose - 16bit
    float32x4x2_t tmp4, tmp5, tmp6, tmp7; // transpose - 32 bit
    b0 = vld1q_f16(A[0]);
    b1 = vld1q_f16(A[1]);
    b2 = vld1q_f16(A[2]);
    b3 = vld1q_f16(A[3]);
    b4 = vld1q_f16(A[4]);
    b5 = vld1q_f16(A[5]);
    b6 = vld1q_f16(A[6]);
    b7 = vld1q_f16(A[7]);
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
    vst1q_f16(A[0], b0);
    vst1q_f16(A[1], b1);
    vst1q_f16(A[2], b2);
    vst1q_f16(A[3], b3);
    vst1q_f16(A[4], b4);
    vst1q_f16(A[5], b5);
    vst1q_f16(A[6], b6);
    vst1q_f16(A[7], b7);
}


int main(void) 
{
    float A[8][8] =
    {
    {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f},
    {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f},
    {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f},
    {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f},
    {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f},
    {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f},
    {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f},
    {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f},
    };

    transpose(A);

    for (int i = 0; i < 8; ++i) 
    {
        for (int j = 0; j < 8; ++j)
        {
            printf("%f ", A[i][j]);
        }
        print("\n");
    }
}
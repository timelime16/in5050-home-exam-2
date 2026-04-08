#include <arm_neon.h>
#include <stdio.h>


// Tables for arm neon instruction convenience
float16_t dctlookup_f16[8][8] =
{
  {1.0f,  0.980785f,  0.923880f,  0.831470f,  0.707107f,  0.555570f,  0.382683f,  0.195090f, },
  {1.0f,  0.831470f,  0.382683f, -0.195090f, -0.707107f, -0.980785f, -0.923880f, -0.555570f, },
  {1.0f,  0.555570f, -0.382683f, -0.980785f, -0.707107f,  0.195090f,  0.923880f,  0.831470f, },
  {1.0f,  0.195090f, -0.923880f, -0.555570f,  0.707107f,  0.831470f, -0.382683f, -0.980785f, },
  {1.0f, -0.195090f, -0.923880f,  0.555570f,  0.707107f, -0.831470f, -0.382683f,  0.980785f, },
  {1.0f, -0.555570f, -0.382683f,  0.980785f, -0.707107f, -0.195090f,  0.923880f, -0.831470f, },
  {1.0f, -0.831470f,  0.382683f,  0.195090f, -0.707107f,  0.980785f, -0.923880f,  0.555570f, },
  {1.0f, -0.980785f,  0.923880f, -0.831470f,  0.707107f, -0.555570f,  0.382683f, -0.195090f, },
};

float16_t dctlookup_f16_T[8][8] = 
{
  {1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f},
  {0.980785f,  0.831470f,  0.555570f,  0.195090f, -0.195090f, -0.555570f, -0.831470f, -0.980785f}, 
  {0.923880f,  0.382683f, -0.382683f, -0.923880f, -0.923880f, -0.382683f,  0.382683f,  0.923880f},
  {0.831470f, -0.195090f, -0.980785f, -0.555570f,  0.555570f,  0.980785f,  0.195090f, -0.831470f},
  {0.707107f, -0.707107f, -0.707107f,  0.707107f,  0.707107f, -0.707107f, -0.707107f,  0.707107f},
  {0.555570f, -0.980785f,  0.195090f,  0.831470f, -0.831470f, -0.195090f,  0.980785f, -0.555570f}, 
  {0.382683f, -0.923880f,  0.923880f, -0.382683f, -0.382683f,  0.923880f, -0.923880f,  0.382683f},
  {0.195090f, -0.555570f,  0.831470f, -0.980785f,  0.980785f, -0.831470f,  0.555570f, -0.195090f},
};

static inline float16x8_t row_mat_mul(float16x8_t row, float16x8_t *dct) 
{
  float16x8_t buf1, buf2;

  buf1 = vdupq_n_f16(0.0f);
  buf2 = vdupq_n_f16(0.0f);

  buf1 = vfmaq_laneq_f16(buf1, dct[0], row, 0);
  buf2 = vfmaq_laneq_f16(buf2, dct[1], row, 1);

  buf1 = vfmaq_laneq_f16(buf1, dct[2], row, 2);
  buf2 = vfmaq_laneq_f16(buf2, dct[3], row, 3);

  buf1 = vfmaq_laneq_f16(buf1, dct[4], row, 4);
  buf2 = vfmaq_laneq_f16(buf2, dct[5], row, 5);

  buf1 = vfmaq_laneq_f16(buf1, dct[6], row, 6);
  buf2 = vfmaq_laneq_f16(buf2, dct[7], row, 7);

  return vaddq_f16(buf1, buf2);
}

static void
dct_2d( const float16_t *in, float16_t *out )
{
    // Loop through all elements of the block
    for ( int v = 0; v < 8; v++ )
    {
        for ( int u = 0; u < 8; u++ )
        {
            /* Compute the DCT */
            float dct = 0;
            for ( int y = 0; y < 8; y++ )
            {
                for ( int x = 0; x < 8; x++ )
                {
                    dct += in[y * 8 + x] * dctlookup_f16[x][u] * dctlookup_f16[y][v];
                }
            }

            out[v * 8 + u] = dct;
        }
    }
}

static void 
dct_neon(float16_t *input, float16_t *out)
{
    float16x8_t dct0, dct1, dct2, dct3, dct4, dct5, dct6, dct7; // dctlookup

    dct0 = vld1q_f16(dctlookup_f16[0]);
    dct1 = vld1q_f16(dctlookup_f16[1]);
    dct2 = vld1q_f16(dctlookup_f16[2]);
    dct3 = vld1q_f16(dctlookup_f16[3]);
    dct4 = vld1q_f16(dctlookup_f16[4]);
    dct5 = vld1q_f16(dctlookup_f16[5]);
    dct6 = vld1q_f16(dctlookup_f16[6]);
    dct7 = vld1q_f16(dctlookup_f16[7]);
    float16x8_t dct[8] = {dct0, dct1, dct2, dct3, dct4, dct5, dct6, dct7};

    float16x8_t t0, t1, t2, t3, t4, t5, t6, t7;

    t0 = vld1q_f16(dctlookup_f16_T[0]);
    t1 = vld1q_f16(dctlookup_f16_T[1]);
    t2 = vld1q_f16(dctlookup_f16_T[2]);
    t3 = vld1q_f16(dctlookup_f16_T[3]);
    t4 = vld1q_f16(dctlookup_f16_T[4]);
    t5 = vld1q_f16(dctlookup_f16_T[5]);
    t6 = vld1q_f16(dctlookup_f16_T[6]);
    t7 = vld1q_f16(dctlookup_f16_T[7]);

    float16x8_t t[8] = {t0, t1, t2, t3, t4, t5, t6, t7};

    float16x8_t r0, r1, r2, r3, r4, r5, r6, r7;

    r0 = vld1q_f16(input);
    r1 = vld1q_f16(input + 8);
    r2 = vld1q_f16(input + 8*2);
    r3 = vld1q_f16(input + 8*3);
    r4 = vld1q_f16(input + 8*4);
    r5 = vld1q_f16(input + 8*5);
    r6 = vld1q_f16(input + 8*6);
    r7 = vld1q_f16(input + 8*7);


    float16x8_t b0, b1, b2, b3, b4, b5, b6, b7;
    b0 = row_mat_mul(r0, dct);
    b0 = vaddq_f16(b0, row_mat_mul(r0, t));
    b1 = row_mat_mul(r1, dct);
    b1 = vaddq_f16(b1, row_mat_mul(r1, t));
    b2 = row_mat_mul(r2, dct);
    b2 = vaddq_f16(b2, row_mat_mul(r2, t));
    b3 = row_mat_mul(r3, dct);
    b3 = vaddq_f16(b3, row_mat_mul(r3, t));
    b4 = row_mat_mul(r4, dct);
    b4 = vaddq_f16(b4, row_mat_mul(r4, t));
    b5 = row_mat_mul(r5, dct);
    b5 = vaddq_f16(b5, row_mat_mul(r5, t));
    b6 = row_mat_mul(r6, dct);
    b6 = vaddq_f16(b6, row_mat_mul(r6, t));
    b7 = row_mat_mul(r7, dct);
    b7 = vaddq_f16(b7, row_mat_mul(r7, t));

    vst1q_f16(out, b0);
    vst1q_f16(out + 8, b1);
    vst1q_f16(out + 2*8, b2);
    vst1q_f16(out + 3*8, b3);
    vst1q_f16(out + 4*8, b4);
    vst1q_f16(out + 5*8, b5);
    vst1q_f16(out + 6*8, b6);
    vst1q_f16(out + 7*8, b7);
}

int main(void) 
{
    int i, j;
    float16_t input[64];
    for (i = 0; i < 8; ++i) 
    {
        for (j = 0; j < 8; ++j)
        {
            input[i*8+j] = (float16_t) j;
        }
    }

    float16_t out[64];
    dct_2d(input, out);

    printf("Benchmark: \n");
    for (i = 0; i < 8; ++i)
    {
        for (j = 0; j < 8; ++j) 
        {
            printf("%f ", out[i*8+j]);
        }
        printf("\n");
    }
    printf("\n");

    float16_t out2[64];
    dct_neon(input, out2);
    printf("Mine: \n");
    for (i = 0; i < 8; ++i)
    {
        for (j = 0; j < 8; ++j) 
        {
            printf("%f ", out2[i*8+j]);
        }
        printf("\n");
    }
    printf("\n");
}
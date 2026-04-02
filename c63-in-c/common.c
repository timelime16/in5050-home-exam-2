#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>

#include "common.h"
#include "dsp.h"
#include "tables.h"

void dequantize_idct_row_neon(int16_t *in_data, uint8_t *prediction, int w, int h,
    int y, uint8_t *out_data, uint8_t *quantization)
{
  int x;

  int16x8_t p0, p1, p2, p3, p4, p5, p6, p7; // prediction rows

  /* Perform the dequantization and iDCT */
  for(x = 0; x < w; x += 8)
  { 
    p0 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(prediction + x))); 
    p1 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(prediction + x + w)));
    p2 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(prediction + x + 2*w)));
    p3 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(prediction + x + 3*w)));
    p4 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(prediction + x + 4*w)));
    p5 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(prediction + x + 5*w)));
    p6 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(prediction + x + 6*w)));
    p7 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(prediction + x + 7*w)));

    dequant_idct_block_8x8_neon(in_data+(x*8), out_data + x, quantization, w, p0, p1, p2, p3, p4, p5, p6, p7);
  }
}

void dequantize_idct_neon(int16_t *in_data, uint8_t *prediction, uint32_t width,
    uint32_t height, uint8_t *out_data, uint8_t *quantization)
{
  int y;

  for (y = 0; y < height; y += 8)
  {
    dequantize_idct_row_neon(in_data+y*width, prediction+y*width, width, height, y,
        out_data+y*width, quantization);
  }
}

static inline float16x8_t load_row(uint8_t *in_data, uint8_t *prediction, int w, int x, int i) 
{
  // block[i*8+j] = ((int16_t)in_data[i*w+j+x] - prediction[i*w+j+x])
  int16x8_t input, p;
  input = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(in_data + x + i*w)));
  p = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(prediction + x + i*w)));
  return vcvtq_f16_s16(vsubq_s16(input, p));
}

void dct_quantize_row(uint8_t *in_data, uint8_t *prediction, int w, int h,
    int16_t *out_data, uint8_t *quantization)
{
  int x;

  float16x8_t b0, b1, b2, b3, b4, b5, b6, b7;

  float16x8_t q0, q1, q2, q3, q4, q5, q6, q7; // quant tbl

  float16x8x4_t dct1, dct2; // dctlookup

  q0 = vcvtq_f16_u16(vmovl_u8(vld1_u8(quantization)));
  q1 = vcvtq_f16_u16(vmovl_u8(vld1_u8(quantization + 8)));
  q2 = vcvtq_f16_u16(vmovl_u8(vld1_u8(quantization + 2*8)));
  q3 = vcvtq_f16_u16(vmovl_u8(vld1_u8(quantization + 3*8)));
  q4 = vcvtq_f16_u16(vmovl_u8(vld1_u8(quantization + 4*8)));
  q5 = vcvtq_f16_u16(vmovl_u8(vld1_u8(quantization + 5*8)));
  q6 = vcvtq_f16_u16(vmovl_u8(vld1_u8(quantization + 6*8)));
  q7 = vcvtq_f16_u16(vmovl_u8(vld1_u8(quantization + 7*8)));

  // Need recirporcal
  q0 = vmulq_f16(vrecpsq_f16(q0, vrecpeq_f16(q0)), vrecpeq_f16(q0));
  q1 = vmulq_f16(vrecpsq_f16(q1, vrecpeq_f16(q1)), vrecpeq_f16(q1));
  q2 = vmulq_f16(vrecpsq_f16(q2, vrecpeq_f16(q2)), vrecpeq_f16(q2));
  q3 = vmulq_f16(vrecpsq_f16(q3, vrecpeq_f16(q3)), vrecpeq_f16(q3));
  q4 = vmulq_f16(vrecpsq_f16(q4, vrecpeq_f16(q4)), vrecpeq_f16(q4));
  q5 = vmulq_f16(vrecpsq_f16(q5, vrecpeq_f16(q5)), vrecpeq_f16(q5));
  q6 = vmulq_f16(vrecpsq_f16(q6, vrecpeq_f16(q6)), vrecpeq_f16(q6));
  q7 = vmulq_f16(vrecpsq_f16(q7, vrecpeq_f16(q7)), vrecpeq_f16(q7));

  #pragma unroll
  for (int i = 0; i < 4; ++i) 
  { 
    dct1.val[i] = vld1q_f16(dctlookup_f16[i]);
    dct2.val[i] = vld1q_f16(dctlookup_f16[i+4]);
  }

  /* Perform the DCT and quantization */
  for(x = 0; x < w; x += 8)
  {
    // /* Store MBs linear in memory, i.e. the 64 coefficients are stored
    //    continous. This allows us to ignore stride in DCT/iDCT and other
    //    functions. */

    // b0 = load_row(in_data, prediction, w, x, 0);
    // b1 = load_row(in_data, prediction, w, x, 1);
    // b2 = load_row(in_data, prediction, w, x, 2);
    // b3 = load_row(in_data, prediction, w, x, 3);
    // b4 = load_row(in_data, prediction, w, x, 4);
    // b5 = load_row(in_data, prediction, w, x, 5);
    // b6 = load_row(in_data, prediction, w, x, 6);
    // b7 = load_row(in_data, prediction, w, x, 7);

    // dct_quant_block_8x8_neon(
    //   b0, b1, b2, b3, b4, b5, b6, b7,
    //   q0, q1, q2, q3, q4, q5, q6, q7,
    //   dct1, dct2, 
    //   out_data + x*8, quantization
    // );

    // top half
    float16x8_t b0 = load_row(in_data, prediction, w, x, 0);
    float16x8_t b1 = load_row(in_data, prediction, w, x, 1);
    float16x8_t b2 = load_row(in_data, prediction, w, x, 2);
    float16x8_t b3 = load_row(in_data, prediction, w, x, 3);

    dct_quant_block_4x8_neon(
      b0, b1, b2, b3,
      q0, q1, q2, q3,
      dct1, dct2,
      out_data + x*8
    );

    // bottom half
    float16x8_t b4 = load_row(in_data, prediction, w, x, 4);
    float16x8_t b5 = load_row(in_data, prediction, w, x, 5);
    float16x8_t b6 = load_row(in_data, prediction, w, x, 6);
    float16x8_t b7 = load_row(in_data, prediction, w, x, 7);

    dct_quant_block_4x8_neon(
      b4, b5, b6, b7,
      q4, q5, q6, q7,
      dct1, dct2,
      out_data + x*8 + 4*8
    );
  }
}

void dct_quantize(uint8_t *in_data, uint8_t *prediction, uint32_t width,
    uint32_t height, int16_t *out_data, uint8_t *quantization)
{
  int y;

  for (y = 0; y < height; y += 8)
  {
    dct_quantize_row(in_data+y*width, prediction+y*width, width, height,
        out_data+y*width, quantization);
  }
}

void destroy_frame(struct frame *f)
{
  /* First frame doesn't have a reconstructed frame to destroy */
  if (!f) { return; }

  free(f->recons->Y);
  free(f->recons->U);
  free(f->recons->V);
  free(f->recons);

  free(f->residuals->Ydct);
  free(f->residuals->Udct);
  free(f->residuals->Vdct);
  free(f->residuals);

  free(f->predicted->Y);
  free(f->predicted->U);
  free(f->predicted->V);
  free(f->predicted);

  free(f->mbs[Y_COMPONENT]);
  free(f->mbs[U_COMPONENT]);
  free(f->mbs[V_COMPONENT]);

  free(f);
}

struct frame* create_frame(struct c63_common *cm, yuv_t *image)
{
  struct frame *f = malloc(sizeof(struct frame));

  f->orig = image;

  f->recons = malloc(sizeof(yuv_t));
  f->recons->Y = malloc(cm->ypw * cm->yph);
  f->recons->U = malloc(cm->upw * cm->uph);
  f->recons->V = malloc(cm->vpw * cm->vph);

  f->predicted = malloc(sizeof(yuv_t));
  f->predicted->Y = calloc(cm->ypw * cm->yph, sizeof(uint8_t));
  f->predicted->U = calloc(cm->upw * cm->uph, sizeof(uint8_t));
  f->predicted->V = calloc(cm->vpw * cm->vph, sizeof(uint8_t));

  f->residuals = malloc(sizeof(dct_t));
  f->residuals->Ydct = calloc(cm->ypw * cm->yph, sizeof(int16_t));
  f->residuals->Udct = calloc(cm->upw * cm->uph, sizeof(int16_t));
  f->residuals->Vdct = calloc(cm->vpw * cm->vph, sizeof(int16_t));

  f->mbs[Y_COMPONENT] =
    calloc(cm->mb_rows * cm->mb_cols, sizeof(struct macroblock));
  f->mbs[U_COMPONENT] =
    calloc(cm->mb_rows/2 * cm->mb_cols/2, sizeof(struct macroblock));
  f->mbs[V_COMPONENT] =
    calloc(cm->mb_rows/2 * cm->mb_cols/2, sizeof(struct macroblock));

  return f;
}

void dump_image(yuv_t *image, int w, int h, FILE *fp)
{
  fwrite(image->Y, 1, w*h, fp);
  fwrite(image->U, 1, w*h/4, fp);
  fwrite(image->V, 1, w*h/4, fp);
}


// Keep original for dec
void dequantize_idct_row(int16_t *in_data, uint8_t *prediction, int w, int h,
    int y, uint8_t *out_data, uint8_t *quantization)
{
  int x;

  int16_t block[8*8];

  /* Perform the dequantization and iDCT */
  for(x = 0; x < w; x += 8)
  {
    int i, j;

    dequant_idct_block_8x8(in_data+(x*8), block, quantization);

    for (i = 0; i < 8; ++i)
    {
      for (j = 0; j < 8; ++j)
      {
        /* Add prediction block. Note: DCT is not precise -
           Clamp to legal values */
        int16_t tmp = block[i*8+j] + (int16_t)prediction[i*w+j+x];

        if (tmp < 0) { tmp = 0; }
        else if (tmp > 255) { tmp = 255; }

        out_data[i*w+j+x] = tmp;
      }
    }
  }
}

void dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width,
    uint32_t height, uint8_t *out_data, uint8_t *quantization)
{
  int y;

  for (y = 0; y < height; y += 8)
  {
    dequantize_idct_row(in_data+y*width, prediction+y*width, width, height, y,
        out_data+y*width, quantization);
  }
}
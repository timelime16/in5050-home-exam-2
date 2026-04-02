#ifndef C63_DSP_H_
#define C63_DSP_H_

#define ISQRT2 0.70710678118654f

#include <inttypes.h>

void dct_quant_block_8x8_neon(
  float16x8_t b0, float16x8_t b1, float16x8_t b2, float16x8_t b3, 
  float16x8_t b4, float16x8_t b5, float16x8_t b6, float16x8_t b7, 
  float16x8_t q0, float16x8_t q1, float16x8_t q2, float16x8_t q3, 
  float16x8_t q4, float16x8_t q5, float16x8_t q6, float16x8_t q7, 
  float16x8x4_t dct1, float16x8x4_t dct2,
  int16_t *out_data, uint8_t *quant_tbl
);

void dct_quant_block_4x8_neon(
  float16x8_t b0, float16x8_t b1,
  float16x8_t b2, float16x8_t b3,
  float16x8_t q0, float16x8_t q1,
  float16x8_t q2, float16x8_t q3,
  float16x8x4_t dct1, float16x8x4_t dct2,
  int16_t *out
);

void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl);

void dequant_idct_block_8x8_neon(
  int16_t *in_data, uint8_t *out_data, uint8_t *quant_tbl, int w,
  int16x8_t p0, int16x8_t p1, int16x8_t p2, int16x8_t p3, 
  int16x8_t p4, int16x8_t p5, int16x8_t p6, int16x8_t p7
);

#endif  /* C63_DSP_H_ */

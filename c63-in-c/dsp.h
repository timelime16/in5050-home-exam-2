#ifndef C63_DSP_H_
#define C63_DSP_H_

#define ISQRT2 0.70710678118654f

#include <inttypes.h>

void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl);

static inline void dct_quant_block_8x8_neon(
  float16x8_t b0, float16x8_t b1, float16x8_t b2, float16x8_t b3, 
  float16x8_t b4, float16x8_t b5, float16x8_t b6, float16x8_t b7, 
  int16_t *out_data, uint8_t *quant_tbl
);

void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl);

void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result);

#endif  /* C63_DSP_H_ */

#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <arm_neon.h>

#include "dsp.h"
#include "me.h"

/* Motion estimation for 8x8 block */
static void me_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
    uint8_t *orig, uint8_t *ref, int color_component)
{
  struct macroblock *mb =
    &cm->curframe->mbs[color_component][mb_y*cm->padw[color_component]/8+mb_x];

  int range = cm->me_search_range;

  /* Quarter resolution for chroma channels. */
  if (color_component > 0) { range /= 2; }

  int left = mb_x * 8 - range;
  int top = mb_y * 8 - range;
  int right = mb_x * 8 + range;
  int bottom = mb_y * 8 + range;

  int w = cm->padw[color_component];
  int h = cm->padh[color_component];

  /* Make sure we are within bounds of reference frame. TODO: Support partial
     frame bounds. */
  if (left < 0) { left = 0; }
  if (top < 0) { top = 0; }
  if (right > (w - 8)) { right = w - 8; }
  if (bottom > (h - 8)) { bottom = h - 8; }

  int x, y;

  int mx = mb_x * 8;
  int my = mb_y * 8;

  // Load orig block [r0 | r1] 16 pixels in one register
  orig = orig + my*w+mx;
  uint8x8_t _r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7;
  _r0 = vld1_u8(orig); _r1 = vld1_u8(orig + w); _r2 = vld1_u8(orig + 2*w); _r3 = vld1_u8(orig + 3*w);
  _r4 = vld1_u8(orig + 4*w); _r5 = vld1_u8(orig + 5*w); _r6 = vld1_u8(orig + 6*w); _r7 = vld1_u8(orig + 7*w);

  uint8x16_t _or0, _or1, _or2, _or3; // [r0 | r1] 16 pixels in one register
  _or0 = vcombine_u8(_r0, _r1); _or1 = vcombine_u8(_r2, _r3); _or2 = vcombine_u8(_r4, _r5); _or3 = vcombine_u8(_r6, _r7);

  uint8x16_t _rr0, _rr1, _rr2, _rr3; // for ref block

  uint8x16_t _diff;
  uint16x8_t _acc;

  uint32_t best_sad = UINT_MAX;

  for (y = top; y < bottom; ++y)
  {
    for (x = left; x < right; ++x)
    {
      // Load ref block
      uint8_t *calc_ref = ref + y*w+x;
      _r0 = vld1_u8(calc_ref); _r1 = vld1_u8(calc_ref + w); _r2 = vld1_u8(calc_ref + 2*w); _r3 = vld1_u8(calc_ref + 3*w);
      _r4 = vld1_u8(calc_ref + 4*w); _r5 = vld1_u8(calc_ref + 5*w); _r6 = vld1_u8(calc_ref + 6*w); _r7 = vld1_u8(calc_ref + 7*w);

      _rr0 = vcombine_u8(_r0, _r1); _rr1 = vcombine_u8(_r2, _r3); _rr2 = vcombine_u8(_r4, _r5); _rr3 = vcombine_u8(_r6, _r7); // combine

      _acc = vdupq_n_u16(0);

      // calculate 
      _diff = vabdq_u8(_or0, _rr0); // abs diff
      _acc = vaddq_u16(_acc, vpaddlq_u8(_diff)); // sum

      _diff = vabdq_u8(_or1, _rr1); 
      _acc = vaddq_u16(_acc, vpaddlq_u8(_diff));
      if (vaddvq_u16(_acc) >= best_sad) continue; // early termination

      _diff = vabdq_u8(_or2, _rr2);
      _acc = vaddq_u16(_acc, vpaddlq_u8(_diff));

      _diff = vabdq_u8(_or3, _rr3); 
      _acc = vaddq_u16(_acc, vpaddlq_u8(_diff));

      uint32_t sad = vaddvq_u16(_acc); // compare 
      if (sad < best_sad)
      {
        mb->mv_x = x - mx;
        mb->mv_y = y - my;
        best_sad = sad;
      }
    }
  }

  /* Here, there should be a threshold on SAD that checks if the motion vector
     is cheaper than intraprediction. We always assume MV to be beneficial */

  // printf("Using motion vector (%d, %d) with SAD %d\n", mb->mv_x, mb->mv_y,
  //    best_sad);

  mb->use_mv = 1;
}

void c63_motion_estimate(struct c63_common *cm)
{
  /* Compare this frame with previous reconstructed frame */
  int mb_x, mb_y;

  /* Luma */
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->Y,
          cm->refframe->recons->Y, Y_COMPONENT);
    }
  }

  /* Chroma */
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
    {
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->U,
          cm->refframe->recons->U, U_COMPONENT);
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->V,
          cm->refframe->recons->V, V_COMPONENT);
    }
  }
}

/* Motion compensation for 8x8 block */
static void mc_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
    uint8_t *predicted, uint8_t *ref, int color_component)
{
  struct macroblock *mb =
    &cm->curframe->mbs[color_component][mb_y*cm->padw[color_component]/8+mb_x];

  if (!mb->use_mv) { return; }

  int left = mb_x * 8 + mb->mv_x;
  int top = mb_y * 8;

  int w = cm->padw[color_component];

  /* Copy block from ref mandated by MV */
  uint8x8_t _r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7;

  _r0 = vld1_u8(ref + (top + mb->mv_y) * w + left); 
  _r1 = vld1_u8(ref + (top + 1 + mb->mv_y) * w + left); 
  _r2 = vld1_u8(ref + (top + 2 + mb->mv_y) * w + left);  
  _r3 = vld1_u8(ref + (top + 3 + mb->mv_y) * w + left); 
  _r4 = vld1_u8(ref + (top + 4 + mb->mv_y) * w + left);  
  _r5 = vld1_u8(ref + (top + 5 + mb->mv_y) * w + left); 
  _r6 = vld1_u8(ref + (top + 6 + mb->mv_y) * w + left); 
  _r7 = vld1_u8(ref + (top + 7 + mb->mv_y) * w + left); 

  vst1_u8(predicted, _r0);
  vst1_u8(predicted + w, _r1);
  vst1_u8(predicted + 2 * w, _r2);
  vst1_u8(predicted + 3 * w, _r3);
  vst1_u8(predicted + 4 * w, _r4);
  vst1_u8(predicted + 5 * w, _r5);
  vst1_u8(predicted + 6 * w, _r6);
  vst1_u8(predicted + 7 * w, _r7);
}

void c63_motion_compensate(struct c63_common *cm)
{
  int mb_x, mb_y;

  /* Luma */
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->Y,
          cm->refframe->recons->Y, Y_COMPONENT);
    }
  }

  /* Chroma */
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
    {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->U,
          cm->refframe->recons->U, U_COMPONENT);
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->V,
          cm->refframe->recons->V, V_COMPONENT);
    }
  }
}

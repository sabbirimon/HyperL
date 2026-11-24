/* SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0 */
#ifndef HYPERL_ERASURE_H
#define HYPERL_ERASURE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define HL_ERASURE_ABI 1
#define HL_ERASURE_MAX_DATA 16
#define HL_ERASURE_MAX_PARITY 4
#define HL_ERASURE_MAX_BYTES (4u * 1024u * 1024u)
enum hl_erasure_status { HL_ERASURE_OK=0, HL_ERASURE_INVALID=1, HL_ERASURE_UNRECOVERABLE=2, HL_ERASURE_CANCELLED=3 };
typedef int (*hl_erasure_cancel)(void *context);
/* Systematic Cauchy Reed-Solomon over GF(256), polynomial 0x11d. Caller
   owns all buffers. Outputs must not overlap each other or any input.
   On any failure discard outputs. Verify trusted hashes before/after repair.
   This corrects verified erasures, not undetected Byzantine data or RAM faults. */
enum hl_erasure_status hl_erasure_encode(unsigned k, unsigned m, size_t bytes,
    const uint8_t *const data[], uint8_t *const parity[], hl_erasure_cancel cancelled, void *context);
/* fragments has k+m slots; mask identifies hash-verified present fragments.
   recovered has k+m slots; only missing slots need non-NULL output buffers.
   Any <=m missing fragments can be restored from the remaining k. */
enum hl_erasure_status hl_erasure_recover(unsigned k, unsigned m, size_t bytes,
    const uint8_t *const fragments[], uint32_t present_mask, uint8_t *const recovered[],
    hl_erasure_cancel cancelled, void *context);
#ifdef __cplusplus
}
#endif
#endif

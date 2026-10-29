/* SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0 */
/* Allocator instrumentation is test-only; the public C ABI stays unchanged. */
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef union { size_t bytes; long double alignment; void *pointer; } allocation;
static size_t live_bytes, peak_bytes, allocations, fail_at;
static void *tracked_malloc(size_t bytes) {
    allocations++;
    if (fail_at && allocations == fail_at) return NULL;
    allocation *block = malloc(sizeof(allocation) + bytes);
    if (!block) return NULL;
    block->bytes = bytes;
    live_bytes += bytes;
    if (live_bytes > peak_bytes) peak_bytes = live_bytes;
    return block + 1;
}
static void tracked_free(void *pointer) {
    if (!pointer) return;
    allocation *block = (allocation *)pointer - 1;
    assert(live_bytes >= block->bytes);
    live_bytes -= block->bytes;
    free(block);
}
#define malloc tracked_malloc
#define free tracked_free
#include "cpu.c"
#undef malloc
#undef free

static void reset(void) {
    assert(live_bytes == 0);
    peak_bytes = allocations = fail_at = 0;
}
static int stop_later(void *context) {
    size_t *calls = context;
    return ++*calls == 9;
}
int main(void) {
    enum { LENGTH = 4096, STEPS = 12 };
    float x[LENGTH], output[LENGTH];
    for (size_t i = 0; i < LENGTH; i++) x[i] = (float)((int)(i % 9) - 4);
    struct hl_vector input = {x, LENGTH};
    struct hl_step chain[STEPS];
    for (size_t i = 0; i < STEPS; i++) chain[i] = (struct hl_step){HL_RELU, i, 0};
    size_t count;
    reset();
    assert(hl_execute(&input, 1, chain, STEPS, STEPS, output, LENGTH, &count, NULL, NULL) == HL_OK);
    for (size_t i = 0; i < LENGTH; i++) assert(output[i] == (x[i] > 0 ? x[i] : 0));
    assert(count == LENGTH && live_bytes == 0);
    assert(peak_bytes == 2 * LENGTH * sizeof(float));
    printf("chain_peak_live_payload_bytes=%zu; previous_all_retained_payload_bytes=%zu\n",
           peak_bytes, (size_t)(STEPS + 1) * LENGTH * sizeof(float));

    /* Last use may be far away; b can equal a. Output can be an early input. */
    struct hl_step fan[] = {{HL_RELU, 0, 0}, {HL_ADD, 0, 1}, {HL_MULTIPLY, 2, 2}};
    reset();
    assert(hl_execute(&input, 1, fan, 3, 3, output, LENGTH, &count, NULL, NULL) == HL_OK);
    for (size_t i = 0; i < LENGTH; i++) {
        float sum = x[i] + (x[i] > 0 ? x[i] : 0);
        assert(output[i] == sum * sum);
    }
    assert(live_bytes == 0);
    assert(hl_execute(&input, 1, fan, 3, 0, output, LENGTH, &count, NULL, NULL) == HL_OK);
    assert(memcmp(output, x, sizeof(x)) == 0 && live_bytes == 0);

    /* Fail every native allocation, including after previous buffers are freed. */
    for (size_t i = 1; i <= STEPS + 1; i++) {
        reset(); fail_at = i; output[0] = 12345;
        assert(hl_execute(&input, 1, chain, STEPS, STEPS, output, LENGTH, &count, NULL, NULL) == HL_MEMORY);
        assert(count == 0 && output[0] == 12345 && live_bytes == 0);
    }
    reset(); size_t calls = 0; output[0] = 12345;
    assert(hl_execute(&input, 1, chain, STEPS, STEPS, output, LENGTH, &count, stop_later, &calls) == HL_CANCELLED);
    assert(count == 0 && output[0] == 12345 && live_bytes == 0);

    /* Unused branches still execute: neither nonfinite input nor overflow hides. */
    reset(); float bad[] = {NAN}; struct hl_vector unused[] = {{x, 1}, {bad, 1}};
    assert(hl_execute(unused, 2, chain, 1, 0, output, LENGTH, &count, NULL, NULL) == HL_NONFINITE);
    assert(count == 0 && live_bytes == 0);
    float huge[] = {FLT_MAX}; struct hl_vector big = {huge, 1};
    struct hl_step overflow = {HL_ADD, 0, 0};
    assert(hl_execute(&big, 1, &overflow, 1, 0, output, LENGTH, &count, NULL, NULL) == HL_NONFINITE);
    assert(count == 0 && live_bytes == 0);

    float ordered[] = {16777216, 1, -16777216};
    struct hl_vector vector = {ordered, 3}; struct hl_step sum = {HL_SUM, 0, 0};
    assert(hl_execute(&vector, 1, &sum, 1, 1, output, LENGTH, &count, NULL, NULL) == HL_OK);
    assert(count == 1 && output[0] == 0 && live_bytes == 0);
    /* Snapshot inputs make an overlapping output safe, with publication last. */
    assert(hl_execute(&vector, 1, &sum, 1, 1, ordered, 3, &count, NULL, NULL) == HL_OK);
    assert(ordered[0] == 0 && live_bytes == 0);
    puts("Liveness, fanout, retained output, allocation failures and cancellation passed.");
    return 0;
}

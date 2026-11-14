/* SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0 */
#include <stdio.h>
#include <hyperl.h>

int main(void) {
    const float x[] = {-1.0f, 2.0f, 3.0f};
    const float w[] = {2.0f, 3.0f, 4.0f};
    const struct hl_vector inputs[] = {{x, 3}, {w, 3}};
    const struct hl_step steps[] = {{HL_MULTIPLY, 0, 1}, {HL_RELU, 2, 0}};
    float result[3];
    size_t count;
    if (hl_execute(inputs, 2, steps, 2, 3, result, 3, &count, NULL, NULL) != HL_OK ||
        count != 3 || result[0] != 0 || result[1] != 6 || result[2] != 12)
        return 1;
    printf("%g %g %g\n", result[0], result[1], result[2]);
    return 0;
}

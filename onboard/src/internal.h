#ifndef OPENADCS_INTERNAL_H
#define OPENADCS_INTERNAL_H

#include <math.h>
#include <stddef.h>
#include <string.h>
#include "openadcs/types.h"

static inline int openadcs_finite(const double *values, size_t count)
{
    size_t i;
    for (i = 0; i < count; ++i) {
        if (!isfinite(values[i])) return 0;
    }
    return 1;
}

static inline int openadcs_rotation_valid(const double *matrix)
{
    size_t i, j, k;
    double determinant;
    if (!openadcs_finite(matrix, 9)) return 0;
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            double product = 0.0;
            for (k = 0; k < 3; ++k)
                product += matrix[i * 3 + k] * matrix[j * 3 + k];
            if (fabs(product - (i == j ? 1.0 : 0.0)) > 1e-9) return 0;
        }
    }
    determinant = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
        - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
        + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
    return fabs(determinant - 1.0) <= 1e-9;
}

static inline void openadcs_rotate(
    const double *matrix, const double *input, double *output)
{
    size_t i;
    for (i = 0; i < 3; ++i)
        output[i] = matrix[i * 3] * input[0]
            + matrix[i * 3 + 1] * input[1]
            + matrix[i * 3 + 2] * input[2];
}

static inline void openadcs_clear_measurement(OpenAdcsMeasurement *output)
{
    memset(output, 0, sizeof(*output));
    output->status = OPENADCS_INVALID_INPUT;
}

#endif

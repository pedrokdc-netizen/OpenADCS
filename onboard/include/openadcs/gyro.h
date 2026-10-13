#ifndef OPENADCS_GYRO_H
#define OPENADCS_GYRO_H

#include "types.h"

typedef struct {
    double reference_temperature_c;
    double minimum_temperature_c;
    double maximum_temperature_c;
    double bias_polynomial_rad_s[9];
    double gain[3];
    double maximum_raw_rate_rad_s;
    double sensor_to_body[9];
} OpenAdcsGyroConfig;

typedef struct {
    uint64_t timestamp_ns;
    uint32_t sensor_id;
    uint32_t sequence;
    OpenAdcsStatus status;
    double temperature_c;
    double rate_rad_s[3];
} OpenAdcsGyroRaw;

OPENADCS_API OpenAdcsStatus openadcs_gyro_process(
    const OpenAdcsGyroConfig *config, const OpenAdcsGyroRaw *raw,
    OpenAdcsMeasurement *output);

#endif

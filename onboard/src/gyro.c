#include "openadcs/gyro.h"
#include "internal.h"

static int config_valid(const OpenAdcsGyroConfig *config)
{
    size_t i;
    if (config == NULL || !isfinite(config->reference_temperature_c)
        || !isfinite(config->minimum_temperature_c)
        || !isfinite(config->maximum_temperature_c)
        || config->minimum_temperature_c < -273.15
        || config->minimum_temperature_c > config->maximum_temperature_c
        || config->reference_temperature_c < config->minimum_temperature_c
        || config->reference_temperature_c > config->maximum_temperature_c
        || !openadcs_finite(config->bias_polynomial_rad_s, 9)
        || !openadcs_finite(config->gain, 3)
        || !isfinite(config->maximum_raw_rate_rad_s)
        || config->maximum_raw_rate_rad_s <= 0.0
        || !openadcs_rotation_valid(config->sensor_to_body)) return 0;
    for (i = 0; i < 3; ++i)
        if (config->gain[i] <= 0.0) return 0;
    return 1;
}

OpenAdcsStatus openadcs_gyro_process(
    const OpenAdcsGyroConfig *config, const OpenAdcsGyroRaw *raw,
    OpenAdcsMeasurement *output)
{
    double corrected[3], temperature;
    size_t i;
    if (output == NULL) return OPENADCS_INVALID_INPUT;
    openadcs_clear_measurement(output);
    if (raw == NULL) return output->status;
    output->timestamp_ns = raw->timestamp_ns;
    output->sensor_id = raw->sensor_id;
    output->sequence = raw->sequence;
    if (!config_valid(config))
        return output->status = OPENADCS_INVALID_CONFIG;
    if (raw->status != OPENADCS_OK) return output->status = raw->status;
    if (!isfinite(raw->temperature_c)
        || !openadcs_finite(raw->rate_rad_s, 3)) return output->status;
    if (raw->temperature_c < config->minimum_temperature_c
        || raw->temperature_c > config->maximum_temperature_c)
        return output->status = OPENADCS_UNQUALIFIED;
    temperature = raw->temperature_c - config->reference_temperature_c;
    for (i = 0; i < 3; ++i) {
        const double *bias = config->bias_polynomial_rad_s + 3 * i;
        double offset = (bias[2] * temperature + bias[1]) * temperature + bias[0];
        if (fabs(raw->rate_rad_s[i]) >= config->maximum_raw_rate_rad_s)
            return output->status = OPENADCS_SATURATED;
        corrected[i] = (raw->rate_rad_s[i] - offset) / config->gain[i];
        if (!isfinite(corrected[i])) return output->status;
    }
    openadcs_rotate(config->sensor_to_body, corrected, output->body);
    if (!openadcs_finite(output->body, 3)) {
        memset(output->body, 0, sizeof(output->body));
        return output->status;
    }
    return output->status = OPENADCS_OK;
}

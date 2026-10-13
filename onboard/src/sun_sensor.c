#include "openadcs/sun_sensor.h"
#include "internal.h"

static int config_valid(const OpenAdcsSunConfig *config)
{
    size_t i;
    if (config == NULL
        || !openadcs_finite(config->active_length_m, 2)
        || !openadcs_finite(config->mask_height_m, 2)
        || !openadcs_finite(config->channel_offset, 4)
        || !openadcs_finite(config->channel_gain, 4)
        || !openadcs_finite(config->minimum_signal, 2)
        || !isfinite(config->half_fov_rad)
        || config->half_fov_rad <= 0.0
        || config->half_fov_rad > OPENADCS_ESTCUBE_SUN_HALF_FOV_RAD
        || !openadcs_rotation_valid(config->sensor_to_body)) return 0;
    for (i = 0; i < 2; ++i) {
        if (config->active_length_m[i] <= 0.0
            || config->mask_height_m[i] <= 0.0
            || config->minimum_signal[i] <= 0.0) return 0;
    }
    for (i = 0; i < 4; ++i)
        if (config->channel_gain[i] <= 0.0) return 0;
    return 1;
}

OpenAdcsStatus openadcs_sun_process(
    const OpenAdcsSunConfig *config, const OpenAdcsSunRaw *raw,
    OpenAdcsMeasurement *output)
{
    double sensor[3] = {0.0, 0.0, 1.0};
    double channel[4], norm;
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
    if (raw->illumination == OPENADCS_DARK)
        return output->status = OPENADCS_NO_ILLUMINATION;
    if (raw->illumination == OPENADCS_ILLUMINATION_UNVERIFIED)
        return output->status = OPENADCS_ILLUMINATION_UNKNOWN;
    if (raw->illumination != OPENADCS_DIRECT_SUN
        || !openadcs_finite(raw->channel, 4)) return output->status;
    for (i = 0; i < 4; ++i) {
        channel[i] = (raw->channel[i] - config->channel_offset[i])
            / config->channel_gain[i];
        if (!isfinite(channel[i]) || channel[i] < 0.0) return output->status;
    }
    for (i = 0; i < 2; ++i) {
        double first = channel[2 * i], second = channel[2 * i + 1];
        double sum = first + second;
        if (!isfinite(sum)) return output->status;
        if (sum < config->minimum_signal[i])
            return output->status = OPENADCS_NO_ILLUMINATION;
        sensor[i] = 0.5 * config->active_length_m[i]
            * ((second - first) / sum) / config->mask_height_m[i];
        if (!isfinite(sensor[i])) return output->status;
    }
    if (atan(hypot(sensor[0], sensor[1])) > config->half_fov_rad + 1e-12)
        return output->status = OPENADCS_OUT_OF_FOV;
    norm = sqrt(sensor[0] * sensor[0] + sensor[1] * sensor[1] + 1.0);
    for (i = 0; i < 3; ++i) sensor[i] /= norm;
    openadcs_rotate(config->sensor_to_body, sensor, output->body);
    return output->status = OPENADCS_OK;
}

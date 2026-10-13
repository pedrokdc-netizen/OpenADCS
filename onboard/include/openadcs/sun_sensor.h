#ifndef OPENADCS_SUN_SENSOR_H
#define OPENADCS_SUN_SENSOR_H

#include "types.h"

#define OPENADCS_ESTCUBE_SUN_HALF_FOV_RAD (36.7 * 0.017453292519943295)

enum {
    OPENADCS_ILLUMINATION_UNVERIFIED = 0,
    OPENADCS_DIRECT_SUN = 1,
    OPENADCS_DARK = 2
};

typedef struct {
    double active_length_m[2];
    double mask_height_m[2];
    double channel_offset[4];
    double channel_gain[4];
    double minimum_signal[2];
    double half_fov_rad;
    double sensor_to_body[9];
} OpenAdcsSunConfig;

typedef struct {
    uint64_t timestamp_ns;
    uint32_t sensor_id;
    uint32_t sequence;
    OpenAdcsStatus status;
    uint32_t illumination;
    double channel[4];
} OpenAdcsSunRaw;

OPENADCS_API OpenAdcsStatus openadcs_sun_process(
    const OpenAdcsSunConfig *config, const OpenAdcsSunRaw *raw,
    OpenAdcsMeasurement *output);

#endif

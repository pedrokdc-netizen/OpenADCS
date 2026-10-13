#ifndef OPENADCS_GPS_H
#define OPENADCS_GPS_H

#include <stddef.h>
#include "types.h"

#define OPENADCS_SKYTRAQ_NAVIGATION_FRAME_SIZE 66

typedef struct {
    uint64_t timestamp_ns;
    uint32_t sensor_id;
    uint32_t sequence;
    OpenAdcsStatus status;
    uint32_t fix_mode;
    uint32_t satellites_used;
    uint32_t gps_week;
    double tow_s;
    double position_ecef_m[3];
    double velocity_ecef_m_s[3];
    double latitude_deg;
    double longitude_deg;
    double ellipsoid_height_m;
    double msl_height_m;
    double dop[5];
} OpenAdcsGpsPvt;

OPENADCS_API OpenAdcsStatus openadcs_gps_decode(
    const uint8_t *frame, size_t length, uint64_t timestamp_ns,
    uint32_t sensor_id, uint32_t sequence, OpenAdcsStatus driver_status,
    OpenAdcsGpsPvt *output);

#endif

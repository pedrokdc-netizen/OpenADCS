#ifndef OPENADCS_TYPES_H
#define OPENADCS_TYPES_H

#include <stdint.h>

#if defined(_WIN32) && defined(OPENADCS_SHARED)
#define OPENADCS_API __declspec(dllexport)
#else
#define OPENADCS_API
#endif

typedef uint32_t OpenAdcsStatus;
enum {
    OPENADCS_OK = 0,
    OPENADCS_INVALID_CONFIG = 1u << 0,
    OPENADCS_INVALID_INPUT = 1u << 1,
    OPENADCS_NO_ILLUMINATION = 1u << 2,
    OPENADCS_OUT_OF_FOV = 1u << 3,
    OPENADCS_SATURATED = 1u << 4,
    OPENADCS_DRIVER_FAULT = 1u << 5,
    OPENADCS_STALE = 1u << 6,
    OPENADCS_TIME_ORDER = 1u << 7,
    OPENADCS_CAPACITY = 1u << 8,
    OPENADCS_UNQUALIFIED = 1u << 9,
    OPENADCS_ILLUMINATION_UNKNOWN = 1u << 10,
    OPENADCS_NO_FIX = 1u << 11,
    OPENADCS_PROTOCOL_ERROR = 1u << 12
};

typedef struct {
    uint64_t timestamp_ns;
    uint32_t sensor_id;
    uint32_t sequence;
    OpenAdcsStatus status;
    double body[3];
} OpenAdcsMeasurement;

enum {
    OPENADCS_SUN = 1,
    OPENADCS_GYRO = 2,
    OPENADCS_COARSE_SUN = 3,
    OPENADCS_IMU = 4,
    OPENADCS_HORIZON = 5,
    OPENADCS_GPS = 6
};

#endif

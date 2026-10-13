#ifndef OPENADCS_ADCS_H
#define OPENADCS_ADCS_H

#include "types.h"
#include "gps.h"

#define OPENADCS_SENSOR_CAPACITY 16
#define OPENADCS_ABI_VERSION 2
#define OPENADCS_GPS_CAPACITY 4

typedef struct {
    uint32_t kind;
    uint32_t occupied;
    OpenAdcsMeasurement measurement;
} OpenAdcsSensorSlot;

typedef struct {
    uint32_t occupied;
    OpenAdcsGpsPvt measurement;
} OpenAdcsGpsSlot;

typedef struct {
    OpenAdcsSensorSlot slots[OPENADCS_SENSOR_CAPACITY];
    OpenAdcsGpsSlot gps_slots[OPENADCS_GPS_CAPACITY];
    uint64_t maximum_gps_age_ns;
} OpenAdcsState;

typedef struct {
    uint32_t valid_sun_count;
    uint32_t valid_gyro_count;
    uint32_t invalid_count;
    uint32_t stale_count;
    uint32_t future_count;
    uint32_t instantaneous_reference_rank;
    uint32_t attitude_available;
    uint32_t valid_gps_count;
    uint32_t navigation_available;
} OpenAdcsHealth;

OPENADCS_API uint32_t openadcs_abi_version(void);
OPENADCS_API void openadcs_init(OpenAdcsState *state);
OPENADCS_API OpenAdcsStatus openadcs_accept(
    OpenAdcsState *state, uint32_t kind,
    const OpenAdcsMeasurement *measurement);
OPENADCS_API OpenAdcsStatus openadcs_configure_gps_age(
    OpenAdcsState *state, uint64_t maximum_age_ns);
OPENADCS_API OpenAdcsStatus openadcs_accept_gps(
    OpenAdcsState *state, const OpenAdcsGpsPvt *measurement);
OPENADCS_API OpenAdcsStatus openadcs_health(
    const OpenAdcsState *state, uint64_t now_ns,
    uint64_t maximum_sun_age_ns, uint64_t maximum_gyro_age_ns,
    OpenAdcsHealth *health);

#endif

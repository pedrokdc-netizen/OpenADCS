#include "openadcs/adcs.h"
#include "internal.h"

uint32_t openadcs_abi_version(void)
{
    return OPENADCS_ABI_VERSION;
}

void openadcs_init(OpenAdcsState *state)
{
    if (state != NULL) memset(state, 0, sizeof(*state));
}

OpenAdcsStatus openadcs_accept(
    OpenAdcsState *state, uint32_t kind,
    const OpenAdcsMeasurement *measurement)
{
    OpenAdcsSensorSlot *target = NULL;
    size_t i;
    if (state == NULL || measurement == NULL) return OPENADCS_INVALID_INPUT;
    if (kind == OPENADCS_GPS) return OPENADCS_INVALID_INPUT;
    if (kind != OPENADCS_SUN && kind != OPENADCS_GYRO)
        return OPENADCS_UNQUALIFIED;
    for (i = 0; i < OPENADCS_GPS_CAPACITY; ++i)
        if (state->gps_slots[i].occupied
            && state->gps_slots[i].measurement.sensor_id == measurement->sensor_id)
            return OPENADCS_INVALID_INPUT;
    for (i = 0; i < OPENADCS_SENSOR_CAPACITY; ++i) {
        OpenAdcsSensorSlot *slot = state->slots + i;
        if (slot->occupied && slot->measurement.sensor_id == measurement->sensor_id) {
            if (slot->kind != kind) return OPENADCS_INVALID_INPUT;
            if (measurement->timestamp_ns <= slot->measurement.timestamp_ns)
                return OPENADCS_TIME_ORDER;
            target = slot;
            break;
        }
        if (!slot->occupied && target == NULL) target = slot;
    }
    if (target == NULL) return OPENADCS_CAPACITY;
    target->kind = kind;
    target->occupied = 1;
    target->measurement = *measurement;
    if (measurement->status == OPENADCS_OK) {
        double norm;
        if (!openadcs_finite(measurement->body, 3))
            target->measurement.status = OPENADCS_INVALID_INPUT;
        if (kind == OPENADCS_SUN) {
            norm = sqrt(measurement->body[0] * measurement->body[0]
                + measurement->body[1] * measurement->body[1]
                + measurement->body[2] * measurement->body[2]);
            if (!isfinite(norm) || fabs(norm - 1.0) > 1e-9)
                target->measurement.status = OPENADCS_INVALID_INPUT;
        }
    }
    if (target->measurement.status != OPENADCS_OK)
        memset(target->measurement.body, 0, sizeof(target->measurement.body));
    return target->measurement.status;
}

OpenAdcsStatus openadcs_configure_gps_age(OpenAdcsState *state, uint64_t maximum_age_ns)
{
    if (state == NULL || maximum_age_ns == 0) return OPENADCS_INVALID_CONFIG;
    state->maximum_gps_age_ns = maximum_age_ns;
    return OPENADCS_OK;
}

OpenAdcsStatus openadcs_accept_gps(OpenAdcsState *state, const OpenAdcsGpsPvt *measurement)
{
    OpenAdcsGpsSlot *target = NULL;
    size_t i;
    if (state == NULL || measurement == NULL) return OPENADCS_INVALID_INPUT;
    if (state->maximum_gps_age_ns == 0) return OPENADCS_INVALID_CONFIG;
    for (i = 0; i < OPENADCS_SENSOR_CAPACITY; ++i)
        if (state->slots[i].occupied
            && state->slots[i].measurement.sensor_id == measurement->sensor_id)
            return OPENADCS_INVALID_INPUT;
    for (i = 0; i < OPENADCS_GPS_CAPACITY; ++i) {
        OpenAdcsGpsSlot *slot = state->gps_slots + i;
        if (slot->occupied && slot->measurement.sensor_id == measurement->sensor_id) {
            if (measurement->timestamp_ns <= slot->measurement.timestamp_ns)
                return OPENADCS_TIME_ORDER;
            target = slot;
            break;
        }
        if (!slot->occupied && target == NULL) target = slot;
    }
    if (target == NULL) return OPENADCS_CAPACITY;
    target->occupied = 1;
    target->measurement = *measurement;
    if (measurement->status == OPENADCS_OK
        && (measurement->fix_mode < 2 || measurement->fix_mode > 3
            || measurement->satellites_used < 4 || measurement->satellites_used > 255
            || measurement->gps_week > 65535 || !isfinite(measurement->tow_s)
            || measurement->tow_s < 0.0 || measurement->tow_s >= 604800.0
            || !openadcs_finite(measurement->position_ecef_m, 3)
            || !openadcs_finite(measurement->velocity_ecef_m_s, 3)
            || !isfinite(measurement->latitude_deg) || fabs(measurement->latitude_deg) > 90.0
            || !isfinite(measurement->longitude_deg) || fabs(measurement->longitude_deg) > 180.0
            || !isfinite(measurement->ellipsoid_height_m)
            || !isfinite(measurement->msl_height_m)
            || !openadcs_finite(measurement->dop, 5)
            || (measurement->position_ecef_m[0] == 0.0
                && measurement->position_ecef_m[1] == 0.0
                && measurement->position_ecef_m[2] == 0.0)))
        target->measurement.status = OPENADCS_INVALID_INPUT;
    if (measurement->status == OPENADCS_OK) {
        for (i = 0; i < 5; ++i)
            if (measurement->dop[i] < 0.0 || measurement->dop[i] > 655.35)
                target->measurement.status = OPENADCS_INVALID_INPUT;
    }
    if (target->measurement.status != OPENADCS_OK) {
        memset(target->measurement.position_ecef_m, 0, sizeof(target->measurement.position_ecef_m));
        memset(target->measurement.velocity_ecef_m_s, 0, sizeof(target->measurement.velocity_ecef_m_s));
    }
    return target->measurement.status;
}

OpenAdcsStatus openadcs_health(
    const OpenAdcsState *state, uint64_t now_ns,
    uint64_t maximum_sun_age_ns, uint64_t maximum_gyro_age_ns,
    OpenAdcsHealth *health)
{
    size_t i;
    OpenAdcsStatus status = OPENADCS_OK;
    if (health == NULL) return OPENADCS_INVALID_INPUT;
    memset(health, 0, sizeof(*health));
    if (state == NULL || maximum_sun_age_ns == 0 || maximum_gyro_age_ns == 0)
        return OPENADCS_INVALID_CONFIG;
    for (i = 0; i < OPENADCS_SENSOR_CAPACITY; ++i) {
        const OpenAdcsSensorSlot *slot = state->slots + i;
        uint64_t maximum_age;
        OpenAdcsStatus problem;
        if (!slot->occupied) continue;
        problem = slot->measurement.status;
        if (slot->measurement.timestamp_ns > now_ns) {
            ++health->future_count;
            problem |= OPENADCS_TIME_ORDER;
        } else {
            maximum_age = slot->kind == OPENADCS_SUN
                ? maximum_sun_age_ns : maximum_gyro_age_ns;
            if (now_ns - slot->measurement.timestamp_ns > maximum_age) {
                ++health->stale_count;
                problem |= OPENADCS_STALE;
            }
        }
        if (problem != OPENADCS_OK) {
            ++health->invalid_count;
            status |= problem;
        } else if (slot->kind == OPENADCS_SUN) {
            ++health->valid_sun_count;
        } else {
            ++health->valid_gyro_count;
        }
    }
    /* Multiple Sun sensors observe the same reference, not independent axes. */
    health->instantaneous_reference_rank = health->valid_sun_count ? 2u : 0u;
    for (i = 0; i < OPENADCS_GPS_CAPACITY; ++i) {
        const OpenAdcsGpsSlot *slot = state->gps_slots + i;
        OpenAdcsStatus problem;
        if (!slot->occupied) continue;
        problem = slot->measurement.status;
        if (slot->measurement.timestamp_ns > now_ns) {
            ++health->future_count;
            problem |= OPENADCS_TIME_ORDER;
        } else if (now_ns - slot->measurement.timestamp_ns > state->maximum_gps_age_ns) {
            ++health->stale_count;
            problem |= OPENADCS_STALE;
        }
        if (problem != OPENADCS_OK) {
            ++health->invalid_count;
            status |= problem;
        } else {
            ++health->valid_gps_count;
        }
    }
    health->navigation_available = health->valid_gps_count ? 1u : 0u;
    return status;
}

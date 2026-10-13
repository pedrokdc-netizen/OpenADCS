#include "openadcs/gps.h"
#include "internal.h"

static uint32_t read_u16(const uint8_t *data)
{
    return ((uint32_t)data[0] << 8) | data[1];
}

static uint32_t read_u32(const uint8_t *data)
{
    return ((uint32_t)data[0] << 24) | ((uint32_t)data[1] << 16)
        | ((uint32_t)data[2] << 8) | data[3];
}

static double read_s32(const uint8_t *data)
{
    uint32_t value = read_u32(data);
    return value > 2147483647u ? (double)value - 4294967296.0 : (double)value;
}

OpenAdcsStatus openadcs_gps_decode(
    const uint8_t *frame, size_t length, uint64_t timestamp_ns,
    uint32_t sensor_id, uint32_t sequence, OpenAdcsStatus driver_status,
    OpenAdcsGpsPvt *output)
{
    const uint8_t *payload;
    uint8_t checksum = 0;
    size_t i;
    if (output == NULL) return OPENADCS_INVALID_INPUT;
    memset(output, 0, sizeof(*output));
    output->timestamp_ns = timestamp_ns;
    output->sensor_id = sensor_id;
    output->sequence = sequence;
    output->status = OPENADCS_PROTOCOL_ERROR;
    if (driver_status != OPENADCS_OK) return output->status = driver_status;
    if (frame == NULL || length != OPENADCS_SKYTRAQ_NAVIGATION_FRAME_SIZE
        || frame[0] != 0xA0 || frame[1] != 0xA1
        || read_u16(frame + 2) != 59 || frame[4] != 0xA8
        || frame[64] != 0x0D || frame[65] != 0x0A) return output->status;
    for (i = 4; i < 63; ++i) checksum ^= frame[i];
    if (checksum != frame[63]) return output->status;
    payload = frame + 4;
    output->fix_mode = payload[1];
    output->satellites_used = payload[2];
    output->gps_week = read_u16(payload + 3);
    output->tow_s = 0.01 * read_u32(payload + 5);
    if (output->fix_mode > 3 || output->tow_s >= 604800.0) return output->status;
    if (output->fix_mode < 2) return output->status = OPENADCS_NO_FIX;
    if (output->satellites_used < 4) return output->status;
    output->latitude_deg = 1e-7 * read_s32(payload + 9);
    output->longitude_deg = 1e-7 * read_s32(payload + 13);
    if (fabs(output->latitude_deg) > 90.0 || fabs(output->longitude_deg) > 180.0)
        return output->status;
    output->ellipsoid_height_m = 0.01 * read_s32(payload + 17);
    output->msl_height_m = 0.01 * read_s32(payload + 21);
    for (i = 0; i < 5; ++i) output->dop[i] = 0.01 * read_u16(payload + 25 + 2 * i);
    for (i = 0; i < 3; ++i) {
        output->position_ecef_m[i] = 0.01 * read_s32(payload + 35 + 4 * i);
        output->velocity_ecef_m_s[i] = 0.01 * read_s32(payload + 47 + 4 * i);
    }
    if (output->position_ecef_m[0] == 0.0 && output->position_ecef_m[1] == 0.0
        && output->position_ecef_m[2] == 0.0) return output->status;
    return output->status = OPENADCS_OK;
}

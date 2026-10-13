import math

import numpy as np
import pytest

from openadcs_sim.black_sun_sensor import (
    BlackSunHeadConfig, BlackSunHeadModel, BlackSunReading, fuse_heads)


def make_config(**overrides):
    values = dict(sensor_from_body=np.eye(3), image_size_px=(250, 250), focal_length_px=125.0,
                  principal_point_px=(125.0, 125.0), half_fov_deg=45.0, centroid_sigma_px=0.0)
    values.update(overrides)
    return BlackSunHeadConfig(**values)


def angle_deg(a, b):
    return math.degrees(math.acos(np.clip(np.dot(a, b), -1, 1)))


def test_noiseless_head_recovers_sun_vector_and_boresight_centroid():
    model = BlackSunHeadModel(make_config())
    sun = np.array([0.2, -0.1, 1.0])
    reading = model.sample(0.0, sun)
    assert reading.valid and reading.status == "OK"
    assert angle_deg(reading.sun_body, sun / np.linalg.norm(sun)) < 1e-4
    assert BlackSunHeadModel(make_config()).sample(0.0, [0, 0, 1]).centroid_px == (125.0, 125.0)


def test_fov_eclipse_and_time_order():
    model = BlackSunHeadModel(make_config())
    assert model.sample(0.0, [1, 0, 0.1]).status == "OUT_OF_FOV"
    assert model.sample(1.0, [0, 0, 1], sunlit=False).status == "ECLIPSE"
    with pytest.raises(ValueError):
        model.sample(1.0, [0, 0, 1])
    with pytest.raises(ValueError):
        model.sample(2.0, [0, 0, 0])


def test_noise_is_seeded_and_scales_with_sigma():
    sun = np.array([0.1, 0.1, 1.0])

    def errors(seed, sigma):
        model = BlackSunHeadModel(make_config(centroid_sigma_px=sigma, seed=seed))
        return [angle_deg(model.sample(float(i), sun).sun_body, sun / np.linalg.norm(sun))
                for i in range(200)]
    assert errors(3, 0.5) == errors(3, 0.5)
    assert np.mean(errors(3, 1.0)) > np.mean(errors(3, 0.2))
    expected_rms = math.degrees(1.0 / 125.0) * math.sqrt(2)
    assert np.sqrt(np.mean(np.square(errors(4, 1.0)))) == pytest.approx(expected_rms, rel=0.25)


def test_dropout_and_bias():
    lost = BlackSunHeadModel(make_config(dropout_probability=1.0)).sample(0.0, [0, 0, 1])
    assert not lost.valid and lost.status == "CENTROID_LOST"
    biased = BlackSunHeadModel(make_config(centroid_bias_px=(1.25, 0.0))).sample(0.0, [0, 0, 1])
    assert biased.centroid_px == (126.25, 125.0)


def test_head_rotation_maps_back_to_body_frame():
    rotation = np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    model = BlackSunHeadModel(make_config(sensor_from_body=rotation))
    sun_body = np.array([-1.0, 0, 0.0])
    reading = model.sample(0.0, sun_body)
    assert reading.valid
    assert angle_deg(reading.sun_body, sun_body) < 1e-4


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        make_config(sensor_from_body=np.diag([1.0, 1.0, -1.0]))
    with pytest.raises(ValueError):
        make_config(half_fov_deg=95.0)
    with pytest.raises(ValueError):
        make_config(focal_length_px=0.0)
    with pytest.raises(ValueError):
        make_config(dropout_probability=1.5)
    with pytest.raises(ValueError):
        make_config(principal_point_px=(300.0, 125.0))


def test_fusion_reduces_error_and_ignores_invalid_heads():
    truth = np.array([0.3, 0.2, 1.0])
    truth /= np.linalg.norm(truth)
    singles, fused_errors = [], []
    for trial in range(100):
        heads = [BlackSunHeadModel(make_config(centroid_sigma_px=1.0, seed=trial * 3 + k))
                 for k in range(3)]
        readings = [h.sample(0.0, truth) for h in heads]
        singles.append(angle_deg(readings[0].sun_body, truth))
        fused = fuse_heads(readings)
        fused_errors.append(angle_deg(fused.sun_body, truth))
    assert np.mean(fused_errors) < np.mean(singles)
    bad = BlackSunReading(np.full(3, np.nan), False, None, 0.01, "OUT_OF_FOV")
    assert fuse_heads([bad]).status == "NO_VALID_HEAD"
    assert not fuse_heads([bad]).valid
    good = BlackSunHeadModel(make_config(centroid_sigma_px=0.5)).sample(0.0, truth)
    assert fuse_heads([bad, good]).valid

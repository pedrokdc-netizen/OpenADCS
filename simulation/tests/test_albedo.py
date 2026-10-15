import math

import numpy as np
import pytest

from openadcs_sim import sun_sensor_array as ssa
from openadcs_sim.albedo import AlbedoContext, albedo_context, sphere_view_factor


def test_view_factor_limits():
    rho = math.asin(6378137 / (6378137 + 500e3))
    assert sphere_view_factor(0.0, rho) == pytest.approx(math.sin(rho) ** 2)
    assert sphere_view_factor(math.pi / 2 + rho + 0.01, rho) == 0.0
    assert sphere_view_factor(math.pi / 2, rho) == pytest.approx((rho - math.sin(rho) * math.cos(rho)) / math.pi)
    just_in = sphere_view_factor(math.pi / 2 - rho - 1e-6, rho)
    just_out = sphere_view_factor(math.pi / 2 - rho + 1e-6, rho)
    assert just_in == pytest.approx(just_out, abs=1e-4)


def test_night_side_has_no_albedo():
    ctx = albedo_context([7e6, 0, 0], [-1, 0, 0], [-7e6, 0, 0])
    assert ctx.reflectance == 0.0


def _context(reflectance=0.3):
    return AlbedoContext(np.array([0.0, 0.0, 1.0]), math.asin(6378137 / 6878137), reflectance)


def test_albedo_biases_nadir_head_toward_earth_and_is_off_by_default_leak():
    sun = np.array([0.6, 0.0, 0.8])
    base = ssa.SunSensorArrayConfig(1e-6, albedo_leak_fraction=0.0)
    biased = ssa.SunSensorArrayConfig(1e-6, albedo_leak_fraction=1.0, veiling_glare=0.5, threshold_fraction=0.0)
    ref = ssa.measure_once(base, sun, True, _context())
    out = ssa.measure_once(biased, sun, True, _context())
    assert out.error_deg > ref.error_deg + 1e-4
    head = out.heads[0]            # +Z head looks at nadir and sees the Sun
    assert head.albedo_ratio > 0.1 and any(abs(v) > 0 for v in head.albedo_bias_px)
    assert ssa.measure_once(biased, sun, True, None).error_deg == pytest.approx(ref.error_deg, abs=1e-5)


def test_eclipse_still_flags_heads_and_albedo_round_trips():
    config = ssa.SunSensorArrayConfig(0.1)
    dark = ssa.measure_once(config, [0, 0, 1], False, _context())
    assert all(r.status == "ECLIPSE" for r in dark.heads)
    lit = ssa.measure_once(config, [0.6, 0, 0.8], True, _context())
    again = ssa.SunSensorMeasurement.from_dict(lit.to_dict())
    assert again.heads[0].albedo_ratio == pytest.approx(lit.heads[0].albedo_ratio)


def test_config_validation():
    with pytest.raises(ValueError):
        ssa.SunSensorArrayConfig(0.1, albedo_coefficient=1.5)
    with pytest.raises(ValueError):
        ssa.SunSensorArrayConfig(0.1, albedo_leak_fraction=-1)
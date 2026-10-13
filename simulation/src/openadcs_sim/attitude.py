from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import astropy.units as u
from astropy.coordinates import CartesianDifferential, CartesianRepresentation, GCRS, ITRS
from astropy.time import Time
import numpy as np
from numpy.typing import NDArray

from .common import Vector, vector3
from .environment import EnvironmentSample

Matrix = NDArray[np.float64]
WGS84_A_M = 6378137.0
WGS84_B_M = 6356752.314245179


def unit_vector(vector: Vector) -> Vector:
    vector = vector3(vector)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Direction requires a non-zero finite vector")
    return vector3(vector / length)


@dataclass(frozen=True)
class AttitudeState:
    """Full body-to-GCRS orientation; a single boresight is not a full attitude."""

    body_to_gcrs: Matrix
    model: str = "External attitude"

    def __post_init__(self) -> None:
        matrix = np.array(self.body_to_gcrs, dtype=np.float64, copy=True)
        if (
            matrix.shape != (3, 3) or not np.all(np.isfinite(matrix))
            or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-10, rtol=0)
            or not np.isclose(np.linalg.det(matrix), 1, atol=1e-10, rtol=0)
        ):
            raise ValueError("Attitude must be a finite, right-handed orthonormal rotation")
        matrix.setflags(write=False)
        object.__setattr__(self, "body_to_gcrs", matrix)


class AttitudeModel(Protocol):
    def orientation(
        self, time_utc: datetime, position_gcrs_m: Vector, velocity_gcrs_m_s: Vector,
    ) -> AttitudeState: ...


class NadirPointingAttitude:
    """Ideal orbit-following reference only: no torques, disturbances or control."""

    def orientation(
        self, time_utc: datetime, position_gcrs_m: Vector, velocity_gcrs_m_s: Vector,
    ) -> AttitudeState:
        z = -unit_vector(position_gcrs_m)
        velocity = vector3(velocity_gcrs_m_s)
        x = unit_vector(velocity - np.dot(velocity, z) * z)
        y = unit_vector(np.cross(z, x))
        return AttitudeState(np.column_stack((x, y, z)), "Ideal nadir / orbit-following reference")


@dataclass(frozen=True)
class BodyView:
    attitude: AttitudeState
    itrs_to_body: Matrix
    earth_center_body_m: Vector
    velocity_body_m_s: Vector
    magnetic_body_t: Vector
    sun_body_direction: Vector
    attitude_body_direction: Vector


def spacecraft_view(
    sample: EnvironmentSample, attitude_model: AttitudeModel | None = None,
) -> BodyView:
    instant = Time(sample.time_utc)
    fixed = ITRS(
        CartesianRepresentation(
            sample.position_itrs_m * u.m,
            differentials=CartesianDifferential(sample.velocity_itrs_m_s * u.m / u.s),
        ),
        obstime=instant,
    )
    inertial = fixed.transform_to(GCRS(obstime=instant))
    position = vector3(inertial.cartesian.xyz.to_value(u.m))
    velocity = vector3(inertial.cartesian.differentials["s"].d_xyz.to_value(u.m / u.s))
    basis = ITRS(CartesianRepresentation(np.eye(3) * u.m), obstime=instant)
    itrs_to_gcrs = basis.transform_to(GCRS(obstime=instant)).cartesian.xyz.to_value(u.m)
    model = attitude_model if attitude_model is not None else NadirPointingAttitude()
    attitude = model.orientation(sample.time_utc, position, velocity)
    rotation = attitude.body_to_gcrs.T @ itrs_to_gcrs
    rotation.setflags(write=False)
    return BodyView(
        attitude, rotation, vector3(rotation @ -sample.position_itrs_m),
        vector3(attitude.body_to_gcrs.T @ velocity),
        vector3(rotation @ sample.magnetic_field_itrs_t),
        unit_vector(rotation @ sample.sun_direction_itrs),
        vector3([0, 0, 1]),
    )


def earth_surface_body_m(view: BodyView, longitude_count: int = 73, latitude_count: int = 37) -> NDArray[np.float64]:
    if longitude_count < 4 or latitude_count < 3:
        raise ValueError("Earth mesh needs at least four longitudes and three latitudes")
    longitude, latitude = np.meshgrid(
        np.linspace(0, 2 * np.pi, longitude_count),
        np.linspace(-np.pi / 2, np.pi / 2, latitude_count),
    )
    fixed = np.stack((
        WGS84_A_M * np.cos(latitude) * np.cos(longitude),
        WGS84_A_M * np.cos(latitude) * np.sin(longitude),
        WGS84_B_M * np.sin(latitude),
    ), axis=-1)
    return fixed @ view.itrs_to_body.T + view.earth_center_body_m

"""Static architecture contracts; this catalog does not execute connections."""

from dataclasses import dataclass
from enum import Enum

from .planned.actuators import ActuatorModel
from .planned.adcs_mcu import AdcsMicrocontroller
from .planned.attitude_estimator import AttitudeEstimator
from .planned.base import PlaceholderModule
from .planned.dynamics import DisturbanceTorques

from .planned.environment_conditions import EnvironmentConditions
from .planned.horizon import HorizonSensor
from .planned.onboard_computer import OnboardComputer
from .planned.sun_sensor_array import SunSensorArrayIngress


class ModuleStatus(str, Enum):
    IMPLEMENTED = "IMPLEMENTED"
    PARTIAL = "PARTIAL"
    PLACEHOLDER = "PLACEHOLDER"


@dataclass(frozen=True)
class Module:
    id: str
    title: str
    status: ModuleStatus
    summary: str
    bounds: tuple[int, int, int, int]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    sources: tuple[str, ...]
    placeholders: tuple[type[PlaceholderModule], ...] = ()

    @property
    def source_paths(self) -> tuple[str, ...]:
        return self.sources + tuple(
            "simulation\\src\\" + placeholder.__module__.replace(".", "\\") + ".py"
            for placeholder in self.placeholders
        )


@dataclass(frozen=True)
class Connection:
    id: str
    source: str
    target: str
    variables: str
    planned: bool
    points: tuple[tuple[int, int], ...]
    label_position: tuple[int, int]


MODULES = (
    Module("orbit", "Orbit / environment state", ModuleStatus.IMPLEMENTED,
           "SGP4 -> ITRS; environment truth, not GPS measurements.", (40, 100, 300, 230),
           ("TLE, UTC",), ("r_ITRS [m], v_ITRS [m/s], UTC",),
           ("simulation\\src\\openadcs_sim\\orbit.py",)),
    Module("magnetic", "Earth magnetic field", ModuleStatus.IMPLEMENTED,
           "XYZgeomag / WMM2025 reference field.", (40, 310, 300, 440),
           ("r_ITRS [m], UTC",), ("B_ref_ITRS [T]",),
           ("simulation\\src\\openadcs_sim\\magnetic.py",)),
    Module("sun", "Sun geometry", ModuleStatus.IMPLEMENTED,
           "Huira / SPICE geometric direction; not illumination.", (40, 520, 300, 650),
           ("r_ITRS [m], UTC",), ("s_ITRS [unit], Sun distance [m]",),
           ("simulation\\src\\openadcs_sim\\sun.py",)),
    Module("conditions", "External environment inputs", ModuleStatus.PARTIAL,
           "Manual/driver conditions exist; environmental generators are PLACEHOLDER.", (40, 730, 300, 900),
           ("User characterization / hardware inputs",),
           ("Illumination / T [deg C], GNSS SV/signal/DOP; planned: albedo generation",),
           ("simulation\\src\\openadcs_sim\\orion_b16.py",
            "simulation\\src\\openadcs_sim\\sensor_processing.py"), (EnvironmentConditions,)),
    Module("dynamics", "Spacecraft attitude dynamics", ModuleStatus.PARTIAL,
           "Rigid-body RK4 + gravity gradient, opt-in (GUI/--dynamics); no actuator/disturbance torques yet.", (40, 1010, 300, 1170),
           ("Applied body torque [N m], mass/inertia, time",),
           ("R_BG (body -> GCRS), omega_B [rad/s], f_B [m/s2]",),
           ("simulation\\src\\openadcs_sim\\dynamics.py",), (DisturbanceTorques,)),
    Module("spacecraft", "Spacecraft config", ModuleStatus.IMPLEMENTED,
           "Mass, CoM, inertia (JSON); Pilot1 V2 baseline; GUI editor.",
           (40, 1440, 300, 1590),
           ("CAD mass properties (JSON file / GUI window)",), ("mass [kg], CoM [m], inertia I_B [kg m2]",),
           ("simulation\\src\\openadcs_sim\\spacecraft.py", "simulation\\src\\openadcs_sim\\spacecraft_window.py")),
    Module("sunsensor", "Sun sensor (6x OV5640 cube)", ModuleStatus.PARTIAL,
           "120 deg pinhole x6 (fisheye TBD); eclipse, albedo, mount (2 deg) and bias errors; envelope MC tool. Assumed params, NOT flight proven; ideal-nadir truth; ADCS ingress/C99 PLACEHOLDER.",
           (740, 585, 1020, 775),
           ("s_ITRS [unit], sunlit, albedo (Earth view factor), ideal nadir attitude (planned: R_BG)",),
           ("Per-head status, fused s_body [unit], error [deg]",),
           ("simulation\\src\\openadcs_sim\\sun_sensor_array.py",
            "simulation\\src\\openadcs_sim\\black_sun_sensor.py"), (SunSensorArrayIngress,)),
    Module("gyro", "Gyro IMU (gyro-only)", ModuleStatus.IMPLEMENTED,
           "Assumed gyro error model + C99 correction; unvalidated.", (740, 460, 1020, 570),
           ("omega_B truth [rad/s], T [deg C]",),
           ("Corrected omega_B [rad/s], timestamp, validity",),
           ("simulation\\src\\openadcs_sim\\gyro_model.py", "onboard\\src\\gyro.c")),
    Module("horizon", "Horizon sensor (2x LWIR camera)", ModuleStatus.PARTIAL,
           "Digital model + MC error law (NOT flight proven); ADCS ingress/firmware PLACEHOLDER.",
           (740, 790, 1020, 960),
           ("Orbit, WGS84 Earth geometry, body orientation",),
           ("Measured nadir_body [unit], roll/pitch covariance, validity",),
           ("simulation\\src\\openadcs_sim\\horizon_sensor.py",
            "simulation\\src\\openadcs_sim\\horizon_processing.py"), (HorizonSensor,)),
    Module("gps", "Orion B-16 GPS / GNSS", ModuleStatus.IMPLEMENTED,
           "Assumed PVT behavior + C99 binary decoder; NOT flight-calibrated.", (740, 1000, 1020, 1130),
           ("Orbit truth, UTC, explicit receiver conditions/profile",),
           ("WGS84 ECEF r [m], v [m/s], GPS week/TOW, fix/status",),
           ("simulation\\src\\openadcs_sim\\orion_b16.py", "onboard\\src\\gps.c")),
    Module("mag", "Magnetometer (TAM)", ModuleStatus.IMPLEMENTED,
           "Assumed linear error model; unvalidated, no ADCS ingress yet.",
           (740, 1200, 1020, 1310),
           ("B_body [T] (needs ITRS->GCRS->body conversion), T [deg C], user characterization",),
           ("B_meas_sensor [T], valid/saturated/status",),
           ("simulation\\src\\openadcs_sim\\magnetometer.py",)),
    Module("adcs", "ADCS MCU boundary", ModuleStatus.PARTIAL,
           "Python MEKF + control law run closed-loop; C99 port of estimator/control is PLACEHOLDER.",
           (1540, 450, 1880, 680),
           ("Sensor records, reference vectors; planned: mode/orientation target",),
           ("Attitude estimate, wheel torque demand, dipole; C99 input health",),
           ("onboard\\src\\adcs.c", "simulation\\src\\openadcs_sim\\adcs_control.py",
            "simulation\\src\\openadcs_sim\\closed_loop.py"), (AdcsMicrocontroller, AttitudeEstimator)),
    Module("obc", "Onboard computer MCU", ModuleStatus.PLACEHOLDER,
           "No OBC execution, command or telemetry model.", (2100, 450, 2440, 680),
           ("ADCS telemetry / health / planned estimates",),
           ("Mode, target orientation, clock synchronization",), (), (OnboardComputer,)),
    Module("actuators", "Actuators", ModuleStatus.PARTIAL,
           "4x10 g RW pyramid + 3 MTQ, simple model; no ADCS.", (1540, 1010, 1880, 1170),
           ("Desired body torque [N m], time; hardware commands TBD",),
           ("Applied body torque [N m], wheel speeds, dipole",),("simulation\\src\\openadcs_sim\\actuators.py",), (ActuatorModel,)),
)


CONNECTIONS = (
    Connection("orbit_magnetic", "orbit", "magnetic", "r_ITRS [m], UTC", False,
               ((170, 230), (170, 310)), (185, 265)),
    Connection("orbit_sun", "orbit", "sun", "r_ITRS [m], UTC", False,
               ((80, 230), (20, 230), (20, 490), (170, 490), (170, 520)), (170, 485)),
    Connection("orbit_gps", "orbit", "gps", "r_ITRS [m], v_ITRS [m/s], UTC", False,
               ((300, 150), (350, 150), (350, 1035), (740, 1035)), (550, 1015)),
    Connection("sun_sunsensor", "sun", "sunsensor", "s_ITRS [unit], sunlit -> s_body (ideal nadir attitude)", False,
               ((300, 635), (740, 635)), (520, 655)),
    Connection("sunsensor_adcs", "sunsensor", "adcs", "fused s_body [unit], validity (closed loop)", False,
               ((1020, 670), (1540, 670)), (1250, 690)),
    Connection("dynamics_gyro", "dynamics", "gyro", "omega_B [rad/s] -> raw rate (planned)", True,
               ((300, 1045), (470, 1045), (470, 505), (740, 505)), (595, 485)),
    Connection("orbit_horizon", "orbit", "horizon", "r_ITRS [m], WGS84 geometry (GUI)", False,
               ((300, 185), (385, 185), (385, 850), (740, 850)), (560, 830)),
    Connection("dynamics_horizon", "dynamics", "horizon", "R_BG [rotation] (planned)", True,
               ((300, 1115), (540, 1115), (540, 900), (740, 900)), (640, 885)),
    Connection("conditions_gps", "conditions", "gps", "SV count, signal [dBm], DOP, power/T", False,
               ((300, 800), (575, 800), (575, 1100), (740, 1100)), (520, 1190)),
    Connection("conditions_sensors", "conditions", "sunsensor", "sunlit fed; albedo fed from geometry (Bond 0.30, glare vs threshold)", False,
               ((300, 765), (610, 765), (610, 720), (740, 720)), (530, 745)),
    Connection("dynamics_sunsensor", "dynamics", "sunsensor", "R_BG [rotation] (environment, opt-in)", False,
               ((300, 1155), (705, 1155), (705, 750), (740, 750)), (535, 1135)),
    Connection("conditions_gyro", "conditions", "gyro", "T [deg C] (external/driver)", False,
               ((300, 850), (645, 850), (645, 555), (740, 555)), (540, 595)),
    Connection("magnetic_reference", "magnetic", "adcs", "B_ref_ITRS [T]: reference, NOT a measurement (closed loop)", False,
               ((300, 350), (330, 350), (330, 70), (1710, 70), (1710, 450)), (1190, 95)),
    Connection("sun_reference", "sun", "adcs", "s_ref_ITRS [unit]: reference (closed loop)", False,
               ((300, 620), (320, 620), (320, 30), (1790, 30), (1790, 450)), (1200, 45)),
    Connection("gyro_adcs", "gyro", "adcs", "omega_B [rad/s], timestamp, validity", False,
               ((1020, 515), (1320, 515), (1320, 550), (1540, 550)), (1200, 500)),
    Connection("horizon_adcs", "horizon", "adcs", "nadir_body [unit], validity (closed loop)", False,
               ((1020, 875), (1380, 875), (1380, 620), (1540, 620)), (1200, 890)),
    Connection("gps_adcs", "gps", "adcs", "ECEF PVT, GPS time, fix/status", False,
               ((1020, 1065), (1410, 1065), (1410, 655), (1540, 655)), (1200, 1050)),
    Connection("adcs_obc", "adcs", "obc", "Input health; estimate/telemetry (planned)", True,
               ((1880, 500), (2100, 500)), (1990, 380)),
    Connection("obc_adcs", "obc", "adcs", "Mode, target orientation, clock sync (planned)", True,
               ((2100, 630), (1880, 630)), (1990, 740)),
    Connection("adcs_actuators", "adcs", "actuators", "Torque [N m], dipole [A m^2] (closed loop)", False,
               ((1710, 680), (1710, 1010)), (1930, 845)),
    Connection("actuators_dynamics", "actuators", "dynamics", "Applied body torque [N m] + status", False,
               ((1540, 1090), (1500, 1090), (1500, 1400), (150, 1400), (150, 1170)), (1040, 1380)),
    Connection("spacecraft_dynamics", "spacecraft", "dynamics", "mass, inertia I_B (JSON)", False,
               ((70, 1440), (70, 1170)), (190, 1420)),
    Connection("magnetic_mag", "magnetic", "mag", "B_ref_ITRS [T] -> B_body after frame conversion (planned)", True,
               ((300, 420), (360, 420), (360, 1235), (740, 1235)), (540, 1250)),
    Connection("dynamics_mag", "dynamics", "mag", "R_BG [rotation] (planned)", True,
               ((300, 1165), (720, 1165), (720, 1260), (740, 1260)), (540, 1300)),
    Connection("mag_adcs", "mag", "adcs", "B_meas_sensor [T], validity (closed loop)", False,
               ((1020, 1255), (1440, 1255), (1440, 670), (1540, 670)), (1230, 1235)),
)

MODULE_BY_ID = {module.id: module for module in MODULES}
DIAGRAM_SIZE = (2490, 1640)

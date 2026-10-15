# OpenADCS
Open-source spacecraft attitude determination and control system, starting with simulation.

## Horizon sensor digital model (original concept, NOT flight proven)

Two Waveshare Thermal-45 LWIR cameras (80x62, MI0802, ESP32-S3) image Earth's limb; a range-aided
cone fit gives the nadir vector (roll/pitch; yaw is unobservable). `openadcs_sim/horizon_*.py` simulate
image acquisition and the processing chain (ESP32-portable reference), run a Monte Carlo campaign and
provide an equivalent error law for three architectures (1 camera, 2 independent, 2 fused) with ESP32
timing/rate limits. Pointing p95 at 500 km, full scene: 3.7 deg / 2.85 deg / 2.84 deg; recommended
architecture: two independent cameras 90-120 deg apart. Details, caveats and assumptions:
[simulation/docs/HORIZON_SENSOR.md](simulation/docs/HORIZON_SENSOR.md); firmware port plan and golden
vectors: [onboard/esp32_horizon/README.md](onboard/esp32_horizon/README.md).

## Gyro IMU digital model (gyro-only, assumption-based)

`openadcs_sim/gyro_model.py` is the forward model of a three-axis rate gyro; a gyro-only IMU
is sufficient for this project, so the former "Full IMU" placeholder was removed from the
architecture and the Gyro block now represents the IMU.

- Terms: scale factor, misalignment, constant bias, quadratic temperature bias (same polynomial
  form the flight-derived `onboard/src/gyro.c` corrects), angle random walk (rad/s/sqrt(Hz),
  discretised as sigma/sqrt(dt)), rate random walk, first-order Gauss-Markov bias instability,
  quantization, saturation and a temperature-validity domain. Seeded and reproducible.
- The output is the raw sensor-frame rate, the input of the C99 correction.
- Structure follows Basilisk `imuSensor` and the AcubeSAT gyro-noise studies; the temperature
  calibration follows ESTCube-1. **No parameter has a default and nothing is flight validated.**
- No accelerometer: specific force is not modeled.

## Black-sun micro Sun sensor model (unvalidated exception)

`openadcs_sim/black_sun_sensor.py` models the image-based Sun sensor of Lee and Shah,
*Sensors* 2019, 19(4), 739 ([PMC6386896](https://pmc.ncbi.nlm.nih.gov/articles/PMC6386896/)).
The paper is ground-tested only; **no flight heritage was found**, so this is an explicit,
user-approved exception (like the GPS and magnetometer models).

- `BlackSunHeadModel`: pinhole camera, Sun-in-FOV and image-bounds checks, eclipse, centroid
  bias and Gaussian noise (pixels), centroid-loss probability, seeded and reproducible. The
  output is a body-frame unit Sun vector with a per-head angular sigma.
- `fuse_heads`: inverse-covariance fusion of independent valid heads. It is **not** the paper's
  Covariance Projection Method, which handles unknown cross-correlation.
- Not modeled: the black-sun image processing itself (blur, thresholds, corner refinement),
  the stationary Kalman filter, radiation, thermal drift, albedo and Earth glare.
- No defaults: all parameters are user characterizations. The paper reports about 0.125 deg raw
  per head and 0.071 deg fused (ground, open sky); these are reference figures only.
- The flown slit/PSD processing in `onboard/src/sun_sensor.c` is unchanged.

### Six-head OV5640 cube array (`openadcs_sim/sun_sensor_array.py`)

Six OV5640 heads, one per cube face (boresights +/-X, +/-Y, +/-Z), use the head model above with
the OV5640 geometry (2592x1944 px, 1.4 um pixel, `ov5640_cube_heads` in `black_sun_sensor.py`).

- **Environment:** `EnvironmentSimulator(..., sun_sensor=SunSensorArray(...))` computes, per
  sample, the body-frame Sun vector (ideal nadir attitude unless `attitude_model` is given), the
  cylindrical-shadow eclipse flag and the six head readings. `EnvironmentSample.sun_sensor`
  holds per-head status/centroid, the fused body vector, the number of valid heads and the fused
  error in degrees. The JSON gains an optional `sun_sensor` key (schema 1 stays additive); the
  CSV gains `sun_sensor_*` columns only when the sensor ran. CLI: `--sun-sensor`,
  `--sun-sensor-sigma-px` (default 0.1), `--sun-sensor-diagonal-fov-deg` (default 120).
- **GUI:** the Sun tab has a "Sun sensor (OV5640)" panel: body az/el coverage map (number of
  heads seeing each direction, Sun marked), a per-head status table with the fused vector and
  error, and a fused-error history (eclipse shaded, valid-head count). The centroid-error entry
  re-evaluates the selected time; "Compute history" runs the whole recorded run in a background
  thread (decimated to the chosen step). The panel recomputes from the selected sample and the
  entered sigma; it does not require `--sun-sensor`.
- **Results (baseline 500 km / 98.5 deg orbit, one orbit, 20 s steps, sunlit samples):**
  coverage of the sphere 100 %, 18.8 % of directions seen by 2+ heads, at most 3 heads at once;
  fused error RMS / p95 = 0.007 / 0.013 deg at 0.1 px, 0.021 / 0.038 deg at 0.3 px,
  0.069 / 0.126 deg at 1.0 px. Eclipse: no valid heads, no estimate (by design).
- **Assumptions to confirm:** the "120 deg" module is taken as the DIAGONAL FOV (horizontal is
  then about 108 deg, vertical about 92 deg); with the standard 68.7 deg lens coverage drops to
  about 35 %. The 0.1 px centroid error is a placeholder, not a datasheet figure. The reported
  error is therefore optimistic: not modeled are lens distortion, off-axis pixel scale, rolling
  shutter, saturation/ND filtering, glare and Earth albedo, thermal drift, radiation and
  autofocus. Nothing here is flight validated.
- Not yet connected: ADCS ingress (architecture block `Sun sensor (6x OV5640 cube)` is PARTIAL),
  the C99 fusion, and torque-driven attitude (the body attitude is the ideal nadir view).

## Magnetometer digital model (assumption-based)

Benchmark of existing magnetometer models (primary sources checked 2026-10-11):

| Source | Model content | Flight linkage found |
|---|---|---|
| Basilisk `magnetometer` ([docs](https://avslab.github.io/basilisk/_sources/Documentation/simulation/sensors/magnetometer/magnetometer.rst.txt)) | `(B + noise + bias) * scale`, Gauss-Markov/white noise, saturation, stuck/spike faults, sensor mounting | None stated |
| AcubeSAT/adcs-simulation | Uses magnetometer measurement/noise data files in a Kalman tuning study | None established |
| ArthurThiam/Cubesat-Attitude, rammak/CubeSat-ADCS | No magnetometer model found | n/a |
| ESTCube-1 (Ehrpais thesis, [handle](https://hdl.handle.net/10062/47179)) | In-orbit: residual magnetic moment biased the magnetometers; UKF used to estimate/remove it; temperature calibration of magnetometers and gyros built from orbit data | Yes (flown), but the summary-level text read gives no reproducible equations or coefficients |

No source provided a complete, flight-validated forward noise model. Per your
decision, `openadcs_sim.magnetometer` implements an explicitly assumed, unvalidated
model: sensor mounting rotation, per-axis scale error, misalignment matrix,
temperature-dependent bias and scale (ESTCube-1-style), a constant residual
spacecraft field in the sensor frame, white and Gauss-Markov noise, quantization
and full-scale saturation. There are **no default parameters**; all values are
user characterization. Input must be the true field already in the body frame
(ITRS to GCRS to body conversion is not done by this model), and it has no ADCS
ingress yet; the diagram shows those links as planned. Fixed seeds make runs repeatable.


Use [CodeGraph for VS Code](https://marketplace.visualstudio.com/items?itemName=aStudioPlus.codegraph)
to explore module dependencies, callers/callees and symbols. The validated extension
and native engine version is **0.21.0**; VS Code 1.90 or newer is required.
This is a development tool, separate from the spacecraft simulator GUI.

The repository's `.vscode/` configuration recommends CodeGraph and the Python
extension, indexes the whole workspace, includes tests and private symbols, and
excludes virtual environments, portable tools, generated/build outputs, vendored
code and simulation datasets. Future onboard and HIL source folders under the
workspace are included without hard-coded folder names. CodeGraph telemetry is
disabled; full-body embeddings and embedding-on-open are disabled to reduce work.
Graph construction is local. Semantic search can still download public model
weights; it is not required for structural graphs. Using an AI assistant with
graph context is a separate action governed by that assistant's data policy.

### Open and navigate

For a normal VS Code installation, open the repository folder and install:

```powershell
code --install-extension aStudioPlus.codegraph@0.21.0
code .
```

The Windows worktree prepared for this project also has an ignored portable
VS Code installation with CodeGraph already installed:

```powershell
.\.tools\vscode\bin\code.cmd .
```

Its editor telemetry is disabled without changing other VS Code installations.
The engine and Windows ONNX runtime are installed in `%USERPROFILE%\.codegraph\bin`;
the graph cache is also outside the repository. The portable editor and engine
downloads were checked against official SHA-256 metadata, and the editor executable
has a valid Microsoft signature. `.tools/` is local installation state, not part
of a clone; other machines should follow the normal installation instructions.
On a new machine, use **CodeGraph: Download Analysis Engine** if prompted.

With a source file open, use the Command Palette:

- **CodeGraph: Show Dependency Graph** for file/module relationships.
- **CodeGraph: Show Call Graph** with the cursor on a function.
- **CodeGraph: Analyze Impact** before changing shared interfaces.
- **CodeGraph: Reindex Workspace** after large source changes or adding components.

CodeLens above functions and hover information provide additional navigation.
The 0.21.0 visualizer omits the separately returned call-graph root, which causes
its connected edges to disappear. This worktree has a local adapter repair.
To apply it again after reinstalling the same extension version:

```powershell
.\.venv\Scripts\python tools\repair_codegraph.py
```

Then run **Developer: Reload Window** and reopen the graph. The repair preserves
the selected function, normalizes node/edge identifiers, and keeps the correct
graph type when switching views. It does not invent missing call relationships
or fix parser limitations. It saves the original extension bundle beside the
patched file and refuses unknown versions or modified bundles; an extension
update may supersede or remove this local fix.

The default visualization depth is two to keep graphs readable. For example,
start with `simulation\src\openadcs_sim\environment.py` for provider dependencies,
`common.py` for shared helpers and test callers, and
`simulation\native\xyzgeomag.cpp` for native symbols.
**Tasks: Run Test Task** runs the existing simulator regression suite through the
repository virtual environment and source tree. The optional recommended Python
extension adds pytest discovery in VS Code's Test Explorer.

### Embedded software and HIL scope

CodeGraph supports C/C++ and Python source analysis, so it can help explore future
firmware, host-side HIL harnesses and their test relationships. It does **not**
execute HIL tests, flash a target, access serial/JTAG hardware, measure timing,
or validate hardware behavior. The repository now has portable C99 sensor
processing and ADCS input-health interfaces, but no attitude controller, board
drivers or HIL transport. The test task runs host tests, including compiled C99
processing through its Python adapter; these are not hardware HIL tests.

Graphs are heuristic, not compiler/linker truth. Dynamic Python dispatch,
native-language bindings, C/C++ function pointers/macros and excluded SDK/vendor
code may leave relationships unresolved. In the local check, CodeGraph found
the native `field` symbol and Python test symbols, and linked `utc_time` to callers
including tests. However, the `EnvironmentSimulator.sample` call graph had no
edges despite known calls in its body. An empty graph is therefore **not** evidence
that a function is unused or isolated. Keep source inspection, the target compiler,
debugger and real HIL runner authoritative.

### Tool comparison and benchmark

Reviewed online documentation and checked the selected tool locally on
2026-10-09. This capability comparison does not claim equal-workload performance
measurements of the alternatives.

| Tool | Python / C++ | Main use | Trade-off |
| --- | --- | --- | --- |
| [CodeGraph](https://github.com/codegraph-ai/CodeGraph) | Both | Local symbol, dependency and heuristic call graphs in VS Code | Apache-2.0 core; native engine required; call resolution is incomplete |
| [Understand](https://docs.scitools.com/help/editor/understand-for-vscode.html) | Both | Commercial source analysis and navigation | Desktop installation required; [API/export capabilities depend on licensing](https://scitools.com/pricing) |
| [pydeps](https://github.com/thebjorn/pydeps) | Python only | Module dependency diagrams | BSD-2-Clause; requires Graphviz for rendering; not a function call graph |
| [Sourcetrail](https://github.com/CoatiSoftware/Sourcetrail) | Both | Offline graphical source exploration | Original maintainers archived the project in 2021 |

CodeGraph's [published performance figures](https://marketplace.visualstudio.com/items?itemName=aStudioPlus.codegraph)
are vendor claims, not independently comparable timings. On this Windows worktree,
the 0.21.0 engine's **graph-only** mode indexed 23 source files into 387 nodes and
634 edges. One cold invocation, including startup, indexing, a dependency query
and shutdown, took **0.261 s**. Subsequent invocations loading the saved graph and
performing symbol/call queries took **0.135-0.177 s** each. These are single-run
smoke measurements, not query-only latency, editor rendering time, embedding
performance or a large-firmware benchmark. They do not establish that CodeGraph
is faster or more accurate than the alternatives.

## Attitude determination and sensor qualification

The requested sensor suite comprises a six-head OV5640 cube Sun sensor (which replaces the coarse and fine Sun sensors), an IMU, a horizon sensor and GPS. **Flight-derived Sun/gyro
processing and an explicitly assumed Orion B-16 PVT behavioral model are
implemented; attitude estimation remains disabled.** General qualification
requires documented in-orbit use of
the specific model or algorithm, not just flight heritage of its hardware, a
mission name in a README, ground testing, or preflight simulation.
For the specifically selected Orion B-16, the user subsequently authorized a
datasheet-based behavioral model with explicit statistical assumptions, rather
than requiring flight-calibrated receiver behavior. This exception does not
qualify the other deferred sensors.

### Reference benchmark

The following is a source/documentation comparison, not a runtime or accuracy
benchmark against flight telemetry. Reviewed 2026-10-09 at these revisions:

| Reference | Inspected revision | Relevant implementation | Assessment |
| --- | --- | --- | --- |
| [ArthurThiam/Cubesat-Attitude](https://github.com/ArthurThiam/Cubesat-Attitude/tree/fd59a3dc10be444eff649d9a66841d6682024d2d) | `fd59a3dc10be444eff649d9a66841d6682024d2d` | Arduino readings from 12 LDRs; Python calibration, thresholding and Sun-vector reconstruction | Bench prototype; no qualifying flight evidence found. Not a realistic space/noise sensor model. No repository license declared in the inspected metadata. |
| [rammak/CubeSat-ADCS](https://github.com/rammak/CubeSat-ADCS/tree/9dae14c4a5209d84ede7a3d5d1c6ec08c3eae0b3) | `9dae14c4a5209d84ede7a3d5d1c6ec08c3eae0b3` | The inspected `main.cpp` parses IGRF12 coefficients | README describes intended LORIS ADCS functions, but the inspected implementation does not provide the requested sensor models. A mission association does not establish flight use of this revision. No repository license declared in the inspected metadata. |
| [AcubeSAT/adcs-simulation](https://github.com/AcubeSAT/adcs-simulation/tree/53e28f5e07e3859172d3188946477df381804745) | `53e28f5e07e3859172d3188946477df381804745` | MATLAB CSS/albedo/noise models, gyro noise and attitude-filter simulation | Useful research reference, especially for gyro noise. No qualifying in-orbit evidence found for the inspected models. MIT repository with additional third-party license notices. |
| [Basilisk](https://github.com/AVSLab/basilisk/tree/1c89eed20c98ef36f3fcac10d6516d12d3307d80) | `1c89eed20c98ef36f3fcac10d6516d12d3307d80` | Segregated C++ CSS and IMU simulation modules with message interfaces and configurable errors | Strongest inspected CSS/IMU simulation starting point. The relevant files carry ISC licenses. Documentation/source inspected does not establish flight use of these specific models. |

The [Basilisk CSS documentation](https://hanspeterschaub.info/basilisk/Documentation/simulation/sensors/coarseSunSensor/coarseSunSensor.html)
and [inspected header](https://github.com/AVSLab/basilisk/blob/1c89eed20c98ef36f3fcac10d6516d12d3307d80/src/simulation/sensors/coarseSunSensor/coarseSunSensor.h)
provide an explicit FOV **half-angle**, sensor mounting geometry, a configurable
cosine-response correction, bias, white/correlated error configuration, saturation,
eclipse/albedo message inputs and stuck/off/random fault modes. Eclipse and albedo
are separate inputs, not effects obtained merely from a Sun direction.

The [Basilisk IMU header](https://github.com/AVSLab/basilisk/blob/1c89eed20c98ef36f3fcac10d6516d12d3307d80/src/simulation/sensors/imuSensor/imuSensor.h)
separates gyro and accelerometer error states, biases, scale factors, sensor-frame
mounting, offset from the spacecraft origin, discretization and saturation.
Noise propagation matrices can produce white or correlated errors. These
capabilities alone do not establish a fit to a flight instrument.

AcubeSAT's
[`gyro_noise_func.m`](https://github.com/AcubeSAT/adcs-simulation/blob/53e28f5e07e3859172d3188946477df381804745/src/gyro%20modeling/gyro_noise_func.m)
implements time-step-dependent white rate noise and a bias random walk, citing
Markley/Crassidis. Its separate
[`calculate_gyro_noise.m`](https://github.com/AcubeSAT/adcs-simulation/blob/53e28f5e07e3859172d3188946477df381804745/src/gyro%20modeling/calculate_gyro_noise.m)
combines white, pink and red noise for ARW/BI/RRW studies, but hard-codes a 0.1 s
characterization interval. A bias random walk is not by itself a physical model
of flicker-driven bias instability.

AcubeSAT's
[`css_noise.m`](https://github.com/AcubeSAT/adcs-simulation/blob/53e28f5e07e3859172d3188946477df381804745/src/sun%20vectors/css_noise.m)
uses six cosine responses, assumes albedo comes from nadir, adds a signed
multiplicative Poisson perturbation and normalizes reconstructed face differences.
This is not automatically a calibrated photon shot-noise model, nor a substitute
for explicit FOV, eclipse, invalid-measurement and instrument-response handling.
None of these four references establishes a mission-calibrated radiation-dose or
particle-flux sensor-error model. Generic fault injection must not be presented
as validated radiation physics.

### Expanded flight-evidence review

Research beyond the four repositories found primary flight evidence for
ESTCube-1 sensor processing and attitude estimation, and measured Phoenix GPS
flight performance. **Flight-qualified processing algorithms are not the same
as flight-validated stochastic sensor simulators.** Neither finding licenses
copying unreleased flight software or establishes that an arbitrary simulation
noise implementation is accurate.

The inspected primary sources are:

- Andris Slavinskis, *ESTCube-1 attitude determination* (2015),
  [University of Tartu record](https://hdl.handle.net/10062/49230) and
  [full thesis](https://dspace.ut.ee/server/api/core/bitstreams/f073b052-8faf-40e6-a515-b37a80ead0fa/content),
  especially Sections 3.4-3.5 and 4.1-4.3, printed pages 26-32.
- Hendrik Ehrpais, *EST-Cube-1 in-orbit attitude determination validation*
  (2015), [University of Tartu record](https://hdl.handle.net/10062/47179) and
  [full thesis](https://dspace.ut.ee/server/api/core/bitstreams/04981d07-ec57-435f-8306-ff3c374d5728/content),
  especially Sections 3.1-3.6 and Chapter 4, printed pages 12-24.
- Robert Valner, *Characterization of custom built Sun sensors for ESTCube-1*
  (2013), [University of Tartu record](https://hdl.handle.net/10062/30981).
  This is preflight characterization, not proof of successful in-orbit
  calibration.
- Slavinskis et al., *Attitude determination and control for centrifugal
  tether deployment on the ESTCube-1 nanosatellite* (2014),
  [publisher PDF](https://kirj.ee/public/proceedings_pdf/2014/issue_2S/Proc-2014-2S-242-249.pdf).
  This describes hardware, uncertainty budgets and simulation; the later
  theses establish the changes needed for successful flight operation.
- Markus Markgraf, *The Phoenix GPS Receiver - A Practical Example of the
  Successful Application of a COTS-Based Navigation Sensor in Different Space
  Missions* (WISEE 2020),
  [DLR record](https://elib.dlr.de/137677/) and
  [presentation](https://elib.dlr.de/137677/1/WISEE20-Markgraf-Phoenix.pdf),
  especially charts 5, 8-12 and 15-18.

**ESTCube-1 Sun processing and gyroscope calibration.** The mission used
two-axis slit/position-sensitive-detector (PSD) Sun sensors, ITG-3200
three-axis gyroscopes and magnetometers. Its initial Sun calibration did not
work in orbit. The flight correction used the PSD datasheet's normalized
differential-current position relation, restricted each Sun sensor's angular
acceptance from the original +/-45 degrees to +/-36.7 degrees, and reduced
the estimator's trust as incidence approached the acceptance boundary.
Measurements inside +/-20 degrees received the highest trust. This is a
directional analogue Sun sensor candidate, not a cosine-response coarse Sun
sensor, and not evidence of sub-degree standalone fine-Sun accuracy.

Gyroscope offsets were refined by comparing measured Sun vectors with vectors
extrapolated using measured angular rates; temperature-dependent offsets were
then compensated with polynomial fits. This establishes flight use of the
processing/calibration approach, **not** a full accelerometer-plus-gyroscope
IMU model. The published preflight uncertainty budgets cannot be substituted
for the recalibrated instrument's stochastic noise.

The ESTCube-1 unscented Kalman filter (UKF) ran onboard and was checked against
attitude reconstructed from onboard Earth images. Across 15 validation samples,
reported differences were below 1.44 degrees; 1.75 degrees is the **combined
expanded uncertainty of the comparison** (95% confidence, coverage factor 2),
not Sun-sensor noise, per-axis RMS, or a general estimator performance guarantee.
The validation thesis limits its uncertainty budgets to spin rates up to
7.1 degrees/s. It also reports problems recovering from nearly aligned Sun and
magnetic vectors at high spin rates. Copying its covariance values without
resolving state definitions, units and timing would not reproduce this result.

**Phoenix GPS.** DLR describes flight firmware functions including carrier-phase
smoothing and range-rate from carrier phase, RF-simulator HIL characterization,
and actual PROBA-2 operation. Charts 16-17 report approximately 2 m 3D RMS
position error, 5 cm/s velocity error, and one to two latch-up events every
48 hours on average, mainly near the South Atlantic Anomaly. Those are
instrument/mission-specific observations, not independent Cartesian Gaussian
standard deviations or a location-independent radiation-event law. The TID
and proton-test results provide useful qualification evidence, but do not
expose a complete reproducible receiver/error model or released flight firmware.
Position truth plus 2 m white noise would not reproduce this receiver.

**Other candidates.** CubeSpace's
[fine-Sun product description](https://www.cubespace.co.za/products/fine-sun-sensor/)
mentions a calibrated CMOS distortion model but does not expose its equations
or a specific flight implementation in the inspected text.
[`gotenham/infraredHorizonSensor`](https://github.com/gotenham/infraredHorizonSensor)
provides a thermopile-array prototype, subpixel edge processing and a simulated
orbital-view test stand, not successful orbital validation. Neither is qualified
for copying as a flight-proven model on this evidence.

The user selected independent implementations of verified processing and
interfaces, with unqualified models disabled, using portable C99 with fixed-size
data structures and no heap allocation. No flight-calibrated nominal noise profile, complete
flight-filter reproduction or HIL transport is enabled by the literature review
alone.

### Sensor disposition

| Requested sensor | Best inspected reference | Current disposition |
| --- | --- | --- |
| Coarse Sun sensor with FOV | Basilisk `coarseSunSensor` | Dropped from the system design: the six-head black-sun camera arrangement replaces it (modelled as six OV5640 cube heads, 120 deg diagonal-FOV variant: 100% sphere coverage, see above) |
| Fine Sun sensor | ESTCube-1 directional slit/PSD Sun processing | Removed from the system architecture (Phase 2): superseded by the six-head OV5640 cube Sun sensor. The legacy C99 `onboard/src/sun_sensor.c` is kept in the repository but is not part of the architecture |
| IMU | ESTCube-1 gyro calibration; Basilisk `imuSensor`; AcubeSAT gyro-noise studies | Independent gyro correction implemented; full IMU and stochastic/environment model remain deferred |
| Horizon sensor | `gotenham/infraredHorizonSensor` prototype | Deferred: inspected tests use a simulated orbital view, not orbital flight; AcubeSAT `GHorizon.m` is geometry, not an instrument |
| GPS | User-selected NavSpark/SkyTraq Orion B-16 | Datasheet-based PVT behavioral model and C99 binary navigation ingress implemented under explicitly authorized assumptions; not flight-calibrated or a clone of receiver firmware |

"Evidence not established" is not a claim that the algorithm has never flown.
Approval requires a traceable mission, successful on-orbit operation, an identified
model/algorithm and version or implementation mapping, and published performance
or validation evidence sufficient to justify its use. Copying any source also
requires a usable license.

### Embedded and HIL boundary

The intended separation is independent sensor producers, a transport-independent
ADCS estimator core, and interchangeable simulation, embedded-driver and HIL
adapters. Simulator truth must never silently replace missing sensor measurements.
The current ideal nadir reference is visualization truth, **not** an attitude
estimate or controller. The existing environment field/vector records are not
sensor outputs.

Before enabling a model, its configuration must specify mounting and frames,
units, sample timing, FOV/validity, bias and correlation parameters, scale and
alignment errors, quantization, saturation and supported environmental effects.
Future characterization and extreme-case profiles must be reproducible, distinguish
calibrated ranges from extrapolation, and preserve explicit out-of-FOV, eclipse,
stale-data, dropout and fault status. Parameters without a physical model or
measured basis must not be assigned invented "realistic" defaults.

A single-antenna GPS position measurement is not a direct attitude measurement.
A single Sun direction cannot constrain rotation about that direction. Sensor
availability and non-collinear reference geometry must therefore be reflected in
estimator observability and health, rather than returning a fully determined
attitude regardless of the available measurements.

### Implemented processing and interfaces

`onboard/` is a standalone C99 library using fixed-size records, no heap
allocation and only the standard C/math runtime. It is independent of Python,
SPICE, Huira, the orbit propagator and the GUI. The segregated modules are:

| Module | Supported behavior |
| --- | --- |
| `sun_sensor.c` / `sun_sensor.h` | Two slit/PSD channels per axis; offset/gain correction, normalized differential position, mask geometry, body-frame rotation and explicit signal/FOV/illumination validity |
| `gyro.c` / `gyro.h` | Per-axis quadratic temperature-offset correction, fixed gain correction, body-frame rotation, raw-rate saturation and calibrated temperature-domain checking |
| `gps.c` / `gps.h` | Bounded SkyTraq 0xA8 navigation-frame validation/decoding into typed WGS84 ECEF PVT records |
| `adcs.c` / `adcs.h` | Up to 16 Sun/gyro identities plus four GPS identities, typed ingress, timestamp ordering, independent freshness limits, redundant health and measurement availability |
| `types.h` | Versioned native ABI, nanosecond timestamps, sequence/identity fields and explicit status flags |

The Sun processor conservatively applies the published 36.7-degree limit as a
**boresight cone**, not a square accepting larger diagonal incidence angles.
Configurations may narrow this cone but cannot expand it beyond the published
flight limit. This is a documented conservative acceptance policy, not a claim
to reproduce the mission's exact aperture or firmware. Channel pairs are ordered
first/second according to the PSD differential-current relation; reversing a pair
reverses that axis. Mounting matrices are row-major, right-handed orthonormal
sensor-to-body rotations. Processed Sun vectors point **towards the Sun**.
Gyroscope inputs/outputs use rad/s; temperatures use degrees Celsius.

All geometry, offsets, gains, signal thresholds, thermal coefficients and
calibrated limits must be supplied explicitly. There are no invented nominal
hardware values or default noise profiles. The gyro module **applies**
characterization coefficients; it does not estimate them from flight data.
The flight UKF's angle-dependent covariance tuning is not implemented because
its state/units/timing must be resolved before a faithful port. These routines
are independent implementations of published processing approaches, not the
original flight source or a flight-qualified OpenADCS implementation.

Simulation, embedded-driver and future HIL producers share the measurement
ingress API. Use one common clock domain and persistent sensor identities;
restart the input state explicitly when starting a new playback/run or resetting
the clock. Duplicate or older timestamps are rejected without overwriting newer
records. A newer fault record invalidates that sensor's previous reading without
disabling redundant sensors. Future and stale records are excluded from health.
Sequence numbers are carried for transport diagnostics, not used as a clock.

Check every C status return and measurement status before reading its body
components. Invalid C outputs are cleared; the Python adapter exposes their
vector as `None`, not a zero vector pretending to be a measurement. Direct
illumination must be explicitly verified by the producer: an unknown
illumination state is invalid, and a geometric Sun direction does not establish
illumination during eclipse or exclude albedo.

`AdcsInputs` reports measurement availability only. Multiple Sun heads still
observe one reference direction: their instantaneous reference-rank upper bound
is two. `attitude_available` remains false, even with valid Sun and gyro readings.
There is no quaternion estimate, uncertainty claim, controller or substitution
of the existing ideal nadir reference. CSS, full IMU and horizon kinds are
explicitly rejected as unqualified rather than filled from environment truth.
GPS requires its own ECEF PVT record; a GPS label on a body-vector record is
rejected. Valid GPS can establish navigation availability, not attitude.
The current GUI remains an environment/reference visualization.

#### Build and use on the ground

From this checkout in PowerShell:

```powershell
# Optional host compiler, also declared in the onboard-test extra.
.\.venv\Scripts\python.exe -m pip install ziglang==0.16.0
.\.venv\Scripts\python.exe tools\build_onboard.py
$env:PYTHONPATH = Join-Path (Get-Location) 'simulation\src'
.\.venv\Scripts\python.exe -m pytest -q simulation\tests\test_sensor_processing.py
```

The build writes `build\onboard\openadcs_processing.dll` on Windows. The
`--cc` option selects an existing GCC/Clang-compatible compiler or Zig executable;
`onboard\CMakeLists.txt` also supports static builds with an embedded toolchain,
or shared builds with `BUILD_SHARED_LIBS=ON`. Host tests compile the current
source with C99 and warnings treated as errors. Building/running on a desktop
does not verify target RAM, CPU timing, board drivers or hardware behavior.

`simulation\src\openadcs_sim\sensor_processing.py` supplies `NativeProcessing`,
explicit `SunConfig`/`GyroConfig`, `SampleHeader`, `Measurement` and `AdcsInputs`.
Load the built library using `NativeProcessing(library_path)`, process producer
readings with `sun(...)` or `gyro(...)`, submit each result to
`AdcsInputs.submit(...)`, and inspect `health(...)` with explicit maximum ages.
For GPS ingress, pass `maximum_gps_age_ns` to `AdcsInputs` and decode a complete
SkyTraq frame with `NativeProcessing.gps_decode(frame, header)`. Sensor timestamps
must already be mapped into the shared clock domain by the producer.
Fault/status results must be checked; library/ABI/configuration errors raise
explicit exceptions. No library is loaded automatically by the simulator.

The native struct ABI is **not a serial/network wire format**: padding and
alignment are platform-dependent. The GPS subset uses the separately documented
SkyTraq wire format. UART streaming/drivers, time synchronization,
covariance-bearing estimator messages and the flight-derived UKF remain future
work. The native ABI is now version **2**; rebuild the shared library when
updating the Python adapter. The source distribution includes the C
core and build helper; simulation wheels do not bundle a platform-specific
processing library.

### Orion B-16 GPS digital model

The selected receiver is the
[NavSpark/SkyTraq Orion B-16](https://navspark.mybigcommerce.com/12mm-x-16mm-gnss-receiver-module-for-leo-applications/).
The manufacturer advertises orbital flight heritage; the inspected page does
not identify a mission/firmware revision or publish flight error telemetry.
The SkyTraq **Phoenix chipset is not the DLR Phoenix receiver** discussed above.
Do not transfer DLR flight accuracy or radiation-event rates to this module.

Primary documents inspected:

| Document | Actual downloaded revision |
| --- | --- |
| [Orion B-16 datasheet](https://navspark.mybigcommerce.com/content/Orion_B16_DS.pdf) | 1.5, 2022-09-16 |
| [SkyTraq binary interface AN0037](https://navspark.mybigcommerce.com/content/AN0037.pdf) | 1.4.69, 2025-03-26; newer than the product page's download label |

SHA-256 identities and revisions are recorded in the model's exported
provenance. The model is implemented in
`simulation\src\openadcs_sim\orion_b16.py`; the portable onboard decoder lives
in `onboard\src\gps.c`. Receiver behavior is kept separate from orbit truth.

**Supported behavior:** configured hot/warm/cold acquisition, signal loss,
reacquisition, main-power cycling, supported 1/2/4/5/8/10/20 Hz PVT sampling,
and explicit no-fix/out-of-domain states. Datasheet defaults are 18/46/50 seconds
for hot/warm/cold acquisition and 16 seconds for reacquisition. These are
**open-sky averages represented as deterministic dwell-time assumptions**, not
guaranteed LEO acquisition times. Interrupting qualifying conditions restarts the
dwell. Backup-SRAM retention is not simulated; a power cycle uses the configured
start mode. Powered-off/out-of-domain samples are diagnostics and emit no modeled
wire packets. During acquisition/no fix, GPS time and navigation fields are
unavailable; they are not filled with orbit truth.

The behavioral domain uses 3.0-3.6 V, -40 to +85 degrees Celsius, inertial speed
up to 10,000 m/s and acceleration up to 39.2 m/s2. Conditions outside that domain
invalidate the solution; the actual physical failure response is unknown.
Cold/warm/hot acquisition uses -148 dBm as a conservative assumption;
reacquisition uses -160 dBm and tracking -165 dBm. Four available satellites
are necessary for a modeled 3D fix. Availability, representative signal power,
antenna status, DOP and dynamics are supplied conditions, **not computed by a
GNSS constellation/RF solver**.
The manufacturer accuracy/sensitivity figures assume a good external LNA; this
model does not simulate LNA gain/noise or establish that those conditions hold.

**Error assumptions:** the published 2 m CEP is not a 3D RMS or a per-axis
standard deviation. Under an explicitly assumed isotropic horizontal Gaussian,
east/north sigma is CEP divided by sqrt(2 ln(2)). Vertical sigma, velocity sigma,
correlation times and seed must be supplied. The published 0.1 m/s velocity
accuracy has no stated probability convention and is not automatically converted
to sigma. Configured errors are independent ENU position/velocity Gauss-Markov
processes, rotated into ECEF; zero correlation time selects white errors.
This is a behavioral approximation, not calibrated pseudorange/clock/tracking
physics. DOP is supplied metadata, not a solved geometry or noise multiplier.

GPS week/TOW uses Astropy UTC-to-GPS conversion with leap-second handling.
Configured clock bias/jitter affects PVT time; the datasheet's 5 ns **1PPS**
specification is not claimed for this model. The binary TOW is quantized to
0.01 seconds, position/velocity to 0.01 m and 0.01 m/s, and latitude/longitude to
1e-7 degrees. A supplied geoid undulation separates ellipsoid height from MSL
height. There is no implicit zero-geoid assumption in the model.

**Protocol/HIL boundary:** encode/decode the 59-byte navigation payload
(`0xA8`), big-endian fields, XOR checksum, framing, GPS time, fix mode,
satellite count, geodetic fields, DOP and ECEF position/velocity. The decoder
requires a complete 66-byte frame and rejects corruption and invalid fields.
No-fix and 2D records do not provide a usable 3D PVT solution. The model does
not emulate NMEA, receiver commands/ACKs, UART byte timing, carrier phase,
navigation firmware, antenna gain patterns, atmospheric propagation or
radiation. Main ADCS input health counts valid/stale GPS records separately and
never treats a single-antenna GPS position as an attitude observation.

#### Run the GPS model

Use a recent TLE and the intentionally **illustrative**, non-flight-calibrated
profile in `simulation\examples\orion_b16_assumptions.json`. Its vertical and
velocity sigmas, correlation times, satellite/DOP conditions, acceleration and
zero geoid are stated assumptions, not measured Orion hardware parameters.
The CLI derives inertial speed from the propagated GCRS state; the profile's
speed placeholder is not used. Edit the other conditions/parameters to represent
characterization or deliberately exaggerated cases.

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'simulation\src'
.\.venv\Scripts\python.exe -m openadcs_sim.gps_cli `
  --tle .\spacecraft.tle `
  --profile .\simulation\examples\orion_b16_assumptions.json `
  --start 2026-10-09T12:00:00Z --samples 120 `
  --output .\gps.jsonl --binary-output .\gps.bin
```

No SPICE kernels or magnetic model are required. The JSON-lines report starts
with provenance and assumptions, then separates measured navigation records
from orbit truth for comparison. Binary output uses the documented default
one-second navigation-message interval even when internal PVT updates are
faster. Actual hardware command configuration and serial transport remain
separate future work. The existing GUI does not yet show GPS receiver results.

## Sun sensor albedo

Each OV5640 head receives Earth-reflected light computed from the exact sphere-to-plane view factor (Lambertian Earth, Bond albedo 0.30, scaled by the Sun zenith at the sub-satellite point; zero on the night side). The centroid is pulled toward the Earth-lit direction by `leak x (E_albedo/E_sun) x pixel offset` (ASSUMPTION: leak 0.002, no characterisation data). CLI: `--albedo` (0 disables), `--albedo-leak`; GUI: Sun tab fields. Over one 500 km / 98.5 deg orbit the fused RMS rises from 0.006 to 0.007 deg (leak 0.002) or 0.020 deg (leak 0.01). Eclipse already marks every head `ECLIPSE`. Not modelled: penumbra, regional albedo, specular glint.

## Sun sensor calibration inputs and envelope validation

Baseline (GUI and default simulation): 120 deg DIAGONAL FOV pinhole (vendor f = 1.78 mm, aperture 2 mm), principal point at the image centre, centroid sigma 0.1 px, centroid bias 0, dropout 0, heads perfectly aligned. Open point: f = 1.78 mm at 1.4 um/px is 1271 px, which as a pinhole gives 103.8 deg diagonal, not 120 deg (the model uses f = 935 px for 120 deg); the lens is a fisheye, so a real projection and distortion model needs hardware calibration. `centroid_bias_px` is the residual constant centroid offset left after calibration (a systematic error that averaging cannot remove). Head misalignment is modelled per head (`mount_sigma_deg` Gaussian per axis or explicit `mount_errors_deg`): the head physically sits misaligned while the estimator assumes the nominal cube axes.

Envelope run: `python -m openadcs_sim.sun_sensor_envelope --output report.json` evaluates the baseline, all 2^6 low/high corners of (sigma, bias x/y, albedo leak, albedo, FOV) with and without the +-2 deg mount bound (128 corners) plus Gaussian Monte Carlo. All extreme ranges are PLACEHOLDERS (`EnvelopeSpec`). First result (placeholder ranges, 200 Sun directions): baseline RMS 0.011 deg; worst corner 2.7 deg, dominated by the 2 deg misalignment.

## Spacecraft definition (mass properties input)

`Spacecraft > Define spacecraft...` (or **Define...** in the Spacecraft box) opens a separate window to edit the name, mass, centre of mass and the 3x3 inertia tensor about the centre of mass. Entries are in CAD units (g, mm, g mm^2); the SolidWorks "positive tensor" convention (off-diagonals are products of inertia) is handled by a checkbox. **Apply** sets the spacecraft used by the GUI, **Save/Load JSON** use SI (`mass_kg`, `com_m`, `inertia_kg_m2`). The baseline is `simulation/data/spacecraft/pilot1_v2.json` (Pilot1_V2: 500 g, principal moments 8.775e-4 / 2.072e-3 / 2.483e-3 kg m^2). The CLI accepts `--spacecraft file.json` (validated), and the JSON export embeds the spacecraft. Assumption: the CAD output axes are the body axes. The Dynamics model will consume this configuration.
## Attitude dynamics (torque-driven truth)

`simulation/src/openadcs_sim/dynamics.py` integrates the rigid-body equations (quaternion body to GCRS, body rate, RK4, 2 s max step) with the inertia from the spacecraft JSON. Torques: gravity gradient plus an `external_torque(t, q, omega_B)` hook where the actuator and disturbance models will plug in. It implements `AttitudeModel`, so it replaces the ideal nadir reference when enabled: GUI checkbox "Torque-driven attitude (dynamics)" in the Spacecraft box, or `python -m openadcs_sim.cli --dynamics [--initial-rate WX WY WZ]`. The start state is nadir-pointing at the orbit rate. Scrubbing backwards restarts from stored checkpoints.

Status: PARTIAL. No actuator, aerodynamic, solar-radiation or residual-magnetic torques and no specific-force output yet. Note: the Pilot1 V2 inertia (Iz largest) is gravity-gradient unstable about nadir, so with dynamics enabled it drifts away from nadir without control.
## Actuators (reaction wheels and magnetorquers)

`openadcs_sim.actuators` provides a 4-wheel pyramid (54.74 deg skew, isotropic) of custom 10 g wheels and 3 orthogonal magnetorquers, integrated by DynamicsAttitudeModel (wheels=, magnetorquers=, controller=, magnetic_field=).

- Wheel: J_w w_dot = u - tau_friction, torque and speed limited (smooth taper near the speed limit), Coulomb + viscous friction. Body: I w_dot = tau - w x (I w + h_w) - A J_w w_dot; total momentum is conserved (tested).
- The 10 g wheel is an **assumed** scaling of the Sinclair RW-0.003 (50 g, 1 mNm, 3 mNms, 8500 rpm): geometry scaled by (m/50 g)^(1/3), inertia and torque by (m/50 g)^(5/3) -> about 0.069 mNm and 0.21 mNms; friction terms are assumed fractions of the torque limit.
- Magnetorquer: 	au = m x B_body, per-axis dipole limit 0.2 A m^2 (**assumed**, generic CubeSat value); dot_dipole gives the B-dot detumble law.
- Not modelled yet: wheel imbalance, back-EMF, torque ripple, current loop, core hysteresis. The ADCS control law is described next.

## ADCS (estimation and control)

`openadcs_sim.adcs_control` (Python reference, C99 port NOT done) and `openadcs_sim.closed_loop` (test harness: truth dynamics + sensor models + ADCS, 1 Hz).

- **Estimator**: a 6-state multiplicative EKF (attitude error + gyro bias), gyro propagation, vector updates from the Sun sensor (sunlit), horizon nadir (eclipse; "law" mode) and magnetometer. TRIAD initialises it from any pair 15-165 deg apart. A UKF or IMM was not needed: measurement dropouts (eclipse, Sun loss) are handled by gating, and the measured error is small.
- **Modes**: DETUMBLE (B-dot, `m = k w x B`, wheels idle) -> ACQUISITION (wheel rate damping) -> NOMINAL (nadir-pointing PD + integral, gyroscopic and orbit-rate feed-forward, wheel friction feed-forward).
- **Wheels / magnetorquers**: wheels do all pointing; magnetorquers (`m = B x tau / |B|^2`, GNSS position + field model for B_ref) unload wheel momentum with hysteresis, and are switched off one cycle in five so the magnetometer reads clean. A zero-net-momentum null-space bias (about 250 rad/s per wheel) keeps wheels away from zero speed, where friction sign changes made them stall.
- **Measured (analytic 500 km / 51.6 deg environment, Pilot1 inertia, 0.05 rad/s initial tumble, large initial attitude error)**: detumble in minutes; after 1000 s pointing error median 0.28 deg in sunlight (max 1 deg) and 0.7 deg in eclipse (max 2 deg), limited by the estimator.
- **Assumptions (optimistic)**: ideal GNSS (true r, v), exact inertia and wheel friction known to the ADCS, "law" horizon model (not flight proven), no magnetorquer interference on the magnetometer, no wheel/torquer imperfections, `RealEnvironment` (SGP4 + XYZgeomag + Huira, baseline TLE, 3500 s) checked once manually: pointing median 0.5 deg, max 2.3 deg, ~47 % sunlit; not covered by a test. Gyro and magnetometer baselines are assumptions.

## Ground simulation

The ground simulator lives in `simulation/` and prioritizes accuracy, inspectability,
and ease of integration. The portable sensor-processing/input-health core is
separate in `onboard/`. Neither an onboard attitude estimator/controller nor a
hardware HIL transport is implemented yet; environment truth and processed sensor
measurements remain separate boundaries.

Initial models:

| Component | Implementation | Scope |
| --- | --- | --- |
| Orbit | Vallado SGP4 through `python-sgp4`, WGS72 | TLE-based propagation; TEME converted to ITRS using Astropy |
| Magnetic field | [XYZgeomag](https://github.com/nhz2/XYZgeomag), WMM2025 | Main field in Earth-fixed Cartesian coordinates, tesla |
| Sun | [Huira 0.9.9](https://github.com/huira-render/huira) scene and time, NAIF SPICE via SpiceyPy | Geometric spacecraft-to-Sun direction and distance |

The [linked propagation article](https://www.opensourcesatellite.org/open-source-solutions-multi-satellite-orbit-propagation-challenge/)
benchmarks several SGP4 implementations; it is not a propagator itself. Vallado SGP4
is recommended here for compatibility with TLEs, not as a precision numerical
propagator. Accuracy depends on TLE quality and age. A seven-day age limit is enforced
in both directions; this is a guardrail, not an accuracy guarantee. For precision
mission analysis, add a numerical orbit model with gravity harmonics, atmosphere,
third-body gravity, and SRP behind the existing orbit interface, initialized from
an appropriate Cartesian state rather than treating TLE mean elements as osculating.

Huira's published 0.9.9 roadmap still lists SRP modeling as in progress. This version
creates an actual Huira Sun/spacecraft scene, but uses SpiceyPy for ephemeris queries
because Huira's Python bindings do not expose direct state queries. It does **not**
yet produce radiation forces, irradiance, eclipses, albedo, sensor readings, or
body-frame vectors. The geometric Sun direction remains defined during eclipse.

### Setup and run

Requires Python 3.10 or newer and a C++ compiler (MSVC Build Tools on Windows).
XYZgeomag is compiled as a small Python extension; it is not reimplemented in Python.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
.\.venv\Scripts\python simulation\tools\fetch_kernels.py
.\.venv\Scripts\openadcs-sim --tle satellite.tle --kernels simulation\data\kernels --start 2026-10-09T12:00:00Z --step-seconds 1 --samples 60
.\.venv\Scripts\python -m pytest
```

Supply a real TLE close to the requested start time, including checksums. The TLE
file accepts two element lines or a name followed by two element lines. The CLI
emits one JSON object per timestamp to stdout. Invalid input, SGP4 errors, missing
kernels, and unsupported model coverage fail explicitly.

### Desktop interface

The desktop GUI is designed for aerospace engineering students: numbered setup
steps, visible units and model limits, TLE validation, UTC epoch selection, progress
and cancellation, seven visualization tabs, a sample inspector, and spreadsheet exports.

```powershell
.\.venv\Scripts\python -m pip install -e ".[gui,test]"
.\.venv\Scripts\openadcs-sim-gui
```

Alternatively launch with `python -m openadcs_sim.gui`. Tkinter is bundled with
standard Windows Python; Linux installations may require the distribution's
`python3-tk` package. Matplotlib is an optional GUI dependency, not required for
the CLI. GUI tests are skipped when Matplotlib or a Tk display is unavailable.

Import a TLE file or paste its lines, select the previously downloaded kernel
folder, choose a UTC start (or **Use TLE epoch**), duration in minutes, and time
step in seconds. **Load synthetic LEO** provides a clearly labelled classroom
exercise at 2026-10-09 12:00 UTC; it is not an observed satellite orbit. The
**Guide** tab explains setup, reference frames, units, sampling and model limits.
The ground track is a latitude/longitude coordinate grid, not a geographic basemap.
The 3D trajectory is Earth-fixed, not an inertial Kepler ellipse.

After a run, use the shared **Animation** toolbar to rewind, step through recorded
samples, play forward/reverse, pause, or jump to the end. The default **60x** speed
advances 60 simulation seconds per real second; choose a preset, enter any finite
positive multiplier, or use **Slower/Faster**. Dragging the timeline pauses playback.
One UTC cursor synchronizes the spacecraft views, ground-track/altitude and
trajectory markers, magnetic/Sun plot cursors, and sample readouts. The **Sun** tab also
shows the OV5640 cube Sun-sensor panel (coverage, per-head status, fused error).

Playback uses actual recorded timestamps and monotonic wall time, not rendering
frame counts. The displayed state is the recorded sample at or immediately before
the animation UTC: there is no interpolation or physics rerun. High speeds may skip
displayed samples, but never remove them from the dataset. Playback stops at either
endpoint; playing forward from the end restarts at the beginning, and reverse from
the beginning starts at the end. Partial results can be replayed but remain labelled
partial. A single recorded sample can be inspected, but cannot be animated.
Controls follow STK's shared animation/time and graphics-window conventions
([Animate reference](https://help.agi.com/stk/Subsystems/connectCmds/Content/cmd_Animate.htm));
STK is not required to run this tool.

The **Window** menu opens/closes individual visualizations or the active visualization,
including **System architecture**, which is available even before a simulation.
The architecture diagram includes environment providers, the requested sensors,
generic actuators, the ADCS MCU boundary and onboard-computer MCU. Click a block
or connector to inspect variables, units/frames, implementation limits and source
paths. **Fit diagram** shows an overview with compact variable labels; **100%**,
zoom buttons, scrollbars and middle-button panning provide readable full labels.
The module dropdown selects and centers a block at readable zoom.

Blue blocks are implemented (GPS under its documented assumptions), amber blocks
are partial, and gray dashed blocks are **PLACEHOLDER**. Solid arrows indicate
available API/offline paths; dashed arrows are planned. This is a **static
architecture map, not live bus activity or an executing sensor/control pipeline**.
Sun/gyro processing is not a complete forward/noise model; C99 input health is
not an attitude estimator or MCU emulator. Magnetic reference truth is not a
magnetometer measurement, GPS PVT is not an attitude estimate, and the existing
ideal nadir visualization is not torque-driven attitude dynamics.
Sun/magnetic references are ITRS vectors, while `R_BG` maps body to GCRS;
future forward/estimation paths must perform the time-dependent frame conversion,
not apply a body/GCRS rotation directly to ITRS data.

Segregated Python boilerplates live in
`simulation\src\openadcs_sim\planned\`. Their `evaluate()` interfaces explicitly
raise `NotImplementedError`; they never return nominal measurements, successful
MCU execution or zero actuator torque. They describe future model boundaries,
not embedded firmware or working HIL transports. Implementation metadata and
labeled connections are shared in `architecture.py`; the view does not execute
the modules. Hide/reopen, playback and simulation reset preserve diagram state.

The Window menu can also open or close all visualizations.
Closing a tab hides it without discarding results, animation
time, or its camera/zoom. Reopening displays the current shared time. **Samples**,
**Guide**, and **Diagnostics** remain available even with all graphics closed.

The separate **Spacecraft** and **Earth context** tabs follow the selected sample.
The local view contains a procedural 3U CubeSat
(0.1 x 0.1 x 0.34 m bus with deployable panels), body axes, and normalized direction
arrows for inertial velocity, magnetic field, Sun and the +Z attitude boresight.
Arrow lengths are schematic, not physical magnitudes.

The initial attitude is the selected ideal nadir/orbit-following reference:
body +Z points to Earth's center, +X follows the horizontal projection of inertial
velocity, and +Y completes the right-handed frame. Inertial velocity includes
Earth rotation; the Earth-fixed velocity is not simply relabelled as inertial.
This reference follows the orbit without disturbances or ADCS dynamics; it is
not a constant inertial attitude. `AttitudeState` stores a full body-to-GCRS
rotation matrix and `AttitudeModel` is the replacement interface for future ADCS.
A boresight arrow alone does not describe the complete attitude.

The **Earth context** tab also uses spacecraft-fixed coordinates. Earth's
center and its WGS84 surface are translated and rotated into the same body frame,
using 6378.137 km equatorial and 6356.752314 km polar radii. Earth and its distance
are to scale in kilometers; the spacecraft is a marker, not an enlarged physical
satellite. The local model uses meters. Separating these scales prevents a
visible spacecraft model from misrepresenting Earth's radius or altitude.
Environment JSON/CSV schemas remain unchanged; attitude currently belongs to the
visualization layer, not a new simulated controller or torque model.

Runs use fresh CLI subprocesses to isolate SPICE kernel pools and avoid freezing
the interface. Inputs are locked while running. **Stop** preserves received
samples as partial results; failed runs are also labelled partial, with errors
and coverage/IERS warnings visible in **Diagnostics**. Samples remain in memory;
large-run confirmation warns about memory consumption. Plots display at most
3000 points, while exports retain all samples.

CSV column headers specify student-friendly units (km, Earth-fixed km/s,
microtesla, degrees and AU). JSON preserves SI environment records and includes
the exact TLE, requested timeline, kernel filenames/path, and completion status.
Exports describe the scenario used for the run, even if the form is later edited.
Archive the actual kernel/EOP files separately; export metadata is not a copy of
those files.

Kernel downloads are explicit and go into ignored `simulation/data/`. DE440s
covers 1849-2150, but the Earth orientation kernel has a narrower, rolling time
range. Archive the exact kernels and Earth orientation data with each experiment
for reproducibility; rerunning the download command refreshes the rolling kernel.
Huira and SpiceyPy each maintain process-global kernel pools: run independent kernel
configurations in separate processes, not concurrent simulator threads.

### Coordinate, time, and fidelity contract

All sample times are timezone-aware UTC. Position is in meters, Earth-fixed velocity
in meters per second, magnetic field in tesla, and Sun distance in meters.
`sun_direction` is a unit vector pointing **from spacecraft to Sun**.
JSON includes `schema_version: 1` and `frame: "ITRS"`; no frame-less ECI values cross
the boundary. SPICE realizes ITRS as ITRF93. Astropy uses IERS Earth orientation
data for TEME-to-ITRS transformations. Its data downloads/warnings remain enabled;
do not ignore degraded Earth orientation warnings in accuracy studies. Differences
between terrestrial realizations and EOP products remain in this initial model.

WMM2025 is restricted to 2025-2029 and geodetic altitudes -1 to 850 km; higher
orbits require another model. XYZgeomag uses single-precision arithmetic internally
even though the Python simulator otherwise uses double precision. WMM models the
main field, not magnetic storms or other external fields. For higher-fidelity
magnetic studies, evaluate IGRF plus external-field models as a separate provider,
without silently replacing the selected XYZgeomag model.

The orbit, magnetic, and Sun providers are replaceable through small typed
interfaces. A future HIL bridge will additionally need spacecraft attitude,
body-frame transformations, simulated sensors, synchronization, and a wire protocol.

### Upstream code

The vendored XYZgeomag header is pinned to commit
`a7d09fa06a0392421d2c353c7d106261bf0e2f76` (including WMM2025), not the older
v2.1.0 release header. Its MIT license is retained in the header. Huira is pinned
to 0.9.9; other dependency ranges are declared in `pyproject.toml`.
Header SHA-256: `cf9d397e36800d76ce365de297f824f78e632fdc9e5af8a09762bd0c562bea0e`.

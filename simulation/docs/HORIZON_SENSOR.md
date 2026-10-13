# Horizon sensor (2 × Waveshare Thermal-45) — digital model, error law, ESP32 plan

> **STATUS: NOT flight proven.** This is an original concept developed in this project. Every number
> below comes from a *simulation* with the ASSUMPTIONS listed in §8, not from hardware. Re-derive the
> error law once real thermal images exist.

## 1. Hardware modelled

| Item | Value | Source |
|---|---|---|
| Camera | Waveshare Thermal-45, Meridian MI0802 LWIR, ESP32-S3 (8 MB PSRAM) | Waveshare wiki / datasheet |
| Array / FOV | 80 × 62 px, 45° × ≈34° | wiki |
| Pixel pitch / band | 35 µm, 8–14 µm | datasheet |
| Noise | 0.2 / 0.63 / 1.09 K at 1 / 10 / 29.3 FPS (÷√N when averaging) | datasheet |
| Scene range | −20 … +175 °C (253 K floor) | datasheet |
| Dead pixels | < 0.5 % | datasheet |

## 2. Model chain (all in `simulation/src/openadcs_sim/`)

1. `horizon_camera.py` – spherical-shell Earth limb with warm plateau, 8–14 µm Planck radiance, optical
   PSF, motion smear (thermal τ), FPN, temporal noise, dead pixels, floor/ceiling clipping, 0.1 K quantisation.
2. `horizon_processing.py` – the **reference processing chain to port to the ESP32**: dead-pixel repair,
   smoothing, radial gradient ridge search, sub-pixel limb points, floor-shift compensation, Huber
   Gauss–Newton fit of the cone \(d\cdot n=\cos\gamma\), \(\sin\gamma=(R_t+h)/r\).
3. `horizon_sensor.py` – architectures, ESP32 timing, error law, ADCS-level fusion, stateful model
   (rate limit, latency, dropouts). Two modes: `image` (full simulation) and `law` (fast equivalent law).
4. `horizon_montecarlo.py` – Monte Carlo campaign that derives the law (`simulation/docs/horizon_montecarlo.json`).

Frame: +Z nadir, +X velocity. `roll = atan2(−n_y, n_z)`, `pitch = asin(n_x)`. **Yaw is unobservable.**

## 3. Why a range-aided cone fit

With a 45° camera the visible limb arc is only ≈45–50°. A free conic fit gave **23° rms** error; the
cone fit with the known orbit radius gives **0.18° rms** (single camera, 500 km). The sensor therefore
needs `r` (GPS/orbit) and assumes an apparent limb height of 38 km (ASSUMPTION).

## 4. Floor-clipping limb shift (key finding)

Space is colder than the 253 K sensor floor, so the ridge of the clipped image moves with the Earth
plateau temperature (several tenths of a degree). The processing samples the plateau temperature behind
each edge point and corrects with a table calibrated by running the real chain noiselessly
(`limb_offset_table`). Compensation reduces the plateau-temperature dependence by > 60 %. The table must be
re-derived from hardware data.

## 5. Error law

\(e = b + \sigma_w w + \sigma_c\,\mathrm{GM}(\tau)\), per axis, by |latitude| band (30/60/90°), with
availability = valid fraction. \(\sigma_w\) is the noise-only robust σ; \(\sigma_c=\sqrt{\max(\sigma_{cons}^2-\sigma_w^2,0)}\)
with \(\sigma_{cons}=\max(\mathrm{std},\,p_{95}/1.96)\) (conservative: heavy-tailed cloud/cold-scene errors are
folded into the correlated term). \(\tau=120\) s is an ASSUMPTION. Values: `DEFAULT_ERROR_LAWS` (500 km).

## 6. Pointing error (300 trials, seed 7, 500 km, full scene)

| Architecture | Valid | σ roll / pitch (°) | Pointing p95 (°) | Bias roll / pitch (°) | Noise-only p95 (°) |
|---|---|---|---|---|---|
| 1 camera | 0.82 | 1.89 / 0.76 | 3.7 | −0.23 / −0.43 | 0.84 |
| 2 independent (90°) | 0.91 | 1.34 / 0.90 | 2.85 | +0.22 / −0.32 | 0.47 |
| 2 fused | 0.91 | 1.35 / 0.98 | 2.84 | +0.17 / −0.26 | 0.61 |

By |latitude| (independent): 0–30° p95 0.96°; 30–60° 2.0°; 60–90° 5.5° (cold plateau near the 253 K floor, clouds).
400 km and 700 km behave similarly (700 km slightly worse).

## 7. Findings and recommendation

* **Weak axis**: one camera measures one axis well (along its azimuth) and the orthogonal one poorly.
  Two cameras at **90–120°** azimuth separation are best (180° measures one axis twice).
* **Scene errors dominate**: clouds and cold high-latitude regions, not sensor noise.
* **Limb-height mismatch** ≈ 0.023°/km (≈0.24° per 10 km). The fused architecture solves γ jointly and cuts
  this to ≈0.08°, but adds variance and ≈50 ms of UART transfer latency, so it is *not* better than
  independent in noise.
* **Recommendation**: INDEPENDENT at 90–120°, fusing at the ADCS (`fuse_nadir_estimates`, information form).
  If a fused variant is wanted, share limb edge points (~400 B) instead of frames (9,920 B).
* Frame averaging helps mostly the single camera (noise-only rms 0.31° at 29 fps×1 vs 0.46° at 10×1 vs 0.31° at 10×10 for two cameras).
* A hypothetical 213 K floor showed larger biases in simulation: **inconclusive**, do not rely on it.

## 8. Timing (ESP32-S3)

Period 0.1 s (10 FPS); latency 66 ms (single/independent), 121 ms (fused); max trackable rate ≈ 49 °/s;
CPU load 16–21 %. ASSUMPTIONS to measure on target: 100 cycles/px filter, 400/candidate, 350/point-iteration,
30 k fixed, ×1.5 memory factor, 240 MHz, SPI 10 MHz ×1.3, UART 2 Mbaud, thermal τ 12 ms.

## 9. ASSUMPTIONS (all unvalidated)

Atmosphere/cloud statistics (warm 285 K, polar drop 35 K, cloud mean 8 K / σ 12 K, limb σ 4/3 km, correlation 10°),
PSF σ 0.6 px, FPN 0.3 K, edge-point σ 0.35 px, apparent limb height 38 km, SSO 97.5° scene, attitude error ±3°,
yaw ≈ 0, GM τ = 120 s, timing costs (§8). Yaw is not observable from a horizon sensor.

## 10. Tests

`test_horizon_camera.py`, `test_horizon_processing.py`, `test_horizon_sensor.py` and
`test_horizon_golden.py` (ESP32 golden-vector contract, see `onboard/esp32_horizon/README.md`).

Reproduce: `python -m openadcs_sim.horizon_montecarlo --trials 300 --seed 7 --out simulation/docs/horizon_montecarlo.json` (~5 min).

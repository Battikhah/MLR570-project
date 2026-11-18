# Screen results after the first weekend (snapshot 2026-11-16)

## What is in this folder

- **`runs.csv`:** one row per finished run, 655 runs.
- **`REPORT.md`:** the output of `scripts/analyze_suite.py`.

Both were generated from the `result.json` files on Google Drive (`gdrive:pcd-suite/outputs/screen/`), which were copied on 2026-11-16 after the session ended.

Each `result.json` records the code commit it ran:
- `8ef78cc` (204 runs)
- `57b787c` (148)
- `791c1b1` (117)
- `9ff7b81` (111)
- `378c330` (64)
- `e7bfcab` (8)
- `da90af7` (2)
- `9d6e4e9` (1)

The raw `result.json` files (with time series) stay on Drive. They are not committed.

To regenerate:

```bash
rclone copy gdrive:pcd-suite/outputs <dir> --include "result.json"
uv run python scripts/analyze_suite.py --out <dir> --profile screen
```

These are screening results: 60 epochs (240 for `lit-p1`), CIFAR-100 unless noted, test top-1 in %. Identical CE runs differ by about 0.5 pp (sd), so single-seed gaps under about 1.5 pp carry no information. The intervals below are 95% t-intervals on seed-paired differences. There is no multiplicity correction.

Labels follow `AGENTS.md`:
- **VERIFIED** means read from these runs.
- **INTERPRETATION** is our reading of them.

`REPORT.md`'s headline table picks configurations on validation, including some with 1 seed, so read it together with the tables below.

## Teacher corruption (teacher outputs corrupted with probability ρ), 5 seeds at ρ 0 / 0.5 / 0.7

| Method | ρ0 | ρ0.5 | ρ0.7 | λ_eff ρ0 → ρ0.7 |
|---|---|---|---|---|
| CE | 55.48 | 55.47 | 55.57 | 0 |
| WKD 0.9 | 56.34 | 47.85 | 41.62 | fixed |
| DOT δ0.075 | 58.23 | 48.63 | 41.05 | fixed |
| PCD-reversed τ0.5 | 56.11 | 48.84 | 44.47 | – |
| NWKD 1.0 | 56.35 | 54.21 | 52.60 | 1.0 |
| NWKD 0.5 | 56.46 | 55.38 | 54.90 | 0.5 |
| PCD τ1.0 | 56.17 | 54.83 | 53.76 | 0.42 → 0.75 |
| PCD τ0.5 | 55.52 | 55.61 | 55.20 | 0.01 → 0.27 |
| PCDL τ1.0 | 56.41 | 54.96 | 53.12 | 0.42 → 0.75 |
| PCDLM τ1.0 | 57.21 | 55.42 | 53.11 | 0.65 → 0.92 |

**VERIFIED, paired over 5 seeds:**

| Comparison | ρ | Difference (pp) | 95% interval |
|---|---|---|---|
| PCD τ0.5 − CE | 0.7 | −0.37 | [−0.86, +0.12] |
| PCD τ0.5 − NWKD 0.5 | 0.7 | +0.30 | [−0.20, +0.79] |
| PCD τ1.0 − NWKD 1.0 | 0.7 | +1.15 | [+0.42, +1.88] |
| PCD τ1.0 − NWKD 0.5 | 0.7 | −1.14 | [−2.08, −0.21] |
| PCD τ1.0 − DOT | 0.7 | +12.70 | [+11.65, +13.76] |
| PCD τ0.5 − NWKD 0.5 | 0 | −0.94 | [−1.40, −0.48] |
| PCD τ1.0 − DOT | 0 | −2.06 | [−2.70, −1.43] |
| PCDLM τ1.0 − PCD τ1.0 | 0 | +1.05 | [+0.29, +1.80] |
| PCDLM τ1.0 − PCD τ1.0 | 0.7 | −0.65 | [−1.13, −0.17] |

**INTERPRETATION:**
- **PCD never drops meaningfully below CE when the teacher is bad,** while WKD and DOT lose 15–17 pp.
- **Its weight is what protects it.** PCD's KD weight rises as the teacher gets worse, because the CE–KD cosine falls. It stays bounded by τ, and a fixed NWKD weight of 0.5 protects as well.
- **PCD τ0.5 gives up KD's benefit on the clean teacher,** because its λ_eff is about 0.01 there.
- **PCDL changes nothing.** Every layer sees the same cosine.
- **PCDLM measures the cosine on momentum buffers,** where it is lower. So it uses more KD: that helps with the clean teacher and costs a little with the bad one.

## Label noise (noisy student labels, clean teacher), 2 seeds

| Method | sym 0.0 | sym 0.4 | sym 0.8 | asym 0.4 |
|---|---|---|---|---|
| CE | 55.56 | 44.72 | 13.33 | 36.43 |
| DOT | 57.93 | 58.51 | 58.01 | 58.20 |
| NWKD 4.0 | 55.62 | 56.31 | 56.43 | 56.02 |
| WKD 0.9 | 56.33 | 55.99 | 55.84 | 56.05 |
| PCD τ1.0 | 56.45 | 54.77 | 54.65 | 50.06 |
| PCD τ0.1 (1 seed) | 55.33 | 45.35 | 21.00 | 37.05 |

**INTERPRETATION:**
- **The methods that lean on the teacher win here.**
- **PCD's λ_eff moves the right way,** rising from 0.42 to 1.32 for τ1.0 at sym 0.8. But it stays below the heavy fixed weights, and small τ follows the noisy labels.

## Clean teacher

**core-p0, ResNet32 → ResNet8, own teachers, 3 seeds:**
- PCGrad 58.30, DOT 58.14
- PCD τ1.0 56.95, WKD 0.5 56.82, NWKD 1.0 56.61, MGDA 55.92
- CE 55.44

Paired:
- PCD τ1.0 − NWKD 1.0: +0.33 [−0.47, +1.14]
- DOT − PCD τ1.0: +1.19 [+0.74, +1.65]

**lit-p1, CRD ResNet32x4 → ResNet8x4, mdistiller protocol, 2 seeds:**
- DOT 74.87, PCGrad 74.43, NWKD 1.0 73.62, PCD τ2.0 73.20, WKD 0.9 73.05, CE 72.16.
- **VERIFIED against mdistiller's tables:** KD is 73.33 published vs our 73.05, and DOT 75.12 vs our 74.87, so the pipeline reproduces the published protocol.

## Single-seed families (leads only)

- **temperature:** WKD's spread over T ∈ {1, …, 16} is 5.7 pp, against 0.4 pp for PCD τ1.0 and 1.7 for NWKD.
- **pcd-ablations:**
  - Removing momentum costs every method 4–5 pp.
  - Removing PCD's rescaling costs 3.7 pp.
  - The EMA β does nothing.
  - Batch 256 hurts every method.
- **multi-loss-dkd:** DKD 72.53, PCD τ1.0 71.47, NWS 71.05–71.70, MGDA 64.50.
- **capacity-gap, long-tail, core-p1:** DOT or PCGrad lead. PCD sits with NWKD.

## Not reportable yet

**DTO-KD** is below CE in every family, for example 50.08 vs 55.44 on core-p0. That points to an implementation problem. It is held back until audited (decision 0005).

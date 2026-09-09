# Canonical Local-Volatility Surface Construction — Heston-Nandi (ρ = −1)

Numerical implementation of the canonical Dupire/Markovian local-variance surface for the
Heston model under perfect negative spot-variance correlation **ρ = −1** (Heston-Nandi
model), based on the paper's exact gap-density Laplace-transform characterisation.

## Research Summary

The project verifies five theoretical claims:

| Exp | Title | Key Result |
|-----|-------|------------|
| 1 | Surface construction & structural validation | λ(t,s) ≥ 0, λ ≤ A⁺, exact hard barrier |
| 2 | Mimicking validation — spot law equality | Law(S_T^H) = Law(S̄_T^LV) within tolerance |
| 3 | Convex-order inequality for variance calls | E[(I_T^LV-K)⁺] > E[(I_T^H-K)⁺] for K > 0 |
| 4 | Equal means, variance ordering, vol-swap | All corollaries confirmed numerically |
| 5 | Proof mechanisms — pathwise identity, PDE | Identity RMSE ≈ 10⁻¹⁷, Duhamel consistent |

Full results are in [`results/RESULTS.md`](results/RESULTS.md).

## Model Parameters (baseline)

```
S0 = 100, v0 = 0.04, κ = 2.0, θ = 0.04, ξ = 0.3, ρ = −1
β = κ + ξ/2 = 2.15,  a = κθ = 0.08
```

## Repository Structure

```
├── data/               # Synthetic only — no market data
├── src/
│   ├── config.py           # Model parameters & numerical settings
│   ├── transforms.py       # Laplace transform ingredients L_t, M_t
│   ├── inversion.py        # Numerical inverse Laplace (Euler/Abate-Whitt)
│   ├── localvol_surface.py # Surface φ_t(q) → λ(t,s) → μ(t,x) construction
│   ├── heston_nandi.py     # Heston-Nandi CIR simulation (ρ = −1 exact)
│   ├── localvol_sim.py     # Log-Euler local-vol SDE simulation
│   ├── payoffs.py          # Integrated-variance call, vol-swap, soft-call payoffs
│   └── diagnostics.py      # Structural validation (barrier, nonnegativity, …)
├── exp/
│   ├── exp1_surface_construction.py
│   ├── exp2_mimicking_validation.py
│   ├── exp3_convex_order.py
│   ├── exp4_secondary_consequences.py
│   ├── exp5_proof_mechanisms.py
│   └── run_all.py          # Orchestrator — runs all 5 experiments
├── tests/              # 96 pytest tests (unit + integration)
└── results/            # PNGs, CSVs, RESULTS.md
```

## Quick Start

```bash
# Install dependencies (all standard quant Python)
pip install numpy scipy pandas matplotlib statsmodels mpmath

# Run all experiments (≈ 60 s with 50 000 paths)
python exp/run_all.py

# Or run a single experiment
python exp/exp1_surface_construction.py

# Run the test suite
pytest tests/ -q
```

## Key Numerical Choices

| Setting | Value | Rationale |
|---------|-------|-----------|
| Inverse Laplace method | Euler / Abate-Whitt (N=14, A=19) | Stable for sub-exponential growth |
| CIR simulation | Noncentral χ² exact transition | Zero-bias for variance process |
| I_t^H quadrature | Trapezoidal on CIR path | O(dt²) for smooth v_t |
| Local-vol SDE | Log-Euler with barrier clipping | Preserves nonnegativity |
| Near-barrier asymptotics | Proposition 11: φ_t(q) ≈ 4a/(β(v0+at)²)·q² | Engaged for q < 10⁻⁶ |
| Branch convention | Principal √ and log, real-axis continuation | Documented in transforms.py |

## References

Paper: *Convex ordering of integrated variances for the Heston-Nandi (ρ = −1) model
and its canonical local-volatility projection* (canonical gap-density construction).

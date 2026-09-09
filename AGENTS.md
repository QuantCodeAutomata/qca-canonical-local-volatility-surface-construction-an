# AGENTS.md — Repository Knowledge

## Project Purpose
Numerical validation of the canonical local-volatility surface for the Heston-Nandi (ρ = −1) model.
Five experiments test: surface construction (Exp 1), spot-law mimicking (Exp 2), convex-order inequality (Exp 3), secondary corollaries (Exp 4), and proof mechanisms (Exp 5).

## Key Numerical Conventions
- **Branch convention**: `d(z) = np.sqrt(b² + 2β·c²·z)` uses NumPy principal square root (positive real part).  Logarithm uses `np.log` (principal branch) with real-axis continuation — documented in `src/transforms.py`.
- **Inverse Laplace**: Abate-Whitt Euler method, `N=14` terms, `A=19` (gives ~6 decimal digits). Implemented in `src/inversion.py::euler_invert` / `euler_invert_array`.
- **Near-barrier asymptotics**: for `q < 1e-6`, use Proposition 11 approximation `φ_t(q) ≈ 4a / (β·(v0+a·t)²) · q²` to avoid division-by-near-zero in `g_t(q)`.
- **CIR simulation**: exact noncentral chi-square transition via `scipy.stats.ncx2`. Integral `I_t^H` accumulated with trapezoidal rule.
- **Log-Euler local-vol**: `dX̄_t = -0.5·μ(t,X̄_t)·dt + √μ(t,X̄_t)·dB_t`. Hard barrier: `μ(t,x)=0` for `x ≥ x*(t)`.
- **Pathwise identity**: `log(S_t/S0) = (v0 + κθt − v_t − β·I_t^H)/ξ` — verified to RMSE ≈ 1e-17 (machine precision).

## Module Map
| File | Responsibility |
|------|---------------|
| `src/config.py` | `ModelParams` dataclass, `NumericalParams` dataclass |
| `src/transforms.py` | `laplace_ingredients`, `compute_laplace_transforms` |
| `src/inversion.py` | `euler_invert`, `euler_invert_array`, `build_gt_mt_arrays` |
| `src/localvol_surface.py` | `build_phi_table`, `LocalVolSurface` (callable), `build_surface` |
| `src/heston_nandi.py` | `simulate_heston_nandi` → returns `v, I_H, S` |
| `src/localvol_sim.py` | `simulate_localvol` → returns `X_bar, I_LV` |
| `src/payoffs.py` | `integrated_variance_call`, `volswap_expectation`, `soft_call` |
| `src/diagnostics.py` | `validate_surface`, `check_pathwise_identity`, `mimicking_check` |
| `exp/run_all.py` | Orchestrates all 5 experiments, writes `results/RESULTS.md` |

## Test Suite
- 96 tests in `tests/` — run `pytest tests/ -q` (≈ 3 s).
- Tests use small grids / few paths to stay fast; no live API calls.

## Experiment Runtime (baseline hardware)
With `n_paths=50000, n_steps_per_year=150, maturities=[0.5,1.0,2.0]`:
- Exp 1 (surface): ~15 s
- Exp 2 (mimicking): ~9 s
- Exp 3 (convex order): ~8 s
- Exp 4 (secondary): ~16 s
- Exp 5 (mechanisms): ~7 s
- Total: ~55 s

## Known Results (baseline parameters)
- Exp 1: All 4 structural properties PASS.
- Exp 2: Max call error 0.04–0.11; 0 barrier violations; pathwise residual RMSE ~1e-17.
- Exp 3: 59/60 strikes show ΔC > 0; 0 statistically significant failures (p < 0.05). Theorem supported.
- Exp 4: Equal means ✅, Var^LV > Var^H ✅, E[√I^H/T] > E[√I^LV/T] ✅ at all maturities.
- Exp 5: Pathwise identity RMSE ≈ 1.3e-17; Duhamel LHS/RHS same sign; conditional covariance > 0.

## Pitfalls Discovered
1. **euler_invert factor**: the Abate-Whitt formula uses factor `exp(A/2)/t` NOT `exp(A)/t` — using `A/2` in the exponent and dividing by `t` gives the correct scaling.
2. **exp3 theorem_supported logic**: a single near-zero negative ΔC with p ≈ 0.52 is Monte Carlo noise; `theorem_supported` should only flip to False when there are *statistically significant* failures (p < 0.05 and ΔC < 0).
3. **Branch cut for d(z)**: when `b² + 2βc²z` passes close to the negative real axis (large negative Im(z)), NumPy's `sqrt` can flip branches discontinuously.  Keep inversion contour shift `c0 ≥ 1.0` to stay safely in the right half-plane.
4. **`g_t(q)` near zero**: at very small q the density can be numerically small but positive; always blend with the Proposition 11 quadratic for `q < q_small = 1e-6`.

# LLM and agent context

Kraken ships a compact agent-oriented context at `/llms.txt` in the built
site. It is intentionally stricter than a marketing page: it tells an agent
what this branch can run, what it must not claim, and where the longer docs
live.

## What the agent should know first

- Public workflow starts from `.krk` files.
- Main entry point: `run_simulation("examples/cavity.krk")`.
- Parsed setup entry point: `setup = load_kraken("file.krk")`;
  `run_simulation(setup)`.
- File-path `run_simulation` accepts `max_steps` and parser kwargs.
- Parsed-setup `run_simulation(setup)` does not accept `max_steps`.
- v0.1.0 public physics: BGK single-phase, Guo forcing, thermal DDF,
  Boussinesq natural convection.
- Not public in this branch: MRT, axisymmetric, grid refinement, VOF,
  phase-field, Shan-Chen, species, rheology, viscoelasticity, SLBM.

## Result policy

An agent should cite only benchmark values that are either on
[Accuracy](benchmarks/accuracy.md), [Performance](benchmarks/performance.md),
or backed by a committed CSV in `benchmarks/results/`.

The currently rerun convergence checks are:

- Poiseuille: order 2.00.
- Taylor-Green: order 1.99/2.00/2.00.
- Thermal conduction: order 1.00 with the current wall treatment.
- Natural convection `Ra = 1e3`, `N = 64`: `Nu = 1.1423`, error `2.17%`.

## Long-form references

- [Getting started](getting_started.md)
- [Capabilities](capabilities.md)
- [Integration roadmap](integration_roadmap.md)
- [.krk overview](krk/overview.md)
- [Public API inventory](api/public_api.md)
- [Julia ecosystem docs](julia_docs.md)

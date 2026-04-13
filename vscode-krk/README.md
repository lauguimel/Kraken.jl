# Kraken Simulation (.krk)

VSCode language extension for Kraken LBM simulation configuration files.

## Features

- **Syntax highlighting** for all .krk keywords, parameters, lattice types, boundary conditions, and numeric literals
- **Diagnostics** with real-time validation: missing required blocks, unknown keywords with fuzzy suggestions, stability warnings (tau check)
- **Auto-completion** context-aware: keywords at line start, parameters after Physics, boundary types after face names, modules, presets, rheology models
- **Hover documentation** for all keywords, physics parameters, lattice types, boundary conditions, modules, and rheology models
- **Snippets** for common simulation setups: cavity, poiseuille, couette, taylor-green, rayleigh-benard, droplet, cylinder

## Snippets

Type `sim:` to see available templates:

| Prefix | Description |
|--------|-------------|
| `sim:cavity` | Lid-driven cavity flow |
| `sim:poiseuille` | Poiseuille channel flow |
| `sim:couette` | Couette flow |
| `sim:taylor-green` | Taylor-Green vortex decay |
| `sim:rayleigh-benard` | Rayleigh-Benard convection |
| `sim:droplet` | Static droplet (Laplace test) |
| `sim:cylinder` | Flow around a cylinder |

## Build

```bash
npm install
npm run compile
```

## Install locally

```bash
# From the extension directory
code --install-extension .
```

Or copy the folder to `~/.vscode/extensions/krk-language-0.1.0/`.

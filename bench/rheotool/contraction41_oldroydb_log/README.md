# rheoTool 4:1 contraction reference — Oldroyd-BLog

Project-local copy of rheoTool's `rheoFoam/Contraction41/Oldroyd-BLog`
tutorial for the axis-aligned log-FV comparison.

Parameters from `constant/constitutiveProperties`:

- `rho = 0.01`
- `etaS = 0.11111`
- `etaP = 0.88889`
- `lambda = 1`
- model: `Oldroyd-BLog`

The stock tutorial uses a ramped inlet and runs to `t = 20`. It writes field
snapshots every `2` time units and runs `postProcess -func sampleDict` at the
end.

Run locally with:

```sh
./run_docker.sh
```

The comparison harness consumes the latest nonzero time directory containing
`U` and `tau`, plus optional sampled/probe outputs when present.

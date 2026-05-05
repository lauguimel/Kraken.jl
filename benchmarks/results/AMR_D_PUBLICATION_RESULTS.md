# AMR D Publication Result Manifest

Date: 2026-05-05

## Primary Aqua Results

- `amr_d_publication_raw_2d_aqua_D_pub_20260505_long.csv`
- `amr_d_publication_summary_2d_aqua_D_pub_20260505_long.csv`
- `logs/amr_d_publication_2d_aqua_D_pub_20260505_long.log`

PBS job: `20812821.aqua`

## Local Canary Results

- `amr_d_publication_raw_2d_local_D_pub_canary_20260505.csv`
- `amr_d_publication_summary_2d_local_D_pub_canary_20260505.csv`
- `amr_d_publication_raw_2d_from_krk_from_krk_smoke_20260505.csv`
- `amr_d_publication_summary_2d_from_krk_from_krk_smoke_20260505.csv`

The local canary is only a reporting-path smoke. Use the aqua files for
publication discussion and figures.

## Reproducibility

The source cases are in:

- `benchmarks/krk/amr_d_publication_2d/*.krk`

The `.krk` runner is:

- `benchmarks/amr_d_publication_from_krk_2d.jl`

The env-driven runner used for the recorded aqua job is:

- `benchmarks/amr_d_publication_table_2d.jl`
- `hpc/amr_d_publication_table_2d_aqua.pbs`

Both runners produce the same raw/summary schema.

## Figure Generation

Use:

```bash
julia --project=. benchmarks/plot_amr_d_publication_2d.jl
```

Outputs:

- `figures/amr_d_publication_2d_summary.png`
- `figures/amr_d_publication_2d_summary.pdf`

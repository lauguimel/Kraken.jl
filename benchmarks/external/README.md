# External benchmark data

This directory contains published performance and accuracy data from
other LBM codes, used for comparison in the Kraken.jl publication.

## Files

| File | Description |
|------|-------------|
| `literature_data.csv` | Machine-readable table of all external numbers |
| `comparisons.md` | Human-readable summary with DOI links |

## Data sources

Every row in `literature_data.csv` has a `source_doi` column.
If the DOI is empty, the `notes` column explains the provenance
(e.g. "estimated from ...").

Primary references:

1. Bauer et al. (2021) — waLBerla, `10.1016/j.cpc.2020.107746`
2. Latt et al. (2021) — Palabos, `10.1016/j.camwa.2020.03.022`
3. Krause et al. (2021) — OpenLB, `10.1016/j.camwa.2020.04.033`
4. Januszewski & Kostur (2014) — Sailfish, `10.1016/j.cpc.2014.04.018`
5. Popinet (2009) — Basilisk/Gerris, `10.1016/j.jcp.2009.04.042`

## How to update

1. Add rows to `literature_data.csv` following the existing column format.
2. Always include `source_doi` for published numbers.
3. Mark estimated values with "estimated" in the `notes` column.
4. Update `comparisons.md` to reflect any new entries.

## How docs reference this data

The documentation pages under `docs/` can cite specific rows by
(code, case, metric) key. The CSV is the single source of truth
for all external numbers mentioned in the paper or docs.

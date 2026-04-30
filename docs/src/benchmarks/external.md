# External comparisons

External rankings are intentionally not published from this branch yet.

The repository contains draft comparison material under
`benchmarks/external/`, but the Kraken numbers in those files must be
reconciled with committed benchmark CSVs before they are cited as results.
This matters because older docs contained H100 throughput claims that are not
traceable to a current `benchmarks/results/` CSV.

Use this page as the publication rule:

- cite external projects only with a primary source and hardware context;
- cite Kraken numbers only when a matching CSV, command, hardware id and
  commit are available;
- never compare D2Q9 and D3Q19 MLUPs without explaining the memory traffic
  difference;
- keep development-branch features such as SLBM/body-fitted meshes out of the
  v0.1.0 comparison until they have their own validation artifacts.

For current in-branch checks, see [Accuracy](accuracy.md) and
[Performance](performance.md).

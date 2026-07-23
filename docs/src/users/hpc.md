# Running Kraken on HPC clusters

Kraken is **GPU-first**: production runs use `CUDABackend()` on NVIDIA GPUs, and
the identical KernelAbstractions source also runs on CPU for short tests. On a
typical HPC cluster the workflow is:

1. install Julia in user space with **juliaup** (no root required),
2. clone or `Pkg.add` Kraken and **instantiate/precompile on the login node**
   (CPU-only — this is fine, precompilation does not need a GPU),
3. submit GPU runs as **batch jobs** through the scheduler (PBS Pro, SLURM, …).

This page walks through each step, with a complete, working PBS Pro example
(the Aqua cluster at QUT) and a short SLURM variant. For a workstation or
laptop install, see [Installation](../installation.md).

## User-space installation

### Julia via juliaup

juliaup installs entirely under your home directory — no administrator needed:

```bash
curl -fsSL https://install.julialang.org | sh
# then make it visible in batch jobs too:
export PATH="$HOME/.juliaup/bin:$PATH"   # add to ~/.bashrc
```

### Julia depot location

Julia stores packages, compiled caches, and artifacts in its *depot*
(`~/.julia` by default). On clusters where home is small or slow, point it at
your work/scratch filesystem — but beware of scratch **purge policies** (files
deleted after N days of inactivity would silently wipe your packages):

```bash
# Only if your home quota is too small; prefer a non-purged filesystem.
export JULIA_DEPOT_PATH="/work/$USER/.julia"
```

### Project setup and precompilation

Clone your project (or a Kraken checkout) into your home or work directory and
instantiate it **once on the login node**:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

This resolves the environment, downloads packages, and compiles everything on
CPU, so your first GPU job does not burn walltime on package downloads. Always
run with `--project=.` (or an absolute project path) for reproducibility.

### Pinning the CUDA runtime (driver older than the default runtime)

CUDA.jl downloads a recent CUDA runtime as an artifact. If the cluster's
**driver is older** than that runtime, jobs can fail in a confusing way: the
solver starts, then **kernels die during `CuModule` load with an empty error
message** (or a bare `CUDA error`), even though `CUDA.functional()` is `true`.

The fix is to pin the runtime to what the driver supports. Check the driver's
maximum supported CUDA version with `nvidia-smi` (top-right of the header),
then, once per depot:

```julia
using CUDA
CUDA.set_runtime_version!(v"12.2")   # match your driver, e.g. driver 535 → 12.2
```

Restart Julia and verify with `CUDA.versioninfo()`. Some clusters instead ship
a system CUDA compatibility layer that conflicts with the artifact runtime; in
that case set `JULIA_CUDA_USE_COMPAT=false` in the job script (the Aqua
example below does).

## PBS Pro example (Aqua, QUT)

Aqua uses PBS Pro with GPU nodes carrying H100 (80 GB) or A100 (40 GB) cards,
selected via the `gpu_id` resource. The script below is the same mechanics as
Kraken's own benchmark jobs
(`benchmarks/krk/inc_ns/cavity_gpu_bench.pbs`): request one GPU, point Julia
at the project, verify the GPU, run.

Save as `kraken_gpu.pbs`:

```bash
#!/bin/bash -l
#PBS -N kraken_cavity
#PBS -l select=1:ncpus=4:ngpus=1:mem=32GB:gpu_id=A100
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -V

# Project directory (clone/instantiate beforehand on the login node)
WORKDIR=$HOME/kraken_runs/cavity
cd $WORKDIR

echo "Job:   $PBS_JOBID"
echo "Node:  $(hostname)"
echo "Start: $(date)"

# juliaup lives in user space; make it visible to the batch shell
export PATH=$HOME/.juliaup/bin:$PATH
# Use the node's CUDA driver stack, not CUDA.jl's compat layer (required on Aqua)
export JULIA_CUDA_USE_COMPAT=false

# Pre-flight GPU check: fail fast instead of wasting walltime
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  || { echo "ERROR: GPU not available"; exit 1; }

# Ensure the environment is ready and CUDA is functional
julia --project=. -e 'using Pkg; Pkg.instantiate(); using CUDA; CUDA.versioninfo()'

# The actual run (any script that uses CUDABackend())
julia --project=. benchmarks/krk/inc_ns/cavity_gpu_bench.jl 2>&1
EXIT_CODE=$?

echo "End: $(date)  exit=$EXIT_CODE"
exit $EXIT_CODE
```

Line by line:

- `#!/bin/bash -l` — **login shell**, required on PBS clusters so the module
  system and your profile are available.
- `select=1:ncpus=4:ngpus=1:mem=32GB:gpu_id=A100` — one node chunk with 4 CPU
  cores, one GPU, 32 GB RAM. `gpu_id=H100` requests an H100 instead; A100s
  usually queue faster. Request a few GB more RAM than your data size — Julia's
  JIT needs headroom.
- `walltime=04:00:00` — hard kill time. Budget the *first* run generously:
  the first `using Kraken; ...` on a new depot triggers precompilation of the
  GPU code paths (minutes, not seconds).
- `#PBS -j oe` — merge stdout and stderr into one log file.
- `#PBS -V` — export your environment variables into the job, so options can
  be passed at submit time with `qsub -v NAME=value` and read in the Julia
  script via `ENV`.
- `nvidia-smi` pre-flight — fail immediately if the GPU allocation is broken,
  rather than after minutes of Julia compilation.
- `Pkg.instantiate()` in the job is a cheap no-op when the login-node
  instantiate already ran; it protects against a stale `Manifest.toml`.

Submit, monitor, and collect outputs:

```bash
qsub kraken_gpu.pbs                 # prints the job id, e.g. 1234567.pbs
qsub -v CAVITY_BENCH_1024=1 kraken_gpu.pbs   # same script, option via env var

qstat -u $USER                      # queue state (Q = queued, R = running)
qstat -f 1234567                    # full details of one job

# After completion: merged log appears next to the script
ls kraken_cavity.o1234567
```

Simulation outputs (VTK files, checkpoints, CSV tables) land wherever your
Julia script writes them — put them on the work/scratch filesystem, not in a
small home quota.

## SLURM variant

On SLURM clusters the same job looks like this (`kraken_gpu.slurm`):

```bash
#!/bin/bash -l
#SBATCH --job-name=kraken_cavity
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x.o%j

cd $SLURM_SUBMIT_DIR
export PATH=$HOME/.juliaup/bin:$PATH

nvidia-smi || { echo "ERROR: GPU not available"; exit 1; }
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. my_kraken_script.jl
```

Submit with `sbatch kraken_gpu.slurm`, monitor with `squeue -u $USER`, cancel
with `scancel <jobid>`. Some sites use `--gpus=1` or a typed request such as
`--gres=gpu:a100:1` — check your cluster's documentation.

## Containers (Apptainer)

On clusters where installing juliaup is impractical (restricted home, exotic
OS), run Kraken from a container: convert the official
[Julia Docker image](https://hub.docker.com/_/julia) to a SIF with
`apptainer build julia.sif docker://julia:1.11` and run your script with
`apptainer exec --nv julia.sif julia --project=. script.jl`. The `--nv` flag
passes the host NVIDIA driver through to the container; keep the Julia depot
bind-mounted on the host filesystem so packages persist across runs.

## Good practice

- **Never run long jobs on the login node.** Login nodes are shared and
  processes are killed without warning; anything beyond
  `Pkg.instantiate()`/precompile belongs in a batch (or interactive) job.
- **Write outputs and checkpoints to scratch/work**, not home — and copy
  results off scratch promptly if the cluster purges inactive files.
- **Check GPU exclusivity** with `nvidia-smi` at job start: you should see
  your allocated GPU idle (0 % utilization, no other processes) before Kraken
  starts. Note that on some clusters *interactive* GPU sessions hand out MIG
  slices rather than full GPUs — use batch jobs for real runs.
- **Budget walltime for first-run precompilation.** A fresh depot compiles
  the CUDA kernels on first use; add comfortable margin to the first job, or
  warm the cache with a tiny run.
- **Fail fast**: keep the `nvidia-smi` and `CUDA.functional()` pre-flight
  checks in every job script.

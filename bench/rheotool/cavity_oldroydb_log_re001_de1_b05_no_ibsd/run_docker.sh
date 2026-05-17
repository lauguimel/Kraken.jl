#!/usr/bin/env bash
set -euo pipefail

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image="guiguitcho/openfoam9-rheotool:v1.2"

docker run --rm \
    --platform linux/amd64 \
    -v "${case_dir}:/data" \
    -w /data \
    "${image}" \
    bash -lc '
        source /opt/openfoam9/etc/bashrc
        export PATH="/home/openfoam/platforms/linux64GccDPInt32Opt/bin:${PATH}"
        export LD_LIBRARY_PATH="/home/openfoam/platforms/linux64GccDPInt32Opt/lib:/home/openfoam/OpenFOAM/openfoam-9/ThirdParty/petsc-3.16.5/arch-linux-c-opt/lib:${LD_LIBRARY_PATH:-}"
        ./Allclean
        ./Allrun
    '

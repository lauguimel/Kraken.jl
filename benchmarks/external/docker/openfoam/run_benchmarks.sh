#!/bin/bash
# Source OpenFOAM environment
source /opt/openfoam10/etc/bashrc 2>/dev/null || true

cd /benchmarks/cases

# --- Cavity ---
echo "=== OpenFOAM Cavity Re=100 ===" >&2
cd cavity

echo "Running blockMesh..." >&2
blockMesh > log.blockMesh 2>&1
if [ $? -ne 0 ]; then
    echo "blockMesh FAILED:" >&2
    tail -5 log.blockMesh >&2
    exit 1
fi
echo "blockMesh OK" >&2

echo "Running icoFoam (endTime=50, dt=0.005)..." >&2
START=$(date +%s%N)
icoFoam > log.icoFoam 2>&1
ICOFOAM_EXIT=$?
END=$(date +%s%N)

if [ $ICOFOAM_EXIT -ne 0 ]; then
    echo "icoFoam FAILED:" >&2
    tail -10 log.icoFoam >&2
    exit 1
fi

CAVITY_NS=$((END - START))
CAVITY_SECS=$(python3 -c "print(round(${CAVITY_NS} / 1e9, 3))")
echo "icoFoam done in ${CAVITY_SECS}s" >&2

# Post-process: sample U at centerline
echo "Post-processing..." >&2
postProcess -func sampleDict -latestTime > log.postProcess 2>&1 || true

# Check if postProcessing directory exists
if [ -d "postProcessing" ]; then
    echo "Post-processing OK" >&2
else
    echo "WARNING: No postProcessing directory. Trying sample utility..." >&2
    sample -latestTime > log.sample 2>&1 || true
fi

cd /benchmarks

# Extract results
python3 extract_results.py "$CAVITY_SECS"

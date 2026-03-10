#!/bin/bash
cd /benchmarks/cases

echo "=== Basilisk Benchmarks ===" >&2

# --- Cavity Re=100 ---
echo "Compiling cavity..." >&2
qcc -O2 -Wall cavity.c -o cavity -lm 2>&2

echo "Running cavity Re=100 (level 6, 64x64)..." >&2
START=$(date +%s%N)
./cavity > /dev/null 2>cavity.log
END=$(date +%s%N)
CAVITY_NS=$((END - START))
CAVITY_SECS=$(python3 -c "print(${CAVITY_NS} / 1e9)")
echo "Cavity done in ${CAVITY_SECS}s" >&2

# --- Taylor-Green vortex ---
TG_SECS="0"
echo "Compiling Taylor-Green..." >&2
if qcc -O2 -Wall taylor_green.c -o taylor_green -lm 2>&2; then
    echo "Running Taylor-Green (level 6, 64x64, t_final=1.0)..." >&2
    START=$(date +%s%N)
    ./taylor_green > /dev/null 2>taylor_green.log
    END=$(date +%s%N)
    TG_NS=$((END - START))
    TG_SECS=$(python3 -c "print(${TG_NS} / 1e9)")
    echo "Taylor-Green done in ${TG_SECS}s" >&2
else
    echo "Taylor-Green compilation failed, skipping." >&2
fi

# --- Extract and format results as JSON ---
python3 /benchmarks/extract_results.py "${CAVITY_SECS}" "${TG_SECS}"

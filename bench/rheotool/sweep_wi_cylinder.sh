#!/usr/bin/env bash
set -euo pipefail
#
# Sweep Wi on the confined cylinder benchmark using rheoTool (Docker).
# Reuses the base case cylinder_oldroydb_log_re1_wi01 with varying lambda.
#
# Geometry: R=1, H=4, blockage D/H=0.5
# Flow:     parabolic inlet, U_mean=1, Re_R=1
# Model:    Oldroyd-BLog, beta=0.59
#
# Wi_R = lambda * U_mean / R = lambda  (since U_mean=1, R=1)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_CASE="${SCRIPT_DIR}/cylinder_oldroydb_log_re1_wi01"
IMAGE="guiguitcho/openfoam9-rheotool:v1.2"

WI_VALUES=(0.05 0.1 0.2 0.5 1.0)

# Per-Wi time step and end time (log-conformation handles high Wi,
# but we reduce deltaT for safety at Wi >= 0.5)
get_dt()    { [[ $(echo "$1 >= 0.5" | bc -l) -eq 1 ]] && echo "1e-2" || echo "2e-2"; }
get_tend()  { [[ $(echo "$1 >= 0.5" | bc -l) -eq 1 ]] && echo "10"   || echo "6";    }

RESULTS_FILE="${SCRIPT_DIR}/sweep_wi_results.txt"
printf "%-8s %-10s %-10s %-18s %-18s %-18s\n" \
       "Wi" "lambda" "dt" "Cd_mean_t08" "Cd_last" "status" > "${RESULTS_FILE}"
echo "-------------------------------------------------------------------" >> "${RESULTS_FILE}"

for WI in "${WI_VALUES[@]}"; do
    LAMBDA="${WI}"
    DT=$(get_dt "${WI}")
    TEND=$(get_tend "${WI}")
    CASE_DIR="${SCRIPT_DIR}/cylinder_wi${WI}"

    echo "================================================================"
    echo "  Wi = ${WI}  (lambda = ${LAMBDA}, dt = ${DT}, tend = ${TEND})"
    echo "================================================================"

    # --- Prepare case directory from base ---
    rm -rf "${CASE_DIR}"
    mkdir -p "${CASE_DIR}"

    # Copy structure (0/, constant/, system/, scripts)
    cp -r "${BASE_CASE}/0"        "${CASE_DIR}/0"
    cp -r "${BASE_CASE}/constant" "${CASE_DIR}/constant"
    cp -r "${BASE_CASE}/system"   "${CASE_DIR}/system"
    cp    "${BASE_CASE}/Allrun"   "${CASE_DIR}/Allrun"
    cp    "${BASE_CASE}/Allclean" "${CASE_DIR}/Allclean"
    cp    "${BASE_CASE}/writeData" "${CASE_DIR}/writeData"
    cp    "${BASE_CASE}/summarize_cd.sh" "${CASE_DIR}/summarize_cd.sh"

    # --- Patch constitutiveProperties: set lambda ---
    sed -i.bak "s|lambda  *lambda \[0 0 1 0 0 0 0\] [^;]*;|lambda           lambda [0 0 1 0 0 0 0] ${LAMBDA};|" \
        "${CASE_DIR}/constant/constitutiveProperties"
    rm -f "${CASE_DIR}/constant/constitutiveProperties.bak"

    # --- Patch controlDict: set deltaT and endTime ---
    sed -i.bak "s/^deltaT.*/deltaT          ${DT};/" \
        "${CASE_DIR}/system/controlDict"
    sed -i.bak "s/^endTime.*/endTime         ${TEND};/" \
        "${CASE_DIR}/system/controlDict"
    rm -f "${CASE_DIR}/system/controlDict.bak"

    # --- Run in Docker ---
    echo "  Running rheoFoam in Docker..."
    if docker run --rm \
        --platform linux/amd64 \
        -v "${CASE_DIR}:/data" \
        -w /data \
        "${IMAGE}" \
        bash -lc '
            source /opt/openfoam9/etc/bashrc
            export PATH="/home/openfoam/platforms/linux64GccDPInt32Opt/bin:${PATH}"
            export LD_LIBRARY_PATH="/home/openfoam/platforms/linux64GccDPInt32Opt/lib:/home/openfoam/OpenFOAM/openfoam-9/ThirdParty/petsc-3.16.5/arch-linux-c-opt/lib:${LD_LIBRARY_PATH:-}"
            ./Allclean
            ./Allrun
        ' 2>&1 | tail -5; then
        STATUS="OK"
    else
        STATUS="FAILED"
    fi

    # --- Extract Cd ---
    if [[ -f "${CASE_DIR}/Cd.txt" && "${STATUS}" == "OK" ]]; then
        CD_SUMMARY=$(bash "${CASE_DIR}/summarize_cd.sh" 2>/dev/null || echo "parse_error")
        CD_T08=$(echo "${CD_SUMMARY}" | grep "mean_t_ge_0p8" | awk '{print $2}')
        CD_LAST=$(echo "${CD_SUMMARY}" | grep "last_Cd" | awk '{print $2}')
        [[ -z "${CD_T08}" ]] && CD_T08="N/A"
        [[ -z "${CD_LAST}" ]] && CD_LAST="N/A"
    else
        CD_T08="N/A"
        CD_LAST="N/A"
    fi

    printf "%-8s %-10s %-10s %-18s %-18s %-18s\n" \
           "${WI}" "${LAMBDA}" "${DT}" "${CD_T08}" "${CD_LAST}" "${STATUS}" \
           >> "${RESULTS_FILE}"
    echo "  Wi=${WI}: Cd(t>=0.8)=${CD_T08}, Cd(last)=${CD_LAST}, status=${STATUS}"
done

echo ""
echo "================================================================"
echo "  SWEEP COMPLETE — results in ${RESULTS_FILE}"
echo "================================================================"
cat "${RESULTS_FILE}"

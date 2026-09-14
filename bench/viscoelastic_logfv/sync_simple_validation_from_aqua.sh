#!/usr/bin/env bash
set -euo pipefail

REMOTE="${KRAKEN_AQUA_REMOTE:-aqua}"
REMOTE_REPO="${KRAKEN_AQUA_REPO:-Kraken.jl}"
LOCAL_ROOT="${KRAKEN_LOCAL_SIMPLE_VALIDATION_ROOT:-tmp/aqua_logfv_simple_validation}"
REMOTE_RUN="${1:-}"

if [[ -z "${REMOTE_RUN}" ]]; then
    REMOTE_RUN="$(
        ssh "${REMOTE}" "cd ${REMOTE_REPO} && find results/viscoelastic_logfv -maxdepth 1 -type d -name 'simple_validation_*' | sort | tail -1"
    )"
fi

if [[ -z "${REMOTE_RUN}" ]]; then
    echo "No remote simple_validation_* run found on ${REMOTE}:${REMOTE_REPO}" >&2
    exit 1
fi

mkdir -p "${LOCAL_ROOT}"
echo "Syncing ${REMOTE}:${REMOTE_REPO}/${REMOTE_RUN}/ -> ${LOCAL_ROOT}/"
rsync -az "${REMOTE}:${REMOTE_REPO}/${REMOTE_RUN}/" "${LOCAL_ROOT}/$(basename "${REMOTE_RUN}")/"

LOCAL_RUN="${LOCAL_ROOT}/$(basename "${REMOTE_RUN}")"
MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/mplconfig-kraken}" \
    python3 bench/viscoelastic_logfv/make_simple_validation_dashboard.py "${LOCAL_RUN}"

echo "Local run: ${LOCAL_RUN}"
echo "Dashboard: ${LOCAL_RUN}/dashboard.html"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f Cd.txt ]]; then
    echo "Cd.txt not found; run ./Allrun first." >&2
    exit 1
fi

awk '
    NF >= 2 {
        time[NR] = $1
        cd[NR] = $2
        if ($1 >= 0.6) {
            stable06_sum += $2
            stable06_n++
        }
        if ($1 >= 0.8) {
            stable08_sum += $2
            stable08_n++
        }
    }
    END {
        if (NR == 0) {
            print "Cd.txt is empty" > "/dev/stderr"
            exit 1
        }
        n = NR < 20 ? NR : 20
        start = NR - n + 1
        sum = 0
        for (i = start; i <= NR; i++) {
            sum += cd[i]
        }
        printf("last_time %.12g\n", time[NR])
        printf("last_Cd_viscous %.12g\n", cd[NR])
        printf("mean_last_%d_Cd_viscous %.12g\n", n, sum/n)
        if (stable06_n > 0) {
            printf("mean_t_ge_0p6_Cd_viscous %.12g\n", stable06_sum/stable06_n)
        }
        if (stable08_n > 0) {
            printf("mean_t_ge_0p8_Cd_viscous %.12g\n", stable08_sum/stable08_n)
        }
    }
' Cd.txt

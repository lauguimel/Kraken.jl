# OpenFOAM v10 for benchmark comparisons
# Uses official OpenFOAM Foundation image
FROM openfoam/openfoam10-paraview510:latest

USER root

# Install Python3 and bc for result extraction
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 bc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /benchmarks
COPY cases/ /benchmarks/cases/
COPY extract_results.py /benchmarks/
COPY run_benchmarks.sh /benchmarks/
RUN chmod +x /benchmarks/run_benchmarks.sh

# Give openfoam user write access to the benchmark directory
RUN chown -R openfoam:openfoam /benchmarks

# Run as openfoam user (OpenFOAM expects this)
USER openfoam

ENTRYPOINT ["/bin/bash", "/benchmarks/run_benchmarks.sh"]

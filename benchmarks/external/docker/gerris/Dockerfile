# Gerris CFD solver for benchmark comparisons
# Ubuntu 18.04 amd64 with Gerris from package repo
FROM --platform=linux/amd64 ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Gerris package + full MPI stack + orted daemon
RUN apt-get update && apt-get install -y \
    gerris \
    openmpi-bin openmpi-common libopenmpi2 libopenmpi-dev \
    openssh-client openssh-server \
    python3 bc \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd

# MPI configuration for Docker (run as root, no ssh needed for 1 proc)
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none
ENV OMPI_MCA_btl=self,vader
ENV OMPI_MCA_plm=isolated

WORKDIR /benchmarks
COPY cases/ /benchmarks/cases/
RUN chmod +x /benchmarks/cases/run_benchmarks.sh
RUN mkdir -p /benchmarks/cases/results

ENTRYPOINT ["/benchmarks/cases/run_benchmarks.sh"]

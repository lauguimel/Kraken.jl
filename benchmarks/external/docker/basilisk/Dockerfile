FROM ubuntu:22.04

LABEL maintainer="Guillaume Lauguimel"
LABEL description="Basilisk CFD solver for benchmark comparisons with Kraken.jl"

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies (flex, bison needed for Basilisk parser)
RUN apt-get update && apt-get install -y \
    build-essential gcc make gawk \
    flex bison \
    wget python3 python3-pip \
    time bc \
    && rm -rf /var/lib/apt/lists/*

# Install Basilisk from source (single-threaded build to avoid race conditions)
RUN cd /opt && \
    wget http://basilisk.fr/basilisk/basilisk.tar.gz && \
    tar xzf basilisk.tar.gz && \
    cd basilisk/src && \
    ln -s config.gcc config && \
    make && \
    rm /opt/basilisk.tar.gz

ENV BASILISK=/opt/basilisk/src
ENV PATH="${BASILISK}:${PATH}"

# Set up benchmark workspace
WORKDIR /benchmarks

# Copy case files and scripts
COPY cases/ /benchmarks/cases/
COPY extract_results.py /benchmarks/extract_results.py

# Make run script executable
RUN chmod +x /benchmarks/cases/run_benchmarks.sh

CMD ["/benchmarks/cases/run_benchmarks.sh"]

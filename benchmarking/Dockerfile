# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG CURATOR_IMAGE=nemo_curator
FROM ${CURATOR_IMAGE} AS nemo_curator_benchmarking

# Add system utilities useful for benchmark and debug
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    less \
    lynx \
    openssh-client \
    vim \
    wget \
 && apt-get autoremove -y \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install Prometheus and Grafana so the metrics tooling can skip its runtime download path.
# Symlinks place them on $PATH; nemo_curator.metrics discovers them via shutil.which.
ARG PROMETHEUS_VERSION=3.4.0
ARG GRAFANA_VERSION=12.0.2

RUN ARCH=$(dpkg --print-architecture) \
 && curl -fsSL "https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-${ARCH}.tar.gz" \
      -o /tmp/prom.tar.gz \
 && mkdir -p /opt/prometheus \
 && tar -xzf /tmp/prom.tar.gz -C /opt/prometheus --strip-components=1 \
 && ln -s /opt/prometheus/prometheus /usr/local/bin/prometheus \
 && ln -s /opt/prometheus/promtool   /usr/local/bin/promtool \
 && rm /tmp/prom.tar.gz

RUN ARCH=$(dpkg --print-architecture) \
 && curl -fsSL "https://dl.grafana.com/enterprise/release/grafana-enterprise-${GRAFANA_VERSION}.linux-${ARCH}.tar.gz" \
      -o /tmp/grafana.tar.gz \
 && mkdir -p /opt/grafana \
 && tar -xzf /tmp/grafana.tar.gz -C /opt/grafana --strip-components=1 \
 && ln -s /opt/grafana/bin/grafana-server /usr/local/bin/grafana-server \
 && rm /tmp/grafana.tar.gz

# Ensure all extras and dependency groups (including test) are synced
RUN cd /opt/Curator \
 && uv sync --extra all --all-groups \
 && uv cache prune

# Add the Curator repo to the safe.directory list to avoid GitPython warnings
RUN git config --global --add safe.directory /opt/Curator

 # Set the entrypoint to the main benchmarking runner script
ENTRYPOINT ["python", "/opt/Curator/benchmarking/run.py"]

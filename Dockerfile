# syntax=docker/dockerfile:1

# ---- build -------------------------------------------------------------------
FROM ubuntu:24.04 AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ cmake ninja-build git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DQUIVER_BUILD_TESTS=OFF -DQUIVER_BUILD_TOOLS=ON \
 && cmake --build build \
 && strip build/quiver-server build/quiver-bench

# ---- runtime -----------------------------------------------------------------
FROM ubuntu:24.04
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 --home /data quiver \
 && mkdir -p /data && chown quiver /data
COPY --from=build /src/build/quiver-server /src/build/quiver-bench /usr/local/bin/
USER quiver
VOLUME ["/data"]
EXPOSE 7070
ENV QUIVER_DATA_DIR=/data
HEALTHCHECK --interval=10s --timeout=2s --retries=3 CMD curl -sf http://localhost:7070/health || exit 1
# Docker sends SIGTERM on stop; the server checkpoints every collection before exiting.
ENTRYPOINT ["quiver-server", "--port", "7070"]

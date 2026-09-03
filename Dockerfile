# =============================================================================
# face-gpu-worker — ephemeral RunPod GPU inference worker
# Plan reference: runpod_burst_implementation_plan.md §32
#
# KEY DECISIONS, each with a reason:
#
#  * MODELS ARE BAKED IN AT BUILD TIME. InsightFace otherwise downloads its model
#    pack on first prepare(). On a disposable Pod that is a cold download on EVERY
#    burst, adding minutes to GPU_READY_TIME and putting a third-party CDN on the
#    provisioning critical path. Baking makes the model immutable per image tag,
#    which is exactly the guarantee FAISS compatibility needs — and lets /health
#    assert a SHA-256.
#
#  * NO NETWORK VOLUME is needed as a result (plan §9.2).
#
#  * --workers 1. One GPU, one loaded model. Extra uvicorn workers would each load
#    the model into VRAM and contend for the device.
#
#  * --host 0.0.0.0 is MANDATORY. Binding to 127.0.0.1 makes the container
#    unreachable through the RunPod proxy and presents as a permanent 502.
#
#  * NO DATABASE DRIVERS. pyodbc/psycopg2/faiss are deliberately absent. Absence
#    is a stronger guarantee than policy: the worker physically cannot reach
#    production data stores.
#
# BUILD:
#   docker build \
#     --build-arg IMAGE_VERSION=1.0.0 \
#     --build-arg GIT_SHA=$(git rev-parse HEAD) \
#     --build-arg MODEL_NAME=buffalo_l \
#     -t ghcr.io/<org>/face-gpu-worker:1.0.0 .
#
# !!! MODEL_NAME MUST match the live value on the India server. buffalo_l
# (w600k_r50) and buffalo_s (w600k_mbf) are DIFFERENT 512-dim embedding spaces;
# the wrong one produces embeddings that look valid and never match the existing
# index. See plan Appendix A / Q1.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: model fetch
#
# Isolated so that changing application code does not re-download the models, and
# changing the model invalidates only this layer.
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS models

ARG MODEL_NAME=buffalo_l

# THE MODEL IS PINNED. Fail the BUILD, not the deployment, on a wrong value.
#
# buffalo_l (w600k_r50) and buffalo_s (w600k_mbf) are different 512-dim embedding
# spaces. Both produce valid-looking unit vectors; only buffalo_l matches India's
# existing FAISS index, and a mismatch fails silently. Catching it here means a
# wrong image cannot be built, pushed, or deployed.
RUN if [ "$MODEL_NAME" != "buffalo_l" ]; then \
        echo "FATAL: MODEL_NAME=$MODEL_NAME but this image is PINNED to buffalo_l." >&2; \
        echo "       buffalo_l and buffalo_s are different embedding spaces and" >&2; \
        echo "       only buffalo_l matches the production FAISS index." >&2; \
        echo "       Changing the model requires rebuilding the entire index." >&2; \
        exit 1; \
    fi

RUN pip install --no-cache-dir \
        insightface==0.7.3 \
        onnxruntime==1.19.2 \
        numpy==1.26.4

WORKDIR /build
COPY scripts/download_models.py ./
COPY model_manifest.json ./

# Writes /models plus a manifest of SHA-256 per .onnx file. Fails the build on a
# hash mismatch when model_manifest.json is populated, so a silently changed
# upstream model cannot reach production.
RUN python download_models.py \
        --model "${MODEL_NAME}" \
        --dest /models \
        --manifest model_manifest.json \
        --write-manifest /models/model_manifest.lock.json


# -----------------------------------------------------------------------------
# Stage 2: python dependencies
#
# CUDA runtime (not devel) to keep the image small — it is pulled fresh on every
# burst and sits on the GPU_READY_TIME critical path.
#
# !!! VERIFY the CUDA base tag against onnxruntime-gpu's requirements before
# building (plan §32.2). onnxruntime-gpu is tightly coupled to specific
# CUDA/cuDNN majors, and a mismatch makes CUDAExecutionProvider silently fall back
# to CPU — which /health is written to catch, but which is better prevented.
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS deps

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python3.11 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3.11 -m pip install --no-cache-dir -r /tmp/requirements.txt


# -----------------------------------------------------------------------------
# Stage 3: runtime
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG IMAGE_VERSION=dev
ARG GIT_SHA=unknown
ARG MODEL_NAME=buffalo_l

# Same assertion in the runtime stage: the two ARGs are independent, and an image
# whose label disagreed with its baked model would be worse than useless.
RUN if [ "$MODEL_NAME" != "buffalo_l" ]; then \
        echo "FATAL: runtime MODEL_NAME=$MODEL_NAME, expected buffalo_l" >&2; exit 1; \
    fi

LABEL org.opencontainers.image.title="face-gpu-worker" \
      org.opencontainers.image.description="Ephemeral InsightFace GPU inference worker" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.vendor="TheLit.app" \
      face.model.name="${MODEL_NAME}"

ENV DEBIAN_FRONTEND=noninteractive

# libgl1 + libglib2.0-0: OpenCV runtime dependencies, even for the headless build.
# curl: HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        libgl1 \
        libglib2.0-0 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin worker

COPY --from=deps /usr/lib/python3/dist-packages /usr/lib/python3/dist-packages
COPY --from=deps /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages

# Models, owned by the non-root user that will read them.
COPY --from=models --chown=worker:worker /models /home/worker/.insightface

COPY --chown=worker:worker app/ /app/app/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    INSIGHTFACE_HOME=/home/worker/.insightface \
    IMAGE_VERSION=${IMAGE_VERSION} \
    MODEL_NAME=${MODEL_NAME} \
    GPU_WORKER_PORT=8000

USER worker
WORKDIR /app

EXPOSE 8000

# start-period covers image pull + model load. Without it a cold start would
# register as unhealthy before it has had a chance to become healthy.
# The probe is authenticated, because /health requires a bearer token.
HEALTHCHECK --interval=15s --timeout=10s --start-period=180s --retries=4 \
    CMD curl -fsS -H "Authorization: Bearer ${GPU_WORKER_API_KEY}" \
        "http://127.0.0.1:${GPU_WORKER_PORT}/v1/health" || exit 1

ENTRYPOINT ["python3.11", "-m", "uvicorn", "app.main:app", \
            "--host", "0.0.0.0", "--port", "8000", \
            "--workers", "1", "--no-access-log", "--timeout-keep-alive", "120"]

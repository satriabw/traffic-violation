# detection-worker — the queue consumer that decodes video and runs the detector.
# Build from the repo root: the worker depends on `shared` and on
# packages/trajectory-collector, both outside its directory.
#
# Two targets, because the same worker has to run in two places. `gpu` is what this
# workstation uses; `cpu` is the fallback for when the GPU is busy, and the only one
# that builds on a machine without CUDA.
#
#   docker build -f docker/worker.Dockerfile --target gpu -t traffic-violation-worker .
#   docker build -f docker/worker.Dockerfile --target cpu -t traffic-violation-worker:cpu .
#
# Neither target carries the weights. DETECTION_MODEL_PATH points at a mount — see
# docker-compose.yml.


# --- cpu ------------------------------------------------------------------------
# No ffmpeg apt package here, unlike the API image: the worker never shells out to
# ffprobe. It opens video through cv2.VideoCapture, and the opencv-python-headless
# wheel carries its own ffmpeg libraries.
FROM python:3.11-slim AS cpu

WORKDIR /repo

COPY shared/ shared/
COPY packages/trajectory-collector/ packages/trajectory-collector/
COPY workers/detection-worker/ workers/detection-worker/

# One invocation, so `shared` and `trajectory-collector` resolve from the local
# directories rather than being looked for on PyPI.
RUN pip install --no-cache-dir \
        ./shared ./packages/trajectory-collector ./workers/detection-worker

# Fail the build rather than the first job. An image whose cv2 cannot load its shared
# libraries looks perfectly healthy until a worker has already claimed work.
RUN python -c "import cv2, onnxruntime, supervision, trajectory_collector"

ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "detection_worker.worker"]


# --- gpu ------------------------------------------------------------------------
# CUDA 12.2 to match the workstation's 535 driver exactly, rather than leaning on
# CUDA's minor-version compatibility with a newer runtime. The `runtime` flavour
# (not `base`) is what carries cuBLAS, cuFFT and cuRAND, which the CUDA execution
# provider loads alongside cuDNN.
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 AS gpu

ENV DEBIAN_FRONTEND=noninteractive

# Ubuntu 22.04 ships Python 3.10 and every distribution here declares
# requires-python >= 3.11, so pip would refuse the lot. deadsnakes is the PPA that
# carries newer interpreters for it. There is no CUDA 12.2 image on 24.04 — that
# pairing starts at 12.4 — so the choice is this or a newer CUDA than the driver.
RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common gnupg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv \
        # cv2 links libgthread even in the headless build, which the CUDA images do
        # not carry.
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# A venv, purely so the site-packages path is one predictable string: LD_LIBRARY_PATH
# below has to name the directory pip drops cuDNN's shared objects into, and Debian's
# dist-packages layout moves with the interpreter's packaging.
ENV VIRTUAL_ENV=/opt/venv
RUN python3.11 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /repo

COPY shared/ shared/
COPY packages/trajectory-collector/ packages/trajectory-collector/
COPY workers/detection-worker/ workers/detection-worker/

RUN pip install --no-cache-dir \
        ./shared ./packages/trajectory-collector ./workers/detection-worker

# onnxruntime-gpu replaces onnxruntime rather than joining it. Both distributions
# install the same `onnxruntime` package directory, so having the two in one
# environment is not a fallback arrangement — it is one import shadowing the other,
# and which one wins is an installation-order accident. The uninstall is what makes
# the swap deliberate. detection-worker's pyproject keeps the CPU build as its
# dependency because that is the one every machine can install.
#
# cuDNN comes from pip because the CUDA 12.2 images predate cuDNN 9, which is what
# onnxruntime-gpu >= 1.19 links against. The CUDA libraries proper still come from
# the base image.
#
# The version is pinned, and 1.26.0 specifically: it is the last onnxruntime-gpu
# built against CUDA 12. From 1.27.0 the PyPI wheel links libcublasLt.so.13 and
# libcudart.so.13, and CUDA 13 needs a 580-series driver — this workstation runs 535.
# What makes the pin worth stating rather than floating is *how* it fails: the wheel
# installs, the provider still lists as available, and only session creation logs a
# library-load error before quietly running on CPU. A version bump here has to be
# checked against the driver, not just against the tests.
RUN pip uninstall -y onnxruntime \
    && pip install --no-cache-dir \
        "onnxruntime-gpu==1.26.0" \
        "nvidia-cudnn-cu12>=9,<10"

# pip puts cuDNN under the nvidia namespace package, which no loader searches by
# default. Without this the CUDA provider is registered but unusable: onnxruntime
# logs a libcudnn load failure and falls back to CPU, which is a silent 30x
# slowdown rather than an error.
ENV LD_LIBRARY_PATH="/opt/venv/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.11/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}"

# Same reasoning as the CPU target, plus the provider list: a GPU image whose
# onnxruntime cannot see CUDAExecutionProvider is a build failure, not a runtime
# surprise. This checks that the provider is *registered*; whether it can actually
# create a session needs a GPU, which no build has.
RUN python -c "import cv2, supervision, trajectory_collector; \
import onnxruntime as ort; \
providers = ort.get_available_providers(); \
assert 'CUDAExecutionProvider' in providers, providers; \
print('providers:', providers)"

ENV PYTHONUNBUFFERED=1

# The reason this image exists. Overridable per environment, and the CPU entry stays
# in the list so a worker whose GPU has gone missing keeps working slowly rather than
# refusing to start — the same ordering shared/config.py documents.
ENV DETECTION_MODEL_PROVIDERS="CUDAExecutionProvider,CPUExecutionProvider"

CMD ["python", "-m", "detection_worker.worker"]

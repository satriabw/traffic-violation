# evidence-worker — the queue consumer that cuts a thumbnail and a clip out of a
# source video. Build from the repo root: the worker depends on `shared`, which lives
# outside its directory, so the context has to be the repository.
#
#   docker build -f docker/evidence-worker.Dockerfile -t traffic-violation-evidence .
#
# ONE TARGET, AND A SMALL ONE. Contrast docker/worker.Dockerfile, which needs a CUDA
# runtime, a deadsnakes interpreter, an onnxruntime-gpu pinned to the workstation's
# driver, and a cuDNN from pip whose directory has to be on LD_LIBRARY_PATH. None of
# that decodes anything here: the cut is two ffmpeg invocations against a presigned
# URL, so this is the same python:3.11-slim the API runs on, and it builds anywhere.
FROM python:3.11-slim

# The whole of the runtime that is not Python. ffmpeg speaks HTTP and range-requests
# what it needs out of a remote object, which is why a clip costs its own bytes rather
# than the video's.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ffmpeg -version > /dev/null

# The same path compose bind-mounts the repo at, so an installed copy and a mounted one
# never disagree about where the repo root is.
WORKDIR /repo

# One invocation, so `shared` resolves from the local directory rather than being
# looked for on PyPI.
COPY shared/ shared/
COPY workers/evidence-worker/ workers/evidence-worker/
RUN pip install --no-cache-dir ./shared ./workers/evidence-worker

# Fail the build rather than the first job.
RUN python -c "import boto3, redis; import evidence_worker.worker"

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "evidence_worker.worker"]

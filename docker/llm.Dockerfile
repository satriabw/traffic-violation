# llm-service — turning one violation into an explanation. Build from the repo root,
# not from services/llm-service: the service depends on `shared`, which lives outside
# its directory, so the context has to be the repository.
#
#   docker build -f docker/llm.Dockerfile -t traffic-violation-llm .

FROM python:3.11-slim-bookworm AS builder

WORKDIR /repo
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# shared and llm-service installed in separate layers, so a change to one doesn't
# invalidate the other's cache. Not editable: this image runs on its own — development
# gets the working tree back through a bind mount and PYTHONPATH instead, see
# docker-compose.yml.
COPY shared/ shared/
RUN pip install --no-cache-dir ./shared

COPY services/llm-service/ services/llm-service/
RUN pip install --no-cache-dir ./services/llm-service \
    # uvicorn --reload needs a file watcher and refuses to start without one. Only
    # the compose dev override uses it; it lives here so that override needs no
    # second image.
    watchfiles


FROM python:3.11-slim-bookworm

# No ffmpeg, no ffprobe, no CUDA: this one makes an HTTPS request and parses the
# answer, which is why it is its own image rather than another process inside the
# API's — nothing it needs overlaps with what that one carries.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /repo

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER appuser

# Reachable only from the compose network — docker-compose publishes no port for this
# service. Binding to loopback inside the container would make it reachable from
# nothing at all, including site-service.
EXPOSE 8002
CMD ["uvicorn", "llm_service.main:app", "--host", "0.0.0.0", "--port", "8002"]

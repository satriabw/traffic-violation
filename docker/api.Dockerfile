# site-service — the HTTP API. Build from the repo root, not from
# services/site-service: the service depends on `shared`, which lives outside its
# directory, so the context has to be the repository.
#
#   docker build -f docker/api.Dockerfile -t traffic-violation-api .

FROM python:3.11-slim-bookworm AS builder

WORKDIR /repo
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# shared and site-service installed in separate layers, so a change to one doesn't
# invalidate the other's cache. Not editable: this image runs on its own — development
# gets the working tree back through a bind mount and PYTHONPATH instead, see
# docker-compose.yml.
COPY shared/ shared/
RUN pip install --no-cache-dir ./shared

COPY services/site-service/ services/site-service/
RUN pip install --no-cache-dir ./services/site-service \
    # uvicorn --reload needs a file watcher and refuses to start without one. Only
    # the compose dev override uses it; it lives here so that override needs no
    # second image.
    watchfiles


FROM python:3.11-slim-bookworm

# ffprobe, and only for that: creating a video source shells out to it to read the
# object's header, and without it the endpoint returns 502. `ffmpeg` is the package
# that carries the binary — Debian has no smaller one.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ffprobe -version > /dev/null

# The same path compose bind-mounts the repo at, so an installed copy and a mounted
# one never disagree about where the repo root is. Relative settings like
# TRAFFIC_DB_PATH's `./data` default resolve against it exactly as they do on the host.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /repo \
    && chown appuser:appuser /repo
WORKDIR /repo

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER appuser

# 0.0.0.0 inside the container is not a network exposure decision: the container has
# its own stack, and what it is reachable from is the publish address in
# docker-compose.yml. Binding to loopback here would make it reachable from nothing.
EXPOSE 8001
CMD ["uvicorn", "site_service.main:app", "--host", "0.0.0.0", "--port", "8001"]

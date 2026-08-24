# site-service — the HTTP API. Build from the repo root, not from
# services/site-service: the service depends on `shared`, which lives outside its
# directory, so the context has to be the repository.
#
#   docker build -f docker/api.Dockerfile -t traffic-violation-api .
#
FROM python:3.11-slim

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
WORKDIR /repo

# Both distributions, in one pip invocation so `shared` resolves from the local
# directory rather than being looked for on PyPI. Not editable: the image is meant to
# run on its own, and development gets the working tree back through a bind mount and
# PYTHONPATH instead — see docker-compose.yml.
COPY shared/ shared/
COPY services/site-service/ services/site-service/
RUN pip install --no-cache-dir ./shared ./services/site-service \
    # uvicorn --reload needs a file watcher, and refuses to start without one. Only
    # the compose command uses it; it is here so the dev override does not need a
    # second image.
    watchfiles

# Unbuffered, because the interesting output of a container is its log and a buffered
# stream shows up minutes late — or not at all, if the process is killed.
ENV PYTHONUNBUFFERED=1

# 0.0.0.0 inside the container is not a network exposure decision: the container has
# its own stack, and what it is reachable from is the publish address in
# docker-compose.yml. Binding to loopback here would make it reachable from nothing.
EXPOSE 8001
CMD ["uvicorn", "site_service.main:app", "--host", "0.0.0.0", "--port", "8001"]

# llm-service — turning one violation into an explanation. Build from the repo root,
# not from services/llm-service: the service depends on `shared`, which lives outside
# its directory, so the context has to be the repository.
#
#   docker build -f docker/llm.Dockerfile -t traffic-violation-llm .
#
FROM python:3.11-slim

# No ffmpeg, no ffprobe, no CUDA. This one makes an HTTPS request and parses the
# answer, which is the argument for it being its own image rather than another process
# inside the API's: nothing it needs overlaps with what that one carries.

WORKDIR /repo

COPY shared/ shared/
COPY services/llm-service/ services/llm-service/
RUN pip install --no-cache-dir ./shared ./services/llm-service \
    # uvicorn --reload needs a file watcher and refuses to start without one. Only the
    # compose command uses it; it is here so the dev override does not need a second
    # image.
    watchfiles

ENV PYTHONUNBUFFERED=1

# Reachable only from the compose network — docker-compose publishes no port for this
# service. Binding to loopback inside the container would make it reachable from
# nothing at all, including site-service.
EXPOSE 8002
CMD ["uvicorn", "llm_service.main:app", "--host", "0.0.0.0", "--port", "8002"]

# traffic-violation

## Local development

```
python3 -m venv .venv
.venv/bin/pip install -e ./shared -e ./packages/trajectory-collector \
  -e ./services/site-service -e ./workers/detection-worker
```

Note: on macOS, `pip install -e` can silently no-op — files it creates starting
with `__` get the Finder "hidden" flag, and CPython's `site.py` skips hidden
`.pth` files. If `import shared` fails after installing, run
`chflags nohidden .venv/lib/python*/site-packages/__editable__*` or just rely
on `PYTHONPATH` as shown below (tests already do this via `pythonpath` in
`pyproject.toml`, no install needed for running the test suite).

`ffprobe` (part of ffmpeg) must be on `PATH` — the site-service shells out to it to
read a video's metadata when a video source is created. `brew install ffmpeg` on
macOS, `apt-get install ffmpeg` in a container. Without it, creating a video source
returns 502; the probe tests skip rather than fail.

Run tests:
```
.venv/bin/pytest
```

## Running the stack

Two ways, and which one you want follows from where the GPU is.

### In containers (Linux, and the way to use a GPU)

```
cp .env.example .env                                   # then fill in the R2 settings
printf 'DOCKER_UID=%s\nDOCKER_GID=%s\n' "$(id -u)" "$(id -g)" >> .env
docker compose up --build
```

Three services — `redis`, `api`, `worker` — and naming one starts just that one, so
`docker compose up redis` is how you run the queue alone against host processes. The
API is on `127.0.0.1:8001`; reach it from another machine with
`ssh -L 8001:localhost:8001` rather than by widening `API_BIND`, because nothing here
authenticates anything.

The repository is bind-mounted at `/repo` and `PYTHONPATH` puts it ahead of the copy
pip installed into the image, so an edit on the host is what runs — uvicorn reloads,
and the worker needs a `docker compose restart worker`. Rebuild only when a dependency
changes. The `DOCKER_UID` lines are what keep `data/` yours: without them the SQLite
file, its WAL sidecars and every `__pycache__` come back owned by root, and the next
host-side `pytest` cannot write them.

`data/` is the shared mount, which is what keeps SQLite viable here — `api` and
`worker` are still two processes on one local filesystem, exactly the arrangement
below. It is also why the containers are not the thing that lets the worker move to
its own host; the filesystem is still the constraint.

**The detection worker runs on the GPU by default**, through the `nvidia` container
runtime. When the card is busy with something else:

```
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
```

That is a different image, not a flag: `onnxruntime` and `onnxruntime-gpu` install the
same `onnxruntime` package and cannot coexist, so the swap happens at build time in
`docker/worker.Dockerfile`. Expect about a tenth of the throughput — RT-DETR at 640×640
measures ~101 fps on a 2080 Ti against ~10 fps on the CPU.

The pinned `onnxruntime-gpu==1.26.0` is worth knowing about before anyone bumps it. It
is the last release built against CUDA 12; from 1.27.0 the wheel wants CUDA 13 and a
580-series driver. The failure is quiet — the wheel installs, the provider still lists
as available, and only session creation logs a library-load error before falling back
to CPU at a tenth of the speed. Check the worker's first log line, and check it against
`nvidia-smi`, rather than assuming.

### On the host (macOS, or without Docker)

Three processes, one per terminal, all from the repo root:

```
# 1. the queue
redis-server docker/redis.conf   # on Linux: docker compose up redis

# 2. the API
set -a; source .env; set +a
PYTHONPATH=shared/src:services/site-service .venv/bin/uvicorn site_service.main:app --reload --port 8001

# 3. the detection worker
set -a; source .env; set +a
PYTHONPATH=shared/src:packages/trajectory-collector/src:workers/detection-worker \
  .venv/bin/python -m detection_worker.worker
```

Source `.env` in **each** shell that needs it. The worker reads `REDIS_URL` from the
same environment uvicorn does, so a worker started in an unsourced shell quietly talks
to the default localhost instead of wherever you pointed it. Compose does this for you;
this is the path where it is yours to remember.

A note if you are on a conda machine: `.venv/bin/...` above is a plain `venv`, and an
active conda environment does not provide one. Either create the venv anyway — it works
fine underneath a conda base — or drop the `.venv/bin/` prefix and let the conda
interpreter serve, having installed the same four distributions into it. The
`PYTHONPATH` prefix is what matters either way, and it is why the test suite needs no
install at all.

Only the API is needed to browse sites, files, and calibrations. Redis is needed to
accept a detection job, and the worker to consume one — see below.

Config is read from the process environment only (`shared/config.py` is plain
`os.environ`); `.env` is just a convenient way to populate a local shell, and is
gitignored. `.env.example` is the annotated template — copy it and fill in the R2
settings. In deployment the same variables come from the platform instead.

Compose reads that same file twice, and the second one catches people out: once to
pass into the containers, and once for `${...}` interpolation in `docker-compose.yml`.
A literal `$` in a secret has to be written `$$`, or the container gets a truncated
value.

Source `.env` in the *same* shell you start uvicorn from. The service refuses to
start when the object-storage settings are empty and names the missing ones — a
config mistake should not wait until the first upload to surface as an
`Invalid endpoint:` traceback from inside boto3.

Endpoints are served under `/api/v1` — e.g. `http://localhost:8001/api/v1/sites`.
Services mount that prefix themselves and the gateway does not rewrite it, so the
path is identical whether a request arrives through the gateway or goes directly to
the service. `/health` is served at the root, outside the prefix.

All data is stored in one local SQLite file at `./data/traffic_violations.sqlite`
(override with `TRAFFIC_DB_PATH`). Every resource below lives in it — including
files, which the LLD gives to a separate file-service. Splitting that out later is a
gateway config change, since routing is by resource prefix.

SQLite rather than DuckDB because more than one process needs the file. DuckDB allows
a single read-write process *or* several read-only ones, never both, so
detection-worker could not open the database at all while site-service held it. SQLite
in WAL mode gives concurrent readers alongside one writer, which is the shape of this
system: an API taking human-rate writes and a worker appending violations.

That holds only while the file is on a local filesystem shared by every process.
Moving detection-worker to its own host, or running worker replicas across machines,
is where SQLite stops working and Postgres starts. The trigger is the filesystem, not
the row count.

Redis is needed only to *run* detection jobs for real. The two platforms reach the same
thing — one server, alive only while you are working — by different routes.

On macOS, `brew install redis` for the binary, then start it from the repo root:

```
mkdir -p data          # once per checkout
redis-server docker/redis.conf
```

The `mkdir` is only needed the first time. `data/` is gitignored so a fresh clone does
not have it, and unlike the database — whose `get_connection` creates its parent
directory — Redis refuses to start when its `dir` is missing.

On Linux, a container — and it is the compose file's `redis` service, named on its own:

```
docker compose up redis    # from the repo root; Ctrl-C stops it
```

That is the whole story now. There used to be a `dev/redis-docker.sh` holding the same
`docker run` by hand, and once compose described the service there were two spellings
of one thing, publishing the same port and so unable to run at the same time. Naming
the service is what makes "just the queue" and "the whole stack" the same mechanism
rather than two — `docker compose up` starts all three, `docker compose up redis`
starts one, and neither can drift from the other.

Either way everything stays inside the repo: it writes to the gitignored `data/`,
persists nothing, and stops with Ctrl-C. No `brew services`, no launchd job, no systemd
unit, nothing running when you are not working on this. Point `REDIS_URL` elsewhere to
use one you already have.

The split is not arbitrary. Docker containers are Linux processes, so on macOS any
Docker setup runs a Linux VM underneath — a lot of machinery for one 40 MB server, when
everything else here already runs on the host. On Linux there is no VM, and the trade
runs the other way: the distribution package is what carries the cost. On Ubuntu 20.04
`apt install redis-server` is Redis 5.0.7, and it registers a unit that starts at boot
and holds 6379 — which is both the one thing this setup exists to avoid and a port
conflict with the server you actually want. A container installs nothing.

Two details in that service definition are easy to get wrong on your own. The
**published port is `127.0.0.1:6379:6379`, and that address is the entire security
boundary**: the official images ship `protected-mode no` alongside `bind * -::*`, so a
bare `-p 6379:6379` is an open unauthenticated Redis on every interface — and Docker
writes its iptables rules ahead of ufw, so a host firewall does not cover for you. That
is what `REDIS_BIND` defaults to loopback for, and why widening it is a decision rather
than a convenience.

The other is that the container overrides the conf's `bind 127.0.0.1` with
`--bind 0.0.0.0`. Connections arrive on the container's bridge interface rather than
its loopback — the forwarded ones from the host, and `api` and `worker` reaching
`redis` over the compose network — so the conf as written refuses all of them. Redis
then logs a warning about accepting connections from anywhere; that is its view from
inside the container, which cannot see the publish address. The conf and `data/` are
mounted under `/repo` with a matching working directory, so `dir ./data` resolves as it
does on the host and the command stays `redis-server docker/redis.conf`.

The test suite drives the queue through an injected fake and never connects to Redis,
the same way it never touches R2. Object storage is likewise needed only for the file
endpoints.

### Sites and sources

A site is a durable camera location. It holds identity and nothing else — what it is
pointed at lives in `site_sources`, because a location outlives any one video and its
stream address can change.

```
POST /api/v1/sites   {"name": "Junction 5"}
POST /api/v1/sites   {"name": "Junction 5",
                      "source": {"kind": "stream", "stream_url": "rtsp://10.0.0.5/s"}}

POST /api/v1/sites/{site_id}/sources  {"kind": "stream", "stream_url": "rtsp://..."}
POST /api/v1/sites/{site_id}/sources  {"kind": "video",  "file_id": "..."}
GET  /api/v1/sites/{site_id}/sources             -> the active source
GET  /api/v1/sites/{site_id}/sources/{source_id} -> a specific version
```

A site with no source is valid — the url can be added later. The inline `source` on
`POST /sites` is sugar: it runs the same validation the dedicated endpoint does, so the
two entry points cannot drift apart.

Sources are versioned like calibrations: each POST appends, the highest version is
active, superseded versions stay addressable. That is what keeps a past violation
explainable — you can still see which stream url or video file produced it. A site may
hold both kinds over time; a camera you stream *and* upload recordings from is one
location, not two.

Within a source, `kind` decides which column carries it — `stream_url` for a stream,
`file_id` for a video — enforced twice, by `SourceCreate` (422) and by a CHECK
constraint that makes the invalid rows unrepresentable. Video sources validate their
file through `site_service/file_reference.py`, the same helper calibrations use: 422
unknown, 409 still pending, 422 wrong type.

`status` and `metadata` (fps, frame count, duration) live on the **source**, not the
site. A duration describes a video; `active` and `degraded` only ever described a
stream. `GET /sites?kind=video&status=processing` filters on the *active* source, so a
site that used to be a video and is now a stream reads as a stream site.

`SiteResponse` embeds the active source, so the common read stays one request.

Request bodies for sites and sources reject unknown fields. A client still sending the
old top-level `mode`/`url` gets a 422 rather than a silently source-less site.

The LLD hangs job enqueueing off `POST /sites` itself. Here it is a separate
`POST /sites/{id}/detect` (below): creating a site and asking for detection are
different decisions, and a site whose video is uploaded later has nothing to enqueue at
creation time.

### Detection jobs

Detection is asynchronous: the API accepts the work and a separate process does it.

```
POST /api/v1/sites/{site_id}/detect   {}                              -> 202
POST /api/v1/sites/{site_id}/detect   {"types": ["red_light_running"]}
```

```json
{
  "id": "9f0c...",
  "site_id": "3a1b...",
  "source": {
    "source_id": "7d2e...",
    "version": 3,
    "key": "video/4c8a.../clip.mp4",
    "fps": 29.97,
    "total_frames": 27000
  },
  "frame_range": {"start": 0, "end": 27000},
  "types": ["red_light_running", "pedestrian_right_of_way"],
  "calibration_version": 2,
  "configuration_version": 1
}
```

202 rather than 201: nothing was created — the work was accepted, and the `id` that
comes back is what identifies it downstream. The response body *is* the queued message
(`shared/models/detection.py`), which is the contract between site-service and
detection-worker.

`frame_range` is not something the client sends. It is derived from the active source's
probed `total_frames`, in frame indices rather than seconds — variable-rate footage
makes a time offset ambiguous, which is why the source keeps `fps` and `nominal_fps`
apart. `types` defaults to every type the system knows, so the list can grow without
updating callers.

`source` is the video itself, carried in the message rather than looked up by the
worker. Not to save a round trip — that is microseconds against a multi-minute decode —
but because **sources are versioned**: asking site-service for a site's source answers
"what is active now", so a job enqueued against v3 and consumed after v4 was attached
would silently read a different video than the one it was created for. Carrying it pins
the job to the decision that produced it, and lets a backlog drain while site-service is
down.

It carries the object **key**, not a download url. A presigned url expires, and one that
died in a queue — or partway through a long read — fails *after* the worker has started.
A key is immutable (a new upload is a new file id), so the worker signs its own url each
time it opens the video. That is why the worker needs the R2 settings in its
environment; it needs them regardless, since evidence frames will be written there.

`calibration_version` and `configuration_version` are the same idea applied to the
site's other two documents, resolved to a number at enqueue time. The documents
themselves are small files in R2, so the message would only ever carry a pointer —
and a version keeps the queue contract fixed as more per-site context arrives, which a
growing set of keys would not. The worker looks them up by `(site_id, version)` and
**never** by "the site's active calibration": a v4 uploaded while the job sits in the
queue must not change what that job is evaluated against, and a run that used the wrong
camera model would look completely normal. `null` means the site had neither when the
job was created, which is a normal state until there is a rule engine to need them.

Everything that can go wrong is about the site's *state*, so it is 409 rather than 422 —
the request is well formed, and the identical one succeeds once a video is attached:

| Problem | Status |
|---|---|
| unknown site | 404 |
| site has no source | 409 |
| active source is a stream | 409 |
| video was never successfully probed | 409 |

A stream is rejected because a live feed has no frame count to bound a job with. The
LLD gives streams to a supervisor worker that spawns long-running consumers, which is a
different mechanism from enqueueing a bounded job.

One job covers the whole video today. The LLD's 30-second chunks with 10-second overlap
are what make violations at a chunk boundary detectable, and they arrive with the
pipeline that needs them.

### The queue and the worker

A Redis list, not Celery or RQ: this hop needs push and pop, and a list keeps the
worker's entrypoint ordinary Python rather than a framework's. `LPUSH` at the head,
`BRPOP` at the tail — that pairing is what makes it FIFO.

```
PYTHONPATH=shared/src:packages/trajectory-collector/src:workers/detection-worker \
  .venv/bin/python -m detection_worker.worker
```

The worker signs a url from the job's `source.key`, reads the frames the job asks for,
detects and tracks what is in them, and logs a summary. The rest of the pipeline the
LLD describes (evaluate → store) is still missing, and nothing writes violations yet.

The work splits by how often it runs. `make_handler` in `detection_worker/worker.py`
holds what happens **once per job** — resolve the job's context, sign the url, iterate
the reader, aggregate, log — and a `FrameAnalyzer` from `detection_worker/analysis/`
holds what happens **once per frame**: predict, then track, returning a `FrameResult`. Trajectory collection and the rule engine land inside
`analyze`, which is why they are separate at all: neither has any business widening a
function that also knows about presigned urls.

An analyzer belongs to exactly one job, because its tracker does. That lifetime is the
one thing in this design worth being careful about — a tracker holds live state, so
sharing one across jobs would let a track from one site's chunk be re-matched against
another's, and the corruption would arrive silently.

### Trajectories

Where an object is on screen is not where it is. Two cars a hundred pixels apart are
metres apart at the bottom of a frame and tens of metres apart at the top, so any rule
about speed or distance has to work on the ground plane rather than in pixels. Turning
one into the other is what a site's calibration is for.

That work lives in **`packages/trajectory-collector/`**, a separate distribution rather
than another module in the worker:

```python
from trajectory_collector import TrajectoryCollector

collector = TrajectoryCollector.from_calibration(document, fps=30.0)
trajectories = collector.collect(boxes, track_ids, frame_index)
# {track_id: Trajectory(position=(x, y), speed=…)}   metres, metres per second
```

It is separate because the same code has been copied into several projects, and the
thing that makes it liftable back out of this one is that it imports nothing from
here — not `shared`, not the worker, and no detection library either. Its whole
dependency list is numpy and OpenCV, the latter only because calibrations are OpenCV
FileStorage documents. That constraint is what shapes the interface: `collect` takes
plain arrays rather than an `sv.Detections`, and the translation between them is two
attribute reads in `frame_analyzer._tracked`. Nothing in the API mentions a site, a job
or a frame range, because none of those exist outside this system.

`from_calibration` is the only entry point. Which projection a calibration calls for is
the package's decision, so the worker never names a camera model, a filter or a
projection — the same way it opens a video without naming a decoder.

The collector is per-job for the same reason the tracker is, and by the same key:
tracker ids restart at 1 for every job, so a collector outliving one would merge
unrelated objects. A site with no calibration gets a `NullCollector`, which reports
nothing — an uncalibrated site is a normal state, not a failure, and expressing it as a
collector rather than as `None` keeps the null check out of the per-frame path.

Three things happen per frame, in this order:

**Anchor.** A box becomes the middle of its bottom edge, where the object meets the
road. Not the centre — a car's box centre floats a metre above the ground, and
projecting a point that is not on the ground plane onto the ground plane puts it metres
from the car.

**Project.** The whole frame's anchors go through the camera model in one matrix
multiply. Fixing Z=0 is what makes this invertible at all: a pixel is a ray, and only
the assumption that the object is on the ground picks out one point along it.

**Filter.** Each track's projected position goes through its own Kalman filter, and
that is where speed comes from. Speed is never differenced between consecutive
positions — a box jitters by a few pixels, near the horizon a few pixels is metres, and
the resulting speed swings wildly while the object moves smoothly.

A track reports position but no speed for its first few frames: a filter needs a
velocity to start from and one sighting gives none, so the warmup is spent measuring
one. A track that disappears and comes back is stepped over the real elapsed interval
rather than one frame, which is the other reason frame indices are absolute — a job
covering frames 900-1800 measures gaps against the video's clock, not its own.

**A box whose bottom edge lands on or above the horizon has no ground point at all**;
its ray never meets the plane. Such a box is skipped for that frame rather than
reported. Letting the resulting `nan` into a Kalman filter would not cost one frame, it
would poison every estimate that track produced afterwards.

The job summary logs `located=`, the number of distinct tracks that were put on the
ground. Zero means the site has no calibration; short of `tracks=` means some object
never had a box whose bottom edge met the road.

Frames come from `detection_worker/video/reader.py`. OpenCV opens the presigned url
through its ffmpeg backend, which range-requests the object — the same mechanism the probe
relies on — so nothing is downloaded to disk. `read_frames` yields `(index, frame)` over
the half-open range with **absolute** indices: a job covering frames 900-1800 reports
900 for its first frame, because that is the number a violation has to be recorded
against for anyone to find it in the footage later. A video that ends before
`frame_range.end` stops the read rather than failing it — `total_frames` is the
container's claim, and truncated or stream-copied files over-report it.

Because the worker signs its own urls, it needs the same R2 settings site-service does.
Source `.env` in the worker's shell too, not just uvicorn's.

Nothing is cached between jobs. Today one job covers a whole video read once, so a cache
would have a zero hit rate by construction; it earns its place when the LLD's 30-second
chunks make one video the target of many jobs. The object key is immutable, so when that
day comes the cache key is the key and there is no invalidation problem.

`shared/queue/` holds two implementations of one shape, `enqueue` / `consume`:
`RedisQueue`, and an `InMemoryQueue` that lets the worker run a job through without
Redis anywhere. The difference that matters is what `consume()` does when empty —
`InMemoryQueue` returns `None`, `RedisQueue` keeps waiting — so the same loop in
`run()` drains an in-memory queue once and keeps a deployed worker waiting
indefinitely. `None` therefore means "drained", never "nothing yet".

That waiting is a series of bounded `BRPOP` windows (`BLOCK_SECONDS`), not one
unbounded one. A `BRPOP` of `0` blocks forever and outlives redis-py's default 5-second
socket timeout, so an idle worker died with `TimeoutError: Timeout reading from socket`
— a quiet queue looking exactly like a network fault. The connection's `socket_timeout`
is set above the window so the server always answers first, and stays finite so a truly
dead connection still surfaces.

A handler that raises stops the worker rather than dropping the job. Retry and a
dead-letter queue are marked FUTURE in the LLD; until they exist, failing loudly beats
losing work quietly.

### Calibrations and configurations

Both are the same thing structurally — a versioned pointer from a site to a file — so
they share one implementation (`routers/_versioned_document.py`) and differ only in
table, file type, and path. Each belongs to a site, so each is addressed under one:

```
POST /api/v1/sites/{site_id}/calibrations    {"file_id": "..."}
GET  /api/v1/sites/{site_id}/calibrations                  -> the active version
GET  /api/v1/sites/{site_id}/calibrations/{calibration_id} -> a specific version

POST /api/v1/sites/{site_id}/configurations  {"file_id": "..."}
GET  /api/v1/sites/{site_id}/configurations
GET  /api/v1/sites/{site_id}/configurations/{configuration_id}
```

The full flow is three steps — reserve, upload, attach:

```
POST /api/v1/files                {"name": "camera_model.yml",
                                   "type": "calibration", "size_bytes": 2048}
PUT  <upload_url>                 the bytes, straight to R2
POST /api/v1/files/{id}/complete  confirms the object landed
POST /api/v1/sites/{site_id}/calibrations   {"file_id": "<id>"}
```

`file_id` replaced the old `url` field. A url was an unverifiable claim — nothing
checked that anything had ever been uploaded to it. A `file_id` points at a row whose
upload the service confirmed, so `POST` can reject what it cannot use:

| Problem | Status |
|---|---|
| unknown site | 404 |
| unknown `file_id` | 422 |
| file still `pending` — bytes never landed | 409 |
| file is the wrong type (a video, say) | 422 |

### What a calibration file has to contain

The service stores calibrations as opaque files — it checks that the bytes landed and
that the type is right, not what is inside them. The worker is what reads one, and the
format is OpenCV `FileStorage`, which is what camera calibration tools write:

```yaml
%YAML:1.0
---
camera_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ fx, 0., cx, 0., fy, cy, 0., 0., 1. ]
rot_matrix: !!opencv-matrix
   ...
tvec: !!opencv-matrix
   ...
```

Intrinsics, and the rotation and translation that put the camera in the world. That is
the only format read — a calibration is whatever the calibrating tool produced, and
supporting a second one bought nothing but the code to tell them apart. Extra nodes are
ignored, `dist_coeffs` among them: nothing here undistorts, because the only projection
it performs is the ground-plane homography.

**Translations are in metres**, and there is no field to say otherwise. Every position
and speed downstream inherits the unit the calibration was built in, so a calibration
in feet is a wrong calibration rather than a differently-configured one.

A calibration that cannot be projected with — an unreadable document, a missing node, a
wrong shape, a camera whose ground plane is degenerate — fails the job on the way in,
before a frame is decoded, and the worker stops. That is louder than it sounds and
deliberately so: a run that produced plausible-looking metres from a broken camera
model is the failure nobody notices.

Note that the calibration is fetched as raw bytes while the configuration is fetched as
parsed JSON. The asymmetry is deliberate: a configuration is ours, so `context.py`
parses it and nobody thinks about it again, whereas a calibration is whatever the
calibrating tool wrote — so it travels intact and the one component that knows what a
camera model is decides what it means.

The `pending` check is the reason the two-phase upload exists. Without it a client
could reserve a slot, skip the PUT, and attach the resulting id to a site — which is
precisely the broken state the `url` field used to permit. It is a 409 rather than a
422 because the request is well formed; the file is simply not ready yet, and the same
request succeeds once the upload completes.

Each POST appends a new version (`version = max(version) + 1` for that site) rather
than replacing the previous one. The highest version is the one active document that
`GET .../calibrations` returns; superseded versions stay addressable by id. Reads are
scoped by both ids, so one site can never read another's documents.

Deleting a site deletes its sources, calibrations and configurations — `ON DELETE
CASCADE` in the schema, not application code. Foreign keys are off by default in
SQLite and enabled per connection, so `get_connection` sets `PRAGMA foreign_keys = ON`
on every connection it hands out; without it the cascade silently does nothing.

Violations are the exception: a site with any recorded against it cannot be deleted,
and the attempt is a 409. Sources and calibrations describe how a site is configured
and are meaningless once it is gone, but a violation is a record of something that
happened — taking it along as a side effect of tidying up a site is not a decision the
API should make on the caller's behalf.

A `files` row referenced by a site, a calibration, or a configuration cannot be
deleted while that reference stands. Nothing deletes files today, so this only
constrains a future `DELETE /files/{id}`.

### Files

Uploads never pass through the API. The client asks for a slot, PUTs the bytes
straight to R2, then confirms:

```
POST /api/v1/files                  {"name": "clip.mp4", "type": "video",
                                     "size_bytes": 4096,
                                     "content_type": "video/mp4"}
  -> 201 {id, url, status: "pending", upload_url, upload_expires_in}
  -> 413 if size_bytes exceeds S3_MAX_UPLOAD_BYTES (default 100 MiB)

PUT  <upload_url>                   client -> R2 directly, bytes skip the service

POST /api/v1/files/{id}/complete    -> 200 {status: "uploaded", size_bytes}
                                    -> 409 if the object is not in the bucket
                                    -> 404 if the id is unknown

GET  /api/v1/files/{id}             -> 200 {..., download_url}
```

`type` is one of `calibration`, `configuration`, `video`, `evidence_frame`.

The `url` field is an **object key**, not a URL — the name follows the LLD and matches
`sites.url`. Presigned URLs expire, so `upload_url` and `download_url` are computed per
request and never stored. `download_url` is null while a file is `pending`, and a
`GET` never returns an `upload_url`: only file creation mints one, since it grants
write access.

Keys are assigned server-side as `{type}/{file_id}/{sanitised_name}`. The client
supplies a display name, never a path — see `shared/s3/keys.py`.

The `pending` -> `uploaded` transition is what makes a row trustworthy. Because the
client uploads out-of-band, the server only believes an upload happened after
`/complete` verifies it with a HeadObject. A client that never calls `/complete`
leaves a `pending` row; there is no reaper for those yet.

`size_bytes` is required, and is the client's *declared* size — a claim. It is checked
against `S3_MAX_UPLOAD_BYTES` (default 100 MiB) before any URL is minted, then signed
into that URL so R2 refuses a body of a different length. At `/complete` the declared
value is replaced by the size HeadObject actually measured, which is the fact.

Capping at issue time rather than at `/complete` is the point: an upload URL is a
spending authorisation, and once bytes have landed you have already paid for the
transfer.

If `content_type` is given at creation it is signed into the upload URL too, so the
client **must** send the identical `Content-Type` header on its PUT or R2 rejects it.

Calibrations and configurations reference files by `file_id` — see above.

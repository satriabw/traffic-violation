# traffic-violation

## Local development

```
python3 -m venv .venv
.venv/bin/pip install -e ./shared -e ./packages/trajectory-collector \
  -e ./packages/violation-detector -e ./packages/evidence-collector \
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
PYTHONPATH=shared/src:packages/trajectory-collector/src:packages/violation-detector/src:packages/evidence-collector/src:workers/detection-worker \
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
job was created, which stays a normal state: without a calibration a job reports no
trajectories, and without a configuration it runs no rules.

Each violation records the **row** those versions resolved to, in `calibration_id` and
`configuration_id`. Two things need it. A reader asking which of a site's violations
still hold under the setup it runs now has nothing else to compare against; and evidence
drawn with the *current* polygons over a violation found under older ones shows a
vehicle sitting outside the box it was convicted in — evidence that looks falsified
rather than merely stale. The id rather than the version, for the reason `source_id`
gives: it is the primary key of one version's row, so it pins that version by itself,
and a second column holding the version would be free to disagree with it.

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
PYTHONPATH=shared/src:packages/trajectory-collector/src:packages/violation-detector/src:packages/evidence-collector/src:workers/detection-worker \
  .venv/bin/python -m detection_worker.worker
```

The worker signs a url from the job's `source.key`, reads the frames the job asks for,
detects and tracks what is in them, puts them on the ground, runs the rules the site's
configuration asks for, and logs a summary. What a firing rule produces is counted and
logged and goes no further: turning a `Violation` into a row means cropping evidence
frames, uploading them and calling `detection_worker.violations.record`, and that is
the next piece of work. The seam is `FrameResult.violations`.

The work splits by how often it runs. `make_handler` in `detection_worker/worker.py`
holds what happens **once per job** — resolve the job's context, sign the url, iterate
the reader, aggregate, log — and a `FrameAnalyzer` from `detection_worker/analysis/`
holds what happens **once per frame**: predict, track, locate, judge, returning a
`FrameResult`. Trajectory collection and the rules live inside `analyze`, which is why
the split exists at all: neither has any business widening a function that also knows
about presigned urls.

`analyzer.finish()` is called once after the loop. Every rule shipped today returns
nothing from it, because a rule decides on the frame it is given — but a module working
on a *clip* is always holding a partial window when the frames run out, and without the
drain the last seconds of every chunk would go unjudged in silence.

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

### Violations

A rule is a statement about what an object did over time — crossed the stop line after
the light governing its lane turned red — so it runs after tracking, never before. An
untracked box has no history to have done anything in.

That work lives in **`packages/violation-detector/`**, separate from the worker on
exactly the terms trajectory-collector is:

```python
from violation_detector import Configuration, get_detector

detector = get_detector(Configuration.from_document(document), types=[...], fps=30.0)
violations = detector.detect(frame, tracked_objects, frame_index)
# [Violation(type="red_light_running", track_id=7, frame_index=912)]
violations = detector.finish()
```

Its whole dependency list is numpy and OpenCV — not `shared`, not the worker, and no
detection library. A `TrackedObject` carries a **class name**, not a class id, because
an id means nothing without knowing which model produced it; translating one into the
other is `frame_analyzer._tracked_objects`, and it is the whole of the boundary.

Two rules ship today. **`rlr_violation`** is red-light running: crossing into the box
past the stop line after the light governing your lane turned red. **`pdx_violation`**
is pedestrian right of way: driving into a crossing somebody is already in. Both are
reported under the canonical names the queue uses — `red_light_running` and
`pedestrian_right_of_way`.

Both turn on the same distinction, and it is the one worth understanding: a rule fires
on the frame a vehicle *enters* a region, not on every frame it spends there. A car
already inside the box when the light changed was caught by the change rather than
running it; a car already stopped in a crossing when somebody steps off the kerb has
not failed to give way to them. Neither is an offence, and neither is reported.

## The record a violation carries

A violation on its own is a type, a track id and a frame number, which is enough to
find it and nothing like enough to review it. What makes it reviewable is the few
seconds before it: the approach, the speed the vehicle was carrying, whether anybody
was already in the crossing.

That work lives in **`packages/evidence-collector/`**, a fourth distribution with **no
dependencies at all** — not numpy, not OpenCV, and nothing from this repository.

```python
from evidence_collector import EvidenceCollector

collector = EvidenceCollector.over(seconds=5, fps=30)
collector.observe(frame_index, object_states)     # every frame, empty ones included
windows = collector.window_for([track_id])        # the lead-up, oldest first
```

A firing rule becomes a row: `detection_worker.violations.to_create` turns the
`Violation` and the window that came with it into a `ViolationCreate`, and `record`
writes it. Rows are written as violations are found rather than batched to the end, so
a job that dies half way through keeps what it had already seen.

**It keeps records, not pixels.** Where each object was and how fast it was going,
against the frame index that finds the moment in the footage again — some hundreds of
bytes a frame against megabytes for the image. The pixels are re-derived from the
source when somebody opens the detail view, by whatever knows how to draw them then,
rather than guessed at now by the process that happened to be running. `frames` in a
violation's metadata stays empty at write time, deliberately.

**The window ends at the violation.** By the time a rule fires the thing has already
happened, and `prev_in_roi` means a track is convicted once per crossing rather than on
every frame of it. What came after is the consequence and is still in the source.
Buffering forward would mean holding every frame until enough future had arrived to
prove nothing happened, on the overwhelming majority of frames where nothing ever does.

**How long a window is, is the site's decision.** `evidence_seconds` is a top-level
key in the configuration document, beside `violations` and `regions`; a site that says
nothing gets five seconds. A junction is the thing that knows better — an approach with
a long sight line wants more, a tight one-way needs less.

It is read in `detection_worker.context`, where the document is already in hand, and
travels on as a plain number — nothing downstream is handed the document to go digging
in. Not by `Configuration.from_document` either: how long to keep records is a question
about evidence, not about traffic rules, and the rules package has no business holding
an answer to it. Its parser leaves unknown top-level keys alone precisely so a document
can carry something like this. A value that is not a positive number stops the job
before a frame is decoded, by name.

The ring holds `seconds * fps + 1`. The `+ 1` is the moment itself, since the frame is
recorded before the window is read — and the pipeline this is ported from bears the
number out: its output for a convicted car carries ninety-five frames of lead-up at
19fps, and the car is outside the region of interest on every one of them.

**Keep the window no longer than the overlap between chunks.** One that reaches back
past the start of a chunk is truncated in silence, and the record just looks shorter.
Nothing can check that, because a chunk's overlap is not in the document — so the
handler logs `short=`, and a few is ordinary while mostly-all means the window has
outgrown the overlap.

The join between a rule's boxes and a collector's ground positions is
`frame_analyzer._object_states`, the third and last adapter on that boundary.

**The record computes nothing.** Every number in it is a number something else
produced. A job with no calibration produces no trajectories, so its records simply
carry no positions — `None` beside a full set of boxes, rather than image coordinates
standing in for metres. Nothing has to be configured for that; it is the same absence,
arrived at without anyone deciding anything.

## Which rules a job runs

Which rules a job runs is the intersection of two things. The site's configuration says
what this junction is annotated for; the job's `types` say what was asked for. Either
side naming something the other does not is a no-op rather than an error — the site is
the authority on what can be watched for, the job on what was wanted. `types` are
canonical values (`red_light_running`), so the worker holds no table mapping its own
`ViolationType` to the `rlr_violation` a document says; the registry inside the package
is the only thing that knows both vocabularies.

A detector is per-job, for the third time and the same reason: its rules cache what
they have seen keyed by tracker id, and those restart at 1 for every video.

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

### What a configuration file has to contain

The service stores configurations the same opaque way it stores calibrations, so the
schema below is enforced by whoever reads one — `packages/violation-detector/` — and
`Configuration.from_document` is where it is written down.

A configuration says two things: which rules a site runs, and the places they run
against.

```json
{
  "version": 1,
  "violations": ["rlr_violation"],
  "evidence_seconds": 5,
  "regions": {
    "lanes":          [{"id": "lane_1", "points": [[115, 640], [232, 1069], [807, 1067], [354, 639]]}],
    "traffic_lights": [{"id": "tl_1", "points": [[33, 470], [31, 516], [50, 519], [53, 458]],
                        "controls": ["lane_1"]}],
    "rois":           [{"id": "roi_1", "points": [[58, 564], [62, 636], [113, 636], [351, 634]]}]
  }
}
```

`points` are pixels in the source video's coordinate space, at least three per region,
in the order they should be joined. A lane is where a vehicle comes *from*; an ROI is
the area a rule cares about it entering — the junction box, a crossing.

`controls` is the junction's wiring, and the document is the only thing that knows it.
Without it there is no way to say which vehicles a given light is responsible for, and
red-light running is not expressible at all. Every lane it names must be declared in
`lanes`, and that is checked when the document is parsed: a rule handed the mapping
cannot tell a lane that does not exist from one it has not seen a vehicle in yet, so a
typo would simply never fire and never explain itself.

Parsing is strict about anything that could swallow content silently. An unknown
section (`"roi"` for `"rois"`) is refused rather than ignored, because ignoring it
drops every region in it and the only symptom is a rule that never fires. Two regions
in one section may not share an `id`; a lane and an ROI may. Unknown keys *beside* the
content — a name, a note — are left alone. A `version` other than 1 is refused
outright: a configuration is read once and used for a whole run, so a document whose
meaning changed under us would produce a run that looks entirely normal and is
entirely wrong.

Region membership is decided in **pixels**, and deliberately not on the ground plane
despite a site usually having a camera model to hand. Projecting first is what the
pipeline this is ported from did, and it cannot change the answer: the projection is a
homography, a homography maps lines to lines, so a polygon stays a polygon and its
interior stays its interior. Measured before the code was deleted — 20,000 random
points against a lane polygon under a 65-degree pitch, zero disagreements between the
two paths, and still zero for a polygon annotated so generously that its far edge
crossed the horizon and its ground projection turned inside out. What it cost was a
matrix multiply per region per object per frame. A rule about speed or following
distance is a different question and does need the camera model; membership does not.

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

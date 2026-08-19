# traffic-violation

## Local development

```
python3 -m venv .venv
.venv/bin/pip install -e ./shared -e ./services/site-service -e ./workers/detection-worker
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

Run everything locally — three processes, one per terminal, all from the repo root:

```
# 1. the queue
redis-server dev/redis.conf

# 2. the API
set -a; source .env; set +a
PYTHONPATH=shared/src:services/site-service .venv/bin/uvicorn site_service.main:app --reload --port 8001

# 3. the detection worker
set -a; source .env; set +a
PYTHONPATH=shared/src:workers/detection-worker .venv/bin/python -m detection_worker.worker
```

Source `.env` in **each** shell that needs it. The worker reads `REDIS_URL` from the
same environment uvicorn does, so a worker started in an unsourced shell quietly talks
to the default localhost instead of wherever you pointed it.

Only the API is needed to browse sites, files, and calibrations. Redis is needed to
accept a detection job, and the worker to consume one — see below.

Config is read from the process environment only (`shared/config.py` is plain
`os.environ`); `.env` is just a convenient way to populate a local shell, and is
gitignored. In deployment the same variables come from the platform instead.

Source `.env` in the *same* shell you start uvicorn from. The service refuses to
start when the object-storage settings are empty and names the missing ones — a
config mistake should not wait until the first upload to surface as an
`Invalid endpoint:` traceback from inside boto3.

Endpoints are served under `/api/v1` — e.g. `http://localhost:8001/api/v1/sites`.
Services mount that prefix themselves and the gateway does not rewrite it, so the
path is identical whether a request arrives through the gateway or goes directly to
the service. `/health` is served at the root, outside the prefix.

All data is stored in one local DuckDB file at `./data/site_service.duckdb`
(override with `SITE_SERVICE_DB_PATH`). DuckDB permits a single cross-process
writer, so site-service owns the file and every resource below lives in it —
including files, which the LLD gives to a separate file-service. Splitting that
out later is a gateway config change, since routing is by resource prefix.

Redis is needed only to *run* detection jobs for real. `brew install redis` for the
binary, then start it from the repo root:

```
redis-server dev/redis.conf
```

Everything stays inside the repo: it writes to the gitignored `data/`, persists
nothing, and stops with Ctrl-C. No `brew services`, no launchd job, nothing running
when you are not working on this. Point `REDIS_URL` elsewhere to use one you already
have.

No container runtime is involved. Docker containers are Linux processes, so on macOS
any Docker setup runs a Linux VM underneath — a lot of machinery for one 40 MB server,
when everything else here already runs on the host.

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
  "frame_range": {"start": 0, "end": 27000},
  "types": ["red_light_running", "pedestrian_right_of_way"]
}
```

202 rather than 201: nothing was created — the work was accepted, and the `id` that
comes back is what identifies it downstream. The response body *is* the queued message
(`shared/models/detection.py`), which is the contract between site-service and
detection-worker; the two share no database.

`frame_range` is not something the client sends. It is derived from the active source's
probed `total_frames`, in frame indices rather than seconds — variable-rate footage
makes a time offset ambiguous, which is why the source keeps `fps` and `nominal_fps`
apart. `types` defaults to every type the system knows, so the list can grow without
updating callers.

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
PYTHONPATH=shared/src:workers/detection-worker .venv/bin/python -m detection_worker.worker
```

The worker logs each job and moves on. That is the whole stub — the pipeline the LLD
describes (source → detect → track → evaluate → store) hangs off `handle` in
`workers/detection-worker/detection_worker/worker.py` when it exists, and nothing
writes violations yet.

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
POST /api/v1/files                {"name": "homography.json",
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

The `pending` check is the reason the two-phase upload exists. Without it a client
could reserve a slot, skip the PUT, and attach the resulting id to a site — which is
precisely the broken state the `url` field used to permit. It is a 409 rather than a
422 because the request is well formed; the file is simply not ready yet, and the same
request succeeds once the upload completes.

Each POST appends a new version (`version = max(version) + 1` for that site) rather
than replacing the previous one. The highest version is the one active document that
`GET .../calibrations` returns; superseded versions stay addressable by id. Reads are
scoped by both ids, so one site can never read another's documents.

Deleting a site deletes its calibrations and configurations — DuckDB enforces the
foreign keys but has no `ON DELETE CASCADE`, so `delete_site` does it in application
code (see the comment there: the deletes must not be wrapped in a transaction).

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

# traffic-violation

## Local development

```
python3 -m venv .venv
.venv/bin/pip install -e ./shared -e ./services/site-service
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

Run site-service:
```
set -a; source .env; set +a   # R2 credentials — see .env.example
PYTHONPATH=shared/src:services/site-service .venv/bin/uvicorn site_service.main:app --reload --port 8001
```

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

No queue or docker is required yet. Site creation only persists metadata (status
`created`); video metadata extraction and job enqueueing are deferred to a later
slice. Object storage (R2) is needed only for the file endpoints — the test suite
never touches it.

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

`sites.metadata` extraction and job enqueueing — which the LLD hangs off `POST /sites`
— are still deferred; both need real infrastructure.

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

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

Run tests:
```
.venv/bin/pytest
```

Run site-service:
```
PYTHONPATH=shared/src:services/site-service .venv/bin/uvicorn site_service.main:app --reload --port 8001
```

Endpoints are served under `/api/v1` — e.g. `http://localhost:8001/api/v1/sites`.
Services mount that prefix themselves and the gateway does not rewrite it, so the
path is identical whether a request arrives through the gateway or goes directly to
the service. `/health` is served at the root, outside the prefix.

Site data is stored in a local DuckDB file at `./data/site_service.duckdb`
(override with `SITE_SERVICE_DB_PATH`). No AWS/S3, queue, or docker required
for this slice — site creation only persists metadata (status `created`);
video metadata extraction and job enqueueing are deferred to a later slice.

### Calibrations

A camera calibration belongs to a site, so it is addressed under one:

```
POST /api/v1/sites/{site_id}/calibrations   {"url": "s3://bucket/calibration.json"}
GET  /api/v1/sites/{site_id}/calibrations                  -> the active calibration
GET  /api/v1/sites/{site_id}/calibrations/{calibration_id} -> a specific version
```

There is no upload endpoint yet. The client is expected to upload the file to S3
through file-service and pass the resulting `url`; only that url is stored, and the
LLD's `file_id` reference is deferred until a `files` table exists.

Each POST appends a new version (`version = max(version) + 1` for that site) rather
than replacing the previous one, and the highest version is the one valid calibration
that `GET .../calibrations` returns. Deleting a site deletes its calibrations —
DuckDB enforces the foreign key but has no `ON DELETE CASCADE`, so `delete_site`
does it in application code (see the comment there: the two deletes must not be
wrapped in a transaction).

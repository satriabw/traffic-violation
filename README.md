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

Site data is stored in a local DuckDB file at `./data/site_service.duckdb`
(override with `SITE_SERVICE_DB_PATH`). No AWS/S3, queue, or docker required
for this slice — site creation only persists metadata (status `created`);
video metadata extraction and job enqueueing are deferred to a later slice.

import os

DB_PATH = os.environ.get("SITE_SERVICE_DB_PATH", "./data/site_service.duckdb")

# Every service mounts its resource routers under this prefix, and callers use the
# same path whether they reach the service through the gateway or directly by its
# container name. The gateway routes on this prefix but must not rewrite it.
API_V1_PREFIX = "/api/v1"

# Postgres pg15 → pg16 migration (pgvector)

`jarvis-command-center`'s compose files (`docker-compose.prod.yaml`,
`docker-compose.dev.yaml`) now pin `pgvector/pgvector:pg16`, aligning with
`jarvis-installer` (the user-facing source of truth, already on pg16). This
removes the version mismatch tracked in
[roadmap#46](https://github.com/alexberardi/jarvis-roadmap/issues/46).

> **Postgres major versions are not data-compatible.** A pg16 server cannot
> start against a data directory created by pg15 — it fails hard with
> `FATAL: database files are incompatible with server`. There is no in-place,
> zero-touch upgrade; existing operators must run a one-time migration.

## Who is affected

- **Fresh installs / new volumes** — **no action needed.** A brand-new
  `postgres_data` (prod) or `cc-pg-data` (dev standalone) volume is initialized
  directly by pg16.
- **Existing pg15 data directory** — you started postgres on `:pg15` at least
  once, so the volume holds a pg15 data directory. You must migrate it before
  running the pg16 image, or the container will refuse to start.

Check what your volume holds:

```bash
docker run --rm -v jarvis-command-center_postgres_data:/data alpine \
  cat /data/PG_VERSION   # "15" => migrate;  "16" => already good
```

(Substitute your actual volume name — `postgres_data` for prod,
`cc-pg-data` for the dev `standalone` profile. `docker volume ls` lists them.)

## Recommended path: logical dump / restore

This is the safest, most portable route and preserves the pgvector extension
data (vectors survive a logical dump/restore).

1. **Bring up the OLD (pg15) database only** — temporarily pin the postgres
   service back to `pgvector/pgvector:pg15`, or run a throwaway pg15 container
   bound to the existing volume, and start just that service.

2. **Dump the database** from the running pg15 container:

   ```bash
   docker compose exec postgres \
     pg_dump -U jarvis_user -d jarvis_command_center -Fc \
     > jarvis_command_center.pg15.dump
   ```

3. **Stop the stack and replace the volume** so pg16 initializes a fresh data
   directory (do NOT reuse the pg15 volume):

   ```bash
   docker compose down
   docker volume rm jarvis-command-center_postgres_data   # your prod volume
   ```

4. **Start pg16** (the compose files now pin `pgvector/pgvector:pg16`) so it
   creates a new, empty pg16 data directory:

   ```bash
   docker compose up -d postgres
   ```

5. **Restore** into the fresh pg16 database:

   ```bash
   docker compose exec -T postgres \
     pg_restore -U jarvis_user -d jarvis_command_center --clean --if-exists \
     < jarvis_command_center.pg15.dump
   ```

6. **Verify** the `vector` extension and your data are present, then bring the
   rest of the stack up:

   ```bash
   docker compose exec postgres \
     psql -U jarvis_user -d jarvis_command_center -c "\dx vector"
   docker compose up -d
   ```

## Alternative: in-place `pg_upgrade`

`pg_upgrade` (via an image such as `pgautoupgrade/pgautoupgrade`) can upgrade the
data directory in place without a full dump/restore. It is faster for very large
databases but more involved to operate and must be run against a stopped
database with both the old (15) and new (16) binaries available. Prefer the
dump/restore path above unless database size makes it impractical; if you use
`pg_upgrade`, take a full backup of the volume first.

## Rollback

Keep the `jarvis_command_center.pg15.dump` file and/or a snapshot of the
original pg15 volume until you have confirmed the pg16 database is healthy. To
roll back, restore the pg15 volume and re-pin the image to
`pgvector/pgvector:pg15`.

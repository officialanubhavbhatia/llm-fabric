# Backup and recovery

This is the durable-state recovery contract for MyVista LLM Fabric.

A backup that has never been restored is not considered verified. The sqlite
clone path in `src/llm_fabric/storage/backup.py` is the automated verification
job. PostgreSQL `pg_dump`/`pg_restore` is the production procedure.

## What is durable

| Store | Role | Backup |
| --- | --- | --- |
| PostgreSQL | tenants, users, conversations, prompts, traces, eval metadata, audit events | Required. Point-in-time recovery if WAL archiving is enabled. |
| Redis / Valkey | quotas, breakers, revocation denylist, hot cache | Disposable. Reconstruct from Postgres and live traffic. A Redis snapshot is optional, not a recovery target. |
| ClickHouse / analytics buffer | high-volume traces | Not on the request path. Loss is telemetry loss, not serving loss. |
| Object storage | eval datasets, benchmark artifacts | Not implemented in this repository. |

## Redis: disposable vs reconstructable

Disposable (safe to lose on restart):

- exact/semantic cache entries
- circuit-breaker snapshots (workers re-learn from live errors)
- intent cache

Must be reconstructed, not restored from Redis:

- quota counters (fail closed to the configured policy after a flush, or accept a window of uncounted traffic — operators choose; default after Redis loss in production revocation is fail-closed)
- revocation denylist (re-revoke from the identity provider or an audit log)

## PostgreSQL procedure

### Backup frequency

- Base backup: at least daily
- WAL archive: continuous where RPO below 24h is required

### Retention

- Daily backups: 14 days
- Weekly backups: 8 weeks

These are the recommended starting values, not a measured operational history.

### RPO / RTO (targets, not measured)

- RPO target: 24 hours without WAL; minutes with WAL archiving
- RTO target: 1 hour for a single-region restore of the control-plane database

These targets have not been measured against a production incident.

### Encryption

Encrypt backups at rest with the same key-management system used for the
database volume. Do not check backup files into git. Do not store credentials
inside the dump.

### Restore procedure

1. Provision an empty PostgreSQL instance.
2. Restore the base backup (`pg_restore` or `pg_basebackup` recovery).
3. Replay WAL to the desired timestamp if PITR is configured.
4. Run `python -m llm_fabric doctor` against the restored DSN.
5. Run `pytest -m tenant_isolation` against `LLM_FABRIC_TEST_DATABASE_URL` pointing at the restored copy.
6. Only then point the gateway at the restored instance.

### Local / test verification

```text
pytest tests/unit/test_backup.py -q
pytest tests/unit/test_postgres_backup_live.py -q
```

`test_backup.py` dumps a sqlite engine to JSON and restores it into a second
engine, then asserts tenant A cannot read tenant B's conversations.

`test_postgres_backup_live.py` is the current-release PostgreSQL proof: Alembic
head, `pg_dump`/`pg_restore`, RLS isolation, and usage row counts against a live
Postgres. It does not measure multi-region DR or a managed-service RPO/RTO.

### Failure cases

- Restore into a non-empty cluster: refuse; restore only to a new instance.
- Partial dump: the verification query (tenant isolation plus row counts) must fail the job.
- Backup role subject to FORCE ROW LEVEL SECURITY: the dump would see only one tenant. Production dumps must use a `BYPASSRLS` role.
- Redis restored without Postgres: quotas and revocation will not match durable records; do not do this.

### Verification job

CI runs the sqlite restore test on every change. A PostgreSQL `pg_dump` round-trip
is **not** claimed as verified unless `LLM_FABRIC_TEST_DATABASE_URL` is postgresql
and an operator has run `pg_dump` / `pg_restore` against that instance.

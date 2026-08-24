-- Runtime role: DML only. Schema changes belong to POSTGRES_USER (`fabric`),
-- which is the migration/table-owner role used by `alembic upgrade`.
-- Do not GRANT CREATE, CREATEDB, CREATEROLE, or BYPASSRLS to fabric_app.
CREATE USER fabric_app WITH PASSWORD 'fabric' NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS LOGIN;
GRANT CONNECT ON DATABASE fabric TO fabric_app;
GRANT USAGE ON SCHEMA public TO fabric_app;
REVOKE CREATE ON SCHEMA public FROM fabric_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO fabric_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO fabric_app;

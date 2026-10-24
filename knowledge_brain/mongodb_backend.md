# MongoDB Backend Integration

## Overview
MongoDB acts as the primary document and memory storage engine for **AR.I.S**. It stores dynamic brain representations, contextual memory, interaction logs, and cognitive models.

## Connection Architecture
- **Driver**: Official `pymongo` with SRV/DNS resolution (`dnspython`).
- **Connection Module**: `api/db.py`
  - Singleton client management: `get_mongo_client()` caches the active `MongoClient` instance with a 5000ms server selection timeout.
  - Active database accessor: `get_mongo_db()` accesses the configured database (`MONGODB_DB_NAME`).
  - Active health check: `check_mongo_connection()` issues an admin `ping` command and catches `ConnectionFailure` and `ServerSelectionTimeoutError` to return structured diagnostics.

## Configuration (.env)
- `MONGODB_URI`: Connection string (supports both `mongodb://` and `mongodb+srv://`).
- `MONGODB_DB_NAME`: Target database name (defaults to `aris_db`).

## Standards Adherence
- **Zero Hard Patches (Rule 1)**: Rather than hacking Django's relational ORM with unstable third-party monkey patches, PyMongo provides a direct, native, high-performance interface.
- **Fail-Safe Operation**: If the database cluster experiences transient network disconnection, the API gateway reports the status cleanly instead of throwing uncaught server faults.

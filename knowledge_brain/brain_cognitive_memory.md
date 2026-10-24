# Brain Cognitive Memory (AR.I.S)

## Overview
The **AR.I.S Brain** (`ArisBrain`) serves as the foundational cognitive primitive for knowledge ingestion, persistent storage, and retrieval.
It enforces a strict boundary separation between external interfaces (Terminal CLI, REST API Gateway) and persistent storage (MongoDB).

---

## Architectural Principle: Strict Dependency Flow

```
CLI ───────────────┐
                   │
REST API ──────────┼──► ArisBrain (Cognitive Core) ───► MongoDB (brain_facts)
                   │
Future Interfaces ─┘
```

### Key Invariants:
1. **Single Cognitive Boundary**: `ArisBrain` is the sole module that connects to and communicates with MongoDB.
2. **Zero Storage Logic in Interfaces**: Terminal CLI and REST API Views never contain database queries or direct collection access.
3. **Boundary Authentication**: Authentication and authorization sit strictly at the API gateway boundary, never polluting `ArisBrain`.
4. **Token-Ready Design**: Designed so future token authentication (e.g. JWT/Token) attaches cleanly to the REST views without altering the underlying brain contracts.
5. **No Hard Patches**: Malformed inputs, missing records, and network disconnects are handled cleanly and transparently with standard status codes and diagnostic messages.

---

## The Cognitive Primitive: Store → Retrieve → Inspect

### 1. Fact Schema (`brain_facts` collection)
| Field | Type | Description |
|---|---|---|
| `id` | String | Unique MongoDB document identifier (`_id` hex). |
| `statement` | String | The core factual assertion. |
| `topic` | String | Conceptual domain / category (lowercase). |
| `tags` | Array[String] | Normalized tags / entity keywords. |
| `source` | String | Origin identifier (e.g., `terminal_cli`, `api_connection`). |
| `confidence` | Float | Cognitive confidence level (0.0 to 1.0). |
| `metadata` | Dict | Extensible context for future cognitive growth. |
| `created_at` | String (ISO) | UTC creation timestamp. |
| `updated_at` | String (ISO) | UTC last update timestamp. |

---

## Usage Interfaces

### 1. Terminal CLI (`manage.py brain`)
Focused on clean, distraction-free factual knowledge output:
- **Store Fact**:
  ```powershell
  python manage.py brain store "My statement here"
  ```
- **Retrieve Facts**:
  ```powershell
  python manage.py brain retrieve
  python manage.py brain retrieve --query "search term"
  # Optional: add --verbose (-v) if raw ID, tags, and timestamp are needed.
  ```
- **Count Facts**:
  ```powershell
  python manage.py brain count
  ```
- **Interactive Mode**:
  ```powershell
  python manage.py brain interactive
  ```

### 2. REST API Gateway (Token-Ready)
- `POST /api/brain/facts/`:
  - Body: `{"statement": "...", "topic": "...", "tags": [...]}`
  - Response: `201 Created` with serialized fact.
- `GET /api/brain/facts/?q=<query>&topic=<topic>&limit=<n>`:
  - Response: `200 OK` with list of results and total count.
- `GET /api/brain/facts/<fact_id>/`:
  - Response: `200 OK` with fact or `404 Not Found`.

---

## Evolution Roadmap
This cognitive primitive is structured to scale cleanly without dismantling the foundation:
- **Phase 1 (Current)**: Facts (Store → Retrieve → Inspect).
- **Phase 2**: Concepts & Taxonomy.
- **Phase 3**: Relations & Semantic Graphs.
- **Phase 4**: Episodic & Associative Memory.
- **Phase 5**: Autonomous Reasoning & Cognitive Reflection.

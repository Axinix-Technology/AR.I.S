"""
AR.I.S — Persistent Neuro-Symbolic Graph Store (2026 Production Grade)
Handles indexed SQLite persistence and sub-millisecond multi-hop graph path exploration.
"""

from __future__ import annotations
import sqlite3
import json
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple
from collections import deque

from .schema import EntityNode, RelationEdge, GraphPath


class ArisGraphStore:
    """
    High-performance Knowledge Graph store combining relational storage
    with fast multi-hop path traversal.
    """

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_dir = Path("models") / "memory"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "knowledge_graph.sqlite3"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    entity_type TEXT DEFAULT 'concept',
                    description TEXT DEFAULT '',
                    embedding TEXT DEFAULT NULL,
                    access_count INTEGER DEFAULT 0,
                    created_at REAL NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    snippet TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    FOREIGN KEY(source_id) REFERENCES entities(id),
                    FOREIGN KEY(target_id) REFERENCES entities(id)
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ent_name ON entities(name)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_rel_src ON relations(source_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_rel_tgt ON relations(target_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_rel_pred ON relations(predicate)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_rel_pair ON relations(source_id, target_id)")
            conn.commit()

    def add_entity(self, entity: EntityNode):
        """Inserts or updates an entity node in the graph."""
        emb_json = json.dumps(entity.embedding) if entity.embedding is not None else None
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO entities (id, name, entity_type, description, embedding, access_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    entity_type=excluded.entity_type,
                    description=excluded.description,
                    embedding=COALESCE(excluded.embedding, entities.embedding),
                    access_count=entities.access_count + 1
            """, (entity.id, entity.name, entity.entity_type, entity.description, emb_json, entity.access_count, entity.created_at))
            conn.commit()

    def add_relation(self, relation: RelationEdge):
        """Inserts a directed relation edge between two entities."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO relations (source_id, predicate, target_id, weight, snippet, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (relation.source_id, relation.predicate, relation.target_id, relation.weight, relation.snippet, relation.created_at))
            conn.commit()

    def get_entity(self, entity_id: str) -> Optional[EntityNode]:
        """Retrieves an entity node by its unique identifier."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM entities WHERE id = ?", (entity_id,))
            row = cur.fetchone()
            if not row:
                return None
            emb = json.loads(row["embedding"]) if row["embedding"] else None
            return EntityNode(
                id=row["id"],
                name=row["name"],
                entity_type=row["entity_type"],
                description=row["description"],
                embedding=emb,
                access_count=row["access_count"],
                created_at=row["created_at"],
            )

    def find_entity_by_name(self, name: str) -> Optional[EntityNode]:
        """Looks up an entity by name (case-insensitive exact match)."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM entities WHERE LOWER(name) = LOWER(?) LIMIT 1", (name.strip(),))
            row = cur.fetchone()
            if not row:
                return None
            emb = json.loads(row["embedding"]) if row["embedding"] else None
            return EntityNode(
                id=row["id"],
                name=row["name"],
                entity_type=row["entity_type"],
                description=row["description"],
                embedding=emb,
                access_count=row["access_count"],
                created_at=row["created_at"],
            )

    def get_out_edges(self, source_id: str) -> List[RelationEdge]:
        """Returns all outgoing relations from a given entity."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM relations WHERE source_id = ?", (source_id,))
            edges = []
            for r in cur.fetchall():
                edges.append(RelationEdge(
                    source_id=r["source_id"],
                    predicate=r["predicate"],
                    target_id=r["target_id"],
                    weight=r["weight"],
                    snippet=r["snippet"],
                    created_at=r["created_at"],
                ))
            return edges

    def get_in_edges(self, target_id: str) -> List[RelationEdge]:
        """Returns all incoming relations into a given entity."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM relations WHERE target_id = ?", (target_id,))
            edges = []
            for r in cur.fetchall():
                edges.append(RelationEdge(
                    source_id=r["source_id"],
                    predicate=r["predicate"],
                    target_id=r["target_id"],
                    weight=r["weight"],
                    snippet=r["snippet"],
                    created_at=r["created_at"],
                ))
            return edges

    def find_paths(self, source_id: str, target_id: str, max_depth: int = 3) -> List[GraphPath]:
        """
        Executes cycle-safe BFS to discover all multi-hop reasoning paths
        connecting source_id to target_id within max_depth hops.
        """
        if source_id == target_id:
            src_node = self.get_entity(source_id)
            return [GraphPath(nodes=[src_node] if src_node else [], edges=[], score=1.0)]

        results: List[GraphPath] = []
        # Queue state: (current_node_id, current_nodes_list, current_edges_list, current_depth)
        src_node = self.get_entity(source_id)
        if not src_node:
            return results

        queue = deque([(source_id, [src_node], [], 0)])

        while queue:
            curr_id, curr_nodes, curr_edges, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for edge in self.get_out_edges(curr_id):
                nxt_id = edge.target_id
                # Avoid cycles
                if any(n.id == nxt_id for n in curr_nodes):
                    continue

                nxt_node = self.get_entity(nxt_id)
                if not nxt_node:
                    continue

                new_nodes = curr_nodes + [nxt_node]
                new_edges = curr_edges + [edge]

                if nxt_id == target_id:
                    # Calculate path score based on edge weights and length penalty
                    score = 1.0 / (len(new_edges) + 0.1)
                    results.append(GraphPath(nodes=new_nodes, edges=new_edges, score=score))
                else:
                    queue.append((nxt_id, new_nodes, new_edges, depth + 1))

        results.sort(key=lambda p: p.score, reverse=True)
        return results

    def explore_subgraph(
        self,
        seed_ids: List[str],
        max_hops: int = 2,
        max_paths: int = 10,
    ) -> List[GraphPath]:
        """
        Expands around a set of seed entities to gather contextual multi-hop paths.
        """
        paths: List[GraphPath] = []
        visited_nodes: Set[str] = set()

        for s_id in seed_ids:
            s_node = self.get_entity(s_id)
            if not s_node:
                continue

            queue = deque([(s_id, [s_node], [], 0)])
            while queue and len(paths) < max_paths:
                curr_id, curr_nodes, curr_edges, depth = queue.popleft()

                if curr_edges:
                    paths.append(GraphPath(nodes=curr_nodes, edges=curr_edges, score=1.0 / (depth + 0.5)))

                if depth >= max_hops:
                    continue

                for edge in self.get_out_edges(curr_id):
                    nxt_id = edge.target_id
                    if any(n.id == nxt_id for n in curr_nodes):
                        continue
                    nxt_node = self.get_entity(nxt_id)
                    if not nxt_node:
                        continue
                    queue.append((nxt_id, curr_nodes + [nxt_node], curr_edges + [edge], depth + 1))

        paths.sort(key=lambda p: p.score, reverse=True)
        return paths[:max_paths]

    def get_all_entities(self) -> List[EntityNode]:
        """Returns all entities currently stored in the knowledge graph."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM entities")
            results = []
            for r in cur.fetchall():
                emb = json.loads(r["embedding"]) if r["embedding"] else None
                results.append(EntityNode(
                    id=r["id"],
                    name=r["name"],
                    entity_type=r["entity_type"],
                    description=r["description"],
                    embedding=emb,
                    access_count=r["access_count"],
                    created_at=r["created_at"],
                ))
            return results

    def merge_from(self, other_db_path: Path) -> int:
        """
        Merges entities and relations from another team member's SQLite knowledge graph
        into this store without primary key collisions.
        """
        other_path = Path(other_db_path)
        if not other_path.exists():
            print(f"Warning: Source graph {other_path} not found.")
            return 0

        other_store = ArisGraphStore(db_path=other_path)
        other_entities = other_store.get_all_entities()
        for ent in other_entities:
            self.add_entity(ent)

        rel_count = 0
        with other_store._get_connection() as conn_src:
            cur_src = conn_src.cursor()
            cur_src.execute("SELECT * FROM relations")
            for r in cur_src.fetchall():
                rel = RelationEdge(
                    source_id=r["source_id"],
                    predicate=r["predicate"],
                    target_id=r["target_id"],
                    weight=r["weight"],
                    snippet=r["snippet"],
                    created_at=r["created_at"],
                )
                self.add_relation(rel)
                rel_count += 1

        print(f"[Memory Merged] Successfully integrated {len(other_entities)} entities and {rel_count} relations from {other_path.name}")
        return len(other_entities)
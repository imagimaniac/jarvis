"""Storage: raw files + SQLite + FTS5."""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import utils

SCHEMA = """
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS entries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  uuid TEXT NOT NULL UNIQUE,
  created_at_utc TEXT NOT NULL,
  created_at_local TEXT,
  tz TEXT,
  source_type TEXT DEFAULT 'text',
  source TEXT DEFAULT 'cli',
  origin_kind TEXT DEFAULT 'self',
  origin_id TEXT,
  import_batch_id TEXT,
  external_ref TEXT,
  raw_path TEXT NOT NULL,
  raw_relpath TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  text_raw TEXT NOT NULL,
  size_bytes INTEGER DEFAULT 0,
  updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_entries_created_utc ON entries(created_at_utc);
CREATE INDEX IF NOT EXISTS idx_entries_source ON entries(source);
CREATE INDEX IF NOT EXISTS idx_entries_origin_kind ON entries(origin_kind);
CREATE UNIQUE INDEX IF NOT EXISTS idx_entries_source_origin_id ON entries(source, origin_id) WHERE origin_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_entries_content_hash ON entries(content_hash);

CREATE TABLE IF NOT EXISTS structured (
  entry_id INTEGER PRIMARY KEY REFERENCES entries(id) ON DELETE CASCADE,
  date TEXT,
  time_local TEXT,
  type TEXT,
  mood TEXT,
  people TEXT,
  places TEXT,
  projects TEXT,
  topics TEXT,
  events TEXT,
  questions TEXT,
  decisions TEXT,
  learnings TEXT,
  tags TEXT,
  category TEXT,
  importance TEXT DEFAULT 'MEDIUM',
  confidence REAL DEFAULT 0.5,
  inferred_by TEXT DEFAULT 'heuristic',
  needs_review INTEGER DEFAULT 0,
  reason TEXT,
  is_sensitive INTEGER DEFAULT 0,
  epistemic_status TEXT
);

CREATE INDEX IF NOT EXISTS idx_structured_importance ON structured(importance);
CREATE INDEX IF NOT EXISTS idx_structured_category ON structured(category);

CREATE TABLE IF NOT EXISTS relations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  from_entry_id INTEGER NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
  to_entry_id INTEGER NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
  relation TEXT NOT NULL,
  confidence REAL DEFAULT 0.5,
  inferred_by TEXT DEFAULT 'heuristic',
  created_at_utc TEXT NOT NULL,
  rationale TEXT
);

CREATE TABLE IF NOT EXISTS sources (
  entry_id INTEGER PRIMARY KEY REFERENCES entries(id) ON DELETE CASCADE,
  origin TEXT,
  ref TEXT,
  fetched_at_utc TEXT
);

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS imports (
  batch_id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  started_at_utc TEXT NOT NULL,
  finished_at_utc TEXT,
  items_count INTEGER DEFAULT 0,
  errors TEXT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS syntheses (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  query TEXT,
  type TEXT,
  text TEXT NOT NULL,
  entries_json TEXT,
  raw_paths TEXT,
  method TEXT,
  created_at_utc TEXT NOT NULL,
  includes_external INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS events_seen (
  event_id TEXT NOT NULL,
  source_ref TEXT,
  seen_at_utc TEXT,
  dismissed_at_utc TEXT,
  acted_on_at_utc TEXT,
  PRIMARY KEY (event_id, source_ref)
);

"""

FTS5_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
  entry_id UNINDEXED,
  text_raw,
  tokenize='porter'
);
"""

_BASELINE_VERSION = 1  # what the SCHEMA statements below establish

def _has_column(cur, table: str, column: str) -> bool:
    cur.execute(f"PRAGMA table_info({table})")
    return any(r["name"] == column for r in cur.fetchall())


_MEMORY_HISTORY_JUDGE_DDL = """CREATE TABLE {if_not_exists} {table} (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE RESTRICT,
  action TEXT NOT NULL CHECK (action IN ('CANDIDATE_CREATED','CONFIRMED','REJECTED','UPDATED','SUPERSEDED','FORGOTTEN','DEFERRED')),
  changed_at_utc TEXT NOT NULL,
  reason TEXT NOT NULL CHECK (length(trim(reason)) > 0),
  previous_state TEXT,
  new_state TEXT NOT NULL,
  reviewer TEXT
)"""


MEMORY_SCHEMA_STATEMENTS: List[str] = [
    """CREATE TABLE IF NOT EXISTS memories (
      id TEXT PRIMARY KEY,
      memory_type TEXT NOT NULL CHECK (memory_type IN
        ('FACT','PREFERENCE','GOAL','DECISION','PROJECT','HABIT','LEARNING','HISTORY','TEMPORARY')),
      claim TEXT NOT NULL CHECK (length(trim(claim)) > 0),
      lifecycle_status TEXT NOT NULL CHECK (lifecycle_status IN ('ACTIVE','SUPERSEDED','FORGOTTEN')),
      review_state TEXT NOT NULL CHECK (review_state IN ('CANDIDATE','CONFIRMED','REJECTED')),
      confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
      importance INTEGER NOT NULL CHECK (importance BETWEEN 1 AND 5),
      is_sensitive INTEGER NOT NULL CHECK (is_sensitive IN (0,1)),
      source_attribution TEXT NOT NULL CHECK (length(trim(source_attribution)) > 0),
      candidate_reason TEXT NOT NULL DEFAULT 'Existing Phase 1 memory',
      comparison_relation TEXT NOT NULL DEFAULT 'NEW'
        CHECK (comparison_relation IN ('NEW','POSSIBLE_UPDATE','POSSIBLE_CONFLICT','DUPLICATE','TEMPORARY')),
      related_memory_id TEXT REFERENCES memories(id) ON DELETE RESTRICT,
      created_at_utc TEXT NOT NULL,
      updated_at_utc TEXT NOT NULL CHECK (updated_at_utc >= created_at_utc),
      valid_from_utc TEXT NOT NULL,
      valid_until_utc TEXT,
      superseded_by_memory_id TEXT REFERENCES memories(id) ON DELETE RESTRICT,
      CHECK (valid_until_utc IS NULL OR valid_until_utc >= valid_from_utc),
      CHECK (superseded_by_memory_id IS NULL OR superseded_by_memory_id <> id),
      CHECK ((lifecycle_status = 'SUPERSEDED') = (superseded_by_memory_id IS NOT NULL))
    )""",
    "CREATE INDEX IF NOT EXISTS idx_memories_lifecycle_review ON memories(lifecycle_status, review_state)",
    "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)",
    "CREATE INDEX IF NOT EXISTS idx_memories_validity ON memories(valid_from_utc, valid_until_utc)",
    """CREATE TABLE IF NOT EXISTS memory_sources (
      memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE RESTRICT,
      entry_id INTEGER NOT NULL REFERENCES entries(id) ON DELETE RESTRICT,
      relationship TEXT NOT NULL DEFAULT 'supports' CHECK (relationship IN ('origin','supports','contradicts')),
      linked_at_utc TEXT NOT NULL,
      PRIMARY KEY (memory_id, entry_id)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_memory_sources_entry ON memory_sources(entry_id)",
    _MEMORY_HISTORY_JUDGE_DDL.format(if_not_exists="IF NOT EXISTS", table="memory_history"),
    "CREATE INDEX IF NOT EXISTS idx_memory_history_memory_time ON memory_history(memory_id, changed_at_utc, id)",
    """CREATE VIRTUAL TABLE IF NOT EXISTS canonical_memories_fts USING fts5(
      memory_id UNINDEXED, claim, tokenize='porter'
    )""",
]

EXTERNAL_SOURCE_SCHEMA_STATEMENTS: List[str] = [
    """CREATE TABLE IF NOT EXISTS external_source_records (
      id TEXT PRIMARY KEY,
      provider_type TEXT NOT NULL,
      external_id TEXT NOT NULL,
      title TEXT NOT NULL DEFAULT '',
      occurred_at_utc TEXT,
      source_created_at_utc TEXT,
      source_updated_at_utc TEXT,
      author TEXT,
      normalized_text TEXT NOT NULL,
      metadata_json TEXT NOT NULL,
      source_url TEXT,
      imported_at_utc TEXT NOT NULL,
      content_hash TEXT NOT NULL,
      content_version INTEGER NOT NULL CHECK (content_version >= 1),
      current_entry_id INTEGER NOT NULL UNIQUE REFERENCES entries(id) ON DELETE RESTRICT,
      provenance_json TEXT NOT NULL,
      deleted_at_utc TEXT,
      UNIQUE(provider_type, external_id)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_external_source_records_provider_time "
    "ON external_source_records(provider_type, source_updated_at_utc)",
    "CREATE INDEX IF NOT EXISTS idx_external_source_records_imported "
    "ON external_source_records(imported_at_utc)",
    """CREATE TABLE IF NOT EXISTS external_source_versions (
      record_id TEXT NOT NULL REFERENCES external_source_records(id) ON DELETE RESTRICT,
      version INTEGER NOT NULL CHECK (version >= 1),
      content_hash TEXT NOT NULL,
      title TEXT NOT NULL DEFAULT '',
      occurred_at_utc TEXT,
      source_created_at_utc TEXT,
      source_updated_at_utc TEXT,
      author TEXT,
      normalized_text TEXT NOT NULL,
      metadata_json TEXT NOT NULL,
      source_url TEXT,
      imported_at_utc TEXT NOT NULL,
      entry_id INTEGER NOT NULL UNIQUE REFERENCES entries(id) ON DELETE RESTRICT,
      provenance_json TEXT NOT NULL,
      PRIMARY KEY(record_id, version)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_external_source_versions_hash "
    "ON external_source_versions(content_hash)",
]

CONTINUITY_SCHEMA_STATEMENTS: List[str] = [
    """CREATE TABLE IF NOT EXISTS continuity_threads (
      thread_id TEXT PRIMARY KEY,
      status TEXT NOT NULL CHECK (status IN ('ACTIVE','PAUSED','RESOLVED')),
      topic TEXT NOT NULL DEFAULT '',
      current_problem_or_goal TEXT NOT NULL DEFAULT '',
      decisions_json TEXT NOT NULL DEFAULT '[]',
      open_questions_json TEXT NOT NULL DEFAULT '[]',
      temporary_commitments_json TEXT NOT NULL DEFAULT '[]',
      last_activity_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      expires_at TEXT,
      revision INTEGER NOT NULL DEFAULT 1 CHECK (revision >= 1)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_continuity_threads_status_activity "
    "ON continuity_threads(status, last_activity_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_continuity_threads_expiry "
    "ON continuity_threads(expires_at)",
]

CONTEXT_TOPIC_SCHEMA_STATEMENTS: List[str] = [
    """CREATE TABLE IF NOT EXISTS context_topics (
      topic_id TEXT PRIMARY KEY,
      display_name TEXT NOT NULL CHECK (length(trim(display_name)) BETWEEN 1 AND 160),
      normalized_key TEXT NOT NULL UNIQUE CHECK (length(trim(normalized_key)) > 0),
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL CHECK (updated_at >= created_at)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_context_topics_display_name ON context_topics(display_name)",
    """CREATE TABLE IF NOT EXISTS continuity_thread_topics (
      thread_id TEXT NOT NULL REFERENCES continuity_threads(thread_id) ON DELETE CASCADE,
      topic_id TEXT NOT NULL REFERENCES context_topics(topic_id) ON DELETE CASCADE,
      relation TEXT NOT NULL CHECK (relation IN ('PRIMARY','RELATED')),
      confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
      provenance TEXT NOT NULL CHECK (provenance IN ('USER_EXPLICIT','CONTINUITY_EXACT_MATCH')),
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL CHECK (updated_at >= created_at),
      PRIMARY KEY (thread_id, topic_id)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_continuity_thread_topics_topic ON continuity_thread_topics(topic_id, relation, updated_at DESC)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_continuity_thread_topics_primary ON continuity_thread_topics(thread_id) WHERE relation='PRIMARY'",
]

GMAIL_SYNC_STATE_SCHEMA_STATEMENTS: List[str] = [
    """CREATE TABLE IF NOT EXISTS gmail_sync_state (
      account_key TEXT NOT NULL,
      query_key TEXT NOT NULL,
      history_id TEXT NOT NULL,
      updated_at_utc TEXT NOT NULL,
      PRIMARY KEY (account_key, query_key)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_gmail_sync_state_updated ON gmail_sync_state(updated_at_utc)",
]

def _migrate_memory_history_for_judge(cur) -> None:
    """v9: rebuild memory_history to carry a reviewer column and a DEFERRED action.

    A CHECK constraint cannot be altered in place in SQLite, so the table is
    rebuilt inside a transaction. The helper is idempotent: it inspects the
    live table definition and returns immediately when the target shape is
    already present, so a partially applied migration safely self-heals on
    retry. Nothing references memory_history for an FK, so a rebuild is safe.
    """
    ddl = cur.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='memory_history'"
    ).fetchone()
    current = (ddl["sql"] if ddl else "") or ""
    if "DEFERRED" in current and _has_column(cur, "memory_history", "reviewer"):
        return
    target = _MEMORY_HISTORY_JUDGE_DDL.format(if_not_exists="IF NOT EXISTS", table="memory_history_new")
    cur.execute("BEGIN")
    try:
        cur.execute(f"DROP TABLE IF EXISTS memory_history_new")
        cur.execute(target)
        cur.execute(
            "INSERT INTO memory_history_new (id, memory_id, action, changed_at_utc, reason, previous_state, new_state, reviewer) "
            "SELECT id, memory_id, action, changed_at_utc, reason, previous_state, new_state, NULL FROM memory_history"
        )
        cur.execute("DROP TABLE memory_history")
        cur.execute("ALTER TABLE memory_history_new RENAME TO memory_history")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_history_memory_time ON memory_history(memory_id, changed_at_utc, id)"
        )
        cur.execute("COMMIT")
    except Exception:
        cur.execute("ROLLBACK")
        raise


# (version, [statements]) applied in order for any version above the stored one.
# Anything added here must be idempotent so a partially applied migration can
# safely be retried. New databases establish all current schema statements first.
MIGRATIONS: List[Tuple[int, List[str]]] = [
    # v2 - epistemic status on structured facets. origin_kind is untouched and stays
    # binary; this is the separate axis. Derived from origin_kind, so it can never
    # be promoted to FACT.
    (2, [
        "ALTER TABLE structured ADD COLUMN epistemic_status TEXT",
        "UPDATE structured SET epistemic_status = CASE "
        "WHEN (SELECT origin_kind FROM entries WHERE entries.id = structured.entry_id) = 'self' "
        "THEN 'MEMORY' ELSE 'RECORDED' END WHERE epistemic_status IS NULL",
    ]),
    # v3 - canonical memories are additive and reference (never replace) entries.
    (3, MEMORY_SCHEMA_STATEMENTS),
    # v4 - retain candidate rationale and its non-mutating comparison result.
    (4, [
        "ALTER TABLE memories ADD COLUMN candidate_reason TEXT NOT NULL DEFAULT 'Existing Phase 1 memory'",
        "ALTER TABLE memories ADD COLUMN comparison_relation TEXT NOT NULL DEFAULT 'NEW' CHECK (comparison_relation IN ('NEW','POSSIBLE_UPDATE','POSSIBLE_CONFLICT','DUPLICATE','TEMPORARY'))",
        "ALTER TABLE memories ADD COLUMN related_memory_id TEXT REFERENCES memories(id) ON DELETE RESTRICT",
    ]),
    # v5 - normalized external source records and immutable content versions.
    # These rows stay separate from canonical memories; entries are only the
    # compatibility projection used by existing FTS/evidence/citation paths.
    (5, EXTERNAL_SOURCE_SCHEMA_STATEMENTS),
    # v6 - compact thread snapshots, kept separate from transcript, evidence,
    # and canonical personal-memory records.
    (6, CONTINUITY_SCHEMA_STATEMENTS),
    # v7 - lightweight topic groups for explicitly or strongly associated
    # continuity threads. No existing thread is assigned automatically.
    (7, CONTEXT_TOPIC_SCHEMA_STATEMENTS),
    # v8 - durable Gmail History API checkpoints and current-evidence tombstones.
    (8, [
        "ALTER TABLE external_source_records ADD COLUMN deleted_at_utc TEXT",
        *GMAIL_SYNC_STATE_SCHEMA_STATEMENTS,
    ]),
    # v9 - the Memory Judge can only record a reviewer identity and a DEFER
    # decision in memory_history once the table carries a reviewer column and
    # the DEFERRED action. SQLite cannot alter a CHECK or add an action value
    # in place, so the table is rebuilt by a guarded, transactional helper.
    (9, [_migrate_memory_history_for_judge]),
]

# Current version is derived, so adding a migration can never leave this stale.
SCHEMA_VERSION = max([_BASELINE_VERSION] + [v for v, _ in MIGRATIONS])

def _meta_get(cur, key: str) -> str | None:
    cur.execute("SELECT value FROM meta WHERE key=?", (key,))
    row = cur.fetchone()
    return row["value"] if row else None

def _meta_set(cur, key: str, value: str) -> None:
    cur.execute(
        "INSERT INTO meta (key,value,updated_at) VALUES (?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, value, utils.utc_iso()),
    )


def _migrate(cur) -> int:
    """Apply pending migrations. Returns the resulting schema version."""
    cur.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT NOT NULL)")
    stored = _meta_get(cur, "schema_version")
    if stored is None:
        # Fresh or pre-versioning database: SCHEMA already establishes the
        # current column set, so it starts at the derived SCHEMA_VERSION.
        version = SCHEMA_VERSION
        _meta_set(cur, "schema_version", str(version))
    else:
        try:
            version = int(stored)
        except (TypeError, ValueError):
            version = SCHEMA_VERSION
    for target, stmts in sorted(MIGRATIONS, key=lambda m: m[0]):
        if target <= version:
            continue
        for stmt in stmts:
            if callable(stmt):
                stmt(cur)
                continue
            s = stmt.strip()
            # ALTER TABLE ADD COLUMN has no IF NOT EXISTS; guard it so a
            # half-applied migration self-heals instead of raising.
            if s.upper().startswith("ALTER TABLE") and "ADD COLUMN" in s.upper():
                parts = s.split()
                if _has_column(cur, parts[2], parts[5].rstrip(",")):
                    continue
            cur.execute(s)
        version = target
        _meta_set(cur, "schema_version", str(version))
    return version

def _dict_factory(cursor, row):
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}

_ENTRY_COLUMNS = (
    "e.id, e.uuid, e.created_at_utc, e.created_at_local, e.source, e.origin_kind, "
    "e.raw_relpath, e.raw_path, e.external_ref, e.origin_id, e.text_raw"
)

_SQL_FTS = (
    f"SELECT {_ENTRY_COLUMNS}, s.epistemic_status, s.importance, s.category, "
    "s.inferred_by, s.confidence, "
    "snippet(entries_fts, -1, '«', '»', '…', 16) as snippet, rank as fts_rank "
    "FROM entries_fts f JOIN entries e ON e.id=f.entry_id "
    "LEFT JOIN structured s ON s.entry_id = e.id "
    "WHERE entries_fts MATCH ? AND COALESCE(e.source_type, '') NOT IN "
    "('conversation_candidate', 'canonical_memory_seed') "
    "AND (COALESCE(e.source_type, '') <> 'external_source' OR EXISTS "
    "(SELECT 1 FROM external_source_records x WHERE x.current_entry_id=e.id AND x.deleted_at_utc IS NULL)) "
    "ORDER BY rank LIMIT ?"
)

_SQL_LIKE = (
    f"SELECT {_ENTRY_COLUMNS}, s.epistemic_status, s.importance, s.category, "
    "s.inferred_by, s.confidence, "
    "'' as snippet, 0 as fts_rank "
    "FROM entries e LEFT JOIN structured s ON s.entry_id = e.id "
    "WHERE COALESCE(e.source_type, '') NOT IN ('conversation_candidate', 'canonical_memory_seed') "
    "AND (COALESCE(e.source_type, '') <> 'external_source' OR EXISTS "
    "(SELECT 1 FROM external_source_records x WHERE x.current_entry_id=e.id AND x.deleted_at_utc IS NULL)) "
    "AND e.text_raw LIKE ? LIMIT ?"
)


def _external_predicates(*, provider: str | None, start: str | None, end: str | None,
                         sender: str | None, alias: str = "x") -> tuple[str, list[str]]:
    sql = f"{alias}.deleted_at_utc IS NULL"
    params: list[str] = []
    if provider:
        sql += f" AND {alias}.provider_type=?"
        params.append(provider)
    if start:
        sql += f" AND {alias}.occurred_at_utc>=?"
        params.append(start)
    if end:
        sql += f" AND {alias}.occurred_at_utc<?"
        params.append(end)
    if sender:
        safe_sender = sender.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        sql += f" AND {alias}.author LIKE ? ESCAPE '\\' COLLATE NOCASE"
        params.append(f"%{safe_sender}%")
    return sql, params


class JarvisDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = _dict_factory
        self._init()

    def _init(self):
        cur = self.conn.cursor()
        # Connection/durability pragmas for the storage layer. WAL plus a
        # modest per-connection page cache keeps read-heavy recall fast without
        # widening the write surface; busy_timeout avoids immediate lock errors
        # when a sync/import script writes concurrently.
        try:
            cur.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
        try:
            cur.execute("PRAGMA busy_timeout=5000")
        except Exception:
            pass
        try:
            cur.execute("PRAGMA cache_size=-16000")
        except Exception:
            pass
        for stmt in SCHEMA.strip().split(';'):
            s = stmt.strip()
            if s:
                cur.execute(s)
        cur.execute(FTS5_SCHEMA)
        for stmt in MEMORY_SCHEMA_STATEMENTS:
            cur.execute(stmt)
        # Like the canonical memory tables, new databases establish current
        # tables before the migration marker is initialized; v5 also applies
        # these idempotent statements to pre-existing databases.
        for stmt in EXTERNAL_SOURCE_SCHEMA_STATEMENTS:
            cur.execute(stmt)
        for stmt in CONTINUITY_SCHEMA_STATEMENTS:
            cur.execute(stmt)
        for stmt in CONTEXT_TOPIC_SCHEMA_STATEMENTS:
            cur.execute(stmt)
        for stmt in GMAIL_SYNC_STATE_SCHEMA_STATEMENTS:
            cur.execute(stmt)
        self.schema_version = _migrate(cur)
        self.conn.commit()

    def save(self) -> None:
        self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def insert_entry(self, text: str, *, source: str = "cli", origin_kind: str = "self", source_type: str = "text", origin_id: str | None = None, external_ref: str | None = None, import_batch_id: str | None = None, raw_path: Path | None = None, raw_relpath: str | None = None, tz: str | None = None, content_hash: str | None = None) -> int:
        u = utils.new_uuid()
        cu = utils.utc_now()
        cl = utils.local_now()
        text_raw = str(text) if text is not None else ""
        rp_abs = Path(raw_path) if raw_path else Path("")
        if raw_relpath:
            rp_rel = str(raw_relpath)
        else:
            try:
                rp_rel = str(rp_abs.relative_to(rp_abs.parent)) if False else ""  # dummy
                # better: try to get relative to base if we can; but keep simple
                rp_rel = rp_abs.name if rp_abs.name else str(rp_abs)
            except Exception:
                rp_rel = str(rp_abs)
        # sanitize None -> None strings ok
        def s(v):
            if isinstance(v, (str,int,float,bool)) or v is None:
                return v
            return str(v) if v is not None else None
        ch = content_hash if content_hash else utils.content_hash(text_raw)
        sz = len(text_raw.encode('utf-8', errors='replace'))
        cur = self.conn.cursor()
        cur.execute(
            """INSERT OR IGNORE INTO entries (uuid,created_at_utc,created_at_local,tz,source_type,source,origin_kind,origin_id,import_batch_id,external_ref,raw_path,raw_relpath,content_hash,text_raw,size_bytes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (u, utils.utc_iso(cu), cl.isoformat(timespec='seconds'), s(tz), s(source_type), s(source), s(origin_kind), s(origin_id), s(import_batch_id), s(external_ref), str(rp_abs), rp_rel, ch, text_raw, sz),
        )
        self.conn.commit()
        # get id (maybe ignored if dup)
        if cur.lastrowid > 0:
            eid = cur.lastrowid
        else:
            cur.execute("SELECT id FROM entries WHERE uuid=?", (u,))
            row = cur.fetchone()
            eid = row['id'] if row else 0
        # FTS
        if eid:
            cur.execute("INSERT OR REPLACE INTO entries_fts (entry_id, text_raw) VALUES (?,?)", (eid, text_raw))
            self.conn.commit()
        return eid

    def add_structured(self, eid: int, s: "StructuredLike", inferred_by: str = "heuristic") -> None:
        from . import models, epistemic
        if hasattr(s, 'to_dict'):
            d = s.to_dict()
        else:
            d = dict(s)
        def jsonl(v):
            if v is None:
                return '[]'
            if isinstance(v, str):
                # if already JSON-like string
                return v
            if isinstance(v, list):
                import json
                return json.dumps(v, ensure_ascii=False)
            return '[]'
        def sstr(v):
            if v is None: return None
            if isinstance(v, str): return v
            return str(v)
        def fint(v): 
            try: return int(v) if v is not None else 0
            except: return 0
        def fflt(v):
            try: return float(v) if v is not None else 0.5
            except: return 0.5
        cur = self.conn.cursor()
        # Derived from origin_kind, never from the extractor, so a heuristic can
        # never talk an entry up to something the user actually asserted.
        origin_row = cur.execute("SELECT origin_kind FROM entries WHERE id=?", (eid,)).fetchone()
        origin_kind = (origin_row["origin_kind"] if origin_row else None) or "self"
        status = d.get("epistemic_status") or epistemic.entry_status(origin_kind)
        cur.execute(
            """INSERT OR REPLACE INTO structured
               (entry_id,date,time_local,type,mood,people,places,projects,topics,events,questions,decisions,learnings,tags,category,importance,confidence,inferred_by,needs_review,reason,is_sensitive,epistemic_status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                eid, sstr(d.get('date')), sstr(d.get('time_local')), sstr(d.get('type')), sstr(d.get('mood')),
                jsonl(d.get('people')), jsonl(d.get('places')), jsonl(d.get('projects')), jsonl(d.get('topics')),
                jsonl(d.get('events')), jsonl(d.get('questions')), jsonl(d.get('decisions')), jsonl(d.get('learnings')), jsonl(d.get('tags')),
                sstr(d.get('category')), sstr(d.get('importance')) or 'MEDIUM', fflt(d.get('confidence')), sstr(inferred_by) or 'heuristic', fint(d.get('needs_review')), sstr(d.get('reason')), fint(d.get('is_sensitive')), sstr(status)
            ),
        )
        self.conn.commit()

    def search_fts(self, q: str, limit: int = 10, *, include_external: bool = True,
                   external_provider: str | None = None, external_only: bool = False,
                   external_date_start: str | None = None, external_date_end: str | None = None,
                   external_sender: str | None = None) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        if external_only:
            include_external = True
        import re
        terms = re.findall(r'[A-Za-z0-9]+', q or '')
        # When external records are excluded, the user's own store is a small
        # slice of a large external-dominant store. Two cheap changes keep the
        # query fast there:
        #   * drop the snippet() aux column: with a non-FTS predicate ahead of
        #     ORDER BY rank, FTS5 cannot push the LIMIT down, so without this
        #     it would materialise and snippet over every matched row;
        #   * restrict the MATCH to the own rowids on the FTS table instead of
        #     filtering a joined column after ranking, and bound the LIKE
        #     fallback the same way instead of scanning all rows.
        if not include_external:
            fts_sql = _SQL_FTS.replace(
                "snippet(entries_fts, -1, '«', '»', '…', 16)",
                "''",
            ).replace(
                "ORDER BY rank LIMIT ?",
                "AND f.entry_id IN (SELECT id FROM entries "
                "WHERE COALESCE(origin_kind, '') <> 'external_record') "
                "ORDER BY rank LIMIT ?",
            )
        else:
            fts_sql = _SQL_FTS
        like_sql = _SQL_LIKE if include_external else _SQL_LIKE.replace(
            "AND e.text_raw LIKE ? LIMIT ?",
            "AND e.id IN (SELECT id FROM entries WHERE COALESCE(origin_kind, '') <> 'external_record') "
            "AND COALESCE(e.origin_kind, '') <> 'external_record' AND e.text_raw LIKE ? LIMIT ?",
        )
        has_external_constraints = bool(external_provider or external_only or external_date_start or external_date_end or external_sender)
        external_sql, external_params = _external_predicates(
            provider=external_provider, start=external_date_start, end=external_date_end, sender=external_sender
        )
        if has_external_constraints:
            exists = f"EXISTS (SELECT 1 FROM external_source_records x WHERE x.current_entry_id=e.id AND {external_sql})"
            clause = (f"AND COALESCE(e.source_type, '')='external_source' AND {exists} " if external_only
                      else f"AND (COALESCE(e.source_type, '') <> 'external_source' OR {exists}) ")
            fts_sql = fts_sql.replace("ORDER BY rank LIMIT ?", clause + "ORDER BY rank LIMIT ?")
            like_sql = like_sql.replace("AND e.text_raw LIKE ? LIMIT ?", clause + "AND e.text_raw LIKE ? LIMIT ?")

        def with_sender_candidates(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            # Explicit sender questions can match header metadata even when the
            # sender name never appears in the email body indexed by FTS.
            if not (external_only and external_sender):
                return rows
            cur.execute(
                f"SELECT {_ENTRY_COLUMNS}, s.epistemic_status, s.importance, s.category, "
                "s.inferred_by, s.confidence, '' AS snippet, -4.0 AS fts_rank "
                "FROM entries e JOIN external_source_records x ON x.current_entry_id=e.id "
                "LEFT JOIN structured s ON s.entry_id=e.id "
                f"WHERE e.source_type='external_source' AND {external_sql} "
                "ORDER BY x.occurred_at_utc DESC LIMIT ?",
                (*external_params, limit),
            )
            rows.extend(cur.fetchall())
            return rows

        def _fts(match: str) -> List[Dict[str, Any]]:
            try:
                params = (match, *external_params, limit) if has_external_constraints else (match, limit)
                cur.execute(fts_sql, params)
                return with_sender_candidates(cur.fetchall())
            except Exception:
                return []

        if terms:
            # try phrase
            rows = _fts('"' + ' '.join(terms) + '"')
            if rows:
                return rows
            # try OR
            rows = _fts(' OR '.join(terms))
            if rows:
                return rows
            # LIKE fallback
            params = (*external_params, f"%{q}%", limit) if has_external_constraints else (f"%{q}%", limit)
            cur.execute(like_sql, params)
            return with_sender_candidates(cur.fetchall())
        # no terms
        rows = _fts(q)
        if rows:
            return rows
        params = (*external_params, f"%{q}%", limit) if has_external_constraints else (f"%{q}%", limit)
        cur.execute(like_sql, params)
        return with_sender_candidates(cur.fetchall())

    def search_structured(self, q: str, limit: int = 20, *, include_external: bool = True,
                          external_provider: str | None = None, external_only: bool = False,
                          external_date_start: str | None = None, external_date_end: str | None = None,
                          external_sender: str | None = None) -> List[Dict[str, Any]]:
        """Match a query against extracted facets (people, places, projects, topics,
        tags) as well as the category.

        Facets are what make 'who do I work for' findable: the employer lives in
        `people`, and the question itself never says the word 'employer'.

        List facets are matched element-by-element, not by substring: they are
        stored as JSON arrays, so matching '%car%' against the stored blob would
        also hit 'career transition'.
        """
        term = (q or "").strip()
        if not term:
            return []
        if external_only:
            include_external = True
        elem = f'%"' + term.replace('"', '') + '"%'
        cur = self.conn.cursor()
        external_sql, external_params = _external_predicates(
            provider=external_provider, start=external_date_start, end=external_date_end, sender=external_sender
        )
        # Entry-side validity is asserted with a correlated EXISTS keyed on the
        # tiny `structured` table instead of a bare join. On a large store the
        # planner can otherwise choose to SCAN `entries` (every row, every
        # call), which made recall take tens of seconds; driving from
        # `structured` keeps every entry access on the primary key.
        external_exists = (f"EXISTS (SELECT 1 FROM external_source_records x "
                           f"WHERE x.current_entry_id = e2.id AND {external_sql})")
        if external_only:
            entry_filter = f"COALESCE(e2.source_type, '') = 'external_source' AND {external_exists}"
        else:
            entry_filter = f"(COALESCE(e2.source_type, '') <> 'external_source' OR {external_exists})"
        if not include_external:
            entry_filter += " AND COALESCE(e2.origin_kind, '') <> 'external_record'"
        entry_ok = (
            "EXISTS (SELECT 1 FROM entries e2 WHERE e2.id = s.entry_id "
            "AND COALESCE(e2.source_type, '') NOT IN "
            "('conversation_candidate', 'canonical_memory_seed') "
            f"AND {entry_filter})"
        )
        cur.execute(
            f"SELECT {_ENTRY_COLUMNS}, s.epistemic_status, s.importance, s.category, "
            "s.inferred_by, s.confidence, '' as snippet, 0 as fts_rank "
            "FROM structured s JOIN entries e ON e.id = s.entry_id "
            "WHERE s.entry_id IN ("
            "SELECT t.entry_id FROM structured t "
            "WHERE t.category = ? OR t.people LIKE ? OR t.places LIKE ? OR t.projects LIKE ? "
            "OR t.topics LIKE ? OR t.tags LIKE ? "
            "OR t.decisions LIKE ? OR t.learnings LIKE ?) "
            f"AND {entry_ok} LIMIT ?",
            (term, elem, elem, elem, elem, elem, elem, elem)
            + tuple(external_params)
            + (limit,),
        )
        return cur.fetchall()

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id,uuid,created_at_utc,source,origin_kind,raw_relpath,substr(text_raw,1,120) as snippet "
            "FROM entries e WHERE COALESCE(source_type, '') NOT IN "
            "('conversation_candidate', 'canonical_memory_seed') "
            "AND (COALESCE(source_type, '') <> 'external_source' OR EXISTS "
            "(SELECT 1 FROM external_source_records x WHERE x.current_entry_id=e.id AND x.deleted_at_utc IS NULL)) "
            "ORDER BY id DESC LIMIT ?", (limit,)
        )
        return cur.fetchall()

    def get(self, eid: int) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM entries WHERE id=?", (eid,))
        e = cur.fetchone()
        if not e:
            return None
        cur.execute("SELECT * FROM structured WHERE entry_id=?", (eid,))
        s = cur.fetchone()
        cur.execute("SELECT * FROM relations WHERE from_entry_id=? OR to_entry_id=?", (eid,eid))
        r = cur.fetchall()
        return {"entry": e, "structured": s, "relations": r}

    def stats(self) -> Dict[str, Any]:
        cur = self.conn.cursor()
        def c(sql, *p):
            cur.execute(sql, p)
            r = cur.fetchone()
            return list(r.values())[0] if r else 0
        return {
            "entries": c("SELECT COUNT(*) FROM entries"),
            "self": c("SELECT COUNT(*) FROM entries WHERE origin_kind='self'"),
            "external": c("SELECT COUNT(*) FROM entries WHERE origin_kind='external_record'"),
            "cli": c("SELECT COUNT(*) FROM entries WHERE source='cli'"),
            "gmail": c("SELECT COUNT(*) FROM entries WHERE source='gmail'"),
            "calendar": c("SELECT COUNT(*) FROM entries WHERE source='calendar'"),
            "takeout": c("SELECT COUNT(*) FROM entries WHERE source='takeout'"),
            "linkedin": c("SELECT COUNT(*) FROM entries WHERE source='linkedin'"),
        }

    def add_relation(self, a: int, b: int, relation: str = 'related_to', confidence: float = 0.6, inferred_by: str = 'heuristic', rationale: str | None = None) -> int:
        from datetime import datetime, timezone
        cu = datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
        cur = self.conn.cursor()
        cur.execute("""INSERT OR IGNORE INTO relations (from_entry_id,to_entry_id,relation,confidence,inferred_by,created_at_utc,rationale) VALUES (?,?,?,?,?,?,?)""", (a,b,relation,confidence,inferred_by,cu,rationale))
        self.conn.commit()
        return cur.lastrowid

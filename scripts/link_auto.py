#!/usr/bin/env python3
from pathlib import Path
from app.core import storage, relations

base = Path(__file__).resolve().parent.parent
dbp = base / 'data' / 'jarvis.db'
db = storage.JarvisDB(dbp)
cur = db.conn.cursor()
cur.execute("SELECT * FROM entries ORDER BY id DESC LIMIT 50")
rows = cur.fetchall()
linked = 0
for r in rows:
    rels = relations.find_related(db, r, limit=10)
    for rid, rel, conf in rels:
        if rid < r['id']:
            db.add_relation(rid, r['id'], relation=rel, confidence=conf, rationale='company cross-link')
            linked += 1
db.close()
print('linked', linked)

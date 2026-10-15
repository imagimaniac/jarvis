"""Lightweight cross-linking between entries."""
from __future__ import annotations
import re
from typing import Any, Dict, List, Tuple

COMPANY_RE = re.compile(r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\b')

def extract_companies(text: str) -> List[str]:
    t = text or ''
    cands = set()
    for m in COMPANY_RE.findall(t):
        nm = m.strip()
        if len(nm) > 1 and not nm.lower() in ('I','We','You','They','The','This','That','And','But','For'):
            cands.add(nm)
    return sorted(cands)

def find_related(db, entry: Dict[str, Any], limit: int = 20) -> List[Tuple[int, str, float]]:
    text = entry.get('text_raw') or ''
    comps = extract_companies(text)
    if not comps:
        return []
    cur = db.conn.cursor()
    qparts = []
    params = []
    for c in comps:
        qparts.append('e.text_raw LIKE ?')
        params.append(f'%{c}%')
    sql = f"""SELECT e.id, e.source, e.origin_kind, e.text_raw FROM entries e WHERE ({' OR '.join(qparts)}) AND e.id != ? ORDER BY e.id DESC LIMIT ?"""
    params.extend([entry['id'], limit])
    cur.execute(sql, params)
    rows = cur.fetchall()
    out = []
    for r in rows:
        out.append((r['id'], f'refs_company', 0.6))
    return out

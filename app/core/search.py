"""Search: FTS5 + hybrid ranking (recency + type + self-boost)."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        # handle Z
        if s.endswith('Z'):
            s2 = s.replace('Z', '+00:00')
            return datetime.fromisoformat(s2)
        return datetime.fromisoformat(s)
    except Exception:
        try:
            return datetime.fromisoformat((s or '')[:19])
        except Exception:
            return None


def rank_results(results: List[Dict[str, Any]], *, q: str = "", self_boost: float = 0.25, recency_boost: float = 0.15) -> List[Dict[str, Any]]:
    out = []
    # Naive UTC so it compares against _parse_dt() results after tzinfo is
    # stripped below; the aware->naive conversion avoids datetime.utcnow().
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for r in results:
        score = float(r.get('fts_rank') or 0.0)
        # FTS5 rank is smaller better; adjust with boosts
        if r.get('origin_kind') == 'self':
            score -= self_boost
        dt = _parse_dt(r.get('created_at_utc'))
        if dt:
            try:
                # seconds ago
                delta = (now - dt.replace(tzinfo=None)).total_seconds()
                if delta < 86400:
                    score -= recency_boost * 2.0
                elif delta < 7 * 86400:
                    score -= recency_boost
                elif delta < 30 * 86400:
                    score -= recency_boost * 0.5
            except Exception:
                pass
        r2 = dict(r)
        r2['score'] = score
        out.append(r2)
    return sorted(out, key=lambda x: x['score'])
